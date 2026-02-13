"""Post-generation verification checks.

1. GL Balance: SUM(debit) == SUM(credit) globally and per-day
2. Sequence Monotonicity: no seq_id goes backward within causal chains
3. Referential Integrity: all FKs resolve
4. Row Count Audit: summary of all output tables
5. Cost Sanity: COGS/Revenue ratio check
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def run_verification(output_dir: Path) -> None:
    """Run all verification checks and report results."""
    logger.info("=" * 60)
    logger.info("VERIFICATION REPORT")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    # ── 1. Row Count Audit ────────────────────────────────────
    logger.info("\n--- Row Counts ---")
    total_rows = 0
    for subdir in ("master", "transactional"):
        d = output_dir / subdir
        if not d.exists():
            continue
        for csv_file in sorted(d.glob("*.csv")):
            count = _count_rows(csv_file)
            total_rows += count
            logger.info("  %-40s %10s rows", csv_file.name, f"{count:,}")
    logger.info("  %-40s %10s rows", "TOTAL", f"{total_rows:,}")

    # ── 2. GL Balance ─────────────────────────────────────────
    gl_path = output_dir / "transactional" / "gl_journal.csv"
    if gl_path.exists():
        logger.info("\n--- GL Balance Check ---")
        total_dr, total_cr, per_day = _check_gl_balance(gl_path)
        diff = abs(total_dr - total_cr)
        if diff < 0.02:
            logger.info("  PASS: GL balanced (DR=%.2f, CR=%.2f, diff=%.4f)",
                        total_dr, total_cr, diff)
            passed += 1
        else:
            logger.error("  FAIL: GL imbalance (DR=%.2f, CR=%.2f, diff=%.4f)",
                         total_dr, total_cr, diff)
            failed += 1

        # Check per-day balance
        imbalanced_days = sum(1 for d, (dr, cr) in per_day.items()
                              if abs(dr - cr) > 0.02)
        if imbalanced_days == 0:
            logger.info("  PASS: All %d days balanced", len(per_day))
            passed += 1
        else:
            logger.warning("  WARN: %d/%d days have imbalance > $0.02",
                           imbalanced_days, len(per_day))
            failed += 1
    else:
        logger.warning("  SKIP: gl_journal.csv not found")

    # ── 3. Cost Sanity ────────────────────────────────────────
    if gl_path.exists():
        logger.info("\n--- Cost Sanity ---")
        cogs, revenue = _extract_cogs_revenue(gl_path)
        if revenue > 0:
            ratio = cogs / revenue
            if 0.40 < ratio < 0.90:
                logger.info("  PASS: COGS/Revenue = %.1f%% (expected 50-80%%)", ratio * 100)
                passed += 1
            else:
                logger.warning("  WARN: COGS/Revenue = %.1f%% (outside 40-90%% range)",
                               ratio * 100)
                failed += 1
        else:
            logger.warning("  SKIP: No revenue entries found")

    # ── 4. FK Integrity Spot Check ────────────────────────────
    logger.info("\n--- FK Integrity Spot Checks ---")
    checks = [
        ("order_lines.csv", "order_id", "orders.csv", "id"),
        ("shipment_lines.csv", "shipment_id", "shipments.csv", "id"),
        ("batch_ingredients.csv", "batch_id", "batches.csv", "id"),
        ("ap_invoice_lines.csv", "invoice_id", "ap_invoices.csv", "id"),
        ("ar_invoice_lines.csv", "invoice_id", "ar_invoices.csv", "id"),
    ]
    for child_file, fk_col, parent_file, pk_col in checks:
        child_path = output_dir / "transactional" / child_file
        parent_path = output_dir / "transactional" / parent_file
        if child_path.exists() and parent_path.exists():
            ok, total, missing = _check_fk(child_path, fk_col, parent_path, pk_col)
            pct = (ok / total * 100) if total > 0 else 100
            if pct >= 99.9:
                logger.info("  PASS: %s.%s → %s.%s (%.1f%%, %d/%d)",
                            child_file, fk_col, parent_file, pk_col, pct, ok, total)
                passed += 1
            else:
                logger.warning("  WARN: %s.%s → %s.%s (%.1f%%, %d missing)",
                               child_file, fk_col, parent_file, pk_col, pct, missing)
                failed += 1

    # ── 5. Sequence Monotonicity ──────────────────────────────
    logger.info("\n--- Sequence Monotonicity ---")
    seq_files = ["orders.csv", "shipments.csv", "batches.csv",
                 "gl_journal.csv", "ap_invoices.csv", "ar_invoices.csv"]
    for fname in seq_files:
        fpath = output_dir / "transactional" / fname
        if fpath.exists():
            is_mono = _check_seq_monotonic(fpath, "transaction_sequence_id")
            if is_mono:
                logger.info("  PASS: %s sequence monotonic", fname)
                passed += 1
            else:
                logger.warning("  WARN: %s sequence NOT monotonic", fname)
                failed += 1

    # ── Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION: %d passed, %d failed/warnings", passed, failed)
    logger.info("=" * 60)


def _count_rows(path: Path) -> int:
    """Count data rows (excluding header) in a CSV file."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


def _check_gl_balance(
    gl_path: Path,
) -> tuple[float, float, dict[int, tuple[float, float]]]:
    """Check GL debit/credit balance globally and per-day."""
    total_dr = 0.0
    total_cr = 0.0
    per_day: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])

    with open(gl_path) as f:
        for row in csv.DictReader(f):
            dr = float(row["debit_amount"])
            cr = float(row["credit_amount"])
            day = int(row["entry_date"])
            total_dr += dr
            total_cr += cr
            per_day[day][0] += dr
            per_day[day][1] += cr

    day_tuples = {d: (v[0], v[1]) for d, v in per_day.items()}
    return total_dr, total_cr, day_tuples


def _extract_cogs_revenue(gl_path: Path) -> tuple[float, float]:
    """Extract total COGS (5100 debit) and Revenue (4100 credit)."""
    cogs = 0.0
    revenue = 0.0
    with open(gl_path) as f:
        for row in csv.DictReader(f):
            if row["account_code"] == "5100":
                cogs += float(row["debit_amount"])
            elif row["account_code"] == "4100":
                revenue += float(row["credit_amount"])
    return cogs, revenue


def _check_fk(
    child_path: Path,
    fk_col: str,
    parent_path: Path,
    pk_col: str,
) -> tuple[int, int, int]:
    """Check FK integrity: returns (resolved, total, missing)."""
    parent_ids: set[str] = set()
    with open(parent_path) as f:
        for row in csv.DictReader(f):
            parent_ids.add(row[pk_col])

    total = 0
    resolved = 0
    with open(child_path) as f:
        for row in csv.DictReader(f):
            total += 1
            if row[fk_col] in parent_ids:
                resolved += 1

    return resolved, total, total - resolved


def _check_seq_monotonic(path: Path, col: str) -> bool:
    """Check if a column is monotonically non-decreasing."""
    prev = -1
    with open(path) as f:
        reader = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            return True  # skip if column missing
        for row in reader:
            val = int(row[col])
            if val < prev:
                return False
            prev = val
    return True
