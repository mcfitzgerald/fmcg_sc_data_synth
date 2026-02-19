"""Post-generation verification checks.

1. GL Balance: SUM(debit) == SUM(credit) globally and per-day
2. Sequence Monotonicity: no seq_id goes backward within causal chains
3. Referential Integrity: all FKs resolve
4. Row Count Audit: summary of all output tables
5. Cost Sanity: COGS/Revenue ratio check
6. Friction checks (when friction tables present)

GL checks use DuckDB for performance (~47M+ rows in seconds vs minutes).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import duckdb

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

    # ── 2. GL Balance (DuckDB for speed on ~47M+ rows) ────────
    gl_path = output_dir / "transactional" / "gl_journal.csv"
    if gl_path.exists():
        logger.info("\n--- GL Balance Check ---")
        gl_db = duckdb.connect()
        gl_db.execute(f"""
            CREATE TABLE gl AS SELECT * FROM read_csv_auto('{gl_path}')
        """)

        # Global balance
        total_dr, total_cr = gl_db.execute("""
            SELECT SUM(debit_amount), SUM(credit_amount) FROM gl
        """).fetchone()
        diff = abs(total_dr - total_cr)
        # Tolerance: ~58M rows × ROUND(...,4) → cumulative float drift up to ~$5
        if diff < 5.0:
            logger.info("  PASS: GL balanced (DR=%.2f, CR=%.2f, diff=%.4f)",
                        total_dr, total_cr, diff)
            passed += 1
        else:
            logger.error("  FAIL: GL imbalance (DR=%.2f, CR=%.2f, diff=%.4f)",
                         total_dr, total_cr, diff)
            failed += 1

        # Per-day balance (wider tolerance for friction GL entries)
        day_count, imbalanced_days = gl_db.execute("""
            SELECT COUNT(*), SUM(CASE WHEN ABS(dr - cr) > 0.10 THEN 1 ELSE 0 END)
            FROM (
                SELECT entry_date, SUM(debit_amount) as dr, SUM(credit_amount) as cr
                FROM gl GROUP BY entry_date
            )
        """).fetchone()
        if imbalanced_days == 0:
            logger.info("  PASS: All %d days balanced", day_count)
            passed += 1
        else:
            logger.warning("  WARN: %d/%d days have imbalance > $0.10",
                           imbalanced_days, day_count)
            failed += 1

        # ── 3. Cost Sanity ────────────────────────────────────
        logger.info("\n--- Cost Sanity ---")
        cogs, revenue = gl_db.execute("""
            SELECT
                SUM(CASE WHEN account_code = '5100' THEN debit_amount ELSE 0 END),
                SUM(CASE WHEN account_code = '4100' THEN credit_amount ELSE 0 END)
            FROM gl
        """).fetchone()
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

        # Reference ID population check
        logger.info("\n--- Reference ID Coverage ---")
        ref_stats = gl_db.execute("""
            SELECT reference_type,
                   COUNT(*) as total,
                   COUNT(CASE WHEN reference_id != '' AND reference_id IS NOT NULL
                              THEN 1 END) as has_ref
            FROM gl GROUP BY reference_type ORDER BY reference_type
        """).fetchall()
        for ref_type, total, has_ref in ref_stats:
            pct = (has_ref / total * 100) if total > 0 else 0
            logger.info("  %-20s %10s rows, %5.1f%% with reference_id",
                        ref_type, f"{total:,}", pct)

        # node_id coverage by reference_type
        # Physical events should have ~99% coverage (1% friction nulls).
        # Treasury events (payment, receipt, bad_debt) correctly have 0%.
        logger.info("\n--- node_id Coverage ---")
        node_stats = gl_db.execute("""
            SELECT reference_type,
                   COUNT(*) as total,
                   COUNT(CASE WHEN node_id IS NULL OR TRIM(node_id, '"') = ''
                              THEN 1 END) as empty,
                   ROUND(COUNT(CASE WHEN node_id IS NOT NULL
                               AND TRIM(node_id, '"') != ''
                               THEN 1 END)::numeric / COUNT(*) * 100, 1) as pct_filled
            FROM gl GROUP BY reference_type ORDER BY reference_type
        """).fetchall()

        # Physical event types should have node_id; treasury types shouldn't
        physical_types = {
            "goods_receipt", "production", "shipment", "freight",
            "sale", "return", "price_variance", "qty_variance",
        }
        treasury_types = {"payment", "receipt", "bad_debt"}

        for ref_type, total, empty, pct_filled in node_stats:
            status = ""
            if ref_type in physical_types:
                if pct_filled >= 95.0:
                    status = "OK"
                    passed += 1
                else:
                    status = "LOW"
                    failed += 1
            elif ref_type in treasury_types:
                if pct_filled <= 1.0:
                    status = "OK (treasury)"
                    passed += 1
                else:
                    status = "UNEXPECTED"
                    failed += 1
            else:
                status = "?"

            logger.info("  %-20s %10s rows, %5.1f%% filled  [%s]",
                        ref_type, f"{total:,}", pct_filled, status)

        gl_db.close()
    else:
        logger.warning("  SKIP: gl_journal.csv not found")

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
                 "ap_invoices.csv", "ar_invoices.csv"]
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

    # GL monotonicity via DuckDB (too large for Python row-by-row)
    if gl_path.exists():
        is_mono = _check_seq_monotonic_duckdb(gl_path, "transaction_sequence_id")
        if is_mono:
            logger.info("  PASS: gl_journal.csv sequence monotonic")
            passed += 1
        else:
            logger.warning("  WARN: gl_journal.csv sequence NOT monotonic")
            failed += 1

    # ── 6. Friction Table Checks ──────────────────────────────
    _check_friction_tables(output_dir, passed, failed)

    # ── Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION: %d passed, %d failed/warnings", passed, failed)
    logger.info("=" * 60)


def _check_friction_tables(output_dir: Path, passed: int, failed: int) -> None:
    """Verify friction-specific tables if they exist."""
    trans_dir = output_dir / "transactional"

    iv_path = trans_dir / "invoice_variances.csv"
    ap_pay_path = trans_dir / "ap_payments.csv"
    ar_rec_path = trans_dir / "ar_receipts.csv"

    has_friction = iv_path.exists() or ap_pay_path.exists() or ar_rec_path.exists()
    if not has_friction:
        return

    logger.info("\n--- Friction Tables ---")

    if iv_path.exists():
        count = _count_rows(iv_path)
        logger.info("  invoice_variances: %s rows", f"{count:,}")

    if ap_pay_path.exists():
        count = _count_rows(ap_pay_path)
        ap_count = _count_rows(trans_dir / "ap_invoices.csv") if (
            trans_dir / "ap_invoices.csv"
        ).exists() else 0
        logger.info("  ap_payments: %s rows (AP invoices: %s)",
                    f"{count:,}", f"{ap_count:,}")

    if ar_rec_path.exists():
        count = _count_rows(ar_rec_path)
        ar_count = _count_rows(trans_dir / "ar_invoices.csv") if (
            trans_dir / "ar_invoices.csv"
        ).exists() else 0
        logger.info("  ar_receipts: %s rows (AR invoices: %s)",
                    f"{count:,}", f"{ar_count:,}")

    # Duplicate invoices should have line items
    ai_path = trans_dir / "ap_invoices.csv"
    ail_path = trans_dir / "ap_invoice_lines.csv"
    if ai_path.exists() and ail_path.exists():
        fdb = duckdb.connect()
        fdb.execute(f"CREATE TABLE ai AS SELECT * FROM read_csv_auto('{ai_path}')")
        fdb.execute(f"CREATE TABLE ail AS SELECT * FROM read_csv_auto('{ail_path}')")
        dup_total, dup_with_lines = fdb.execute("""
            SELECT
                COUNT(DISTINCT ai.id),
                COUNT(DISTINCT CASE WHEN ail.line_number IS NOT NULL THEN ai.id END)
            FROM ai
            LEFT JOIN ail ON ail.invoice_id = ai.id
            WHERE ai.invoice_number LIKE '%-DUP'
        """).fetchone()
        fdb.close()
        if dup_total > 0:
            if dup_with_lines == dup_total:
                logger.info("  PASS: All %s dup invoices have line items",
                            f"{dup_total:,}")
            else:
                logger.warning("  WARN: %s/%s dup invoices missing line items",
                               f"{dup_total - dup_with_lines:,}", f"{dup_total:,}")


def _count_rows(path: Path) -> int:
    """Count data rows (excluding header) in a CSV file."""
    with open(path) as f:
        return sum(1 for _ in f) - 1



def _check_fk(
    child_path: Path,
    fk_col: str,
    parent_path: Path,
    pk_col: str,
) -> tuple[int, int, int]:
    """Check FK integrity: returns (resolved, total, missing).

    Skips rows with empty/null FK values (friction may null FKs).
    """
    parent_ids: set[str] = set()
    with open(parent_path) as f:
        for row in csv.DictReader(f):
            parent_ids.add(row[pk_col])

    total = 0
    resolved = 0
    with open(child_path) as f:
        for row in csv.DictReader(f):
            fk_val = row[fk_col].strip()
            if not fk_val or fk_val.lower() == "null":
                continue  # skip null FKs (friction)
            total += 1
            if fk_val in parent_ids:
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


def _check_seq_monotonic_duckdb(path: Path, col: str) -> bool:
    """Check sequence monotonicity via DuckDB (for large CSVs).

    Uses the ``id`` column as physical row ordering proxy since DuckDB's
    ``read_csv_auto`` doesn't expose a ``rowid`` pseudo-column.
    """
    db = duckdb.connect()
    try:
        violations = db.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT {col},
                       LAG({col}) OVER (ORDER BY id) as prev_val
                FROM read_csv_auto('{path}')
            ) WHERE {col} < prev_val
        """).fetchone()[0]
        return violations == 0
    finally:
        db.close()
