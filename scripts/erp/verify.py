"""Post-generation verification checks.

1. GL Balance: SUM(debit) == SUM(credit) globally and per-day
2. Sequence Monotonicity: no seq_id goes backward within causal chains
3. Referential Integrity: all FKs resolve
4. Row Count Audit: summary of all output tables
5. Cost Sanity: COGS/Revenue ratio check
6. Friction checks (when friction tables present)

Supports two modes:
  - DuckDB mode (db parameter): queries in-memory tables directly (~seconds)
  - CSV mode (fallback): reads from disk (~80s for GL journal alone)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# DuckDB table name → schema table name
_TABLE_ALIASES = {
    "erp_orders": "orders",
    "erp_order_lines": "order_lines",
    "erp_shipments": "shipments",
    "erp_shipment_lines": "shipment_lines",
    "erp_batches": "batches",
    "erp_batch_ingredients": "batch_ingredients",
    "erp_ap_invoices": "ap_invoices",
    "erp_ap_invoice_lines": "ap_invoice_lines",
    "erp_ar_invoices": "ar_invoices",
    "erp_ar_invoice_lines": "ar_invoice_lines",
    "erp_gl_journal": "gl_journal",
    "erp_goods_receipts": "goods_receipts",
    "erp_gr_lines": "goods_receipt_lines",
    "erp_pos": "purchase_orders",
    "erp_po_lines": "purchase_order_lines",
    "erp_inventory": "inventory",
    "erp_work_orders": "work_orders",
    "erp_returns": "returns",
    "erp_return_lines": "return_lines",
    "erp_disposition_logs": "disposition_logs",
    "erp_demand_forecasts": "demand_forecasts",
    "erp_invoice_variances": "invoice_variances",
    "erp_ap_payments": "ap_payments",
    "erp_ar_receipts": "ar_receipts",
}


def run_verification(
    output_dir: Path,
    *,
    db: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Run all verification checks and report results.

    When ``db`` is provided, queries DuckDB tables directly (fast).
    Otherwise falls back to reading CSV files from ``output_dir``.
    """
    logger.info("=" * 60)
    logger.info("VERIFICATION REPORT")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    if db is not None:
        p, f = _verify_duckdb(db, output_dir)
    else:
        p, f = _verify_csv(output_dir)

    passed += p
    failed += f

    # ── Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION: %d passed, %d failed/warnings", passed, failed)
    logger.info("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DuckDB Mode — queries in-memory tables directly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _verify_duckdb(
    db: duckdb.DuckDBPyConnection, output_dir: Path,
) -> tuple[int, int]:
    """Run all checks against DuckDB in-memory tables."""
    passed = 0
    failed = 0

    # ── 0. Row Count Audit ─────────────────────────────────────
    logger.info("\n--- Row Counts (DuckDB) ---")
    total_rows = 0

    # Master tables from CSV (not in DuckDB)
    master_dir = output_dir / "master"
    if master_dir.exists():
        for csv_file in sorted(master_dir.glob("*.csv")):
            count = _count_rows(csv_file)
            total_rows += count
            logger.info("  %-40s %10s rows", csv_file.name, f"{count:,}")

    # Transactional tables from DuckDB
    for ddb_name, schema_name in sorted(_TABLE_ALIASES.items(), key=lambda x: x[1]):
        try:
            count = db.execute(f"SELECT COUNT(*) FROM {ddb_name}").fetchone()[0]
            total_rows += count
            logger.info("  %-40s %10s rows", schema_name, f"{count:,}")
        except duckdb.CatalogException:
            pass  # table doesn't exist (e.g. friction tables when friction disabled)

    logger.info("  %-40s %10s rows", "TOTAL", f"{total_rows:,}")

    # ── 1. Schema Validation ──────────────────────────────────
    from .schema import TABLE_MAP, get_column_names

    logger.info("\n--- Schema Validation (DuckDB columns vs schema.py) ---")
    schema_errors = 0
    for ddb_name, schema_name in _TABLE_ALIASES.items():
        if schema_name not in TABLE_MAP:
            continue
        try:
            cols_result = db.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{ddb_name}' ORDER BY ordinal_position"
            ).fetchall()
        except Exception:
            continue
        if not cols_result:
            continue

        ddb_cols = {r[0] for r in cols_result}
        expected_cols = set(get_column_names(schema_name))
        missing = expected_cols - ddb_cols
        if missing:
            logger.warning("  FAIL: %s — missing columns: %s", schema_name, missing)
            schema_errors += 1
            failed += 1
        else:
            passed += 1

    if schema_errors == 0:
        logger.info("  PASS: All DuckDB tables have required schema columns")

    # ── 2. GL Balance ──────────────────────────────────────────
    gl_table = "erp_gl_journal"
    try:
        db.execute(f"SELECT 1 FROM {gl_table} LIMIT 1")
        has_gl = True
    except duckdb.CatalogException:
        has_gl = False

    if has_gl:
        logger.info("\n--- GL Balance Check ---")
        total_dr, total_cr = db.execute(f"""
            SELECT SUM(debit_amount), SUM(credit_amount) FROM {gl_table}
        """).fetchone()
        diff = abs(total_dr - total_cr)
        if diff < 10.0:
            logger.info("  PASS: GL balanced (DR=%.2f, CR=%.2f, diff=%.4f)",
                        total_dr, total_cr, diff)
            passed += 1
        else:
            logger.error("  FAIL: GL imbalance (DR=%.2f, CR=%.2f, diff=%.4f)",
                         total_dr, total_cr, diff)
            failed += 1

        # Per-day balance
        day_count, imbalanced_days = db.execute(f"""
            SELECT COUNT(*), SUM(CASE WHEN ABS(dr - cr) > 1.10 THEN 1 ELSE 0 END)
            FROM (
                SELECT entry_date, SUM(debit_amount) as dr, SUM(credit_amount) as cr
                FROM {gl_table} GROUP BY entry_date
            )
        """).fetchone()
        if imbalanced_days == 0:
            logger.info("  PASS: All %d days balanced (tolerance $1.10)", day_count)
            passed += 1
        else:
            logger.warning("  WARN: %d/%d days have imbalance > $1.10",
                           imbalanced_days, day_count)
            failed += 1

        # GL duplicate detection
        dup_count = db.execute(f"""
            SELECT COUNT(*) FROM {gl_table} WHERE reference_id LIKE '%%-DUP'
        """).fetchone()[0]
        if dup_count > 0:
            logger.info("  INFO: %s GL entries are friction duplicates (-DUP)",
                        f"{dup_count:,}")
        else:
            logger.info("  INFO: No GL duplicate entries found")

        # ── 3. Cost Sanity ─────────────────────────────────────
        logger.info("\n--- Cost Sanity ---")
        cogs, revenue = db.execute(f"""
            SELECT
                SUM(CASE WHEN account_code = '5100' THEN debit_amount ELSE 0 END),
                SUM(CASE WHEN account_code = '4100' THEN credit_amount ELSE 0 END)
            FROM {gl_table}
        """).fetchone()
        if revenue and revenue > 0:
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

        # Reference ID coverage
        logger.info("\n--- Reference ID Coverage ---")
        ref_stats = db.execute(f"""
            SELECT reference_type,
                   COUNT(*) as total,
                   COUNT(CASE WHEN reference_id != '' AND reference_id IS NOT NULL
                              THEN 1 END) as has_ref
            FROM {gl_table} GROUP BY reference_type ORDER BY reference_type
        """).fetchall()
        for ref_type, total, has_ref in ref_stats:
            pct = (has_ref / total * 100) if total > 0 else 0
            logger.info("  %-20s %10s rows, %5.1f%% with reference_id",
                        ref_type, f"{total:,}", pct)

        # node_id coverage
        logger.info("\n--- node_id Coverage ---")
        node_stats = db.execute(f"""
            SELECT reference_type,
                   COUNT(*) as total,
                   COUNT(CASE WHEN node_id IS NULL OR TRIM(node_id, '"') = ''
                              THEN 1 END) as empty,
                   ROUND(COUNT(CASE WHEN node_id IS NOT NULL
                               AND TRIM(node_id, '"') != ''
                               THEN 1 END)::numeric / COUNT(*) * 100, 1) as pct_filled
            FROM {gl_table} GROUP BY reference_type ORDER BY reference_type
        """).fetchall()

        physical_types = {
            "goods_receipt", "production", "shipment", "freight",
            "sale", "return", "price_variance", "qty_variance",
        }
        treasury_types = {"payment", "receipt", "bad_debt"}

        for ref_type, total, _empty, pct_filled in node_stats:
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
    else:
        logger.warning("  SKIP: erp_gl_journal table not found")

    # ── 4. FK Integrity Spot Check (DuckDB) ────────────────────
    logger.info("\n--- FK Integrity Spot Checks ---")
    fk_checks = [
        ("erp_order_lines", "order_id", "erp_orders", "id"),
        ("erp_shipment_lines", "shipment_id", "erp_shipments", "id"),
        ("erp_batch_ingredients", "batch_id", "erp_batches", "id"),
        ("erp_ap_invoice_lines", "invoice_id", "erp_ap_invoices", "id"),
        ("erp_ar_invoice_lines", "invoice_id", "erp_ar_invoices", "id"),
    ]
    for child_tbl, fk_col, parent_tbl, pk_col in fk_checks:
        try:
            db.execute(f"SELECT 1 FROM {child_tbl} LIMIT 1")
            db.execute(f"SELECT 1 FROM {parent_tbl} LIMIT 1")
        except duckdb.CatalogException:
            continue

        result = db.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN p.{pk_col} IS NOT NULL THEN 1 END) as resolved
            FROM {child_tbl} c
            LEFT JOIN {parent_tbl} p ON p.{pk_col} = c.{fk_col}
            WHERE c.{fk_col} IS NOT NULL
        """).fetchone()
        total, resolved = result
        missing = total - resolved
        pct = (resolved / total * 100) if total > 0 else 100
        child_schema = _TABLE_ALIASES.get(child_tbl, child_tbl)
        parent_schema = _TABLE_ALIASES.get(parent_tbl, parent_tbl)
        if pct >= 99.9:
            logger.info("  PASS: %s.%s → %s.%s (%.1f%%, %d/%d)",
                        child_schema, fk_col, parent_schema, pk_col, pct, resolved, total)
            passed += 1
        else:
            logger.warning("  WARN: %s.%s → %s.%s (%.1f%%, %d missing)",
                           child_schema, fk_col, parent_schema, pk_col, pct, missing)
            failed += 1

    # ── 5. Sequence Monotonicity (DuckDB) ──────────────────────
    logger.info("\n--- Sequence Monotonicity ---")
    seq_tables = [
        "erp_orders", "erp_shipments", "erp_batches",
        "erp_ap_invoices", "erp_ar_invoices", "erp_gl_journal",
    ]
    for tbl in seq_tables:
        try:
            db.execute(f"SELECT 1 FROM {tbl} LIMIT 1")
        except duckdb.CatalogException:
            continue

        violations = db.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT transaction_sequence_id,
                       LAG(transaction_sequence_id) OVER (ORDER BY id) as prev_val
                FROM {tbl}
            ) WHERE transaction_sequence_id < prev_val
        """).fetchone()[0]
        schema_name = _TABLE_ALIASES.get(tbl, tbl)
        if violations == 0:
            logger.info("  PASS: %s sequence monotonic", schema_name)
            passed += 1
        else:
            logger.warning("  WARN: %s sequence NOT monotonic (%d violations)",
                           schema_name, violations)
            failed += 1

    # ── 6. Friction Table Checks (DuckDB) ──────────────────────
    _check_friction_tables_duckdb(db, passed, failed)

    return passed, failed


def _check_friction_tables_duckdb(
    db: duckdb.DuckDBPyConnection, passed: int, failed: int,
) -> None:
    """Verify friction-specific tables from DuckDB."""
    has_friction = False
    for tbl in ("erp_invoice_variances", "erp_ap_payments", "erp_ar_receipts"):
        try:
            db.execute(f"SELECT 1 FROM {tbl} LIMIT 1")
            has_friction = True
            break
        except duckdb.CatalogException:
            pass

    if not has_friction:
        return

    logger.info("\n--- Friction Tables ---")

    for ddb_name, label in [
        ("erp_invoice_variances", "invoice_variances"),
        ("erp_ap_payments", "ap_payments"),
        ("erp_ar_receipts", "ar_receipts"),
    ]:
        try:
            count = db.execute(f"SELECT COUNT(*) FROM {ddb_name}").fetchone()[0]
            logger.info("  %s: %s rows", label, f"{count:,}")
        except duckdb.CatalogException:
            pass

    # Duplicate invoices should have line items
    try:
        db.execute("SELECT 1 FROM erp_ap_invoices LIMIT 1")
        db.execute("SELECT 1 FROM erp_ap_invoice_lines LIMIT 1")

        dup_total, dup_with_lines = db.execute("""
            SELECT
                COUNT(DISTINCT ai.id),
                COUNT(DISTINCT CASE WHEN ail.line_number IS NOT NULL THEN ai.id END)
            FROM erp_ap_invoices ai
            LEFT JOIN erp_ap_invoice_lines ail ON ail.invoice_id = ai.id
            WHERE ai.invoice_number LIKE '%-DUP'
        """).fetchone()

        if dup_total > 0:
            if dup_with_lines == dup_total:
                logger.info("  PASS: All %s dup invoices have line items",
                            f"{dup_total:,}")
            else:
                logger.warning("  WARN: %s/%s dup invoices missing line items",
                               f"{dup_total - dup_with_lines:,}", f"{dup_total:,}")
    except duckdb.CatalogException:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSV Mode — fallback when no DuckDB connection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _verify_csv(output_dir: Path) -> tuple[int, int]:
    """Run all checks from CSV files on disk."""
    passed = 0
    failed = 0

    # ── 0. Schema Validation ──────────────────────────────────
    from .schema import validate_csv_headers

    logger.info("\n--- Schema Validation (CSV headers vs schema.py) ---")
    schema_errors = 0
    for subdir in ("master", "transactional"):
        d = output_dir / subdir
        if not d.exists():
            continue
        for csv_file in sorted(d.glob("*.csv")):
            table_name = csv_file.stem
            with open(csv_file) as f:
                reader = csv.reader(f)
                headers = next(reader, [])
            errors = validate_csv_headers(table_name, headers)
            if errors:
                for err in errors:
                    logger.warning("  FAIL: %s — %s", table_name, err)
                schema_errors += 1
                failed += 1
            else:
                passed += 1
    if schema_errors == 0:
        logger.info("  PASS: All CSV headers match schema definitions")

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

    # ── 2. GL Balance ──────────────────────────────────────────
    gl_path = output_dir / "transactional" / "gl_journal.csv"
    if gl_path.exists():
        logger.info("\n--- GL Balance Check ---")
        gl_db = duckdb.connect()
        gl_db.execute(f"""
            CREATE TABLE gl AS SELECT * FROM read_csv_auto('{gl_path}')
        """)

        total_dr, total_cr = gl_db.execute("""
            SELECT SUM(debit_amount), SUM(credit_amount) FROM gl
        """).fetchone()
        diff = abs(total_dr - total_cr)
        if diff < 10.0:
            logger.info("  PASS: GL balanced (DR=%.2f, CR=%.2f, diff=%.4f)",
                        total_dr, total_cr, diff)
            passed += 1
        else:
            logger.error("  FAIL: GL imbalance (DR=%.2f, CR=%.2f, diff=%.4f)",
                         total_dr, total_cr, diff)
            failed += 1

        day_count, imbalanced_days = gl_db.execute("""
            SELECT COUNT(*), SUM(CASE WHEN ABS(dr - cr) > 1.10 THEN 1 ELSE 0 END)
            FROM (
                SELECT entry_date, SUM(debit_amount) as dr, SUM(credit_amount) as cr
                FROM gl GROUP BY entry_date
            )
        """).fetchone()
        if imbalanced_days == 0:
            logger.info("  PASS: All %d days balanced (tolerance $1.10)", day_count)
            passed += 1
        else:
            logger.warning("  WARN: %d/%d days have imbalance > $1.10",
                           imbalanced_days, day_count)
            failed += 1

        dup_count = gl_db.execute("""
            SELECT COUNT(*) FROM gl WHERE reference_id LIKE '%-DUP'
        """).fetchone()[0]
        if dup_count > 0:
            logger.info("  INFO: %s GL entries are friction duplicates (-DUP)",
                        f"{dup_count:,}")
        else:
            logger.info("  INFO: No GL duplicate entries found")

        logger.info("\n--- Cost Sanity ---")
        cogs, revenue = gl_db.execute("""
            SELECT
                SUM(CASE WHEN account_code = '5100' THEN debit_amount ELSE 0 END),
                SUM(CASE WHEN account_code = '4100' THEN credit_amount ELSE 0 END)
            FROM gl
        """).fetchone()
        if revenue and revenue > 0:
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

        physical_types = {
            "goods_receipt", "production", "shipment", "freight",
            "sale", "return", "price_variance", "qty_variance",
        }
        treasury_types = {"payment", "receipt", "bad_debt"}

        for ref_type, total, _empty, pct_filled in node_stats:
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

    if gl_path.exists():
        is_mono = _check_seq_monotonic_duckdb(gl_path, "transaction_sequence_id")
        if is_mono:
            logger.info("  PASS: gl_journal.csv sequence monotonic")
            passed += 1
        else:
            logger.warning("  WARN: gl_journal.csv sequence NOT monotonic")
            failed += 1

    # ── 6. Friction Table Checks ──────────────────────────────
    p, f = _check_friction_tables_csv(output_dir)
    passed += p
    failed += f

    return passed, failed


def _check_friction_tables_csv(output_dir: Path) -> tuple[int, int]:
    """Verify friction-specific tables from CSV files."""
    passed = 0
    failed = 0
    trans_dir = output_dir / "transactional"

    iv_path = trans_dir / "invoice_variances.csv"
    ap_pay_path = trans_dir / "ap_payments.csv"
    ar_rec_path = trans_dir / "ar_receipts.csv"

    has_friction = iv_path.exists() or ap_pay_path.exists() or ar_rec_path.exists()
    if not has_friction:
        return passed, failed

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

    return passed, failed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    """Check FK integrity: returns (resolved, total, missing)."""
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
                continue
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
            return True
        for row in reader:
            val = int(row[col])
            if val < prev:
                return False
            prev = val
    return True


def _check_seq_monotonic_duckdb(path: Path, col: str) -> bool:
    """Check sequence monotonicity via DuckDB (for large CSVs)."""
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
