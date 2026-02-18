"""Friction layer — controlled data quality noise for VKG testing.

Phase 3.5 in the ERP pipeline. Reads clean CSVs into DuckDB, applies
targeted mutations across 4 tiers, then re-exports affected tables.

Tier 1: Entity Resolution  — duplicate suppliers, SKU renames
Tier 2: 3-Way Match        — price/qty variance on AP invoice lines
Tier 3: Data Quality       — null FKs, duplicate invoices, status flips
Tier 4: Payment Timing     — ap_payments, ar_receipts, discounts, bad debt

All friction GL entries are balanced (DR == CR) to maintain GL integrity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .config import ErpConfig
from .sequence import CAT_MULTIPLIER, DAY_MULTIPLIER

logger = logging.getLogger(__name__)


def generate_friction(
    output_dir: Path,
    cfg: ErpConfig,
) -> dict[str, int]:
    """Apply friction mutations to clean ERP CSVs.

    Opens a fresh DuckDB connection, imports the CSVs produced by Phases 1-3,
    applies tiers 1-4, and re-exports modified tables.

    Returns a stats dict with counts of affected rows per tier.
    """
    fc = cfg.friction
    stats: dict[str, int] = {}

    db = duckdb.connect()
    db.execute(f"SELECT setseed({fc.seed / 10000.0})")  # DuckDB setseed range 0.0-1.0

    trans_dir = output_dir / "transactional"
    master_dir = output_dir / "master"

    # Import tables we'll read/mutate
    _import_csv(db, "suppliers", master_dir / "suppliers.csv")
    _import_csv(db, "skus", master_dir / "skus.csv")
    _import_csv(db, "ap_invoices", trans_dir / "ap_invoices.csv")
    _import_csv(db, "ap_invoice_lines", trans_dir / "ap_invoice_lines.csv")
    _import_csv(db, "ar_invoices", trans_dir / "ar_invoices.csv")
    _import_csv(db, "gl_journal", trans_dir / "gl_journal.csv")

    # ── Tier 1: Entity Resolution ──────────────────────────────
    t1 = _apply_entity_friction(db, fc.entity_resolution, fc.seed)
    stats.update(t1)
    logger.info("  Tier 1 (entity resolution): %s", t1)

    # ── Tier 2: 3-Way Match ────────────────────────────────────
    t2 = _apply_three_way_match(db, fc.three_way_match, fc.seed)
    stats.update(t2)
    logger.info("  Tier 2 (3-way match): %s", t2)

    # ── Tier 3: Data Quality ───────────────────────────────────
    t3 = _apply_data_quality(db, fc.data_quality, fc.seed)
    stats.update(t3)
    logger.info("  Tier 3 (data quality): %s", t3)

    # ── Tier 4: Payment Timing ─────────────────────────────────
    t4 = _apply_payment_timing(db, cfg, fc.seed)
    stats.update(t4)
    logger.info("  Tier 4 (payment timing): %s", t4)

    # ── Re-export modified tables ──────────────────────────────
    _export_csv(db, "suppliers", master_dir / "suppliers.csv")
    _export_csv(db, "skus", master_dir / "skus.csv")
    _export_csv(db, "ap_invoices", trans_dir / "ap_invoices.csv")
    _export_csv(db, "ap_invoice_lines", trans_dir / "ap_invoice_lines.csv")
    _export_csv(db, "ar_invoices", trans_dir / "ar_invoices.csv")
    _export_csv(db, "gl_journal", trans_dir / "gl_journal.csv")

    # New tables (always exported, even if empty — schema consistency)
    _export_csv(db, "invoice_variances", trans_dir / "invoice_variances.csv")
    _export_csv(db, "ap_payments", trans_dir / "ap_payments.csv")
    _export_csv(db, "ar_receipts", trans_dir / "ar_receipts.csv")

    # ── Post-friction GL balance assertion ─────────────────────
    total_dr, total_cr = db.execute("""
        SELECT SUM(debit_amount), SUM(credit_amount) FROM gl_journal
    """).fetchone()
    diff = abs(total_dr - total_cr)
    # Tolerance scales with row count: ~58M GL rows × ROUND(...,4) → cumulative drift
    if diff > 5.0:
        logger.error("  FRICTION GL IMBALANCE: DR=%.2f CR=%.2f diff=%.4f",
                      total_dr, total_cr, diff)
    else:
        logger.info("  Post-friction GL balanced: diff=%.4f", diff)

    db.close()
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 1: Entity Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_entity_friction(
    db: duckdb.DuckDBPyConnection,
    er_cfg: object,
    seed: int,
) -> dict[str, int]:
    """Create duplicate suppliers and SKU aliases."""
    stats: dict[str, int] = {}

    # ── Duplicate Suppliers ────────────────────────────────────
    # Select ~N% of suppliers, create variant-name duplicates
    dup_rate = er_cfg.duplicate_supplier_rate  # type: ignore[attr-defined]

    # Pick suppliers to duplicate (deterministic via setseed)
    db.execute(f"SELECT setseed({seed / 10000.0})")
    db.execute(f"""
        CREATE TABLE dup_supplier_map AS
        SELECT id as original_id,
               supplier_code as original_code,
               name as original_name,
               city, country, lat, lon, tier
        FROM suppliers
        WHERE random() < {dup_rate}
    """)

    dup_count = db.execute("SELECT COUNT(*) FROM dup_supplier_map").fetchone()[0]

    if dup_count > 0:
        max_id = db.execute("SELECT MAX(id) FROM suppliers").fetchone()[0]

        # Create variant names: "Acme Corp" → "ACME CORPORATION INC."
        # Use multiple transforms for variety
        db.execute(f"""
            INSERT INTO suppliers
            SELECT
                {max_id} + ROW_NUMBER() OVER () as id,
                original_code || '-ALT' as supplier_code,
                CASE (ROW_NUMBER() OVER ()) % 4
                    WHEN 0 THEN UPPER(original_name) || ' INC.'
                    WHEN 1 THEN original_name || ' LLC'
                    WHEN 2 THEN REPLACE(UPPER(original_name), ' ', '-')
                    ELSE original_name || ' Corp'
                END as name,
                city, country, lat, lon, tier, true as is_active
            FROM dup_supplier_map
        """)

        # Redirect ~50% of AP invoices for affected suppliers to the duplicate
        db.execute(f"""
            UPDATE ap_invoices SET supplier_id = (
                SELECT {max_id} + ROW_NUMBER() OVER (ORDER BY dm.original_id)
                FROM dup_supplier_map dm
                WHERE dm.original_id = ap_invoices.supplier_id
                LIMIT 1
            )
            WHERE supplier_id IN (SELECT original_id FROM dup_supplier_map)
              AND random() < 0.5
        """)

    stats["duplicate_suppliers"] = dup_count

    # ── SKU Renames ────────────────────────────────────────────
    sku_rate = er_cfg.sku_rename_rate  # type: ignore[attr-defined]

    db.execute(f"SELECT setseed({seed / 10000.0 + 0.01})")

    # Check if supersedes_sku_id column exists; add if not
    cols = db.execute("SELECT column_name FROM information_schema.columns WHERE table_name='skus'").fetchall()
    col_names = [c[0] for c in cols]
    if "supersedes_sku_id" not in col_names:
        db.execute("ALTER TABLE skus ADD COLUMN supersedes_sku_id INTEGER")

    db.execute(f"""
        CREATE TABLE sku_rename_map AS
        SELECT id as original_id,
               sku_code as original_code,
               name, category, brand,
               units_per_case, weight_kg,
               cost_per_case, price_per_case,
               value_segment
        FROM skus
        WHERE random() < {sku_rate}
    """)

    sku_dup_count = db.execute("SELECT COUNT(*) FROM sku_rename_map").fetchone()[0]

    if sku_dup_count > 0:
        max_sku_id = db.execute("SELECT MAX(id) FROM skus").fetchone()[0]

        # Create old-code aliases: "SKU-OC-001" → "SKU-OC-001-OLD"
        db.execute(f"""
            INSERT INTO skus
            SELECT
                {max_sku_id} + ROW_NUMBER() OVER () as id,
                original_code || '-OLD' as sku_code,
                name || ' (Discontinued)' as name,
                category, brand, units_per_case, weight_kg,
                cost_per_case, price_per_case, value_segment,
                original_id as supersedes_sku_id,
                false as is_active
            FROM sku_rename_map
        """)

    stats["sku_renames"] = sku_dup_count

    # Cleanup temp tables
    db.execute("DROP TABLE IF EXISTS dup_supplier_map")
    db.execute("DROP TABLE IF EXISTS sku_rename_map")

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 2: 3-Way Match Failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_three_way_match(
    db: duckdb.DuckDBPyConnection,
    twm_cfg: object,
    seed: int,
) -> dict[str, int]:
    """Inject price/quantity variances into AP invoice lines."""
    stats: dict[str, int] = {}

    price_rate = twm_cfg.price_variance_rate  # type: ignore[attr-defined]
    price_lo, price_hi = twm_cfg.price_variance_pct_range  # type: ignore[attr-defined]
    qty_rate = twm_cfg.qty_variance_rate  # type: ignore[attr-defined]
    qty_lo, qty_hi = twm_cfg.qty_variance_pct_range  # type: ignore[attr-defined]

    db.execute(f"SELECT setseed({seed / 10000.0 + 0.02})")

    # Create invoice_variances table
    db.execute("""
        CREATE TABLE invoice_variances (
            id INTEGER,
            invoice_id INTEGER,
            line_number INTEGER,
            variance_type VARCHAR,
            expected_value DOUBLE,
            actual_value DOUBLE,
            variance_amount DOUBLE,
            resolution_status VARCHAR
        )
    """)

    # ── Price Variances ────────────────────────────────────────
    # Tag lines for price friction
    db.execute(f"""
        CREATE TABLE price_friction_lines AS
        SELECT invoice_id, line_number, unit_cost, quantity_kg, line_amount,
               -- Random direction: +/- with magnitude in [lo, hi]
               CASE WHEN random() < 0.5 THEN 1 ELSE -1 END
               * ({price_lo} + random() * ({price_hi} - {price_lo})) as pct_change
        FROM ap_invoice_lines
        WHERE random() < {price_rate}
    """)

    price_var_count = db.execute(
        "SELECT COUNT(*) FROM price_friction_lines"
    ).fetchone()[0]

    if price_var_count > 0:
        # Record variances
        db.execute("""
            INSERT INTO invoice_variances
            SELECT
                ROW_NUMBER() OVER () as id,
                invoice_id, line_number,
                'price' as variance_type,
                unit_cost as expected_value,
                ROUND(unit_cost * (1.0 + pct_change), 4) as actual_value,
                ROUND(unit_cost * pct_change * quantity_kg, 4) as variance_amount,
                'open' as resolution_status
            FROM price_friction_lines
        """)

        # Mutate the actual invoice lines
        db.execute("""
            UPDATE ap_invoice_lines SET
                unit_cost = ROUND(pfl.unit_cost * (1.0 + pfl.pct_change), 4),
                line_amount = ROUND(pfl.quantity_kg * pfl.unit_cost * (1.0 + pfl.pct_change), 4)
            FROM price_friction_lines pfl
            WHERE ap_invoice_lines.invoice_id = pfl.invoice_id
              AND ap_invoice_lines.line_number = pfl.line_number
        """)

        # Backfill AP invoice totals
        db.execute("""
            UPDATE ap_invoices SET total_amount = (
                SELECT COALESCE(SUM(line_amount), 0)
                FROM ap_invoice_lines WHERE invoice_id = ap_invoices.id
            )
            WHERE id IN (SELECT DISTINCT invoice_id FROM price_friction_lines)
        """)

        # Add balancing GL entries: DR/CR 5400 (Price Variance)
        # Positive variance_amount → overcharged → DR 5400, negative → CR 5400
        max_gl_id = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
        db.execute(f"""
            INSERT INTO gl_journal
            SELECT
                {max_gl_id} + ROW_NUMBER() OVER () as id,
                CAST(ai.invoice_date AS BIGINT) * {DAY_MULTIPLIER}
                    + 8 * {CAT_MULTIPLIER}
                    + CAST(ROW_NUMBER() OVER () AS BIGINT) as transaction_sequence_id,
                ai.invoice_date as entry_date,
                ai.invoice_date as posting_date,
                CASE WHEN iv.variance_amount >= 0 THEN '5400' ELSE '2100' END as account_code,
                ROUND(ABS(iv.variance_amount), 4) as debit_amount,
                0.0 as credit_amount,
                'price_variance' as reference_type,
                ai.invoice_number as reference_id,
                '' as node_id,
                '' as product_id,
                'Price variance adjustment' as description,
                false as is_reversal
            FROM invoice_variances iv
            JOIN ap_invoices ai ON ai.id = iv.invoice_id
            WHERE iv.variance_type = 'price'
            UNION ALL
            SELECT
                {max_gl_id} + {price_var_count} + ROW_NUMBER() OVER () as id,
                CAST(ai.invoice_date AS BIGINT) * {DAY_MULTIPLIER}
                    + 8 * {CAT_MULTIPLIER}
                    + {price_var_count} + CAST(ROW_NUMBER() OVER () AS BIGINT) as transaction_sequence_id,
                ai.invoice_date as entry_date,
                ai.invoice_date as posting_date,
                CASE WHEN iv.variance_amount >= 0 THEN '2100' ELSE '5400' END as account_code,
                0.0 as debit_amount,
                ROUND(ABS(iv.variance_amount), 4) as credit_amount,
                'price_variance' as reference_type,
                ai.invoice_number as reference_id,
                '' as node_id,
                '' as product_id,
                'Price variance adjustment' as description,
                false as is_reversal
            FROM invoice_variances iv
            JOIN ap_invoices ai ON ai.id = iv.invoice_id
            WHERE iv.variance_type = 'price'
        """)

    stats["price_variances"] = price_var_count
    db.execute("DROP TABLE IF EXISTS price_friction_lines")

    # ── Quantity Variances ─────────────────────────────────────
    db.execute(f"SELECT setseed({seed / 10000.0 + 0.03})")

    db.execute(f"""
        CREATE TABLE qty_friction_lines AS
        SELECT invoice_id, line_number, unit_cost, quantity_kg,
               CASE WHEN random() < 0.5 THEN 1 ELSE -1 END
               * ({qty_lo} + random() * ({qty_hi} - {qty_lo})) as pct_change
        FROM ap_invoice_lines
        WHERE random() < {qty_rate}
    """)

    qty_var_count = db.execute(
        "SELECT COUNT(*) FROM qty_friction_lines"
    ).fetchone()[0]

    if qty_var_count > 0:
        existing_var_count = db.execute(
            "SELECT COUNT(*) FROM invoice_variances"
        ).fetchone()[0]

        db.execute(f"""
            INSERT INTO invoice_variances
            SELECT
                {existing_var_count} + ROW_NUMBER() OVER () as id,
                invoice_id, line_number,
                'qty' as variance_type,
                quantity_kg as expected_value,
                ROUND(quantity_kg * (1.0 + pct_change), 2) as actual_value,
                ROUND(unit_cost * quantity_kg * pct_change, 4) as variance_amount,
                'open' as resolution_status
            FROM qty_friction_lines
        """)

        # Mutate invoice lines
        db.execute("""
            UPDATE ap_invoice_lines SET
                quantity_kg = ROUND(qfl.quantity_kg * (1.0 + qfl.pct_change), 2),
                line_amount = ROUND(qfl.unit_cost * qfl.quantity_kg * (1.0 + qfl.pct_change), 4)
            FROM qty_friction_lines qfl
            WHERE ap_invoice_lines.invoice_id = qfl.invoice_id
              AND ap_invoice_lines.line_number = qfl.line_number
        """)

        # Backfill totals
        db.execute("""
            UPDATE ap_invoices SET total_amount = (
                SELECT COALESCE(SUM(line_amount), 0)
                FROM ap_invoice_lines WHERE invoice_id = ap_invoices.id
            )
            WHERE id IN (SELECT DISTINCT invoice_id FROM qty_friction_lines)
        """)

        # Balancing GL: DR/CR 1100 (RM Inventory adjustment)
        max_gl_id = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
        db.execute(f"""
            INSERT INTO gl_journal
            SELECT
                {max_gl_id} + ROW_NUMBER() OVER () as id,
                CAST(ai.invoice_date AS BIGINT) * {DAY_MULTIPLIER}
                    + 8 * {CAT_MULTIPLIER}
                    + CAST(ROW_NUMBER() OVER () AS BIGINT) as transaction_sequence_id,
                ai.invoice_date as entry_date,
                ai.invoice_date as posting_date,
                CASE WHEN iv.variance_amount >= 0 THEN '1100' ELSE '2100' END as account_code,
                ROUND(ABS(iv.variance_amount), 4) as debit_amount,
                0.0 as credit_amount,
                'qty_variance' as reference_type,
                ai.invoice_number as reference_id,
                '' as node_id,
                '' as product_id,
                'Quantity variance adjustment' as description,
                false as is_reversal
            FROM invoice_variances iv
            JOIN ap_invoices ai ON ai.id = iv.invoice_id
            WHERE iv.variance_type = 'qty'
            UNION ALL
            SELECT
                {max_gl_id} + {qty_var_count} + ROW_NUMBER() OVER () as id,
                CAST(ai.invoice_date AS BIGINT) * {DAY_MULTIPLIER}
                    + 8 * {CAT_MULTIPLIER}
                    + {qty_var_count} + CAST(ROW_NUMBER() OVER () AS BIGINT) as transaction_sequence_id,
                ai.invoice_date as entry_date,
                ai.invoice_date as posting_date,
                CASE WHEN iv.variance_amount >= 0 THEN '2100' ELSE '1100' END as account_code,
                0.0 as debit_amount,
                ROUND(ABS(iv.variance_amount), 4) as credit_amount,
                'qty_variance' as reference_type,
                ai.invoice_number as reference_id,
                '' as node_id,
                '' as product_id,
                'Quantity variance adjustment' as description,
                false as is_reversal
            FROM invoice_variances iv
            JOIN ap_invoices ai ON ai.id = iv.invoice_id
            WHERE iv.variance_type = 'qty'
        """)

    stats["qty_variances"] = qty_var_count
    db.execute("DROP TABLE IF EXISTS qty_friction_lines")

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 3: Data Quality Friction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_data_quality(
    db: duckdb.DuckDBPyConnection,
    dq_cfg: object,
    seed: int,
) -> dict[str, int]:
    """Inject null FKs, duplicate invoices, and status inconsistencies."""
    stats: dict[str, int] = {}

    # ── Null FKs on AP invoices (gr_id) ────────────────────────
    db.execute(f"SELECT setseed({seed / 10000.0 + 0.04})")
    null_rate_ap = dq_cfg.null_fk_rate_ap  # type: ignore[attr-defined]

    db.execute(f"""
        UPDATE ap_invoices SET gr_id = NULL
        WHERE random() < {null_rate_ap}
    """)
    null_ap_count = db.execute(
        "SELECT COUNT(*) FROM ap_invoices WHERE gr_id IS NULL"
    ).fetchone()[0]
    stats["null_fk_ap_invoices"] = null_ap_count

    # ── Null FKs on GL entries (node_id) ───────────────────────
    db.execute(f"SELECT setseed({seed / 10000.0 + 0.05})")
    null_rate_gl = dq_cfg.null_fk_rate_gl  # type: ignore[attr-defined]

    db.execute(f"""
        UPDATE gl_journal SET node_id = NULL
        WHERE random() < {null_rate_gl}
    """)
    null_gl_count = db.execute(
        "SELECT COUNT(*) FROM gl_journal WHERE node_id IS NULL"
    ).fetchone()[0]
    stats["null_fk_gl_entries"] = null_gl_count

    # ── Duplicate AP invoices ──────────────────────────────────
    db.execute(f"SELECT setseed({seed / 10000.0 + 0.06})")
    dup_rate = dq_cfg.duplicate_invoice_rate  # type: ignore[attr-defined]

    db.execute(f"""
        CREATE TABLE dup_invoice_candidates AS
        SELECT * FROM ap_invoices WHERE random() < {dup_rate}
    """)

    dup_inv_count = db.execute(
        "SELECT COUNT(*) FROM dup_invoice_candidates"
    ).fetchone()[0]

    if dup_inv_count > 0:
        max_ap_id = db.execute("SELECT MAX(id) FROM ap_invoices").fetchone()[0]
        db.execute(f"""
            INSERT INTO ap_invoices
            SELECT
                {max_ap_id} + ROW_NUMBER() OVER () as id,
                transaction_sequence_id,
                invoice_number || '-DUP' as invoice_number,
                supplier_id, gr_id, invoice_date, due_date,
                total_amount, currency, status
            FROM dup_invoice_candidates
        """)

    stats["duplicate_invoices"] = dup_inv_count
    db.execute("DROP TABLE IF EXISTS dup_invoice_candidates")

    # ── AR status inconsistencies ──────────────────────────────
    db.execute(f"SELECT setseed({seed / 10000.0 + 0.07})")
    status_rate = dq_cfg.status_inconsistency_rate  # type: ignore[attr-defined]

    db.execute(f"""
        UPDATE ar_invoices SET status =
            CASE WHEN random() < 0.6 THEN 'disputed' ELSE 'partial' END
        WHERE random() < {status_rate}
    """)
    status_count = db.execute(
        "SELECT COUNT(*) FROM ar_invoices WHERE status IN ('disputed', 'partial')"
    ).fetchone()[0]
    stats["status_inconsistencies"] = status_count

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 4: Payment Timing & Cash Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_payment_timing(
    db: duckdb.DuckDBPyConnection,
    cfg: ErpConfig,
    seed: int,
) -> dict[str, int]:
    """Generate ap_payments and ar_receipts with timing noise."""
    stats: dict[str, int] = {}
    pt = cfg.friction.payment_timing

    db.execute(f"SELECT setseed({seed / 10000.0 + 0.08})")

    # ── AP Payments ────────────────────────────────────────────
    # One payment per AP invoice. payment_date = due_date + N(mean, std)
    # Early payments (within window) get discount
    db.execute(f"""
        CREATE TABLE ap_payments AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY id) as id,
            CAST(due_date + ROUND(
                {pt.ap_payment_lag_mean_days} +
                {pt.ap_payment_lag_std_days} * (random() + random() + random() - 1.5)
            ) AS BIGINT) * {DAY_MULTIPLIER}
                + 9 * {CAT_MULTIPLIER}
                + CAST(ROW_NUMBER() OVER (ORDER BY id) AS BIGINT)
                as transaction_sequence_id,
            id as invoice_id,
            CAST(due_date + ROUND(
                {pt.ap_payment_lag_mean_days} +
                {pt.ap_payment_lag_std_days} * (random() + random() + random() - 1.5)
            ) AS INTEGER) as payment_date,
            total_amount as amount,
            CASE
                WHEN random() < {pt.early_payment_discount_rate}
                     AND due_date - invoice_date > {pt.early_payment_window_days}
                THEN ROUND(total_amount * {pt.early_payment_discount_pct}, 4)
                ELSE 0.0
            END as discount_amount,
            CAST(0.0 AS DOUBLE) as net_amount,
            'EFT' as payment_method,
            'completed' as status
        FROM ap_invoices
    """)

    # Backfill net_amount = amount - discount
    db.execute("""
        UPDATE ap_payments SET net_amount = ROUND(amount - discount_amount, 4)
    """)

    # Fix payment_date: ensure >= invoice_date (floor at invoice_date - 5)
    db.execute("""
        UPDATE ap_payments SET payment_date = GREATEST(
            payment_date,
            (SELECT invoice_date FROM ap_invoices WHERE ap_invoices.id = ap_payments.invoice_id)
        )
    """)

    ap_pay_count = db.execute("SELECT COUNT(*) FROM ap_payments").fetchone()[0]
    discount_count = db.execute(
        "SELECT COUNT(*) FROM ap_payments WHERE discount_amount > 0"
    ).fetchone()[0]
    stats["ap_payments"] = ap_pay_count
    stats["early_discounts"] = discount_count

    # GL entries for AP payments:
    # Standard: DR 2100 AP / CR 1000 Cash
    # Discount: DR 2100 AP / CR 1000 Cash (net) + CR 4200 Discount Income
    max_gl_id = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
    gl_offset = max_gl_id

    # DR 2100 AP (full invoice amount)
    db.execute(f"""
        INSERT INTO gl_journal
        SELECT
            {gl_offset} + ROW_NUMBER() OVER () as id,
            ap.transaction_sequence_id,
            ap.payment_date as entry_date,
            ap.payment_date as posting_date,
            '2100' as account_code,
            ROUND(ap.amount, 4) as debit_amount,
            0.0 as credit_amount,
            'payment' as reference_type,
            ai.invoice_number as reference_id,
            '' as node_id,
            '' as product_id,
            'AP payment' as description,
            false as is_reversal
        FROM ap_payments ap
        JOIN ap_invoices ai ON ai.id = ap.invoice_id
    """)

    gl_offset = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]

    # CR 1000 Cash (net amount)
    db.execute(f"""
        INSERT INTO gl_journal
        SELECT
            {gl_offset} + ROW_NUMBER() OVER () as id,
            ap.transaction_sequence_id,
            ap.payment_date as entry_date,
            ap.payment_date as posting_date,
            '1000' as account_code,
            0.0 as debit_amount,
            ROUND(ap.net_amount, 4) as credit_amount,
            'payment' as reference_type,
            ai.invoice_number as reference_id,
            '' as node_id,
            '' as product_id,
            'Cash paid for AP' as description,
            false as is_reversal
        FROM ap_payments ap
        JOIN ap_invoices ai ON ai.id = ap.invoice_id
    """)

    # CR 4200 Discount Income (only for discounted payments)
    gl_offset = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
    db.execute(f"""
        INSERT INTO gl_journal
        SELECT
            {gl_offset} + ROW_NUMBER() OVER () as id,
            ap.transaction_sequence_id,
            ap.payment_date as entry_date,
            ap.payment_date as posting_date,
            '4200' as account_code,
            0.0 as debit_amount,
            ROUND(ap.discount_amount, 4) as credit_amount,
            'payment' as reference_type,
            ai.invoice_number as reference_id,
            '' as node_id,
            '' as product_id,
            'Early payment discount' as description,
            false as is_reversal
        FROM ap_payments ap
        JOIN ap_invoices ai ON ai.id = ap.invoice_id
        WHERE ap.discount_amount > 0.001
    """)

    # ── AR Receipts ────────────────────────────────────────────
    db.execute(f"SELECT setseed({seed / 10000.0 + 0.09})")

    # Bad debt: ~0.5% of AR invoices never get paid
    bad_debt_rate = pt.bad_debt_rate
    db.execute(f"""
        CREATE TABLE ar_bad_debt AS
        SELECT id, invoice_number, invoice_date, total_amount
        FROM ar_invoices
        WHERE random() < {bad_debt_rate}
    """)

    bad_debt_count = db.execute("SELECT COUNT(*) FROM ar_bad_debt").fetchone()[0]
    stats["bad_debt"] = bad_debt_count

    # Receipts for non-bad-debt invoices
    db.execute(f"""
        CREATE TABLE ar_receipts AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY ar.id) as id,
            CAST(ar.due_date + ROUND(
                {pt.ar_receipt_lag_mean_days} +
                {pt.ar_receipt_lag_std_days} * (random() + random() + random() - 1.5)
            ) AS BIGINT) * {DAY_MULTIPLIER}
                + 9 * {CAT_MULTIPLIER}
                + CAST(ROW_NUMBER() OVER (ORDER BY ar.id) AS BIGINT)
                as transaction_sequence_id,
            ar.id as invoice_id,
            GREATEST(
                ar.invoice_date,
                CAST(ar.due_date + ROUND(
                    {pt.ar_receipt_lag_mean_days} +
                    {pt.ar_receipt_lag_std_days} * (random() + random() + random() - 1.5)
                ) AS INTEGER)
            ) as receipt_date,
            ar.total_amount as amount,
            'completed' as status
        FROM ar_invoices ar
        WHERE ar.id NOT IN (SELECT id FROM ar_bad_debt)
    """)

    ar_receipt_count = db.execute("SELECT COUNT(*) FROM ar_receipts").fetchone()[0]
    stats["ar_receipts"] = ar_receipt_count

    # GL entries for AR receipts: DR 1000 Cash / CR 1200 AR
    gl_offset = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
    db.execute(f"""
        INSERT INTO gl_journal
        SELECT
            {gl_offset} + ROW_NUMBER() OVER () as id,
            r.transaction_sequence_id,
            r.receipt_date as entry_date,
            r.receipt_date as posting_date,
            '1000' as account_code,
            ROUND(r.amount, 4) as debit_amount,
            0.0 as credit_amount,
            'receipt' as reference_type,
            ai.invoice_number as reference_id,
            '' as node_id,
            '' as product_id,
            'Cash received' as description,
            false as is_reversal
        FROM ar_receipts r
        JOIN ar_invoices ai ON ai.id = r.invoice_id
    """)

    gl_offset = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
    db.execute(f"""
        INSERT INTO gl_journal
        SELECT
            {gl_offset} + ROW_NUMBER() OVER () as id,
            r.transaction_sequence_id,
            r.receipt_date as entry_date,
            r.receipt_date as posting_date,
            '1200' as account_code,
            0.0 as debit_amount,
            ROUND(r.amount, 4) as credit_amount,
            'receipt' as reference_type,
            ai.invoice_number as reference_id,
            '' as node_id,
            '' as product_id,
            'AR cleared by receipt' as description,
            false as is_reversal
        FROM ar_receipts r
        JOIN ar_invoices ai ON ai.id = r.invoice_id
    """)

    # GL entries for bad debt: DR 5500 Bad Debt / CR 1200 AR
    if bad_debt_count > 0:
        gl_offset = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
        db.execute(f"""
            INSERT INTO gl_journal
            SELECT
                {gl_offset} + ROW_NUMBER() OVER () as id,
                CAST(bd.invoice_date + 90 AS BIGINT) * {DAY_MULTIPLIER}
                    + 8 * {CAT_MULTIPLIER}
                    + CAST(ROW_NUMBER() OVER () AS BIGINT) as transaction_sequence_id,
                bd.invoice_date + 90 as entry_date,
                bd.invoice_date + 90 as posting_date,
                '5500' as account_code,
                ROUND(bd.total_amount, 4) as debit_amount,
                0.0 as credit_amount,
                'bad_debt' as reference_type,
                bd.invoice_number as reference_id,
                '' as node_id,
                '' as product_id,
                'Bad debt writeoff' as description,
                false as is_reversal
            FROM ar_bad_debt bd
        """)

        gl_offset = db.execute("SELECT MAX(id) FROM gl_journal").fetchone()[0]
        db.execute(f"""
            INSERT INTO gl_journal
            SELECT
                {gl_offset} + ROW_NUMBER() OVER () as id,
                CAST(bd.invoice_date + 90 AS BIGINT) * {DAY_MULTIPLIER}
                    + 8 * {CAT_MULTIPLIER}
                    + CAST(ROW_NUMBER() OVER () AS BIGINT) as transaction_sequence_id,
                bd.invoice_date + 90 as entry_date,
                bd.invoice_date + 90 as posting_date,
                '1200' as account_code,
                0.0 as debit_amount,
                ROUND(bd.total_amount, 4) as credit_amount,
                'bad_debt' as reference_type,
                bd.invoice_number as reference_id,
                '' as node_id,
                '' as product_id,
                'AR written off as bad debt' as description,
                false as is_reversal
            FROM ar_bad_debt bd
        """)

    db.execute("DROP TABLE IF EXISTS ar_bad_debt")

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _import_csv(db: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    """Import a CSV file into a DuckDB table."""
    if path.exists():
        db.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{path}')")
    else:
        logger.warning("  Friction: %s not found at %s", name, path)


def _export_csv(db: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    """Export a DuckDB table to CSV."""
    try:
        db.execute(f"COPY {name} TO '{path}' (HEADER, DELIMITER ',')")
    except duckdb.CatalogException:
        logger.warning("  Friction: table %s does not exist, skipping export", name)
