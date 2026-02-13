"""Invoice generation — AP and AR invoices from physical events.

AP invoices: one per goods receipt (supplier → plant deliveries)
AR invoices: one per demand-endpoint shipment arrival (DC → store/FC)

Uses DuckDB tables created during transactional phase (erp_shipments,
erp_goods_receipts) for FK resolution.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import duckdb

from .config import ErpConfig
from .id_mapper import IdMapper
from .sequence import CAT_MULTIPLIER, DAY_MULTIPLIER

logger = logging.getLogger(__name__)


def generate_invoices(
    db: duckdb.DuckDBPyConnection,
    output_dir: Path,
    mapper: IdMapper,
    cfg: ErpConfig,
) -> None:
    """Generate AP and AR invoices from DuckDB parquet data."""
    trans_dir = output_dir / "transactional"

    # Build cost/price maps from master CSVs
    sku_price_map = _build_map(output_dir / "master" / "skus.csv",
                               "sku_code", "price_per_case")
    ing_cost_map = _build_map(output_dir / "master" / "ingredients.csv",
                              "ingredient_code", "cost_per_kg")
    bulk_cost_map = _build_map(output_dir / "master" / "bulk_intermediates.csv",
                               "bulk_code", "cost_per_kg")
    ing_cost_map.update(bulk_cost_map)

    # Register cost maps as DuckDB tables for SQL access
    _register_cost_table(db, "sku_price_tbl", sku_price_map)
    _register_cost_table(db, "ing_cost_tbl", ing_cost_map)

    _generate_ap_invoices_duckdb(db, trans_dir, cfg)
    _generate_ar_invoices_duckdb(db, trans_dir, cfg)


def _register_cost_table(
    db: duckdb.DuckDBPyConnection, table_name: str, cost_map: dict[str, float]
) -> None:
    """Register a sim_id → cost lookup as a DuckDB table."""
    db.execute(f"CREATE OR REPLACE TABLE {table_name} (sim_id VARCHAR, cost DOUBLE)")
    if cost_map:
        db.executemany(
            f"INSERT INTO {table_name} VALUES (?, ?)",
            [(k, v) for k, v in cost_map.items()],
        )


def _generate_ap_invoices_duckdb(
    db: duckdb.DuckDBPyConnection,
    trans_dir: Path,
    cfg: ErpConfig,
) -> None:
    """Generate AP invoices from goods receipts using DuckDB."""
    logger.info("  Generating AP invoices (DuckDB)...")

    dpo = int(cfg.dpo_days)

    # AP invoice headers: one per goods receipt
    db.execute(f"""
        CREATE OR REPLACE TABLE erp_ap_invoices AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY gr.id) as id,
            CAST(gr.receipt_date AS BIGINT) * {DAY_MULTIPLIER} + 7 * {CAT_MULTIPLIER} +
                CAST(ROW_NUMBER() OVER (ORDER BY gr.id) AS BIGINT) as transaction_sequence_id,
            'AP-' || LPAD(CAST(ROW_NUMBER() OVER (ORDER BY gr.id) AS VARCHAR), 7, '0') as invoice_number,
            -- Get supplier from the shipment's source
            COALESCE(sm.pk, 0) as supplier_id,
            gr.id as gr_id,
            gr.receipt_date as invoice_date,
            gr.receipt_date + {dpo} as due_date,
            CAST(0.0 AS DECIMAL(14,4)) as total_amount,  -- backfilled below
            'USD' as currency,
            'open' as status
        FROM erp_goods_receipts gr
        LEFT JOIN pq_shipments s ON 'GR-' || s.shipment_id = gr.gr_number
        LEFT JOIN loc_map sm ON sm.sim_id = s.source_id
        GROUP BY gr.id, gr.receipt_date, gr.gr_number, sm.pk
    """)

    # AP invoice lines: line items from shipments to plants
    db.execute("""
        CREATE OR REPLACE TABLE erp_ap_invoice_lines AS
        SELECT
            ap.id as invoice_id,
            ROW_NUMBER() OVER (PARTITION BY ap.id ORDER BY s.product_id) as line_number,
            COALESCE(pm.pk, 0) as ingredient_id,
            s.quantity as quantity_kg,
            COALESCE(ic.cost, 5.0) as unit_cost,
            ROUND(s.quantity * COALESCE(ic.cost, 5.0), 4) as line_amount
        FROM pq_shipments s
        JOIN erp_goods_receipts gr ON gr.gr_number = 'GR-' || s.shipment_id
        JOIN erp_ap_invoices ap ON ap.gr_id = gr.id
        LEFT JOIN prod_map pm ON pm.sim_id = s.product_id
        LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
        WHERE s.target_id LIKE 'PLANT-%'
    """)

    # Backfill totals
    db.execute("""
        UPDATE erp_ap_invoices SET total_amount = (
            SELECT COALESCE(SUM(line_amount), 0)
            FROM erp_ap_invoice_lines WHERE invoice_id = erp_ap_invoices.id
        )
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_ap_invoices").fetchone()[0]
    line_count = db.execute("SELECT COUNT(*) FROM erp_ap_invoice_lines").fetchone()[0]

    db.execute(f"""
        COPY erp_ap_invoices TO '{trans_dir / "ap_invoices.csv"}'
        (HEADER, DELIMITER ',')
    """)
    db.execute(f"""
        COPY erp_ap_invoice_lines TO '{trans_dir / "ap_invoice_lines.csv"}'
        (HEADER, DELIMITER ',')
    """)
    logger.info("  AP invoices: %s headers, %s lines", f"{count:,}", f"{line_count:,}")


def _generate_ar_invoices_duckdb(
    db: duckdb.DuckDBPyConnection,
    trans_dir: Path,
    cfg: ErpConfig,
) -> None:
    """Generate AR invoices from shipments arriving at demand endpoints."""
    logger.info("  Generating AR invoices (DuckDB)...")

    # Register DSO lookup table
    db.execute("CREATE OR REPLACE TABLE dso_tbl (channel VARCHAR, dso INTEGER)")
    for ch, dso in cfg.dso_by_channel.items():
        db.execute("INSERT INTO dso_tbl VALUES (?, ?)", [ch, dso])

    # AR invoice headers: one per shipment to demand endpoint
    db.execute(f"""
        CREATE OR REPLACE TABLE erp_ar_invoices AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY es.id) as id,
            CAST(es.arrival_date AS BIGINT) * {DAY_MULTIPLIER} + 7 * {CAT_MULTIPLIER} +
                CAST(ROW_NUMBER() OVER (ORDER BY es.id) AS BIGINT) as transaction_sequence_id,
            'AR-' || LPAD(CAST(ROW_NUMBER() OVER (ORDER BY es.id) AS VARCHAR), 7, '0') as invoice_number,
            COALESCE(tm.pk, 0) as customer_location_id,
            es.id as shipment_id,
            es.arrival_date as invoice_date,
            es.arrival_date + COALESCE(dso.dso, 30) as due_date,
            CAST(0.0 AS DECIMAL(14,4)) as total_amount,
            'USD' as currency,
            CASE
                WHEN s.target_id LIKE 'STORE-RET-%' THEN 'MASS_RETAIL'
                WHEN s.target_id LIKE 'STORE-GRO-%' THEN 'GROCERY'
                WHEN s.target_id LIKE 'STORE-CLUB-%' THEN 'CLUB'
                WHEN s.target_id LIKE 'STORE-PHARM-%' THEN 'PHARMACY'
                WHEN s.target_id LIKE 'STORE-DIST-%' THEN 'DISTRIBUTOR'
                WHEN s.target_id LIKE 'ECOM-FC-%' THEN 'ECOMMERCE'
                WHEN s.target_id LIKE 'DTC-FC-%' THEN 'DTC'
                ELSE 'MASS_RETAIL'
            END as channel,
            'open' as status
        FROM erp_shipments es
        JOIN pq_shipments s ON s.shipment_id = es.shipment_number
        LEFT JOIN loc_map tm ON tm.sim_id = s.target_id
        LEFT JOIN dso_tbl dso ON dso.channel = (
            CASE
                WHEN s.target_id LIKE 'STORE-RET-%' THEN 'MASS_RETAIL'
                WHEN s.target_id LIKE 'STORE-GRO-%' THEN 'GROCERY'
                WHEN s.target_id LIKE 'STORE-CLUB-%' THEN 'CLUB'
                WHEN s.target_id LIKE 'STORE-PHARM-%' THEN 'PHARMACY'
                WHEN s.target_id LIKE 'STORE-DIST-%' THEN 'DISTRIBUTOR'
                WHEN s.target_id LIKE 'ECOM-FC-%' THEN 'ECOMMERCE'
                WHEN s.target_id LIKE 'DTC-FC-%' THEN 'DTC'
                ELSE 'MASS_RETAIL'
            END
        )
        WHERE s.target_id LIKE 'STORE-%'
           OR s.target_id LIKE 'ECOM-FC-%'
           OR s.target_id LIKE 'DTC-FC-%'
        GROUP BY es.id, es.arrival_date, es.shipment_number,
                 tm.pk, s.target_id, dso.dso
    """)

    # AR invoice lines
    db.execute("""
        CREATE OR REPLACE TABLE erp_ar_invoice_lines AS
        SELECT
            ar.id as invoice_id,
            ROW_NUMBER() OVER (PARTITION BY ar.id ORDER BY s.product_id) as line_number,
            COALESCE(pm.pk, 0) as sku_id,
            s.quantity as quantity_cases,
            COALESCE(sp.cost, 15.0) as unit_price,
            ROUND(s.quantity * COALESCE(sp.cost, 15.0), 4) as line_amount
        FROM pq_shipments s
        JOIN erp_shipments es ON es.shipment_number = s.shipment_id
        JOIN erp_ar_invoices ar ON ar.shipment_id = es.id
        LEFT JOIN prod_map pm ON pm.sim_id = s.product_id
        LEFT JOIN sku_price_tbl sp ON sp.sim_id = s.product_id
        WHERE s.target_id LIKE 'STORE-%'
           OR s.target_id LIKE 'ECOM-FC-%'
           OR s.target_id LIKE 'DTC-FC-%'
    """)

    # Backfill totals
    db.execute("""
        UPDATE erp_ar_invoices SET total_amount = (
            SELECT COALESCE(SUM(line_amount), 0)
            FROM erp_ar_invoice_lines WHERE invoice_id = erp_ar_invoices.id
        )
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_ar_invoices").fetchone()[0]
    line_count = db.execute("SELECT COUNT(*) FROM erp_ar_invoice_lines").fetchone()[0]

    db.execute(f"""
        COPY erp_ar_invoices TO '{trans_dir / "ar_invoices.csv"}'
        (HEADER, DELIMITER ',')
    """)
    db.execute(f"""
        COPY erp_ar_invoice_lines TO '{trans_dir / "ar_invoice_lines.csv"}'
        (HEADER, DELIMITER ',')
    """)
    logger.info("  AR invoices: %s headers, %s lines", f"{count:,}", f"{line_count:,}")


# ── Helpers ──────────────────────────────────────────────────


def _build_map(csv_path: Path, key_col: str, val_col: str) -> dict[str, float]:
    """Build sim_id → float value map from a CSV."""
    result: dict[str, float] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                result[row[key_col]] = float(row[val_col])
    return result
