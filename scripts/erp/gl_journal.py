"""GL Journal generation — double-entry bookkeeping via DuckDB SQL.

Produces per-shipment / per-batch GL entries. Each entry carries a
``reference_id`` linking back to the source document (shipment_id,
batch_id, or rma_id), enabling full digital-thread traceability from
financial entries to physical movements.

Seven event types generate DR/CR pairs (each with its own reference_type):
  goods_receipt  → DR 1100 Raw Material Inv / CR 2100 AP
  production     → DR 1120 WIP / CR 1100 RM; DR 1130 FG / CR 1120 WIP
  shipment       → DR 1140 In-Transit / CR 1130 FG (dispatch + arrival)
  freight        → DR 5300 Freight Expense / CR 1000 Cash
  sale           → DR 5100 COGS / CR 1130 FG; DR 1200 AR / CR 4100 Revenue
  return         → DR 5200 Returns / CR 1200 AR
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


def generate_gl_journal(
    db: duckdb.DuckDBPyConnection,
    output_dir: Path,
    mapper: IdMapper,
    cfg: ErpConfig,
) -> None:
    """Generate gl_journal.csv with per-shipment/batch double-entry GL via DuckDB."""
    trans_dir = output_dir / "transactional"

    # Register cost/price lookup tables
    _register_cost_tables(db, output_dir)

    # Build all GL entries as a UNION ALL of 7 event type queries
    logger.info("  GL: Building journal entries via DuckDB...")

    db.execute(f"""
        CREATE OR REPLACE TABLE erp_gl_journal AS
        WITH
        -- 1. Goods Receipts: DR 1100 RM Inv / CR 2100 AP (per shipment)
        gr_entries AS (
            SELECT
                s.arrival_day as entry_date,
                '1100' as account_code,
                SUM(s.quantity * COALESCE(ic.cost, 5.0)) as debit_amount,
                0.0 as credit_amount,
                'goods_receipt' as reference_type,
                s.target_id as node_id,
                s.shipment_id as reference_id,
                'RM receipt' as description
            FROM pq_shipments s
            LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
            WHERE s.target_id LIKE 'PLANT-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
            UNION ALL
            SELECT
                s.arrival_day, '2100', 0.0,
                SUM(s.quantity * COALESCE(ic.cost, 5.0)),
                'goods_receipt', s.target_id, s.shipment_id, 'AP accrual'
            FROM pq_shipments s
            LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
            WHERE s.target_id LIKE 'PLANT-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
        ),
        -- 2. Production: DR 1120 WIP / CR 1100 RM, then DR 1130 FG / CR 1120 WIP (per batch)
        --    Cost = SUM(ingredient_input_kg × ingredient_cost_per_kg) from batch_ingredients
        batch_cost AS (
            SELECT bi.batch_id,
                SUM(bi.quantity_kg * COALESCE(ic2.cost, 5.0)) as ingredient_cost
            FROM pq_batch_ingredients bi
            LEFT JOIN ing_cost_tbl ic2 ON ic2.sim_id = bi.ingredient_id
            GROUP BY bi.batch_id
        ),
        prod_entries AS (
            SELECT b.day_produced as entry_date, '1120' as account_code,
                COALESCE(bc.ingredient_cost, 0.0) as debit_amount,
                0.0 as credit_amount,
                'production' as reference_type,
                b.plant_id as node_id, b.batch_id as reference_id, 'WIP intake'
            FROM pq_batches b
            LEFT JOIN batch_cost bc ON bc.batch_id = b.batch_id
            UNION ALL
            SELECT b.day_produced, '1100', 0.0,
                COALESCE(bc.ingredient_cost, 0.0),
                'production', b.plant_id, b.batch_id, 'RM consumed'
            FROM pq_batches b
            LEFT JOIN batch_cost bc ON bc.batch_id = b.batch_id
            UNION ALL
            SELECT b.day_produced, '1130',
                COALESCE(bc.ingredient_cost, 0.0),
                0.0, 'production', b.plant_id, b.batch_id, 'FG completion'
            FROM pq_batches b
            LEFT JOIN batch_cost bc ON bc.batch_id = b.batch_id
            UNION ALL
            SELECT b.day_produced, '1120', 0.0,
                COALESCE(bc.ingredient_cost, 0.0),
                'production', b.plant_id, b.batch_id, 'WIP transfer to FG'
            FROM pq_batches b
            LEFT JOIN batch_cost bc ON bc.batch_id = b.batch_id
        ),
        -- 3. Ship Dispatch: DR 1140 In-Transit / CR 1130 FG (per shipment)
        dispatch_entries AS (
            SELECT s.creation_day as entry_date, '1140' as account_code,
                SUM(s.quantity * COALESCE(sc.cost, ic.cost, 10.0)) as debit_amount,
                0.0 as credit_amount,
                'shipment' as reference_type, s.source_id as node_id,
                s.shipment_id as reference_id, 'In-transit dispatch'
            FROM pq_shipments s
            LEFT JOIN sku_cost_tbl sc ON sc.sim_id = s.product_id
            LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
            WHERE s.source_id NOT LIKE 'SUP-%'
            GROUP BY s.shipment_id, s.creation_day, s.source_id
            UNION ALL
            SELECT s.creation_day, '1130', 0.0,
                SUM(s.quantity * COALESCE(sc.cost, ic.cost, 10.0)),
                'shipment', s.source_id, s.shipment_id, 'FG shipped out'
            FROM pq_shipments s
            LEFT JOIN sku_cost_tbl sc ON sc.sim_id = s.product_id
            LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
            WHERE s.source_id NOT LIKE 'SUP-%'
            GROUP BY s.shipment_id, s.creation_day, s.source_id
        ),
        -- 4. Freight from erp_shipments (per shipment, no aggregation)
        freight_entries AS (
            SELECT ship_date as entry_date, '5300' as account_code,
                freight_cost as debit_amount, 0.0 as credit_amount,
                'freight' as reference_type, source_sim_id as node_id,
                shipment_number as reference_id, 'Freight expense'
            FROM erp_shipments
            WHERE freight_cost > 0.001
            UNION ALL
            SELECT ship_date, '1000', 0.0, freight_cost,
                'freight', source_sim_id, shipment_number, 'Cash paid for freight'
            FROM erp_shipments
            WHERE freight_cost > 0.001
        ),
        -- 5. Ship Arrival: DR 1130 FG / CR 1140 In-Transit (per shipment)
        arrival_entries AS (
            SELECT s.arrival_day as entry_date, '1130' as account_code,
                SUM(s.quantity * COALESCE(sc.cost, ic.cost, 10.0)) as debit_amount,
                0.0 as credit_amount,
                'shipment' as reference_type, s.target_id as node_id,
                s.shipment_id as reference_id, 'Received into inventory'
            FROM pq_shipments s
            LEFT JOIN sku_cost_tbl sc ON sc.sim_id = s.product_id
            LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
            WHERE s.source_id NOT LIKE 'SUP-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
            UNION ALL
            SELECT s.arrival_day, '1140', 0.0,
                SUM(s.quantity * COALESCE(sc.cost, ic.cost, 10.0)),
                'shipment', s.target_id, s.shipment_id, 'In-transit cleared'
            FROM pq_shipments s
            LEFT JOIN sku_cost_tbl sc ON sc.sim_id = s.product_id
            LEFT JOIN ing_cost_tbl ic ON ic.sim_id = s.product_id
            WHERE s.source_id NOT LIKE 'SUP-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
        ),
        -- 6. Sales: DR 5100 COGS / CR 1130 FG; DR 1200 AR / CR 4100 Revenue (per shipment)
        sale_entries AS (
            SELECT s.arrival_day as entry_date, '5100' as account_code,
                SUM(s.quantity * COALESCE(sc.cost, 10.0)) as debit_amount,
                0.0 as credit_amount,
                'sale' as reference_type, s.target_id as node_id,
                s.shipment_id as reference_id, 'COGS'
            FROM pq_shipments s
            LEFT JOIN sku_cost_tbl sc ON sc.sim_id = s.product_id
            WHERE s.target_id LIKE 'STORE-%'
               OR s.target_id LIKE 'ECOM-FC-%'
               OR s.target_id LIKE 'DTC-FC-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
            UNION ALL
            SELECT s.arrival_day, '1130', 0.0,
                SUM(s.quantity * COALESCE(sc.cost, 10.0)),
                'sale', s.target_id, s.shipment_id, 'FG consumed by sale'
            FROM pq_shipments s
            LEFT JOIN sku_cost_tbl sc ON sc.sim_id = s.product_id
            WHERE s.target_id LIKE 'STORE-%'
               OR s.target_id LIKE 'ECOM-FC-%'
               OR s.target_id LIKE 'DTC-FC-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
            UNION ALL
            SELECT s.arrival_day, '1200',
                SUM(s.quantity * COALESCE(sp.cost, 15.0)),
                0.0, 'sale', s.target_id, s.shipment_id, 'AR for sale'
            FROM pq_shipments s
            LEFT JOIN sku_price_tbl sp ON sp.sim_id = s.product_id
            WHERE s.target_id LIKE 'STORE-%'
               OR s.target_id LIKE 'ECOM-FC-%'
               OR s.target_id LIKE 'DTC-FC-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
            UNION ALL
            SELECT s.arrival_day, '4100', 0.0,
                SUM(s.quantity * COALESCE(sp.cost, 15.0)),
                'sale', s.target_id, s.shipment_id, 'Revenue recognized'
            FROM pq_shipments s
            LEFT JOIN sku_price_tbl sp ON sp.sim_id = s.product_id
            WHERE s.target_id LIKE 'STORE-%'
               OR s.target_id LIKE 'ECOM-FC-%'
               OR s.target_id LIKE 'DTC-FC-%'
            GROUP BY s.shipment_id, s.arrival_day, s.target_id
        ),
        -- 7. Returns: DR 5200 Returns Exp / CR 1200 AR (per RMA)
        return_entries AS (
            SELECT r.day as entry_date, '5200' as account_code,
                SUM(r.quantity * COALESCE(sp.cost, 15.0)) as debit_amount,
                0.0 as credit_amount,
                'return' as reference_type, r.target_id as node_id,
                r.rma_id as reference_id, 'Returns expense'
            FROM pq_returns r
            LEFT JOIN sku_price_tbl sp ON sp.sim_id = r.product_id
            GROUP BY r.rma_id, r.day, r.target_id
            UNION ALL
            SELECT r.day, '1200', 0.0,
                SUM(r.quantity * COALESCE(sp.cost, 15.0)),
                'return', r.target_id, r.rma_id, 'AR reversal for return'
            FROM pq_returns r
            LEFT JOIN sku_price_tbl sp ON sp.sim_id = r.product_id
            GROUP BY r.rma_id, r.day, r.target_id
        ),
        -- Combine all entries
        all_entries AS (
            SELECT * FROM gr_entries
            UNION ALL SELECT * FROM prod_entries
            UNION ALL SELECT * FROM dispatch_entries
            UNION ALL SELECT * FROM freight_entries
            UNION ALL SELECT * FROM arrival_entries
            UNION ALL SELECT * FROM sale_entries
            UNION ALL SELECT * FROM return_entries
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY entry_date, reference_id, account_code) as id,
            CAST(entry_date AS BIGINT) * {DAY_MULTIPLIER} + 6 * {CAT_MULTIPLIER} +
                CAST(ROW_NUMBER() OVER (ORDER BY entry_date, reference_id, account_code) AS BIGINT) as transaction_sequence_id,
            entry_date,
            entry_date as posting_date,
            account_code,
            ROUND(debit_amount, 4) as debit_amount,
            ROUND(credit_amount, 4) as credit_amount,
            reference_type,
            reference_id,
            node_id,
            '' as product_id,
            description,
            false as is_reversal
        FROM all_entries
        WHERE debit_amount > 0.001 OR credit_amount > 0.001
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_gl_journal").fetchone()[0]
    logger.info("  GL journal: %s entries", f"{count:,}")

    # Balance check
    balance = db.execute("""
        SELECT SUM(debit_amount) as total_dr, SUM(credit_amount) as total_cr
        FROM erp_gl_journal
    """).fetchone()
    total_dr, total_cr = balance
    diff = abs(total_dr - total_cr)
    if diff > 0.02:
        logger.warning("  GL IMBALANCE: DR=%.2f CR=%.2f diff=%.4f", total_dr, total_cr, diff)
    else:
        logger.info("  GL balanced: DR=CR=%.2f (diff=%.4f)", total_dr, diff)

    # Export
    db.execute(f"""
        COPY erp_gl_journal TO '{trans_dir / "gl_journal.csv"}'
        (HEADER, DELIMITER ',')
    """)


def _register_cost_tables(
    db: duckdb.DuckDBPyConnection, output_dir: Path
) -> None:
    """Register cost/price lookups as DuckDB tables if not already present."""
    # SKU costs (cost_per_case)
    _maybe_register(
        db, "sku_cost_tbl",
        output_dir / "master" / "skus.csv", "sku_code", "cost_per_case",
    )
    # SKU prices (price_per_case)
    _maybe_register(
        db, "sku_price_tbl",
        output_dir / "master" / "skus.csv", "sku_code", "price_per_case",
    )
    # Ingredient costs (cost_per_kg)
    _maybe_register(
        db, "ing_cost_tbl",
        output_dir / "master" / "ingredients.csv", "ingredient_code", "cost_per_kg",
    )
    # Also add bulk costs to ing_cost_tbl
    bulk_csv = output_dir / "master" / "bulk_intermediates.csv"
    if bulk_csv.exists():
        with open(bulk_csv) as f:
            for row in csv.DictReader(f):
                try:
                    db.execute(
                        "INSERT INTO ing_cost_tbl VALUES (?, ?)",
                        [row["bulk_code"], float(row["cost_per_kg"])],
                    )
                except Exception:
                    pass


def _maybe_register(
    db: duckdb.DuckDBPyConnection,
    table_name: str,
    csv_path: Path,
    key_col: str,
    val_col: str,
) -> None:
    """Register a CSV column as a DuckDB lookup table."""
    try:
        db.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        return  # already exists
    except Exception:
        pass

    db.execute(f"CREATE TABLE {table_name} (sim_id VARCHAR, cost DOUBLE)")
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                db.execute(
                    f"INSERT INTO {table_name} VALUES (?, ?)",
                    [row[key_col], float(row[val_col])],
                )
