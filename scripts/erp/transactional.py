"""DuckDB-based transactional table generation.

Reads simulation parquet files and produces normalized transactional CSVs.
Large tables (orders, shipments) use DuckDB SQL + COPY TO for speed.
Small tables (<200K rows) use Python iteration.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import duckdb

from .config import ErpConfig
from .id_mapper import IdMapper
from .master_tables import classify_node
from .sequence import CAT_MULTIPLIER, DAY_MULTIPLIER

logger = logging.getLogger(__name__)

# Route key mapping: (src_echelon, tgt_echelon) → cost_master route key
ROUTE_KEY_MAP: dict[tuple[str, str], str] = {
    ("supplier", "plant"): "supplier_to_plant",
    ("plant", "rdc"): "plant_to_rdc",
    ("plant", "customer_dc"): "plant_to_dc",
    ("plant", "store"): "plant_to_dc",
    ("rdc", "customer_dc"): "rdc_to_dc",
    ("rdc", "store"): "rdc_to_dc",
    ("customer_dc", "store"): "dc_to_store",
}


def generate_transactional_tables(
    db: duckdb.DuckDBPyConnection,
    input_dir: Path,
    output_dir: Path,
    mapper: IdMapper,
    cfg: ErpConfig,
) -> None:
    """Generate all transactional CSV tables from parquet files."""
    trans_dir = output_dir / "transactional"
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Register parquet files as views
    _register_parquets(db, input_dir)

    # Register helper UDFs and tables
    _register_location_map(db, mapper)
    _register_product_map(db, mapper)
    _register_distance_map(db, input_dir)

    # ── Large tables: DuckDB SQL + COPY TO ────────────────────
    _generate_orders_duckdb(db, trans_dir)
    _generate_purchase_orders_duckdb(db, trans_dir)
    _generate_shipments_duckdb(db, trans_dir, cfg)
    _generate_goods_receipts_duckdb(db, trans_dir)
    _generate_inventory_duckdb(db, trans_dir)

    # ── Small tables: Python iteration ────────────────────────
    _generate_work_orders(db, trans_dir, mapper)
    _generate_batches(db, trans_dir, mapper)
    _generate_batch_ingredients(db, trans_dir, mapper)
    _generate_returns(db, trans_dir, mapper)
    _generate_forecasts(db, trans_dir, mapper)

    # Populate IdMapper from DuckDB header tables for downstream use
    _populate_mapper_from_duckdb(db, mapper)

    logger.info("Transactional tables complete")


def _register_parquets(db: duckdb.DuckDBPyConnection, input_dir: Path) -> None:
    """Register parquet files as DuckDB views."""
    parquets = {
        "pq_orders": "orders.parquet",
        "pq_shipments": "shipments.parquet",
        "pq_batches": "batches.parquet",
        "pq_batch_ingredients": "batch_ingredients.parquet",
        "pq_production_orders": "production_orders.parquet",
        "pq_forecasts": "forecasts.parquet",
        "pq_returns": "returns.parquet",
        "pq_inventory": "inventory.parquet",
    }
    for view_name, filename in parquets.items():
        path = input_dir / filename
        if path.exists():
            db.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            count = db.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
            logger.info("  %s: %s rows", view_name, f"{count:,}")
        else:
            logger.warning("  %s not found, skipping", filename)


def _register_location_map(
    db: duckdb.DuckDBPyConnection, mapper: IdMapper
) -> None:
    """Create a DuckDB table mapping sim location IDs → integer PKs."""
    loc_ids = mapper.all_ids("locations")
    if not loc_ids:
        return
    db.execute("CREATE OR REPLACE TABLE loc_map (sim_id VARCHAR, pk INTEGER)")
    db.executemany(
        "INSERT INTO loc_map VALUES (?, ?)",
        [(sid, pk) for sid, pk in loc_ids.items()],
    )


def _register_product_map(
    db: duckdb.DuckDBPyConnection, mapper: IdMapper
) -> None:
    """Create a DuckDB table mapping sim product IDs → integer PKs."""
    prod_ids = mapper.all_ids("products")
    if not prod_ids:
        return
    db.execute("CREATE OR REPLACE TABLE prod_map (sim_id VARCHAR, pk INTEGER)")
    db.executemany(
        "INSERT INTO prod_map VALUES (?, ?)",
        [(sid, pk) for sid, pk in prod_ids.items()],
    )


def _register_distance_map(
    db: duckdb.DuckDBPyConnection, input_dir: Path
) -> None:
    """Register distance lookup from raw_links (already loaded in master phase)."""
    # raw_links is already loaded from master tables phase
    pass


# ── Orders (DuckDB-native) ──────────────────────────────────


def _generate_orders_duckdb(
    db: duckdb.DuckDBPyConnection, trans_dir: Path
) -> None:
    """Generate orders + order_lines via DuckDB SQL + COPY TO."""
    logger.info("  Generating orders (DuckDB)...")

    # Create order headers with integer PKs
    db.execute("""
        CREATE OR REPLACE TABLE erp_orders AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY MIN(day), order_id) as id,
            order_id as order_number,
            MIN(day) as day,
            COALESCE(sm.pk, 0) as source_id,
            COALESCE(tm.pk, 0) as retail_location_id,
            FIRST(status) as status,
            CAST(SUM(quantity) AS INTEGER) as total_cases,
            CAST(MIN(day) AS BIGINT) * 10000000 + 4 * 1000000 +
                CAST(ROW_NUMBER() OVER (ORDER BY MIN(day), order_id) AS BIGINT) as transaction_sequence_id
        FROM pq_orders o
        LEFT JOIN loc_map sm ON sm.sim_id = o.source_id
        LEFT JOIN loc_map tm ON tm.sim_id = o.target_id
        WHERE o.order_id LIKE 'ORD-%'
        GROUP BY o.order_id, sm.pk, tm.pk
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_orders").fetchone()[0]
    db.execute(f"""
        COPY erp_orders TO '{trans_dir / "orders.csv"}'
        (HEADER, DELIMITER ',')
    """)
    logger.info("  Order headers: %s", f"{count:,}")

    # Create order lines
    db.execute(f"""
        COPY (
            SELECT
                eo.id as order_id,
                ROW_NUMBER() OVER (PARTITION BY eo.id ORDER BY o.product_id) as line_number,
                COALESCE(pm.pk, 0) as sku_id,
                o.quantity as quantity_cases,
                0.0 as unit_price,
                o.status
            FROM pq_orders o
            JOIN erp_orders eo ON eo.order_number = o.order_id
            LEFT JOIN prod_map pm ON pm.sim_id = o.product_id
            WHERE o.order_id LIKE 'ORD-%'
        ) TO '{trans_dir / "order_lines.csv"}'
        (HEADER, DELIMITER ',')
    """)

    line_count = db.execute("""
        SELECT COUNT(*) FROM pq_orders WHERE order_id LIKE 'ORD-%'
    """).fetchone()[0]
    logger.info("  Order lines: %s", f"{line_count:,}")


# ── Purchase Orders (DuckDB-native) ─────────────────────────


def _generate_purchase_orders_duckdb(
    db: duckdb.DuckDBPyConnection, trans_dir: Path
) -> None:
    """Generate purchase_orders + purchase_order_lines via DuckDB."""
    logger.info("  Generating purchase orders (DuckDB)...")

    db.execute("""
        CREATE OR REPLACE TABLE erp_pos AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY MIN(day), order_id) as id,
            order_id as po_number,
            COALESCE(sm.pk, 0) as supplier_id,
            COALESCE(tm.pk, 0) as plant_id,
            MIN(day) as order_date,
            FIRST(status) as status,
            CAST(MIN(day) AS BIGINT) * 10000000 + 0 * 1000000 +
                CAST(ROW_NUMBER() OVER (ORDER BY MIN(day), order_id) AS BIGINT) as transaction_sequence_id
        FROM pq_orders o
        LEFT JOIN loc_map sm ON sm.sim_id = o.source_id
        LEFT JOIN loc_map tm ON tm.sim_id = o.target_id
        WHERE o.order_id LIKE 'PO-%'
        GROUP BY o.order_id, sm.pk, tm.pk
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_pos").fetchone()[0]
    db.execute(f"""
        COPY erp_pos TO '{trans_dir / "purchase_orders.csv"}'
        (HEADER, DELIMITER ',')
    """)

    # PO lines
    db.execute(f"""
        COPY (
            SELECT
                po.id as po_id,
                ROW_NUMBER() OVER (PARTITION BY po.id ORDER BY o.product_id) as line_number,
                COALESCE(pm.pk, 0) as ingredient_id,
                o.quantity as quantity_kg,
                0.0 as unit_cost,
                o.status
            FROM pq_orders o
            JOIN erp_pos po ON po.po_number = o.order_id
            LEFT JOIN prod_map pm ON pm.sim_id = o.product_id
            WHERE o.order_id LIKE 'PO-%'
        ) TO '{trans_dir / "purchase_order_lines.csv"}'
        (HEADER, DELIMITER ',')
    """)
    logger.info("  POs: %s headers", f"{count:,}")


# ── Shipments (DuckDB-native) ───────────────────────────────


def _generate_shipments_duckdb(
    db: duckdb.DuckDBPyConnection, trans_dir: Path, cfg: ErpConfig
) -> None:
    """Generate shipments + shipment_lines via DuckDB."""
    logger.info("  Generating shipments (DuckDB)...")

    # Register route cost config as a DuckDB table for lookups
    _register_route_costs(db, cfg)

    db.execute("""
        CREATE OR REPLACE TABLE erp_shipments AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY MIN(creation_day), shipment_id) as id,
            shipment_id as shipment_number,
            MIN(creation_day) as ship_date,
            MIN(arrival_day) as arrival_date,
            COALESCE(sm.pk, 0) as origin_id,
            COALESCE(tm.pk, 0) as destination_id,
            FIRST(status) as status,
            COALESCE(rc.route_key, 'other') as route_type,
            -- Freight: FTL = cost_per_km * dist / batch_size * qty; LTL = cost_per_case * qty
            CASE
                WHEN rc.mode = 'LTL'
                THEN COALESCE(rc.cost_per_case, 0.75) * SUM(quantity)
                ELSE COALESCE(rc.cost_per_km, 1.85) * COALESCE(lk.distance_km, 0) /
                     GREATEST(SUM(quantity), 100) * SUM(quantity)
            END + COALESCE(rc.handling_cost, 0.20) * SUM(quantity) as freight_cost,
            SUM(COALESCE(total_weight_kg, 0)) as total_weight_kg,
            FIRST(s.source_id) as source_sim_id,
            CAST(MIN(creation_day) AS BIGINT) * 10000000 + 2 * 1000000 +
                CAST(ROW_NUMBER() OVER (ORDER BY MIN(creation_day), shipment_id) AS BIGINT) as transaction_sequence_id
        FROM pq_shipments s
        LEFT JOIN loc_map sm ON sm.sim_id = s.source_id
        LEFT JOIN loc_map tm ON tm.sim_id = s.target_id
        LEFT JOIN raw_links lk ON lk.source_id = s.source_id AND lk.target_id = s.target_id
        LEFT JOIN route_cost_map rc ON rc.src_prefix = (
            CASE
                WHEN s.source_id LIKE 'SUP-%' THEN 'supplier'
                WHEN s.source_id LIKE 'PLANT-%' THEN 'plant'
                WHEN s.source_id LIKE 'RDC-%' THEN 'rdc'
                WHEN s.source_id LIKE 'STORE-%' THEN 'store'
                ELSE 'customer_dc'
            END
        ) AND rc.tgt_prefix = (
            CASE
                WHEN s.target_id LIKE 'PLANT-%' THEN 'plant'
                WHEN s.target_id LIKE 'RDC-%' THEN 'rdc'
                WHEN s.target_id LIKE 'STORE-%' THEN 'store'
                ELSE 'customer_dc'
            END
        )
        GROUP BY s.shipment_id, sm.pk, tm.pk, rc.route_key, rc.mode,
                 rc.cost_per_case, rc.cost_per_km, rc.handling_cost, lk.distance_km
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_shipments").fetchone()[0]
    db.execute(f"""
        COPY (
            SELECT id, shipment_number, ship_date, arrival_date,
                   origin_id, destination_id, status, route_type,
                   ROUND(freight_cost, 2) as freight_cost,
                   ROUND(total_weight_kg, 2) as total_weight_kg,
                   transaction_sequence_id
            FROM erp_shipments
        ) TO '{trans_dir / "shipments.csv"}'
        (HEADER, DELIMITER ',')
    """)
    logger.info("  Shipment headers: %s", f"{count:,}")

    # Shipment lines
    db.execute(f"""
        COPY (
            SELECT
                es.id as shipment_id,
                ROW_NUMBER() OVER (PARTITION BY es.id ORDER BY s.product_id) as line_number,
                COALESCE(pm.pk, 0) as sku_id,
                s.quantity as quantity_cases,
                ROUND(COALESCE(s.total_weight_kg, 0), 2) as weight_kg
            FROM pq_shipments s
            JOIN erp_shipments es ON es.shipment_number = s.shipment_id
            LEFT JOIN prod_map pm ON pm.sim_id = s.product_id
        ) TO '{trans_dir / "shipment_lines.csv"}'
        (HEADER, DELIMITER ',')
    """)

    line_count = db.execute("SELECT COUNT(*) FROM pq_shipments").fetchone()[0]
    logger.info("  Shipment lines: %s", f"{line_count:,}")


def _register_route_costs(db: duckdb.DuckDBPyConnection, cfg: ErpConfig) -> None:
    """Register route cost config as a DuckDB table."""
    db.execute("""
        CREATE OR REPLACE TABLE route_cost_map (
            src_prefix VARCHAR, tgt_prefix VARCHAR,
            route_key VARCHAR, mode VARCHAR,
            cost_per_km DOUBLE, cost_per_case DOUBLE,
            handling_cost DOUBLE
        )
    """)
    for (src, tgt), route_key in ROUTE_KEY_MAP.items():
        rc = cfg.route_costs.get(route_key, {})
        db.execute(
            "INSERT INTO route_cost_map VALUES (?, ?, ?, ?, ?, ?, ?)",
            [src, tgt, route_key, rc.get("mode", "FTL"),
             rc.get("cost_per_km", 1.85), rc.get("cost_per_case", 0.75),
             rc.get("handling_cost_per_case", 0.20)],
        )


# ── Goods Receipts (DuckDB-native) ──────────────────────────


def _generate_goods_receipts_duckdb(
    db: duckdb.DuckDBPyConnection, trans_dir: Path
) -> None:
    """Generate goods_receipts + goods_receipt_lines via DuckDB."""
    logger.info("  Generating goods receipts (DuckDB)...")

    db.execute("""
        CREATE OR REPLACE TABLE erp_goods_receipts AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY MIN(creation_day), shipment_id) as id,
            'GR-' || shipment_id as gr_number,
            COALESCE(es.id, 0) as shipment_id,
            COALESCE(tm.pk, 0) as plant_id,
            MIN(s.arrival_day) as receipt_date,
            'received' as status,
            CAST(MIN(s.arrival_day) AS BIGINT) * 10000000 + 0 * 1000000 +
                CAST(ROW_NUMBER() OVER (ORDER BY MIN(s.arrival_day), s.shipment_id) AS BIGINT) as transaction_sequence_id
        FROM pq_shipments s
        LEFT JOIN loc_map tm ON tm.sim_id = s.target_id
        LEFT JOIN erp_shipments es ON es.shipment_number = s.shipment_id
        WHERE s.target_id LIKE 'PLANT-%'
        GROUP BY s.shipment_id, es.id, tm.pk
    """)

    count = db.execute("SELECT COUNT(*) FROM erp_goods_receipts").fetchone()[0]
    db.execute(f"""
        COPY erp_goods_receipts TO '{trans_dir / "goods_receipts.csv"}'
        (HEADER, DELIMITER ',')
    """)

    # GR lines
    db.execute(f"""
        COPY (
            SELECT
                gr.id as gr_id,
                ROW_NUMBER() OVER (PARTITION BY gr.id ORDER BY s.product_id) as line_number,
                COALESCE(pm.pk, 0) as ingredient_id,
                s.quantity as quantity_kg
            FROM pq_shipments s
            JOIN erp_goods_receipts gr ON gr.gr_number = 'GR-' || s.shipment_id
            LEFT JOIN prod_map pm ON pm.sim_id = s.product_id
            WHERE s.target_id LIKE 'PLANT-%'
        ) TO '{trans_dir / "goods_receipt_lines.csv"}'
        (HEADER, DELIMITER ',')
    """)
    logger.info("  Goods receipts: %s", f"{count:,}")


# ── Inventory (DuckDB-native, weekly snapshots) ─────────────


def _generate_inventory_duckdb(
    db: duckdb.DuckDBPyConnection, trans_dir: Path
) -> None:
    """Generate inventory.csv — weekly snapshots via DuckDB COPY TO."""
    logger.info("  Generating inventory (DuckDB, weekly snapshots)...")

    max_day = db.execute("SELECT MAX(day) FROM pq_inventory").fetchone()[0]

    db.execute(f"""
        COPY (
            SELECT
                ROW_NUMBER() OVER () as id,
                i.day,
                CASE
                    WHEN i.node_id LIKE 'PLANT-%' THEN 'plant'
                    WHEN i.node_id LIKE 'RDC-%' THEN 'rdc'
                    WHEN i.node_id LIKE 'STORE-%' THEN 'store'
                    WHEN i.node_id LIKE 'SUP-%' THEN 'supplier'
                    ELSE 'customer_dc'
                END as location_type,
                COALESCE(lm.pk, 0) as location_id,
                COALESCE(pm.pk, 0) as sku_id,
                i.actual_inventory as quantity_cases
            FROM pq_inventory i
            LEFT JOIN loc_map lm ON lm.sim_id = i.node_id
            LEFT JOIN prod_map pm ON pm.sim_id = i.product_id
            WHERE (i.day % 7 = 0 OR i.day = {max_day})
                AND lm.pk IS NOT NULL AND pm.pk IS NOT NULL
        ) TO '{trans_dir / "inventory.csv"}'
        (HEADER, DELIMITER ',')
    """)

    count_result = db.execute(f"""
        SELECT COUNT(*) FROM pq_inventory
        WHERE (day % 7 = 0 OR day = {max_day})
    """).fetchone()[0]
    logger.info("  Inventory snapshots: ~%s rows", f"{count_result:,}")


# ── Small tables: Python iteration ───────────────────────────


def _generate_work_orders(
    db: duckdb.DuckDBPyConnection, trans_dir: Path, mapper: IdMapper
) -> None:
    """Generate work_orders.csv from production_orders.parquet."""
    logger.info("  Generating work orders...")

    rows = db.execute("""
        SELECT po_id, plant_id, product_id, quantity,
               creation_day, due_day, status
        FROM pq_production_orders ORDER BY po_id
    """).fetchall()

    wo_out: list[tuple] = []
    for row in rows:
        po_id, plant_id, prod_id, qty, create_day, due_day, status = row
        wo_pk = mapper.get("work_orders", po_id)
        plant_pk = mapper.lookup("locations", plant_id) or 0
        formula_pk = mapper.lookup("formulas", f"FORM-{prod_id}") or 0
        seq_id = create_day * DAY_MULTIPLIER + 1 * CAT_MULTIPLIER + wo_pk

        wo_out.append((
            wo_pk, po_id, plant_pk, formula_pk, qty,
            create_day, due_day, status, seq_id,
        ))

    _write_csv(
        trans_dir / "work_orders.csv",
        ["id", "wo_number", "plant_id", "formula_id", "planned_quantity_kg",
         "planned_start_date", "due_date", "status", "transaction_sequence_id"],
        wo_out,
    )
    logger.info("  Work orders: %d", len(wo_out))


def _generate_batches(
    db: duckdb.DuckDBPyConnection, trans_dir: Path, mapper: IdMapper
) -> None:
    """Generate batches.csv from batches.parquet."""
    logger.info("  Generating batches...")

    rows = db.execute("""
        SELECT batch_id, production_order_id, plant_id, product_id,
               day_produced, quantity, yield_pct, status
        FROM pq_batches ORDER BY batch_id
    """).fetchall()

    batch_out: list[tuple] = []
    for row in rows:
        (batch_id, po_id, plant_id, prod_id,
         day_produced, qty, yield_pct, status) = row
        batch_pk = mapper.get("batches", batch_id)
        wo_pk = mapper.lookup("work_orders", po_id) or 0
        plant_pk = mapper.lookup("locations", plant_id) or 0
        formula_pk = mapper.lookup("formulas", f"FORM-{prod_id}") or 0
        product_pk = mapper.lookup("products", prod_id) or 0
        product_type = "bulk_intermediate" if prod_id.startswith("BULK-") else "finished_good"
        bom_level = 1 if prod_id.startswith("BULK-") else 0
        seq_id = day_produced * DAY_MULTIPLIER + 1 * CAT_MULTIPLIER + batch_pk

        batch_out.append((
            batch_pk, batch_id, wo_pk, plant_pk, formula_pk, product_pk,
            qty, yield_pct, day_produced, status, product_type, bom_level, seq_id,
        ))

    _write_csv(
        trans_dir / "batches.csv",
        ["id", "batch_number", "wo_id", "plant_id", "formula_id",
         "product_id", "quantity_kg", "yield_percent", "production_date",
         "status", "product_type", "bom_level", "transaction_sequence_id"],
        batch_out,
    )
    logger.info("  Batches: %d", len(batch_out))


def _generate_batch_ingredients(
    db: duckdb.DuckDBPyConnection, trans_dir: Path, mapper: IdMapper
) -> None:
    """Generate batch_ingredients.csv."""
    logger.info("  Generating batch ingredients...")

    rows = db.execute("""
        SELECT batch_id, ingredient_id, quantity_kg
        FROM pq_batch_ingredients ORDER BY batch_id, ingredient_id
    """).fetchall()

    bi_out: list[tuple] = []
    for i, (batch_id, ing_id, qty_kg) in enumerate(rows, 1):
        batch_pk = mapper.lookup("batches", batch_id) or 0
        ing_pk = mapper.lookup("products", ing_id) or 0
        bi_out.append((i, batch_pk, ing_pk, round(qty_kg, 4)))

    _write_csv(
        trans_dir / "batch_ingredients.csv",
        ["id", "batch_id", "ingredient_id", "quantity_kg"],
        bi_out,
    )
    logger.info("  Batch ingredients: %d", len(bi_out))


def _generate_returns(
    db: duckdb.DuckDBPyConnection, trans_dir: Path, mapper: IdMapper
) -> None:
    """Generate returns, return_lines, disposition_logs."""
    logger.info("  Generating returns...")

    rows = db.execute("""
        SELECT rma_id, day, source_id, target_id, product_id,
               quantity, disposition, status
        FROM pq_returns ORDER BY rma_id
    """).fetchall()

    ret_out: list[tuple] = []
    ret_lines: list[tuple] = []
    disp_out: list[tuple] = []
    seen: set[str] = set()
    ret_pk_map: dict[str, int] = {}
    ret_counter = 0

    for rma_id, day, src, tgt, prod, qty, disposition, status in rows:
        if rma_id not in seen:
            seen.add(rma_id)
            ret_counter += 1
            ret_pk = ret_counter
            ret_pk_map[rma_id] = ret_pk
            src_pk = mapper.lookup("locations", src) or 0
            tgt_pk = mapper.lookup("locations", tgt) or 0
            seq_id = day * DAY_MULTIPLIER + 5 * CAT_MULTIPLIER + ret_pk
            ret_out.append((ret_pk, rma_id, day, src_pk, tgt_pk, status, seq_id))

        pk = ret_pk_map[rma_id]
        prod_pk = mapper.lookup("products", prod) or 0
        condition = "sellable" if disposition == "restock" else "damaged"
        ret_lines.append((pk, 1, prod_pk, qty, condition))
        disp_out.append((pk, 1, disposition, qty))

    _write_csv(trans_dir / "returns.csv",
               ["id", "return_number", "return_date", "source_id", "dc_id",
                "status", "transaction_sequence_id"], ret_out)
    _write_csv(trans_dir / "return_lines.csv",
               ["return_id", "line_number", "sku_id", "quantity_cases", "condition"],
               ret_lines)
    _write_csv(trans_dir / "disposition_logs.csv",
               ["return_id", "return_line_number", "disposition", "quantity_cases"],
               disp_out)
    logger.info("  Returns: %d", len(ret_out))


def _generate_forecasts(
    db: duckdb.DuckDBPyConnection, trans_dir: Path, mapper: IdMapper
) -> None:
    """Generate demand_forecasts.csv."""
    logger.info("  Generating forecasts...")

    rows = db.execute("""
        SELECT day, product_id, forecast_quantity
        FROM pq_forecasts ORDER BY day, product_id
    """).fetchall()

    fc_out: list[tuple] = []
    for i, (day, prod, qty) in enumerate(rows, 1):
        prod_pk = mapper.lookup("products", prod) or 0
        version = f"FCAST-DAY-{day:03d}"
        fc_out.append((i, version, prod_pk, "total", day + 1, qty, "statistical"))

    _write_csv(trans_dir / "demand_forecasts.csv",
               ["id", "forecast_version", "sku_id", "location_type",
                "forecast_date", "forecast_quantity_cases", "forecast_type"],
               fc_out)
    logger.info("  Forecasts: %d", len(fc_out))


# ── Helpers ──────────────────────────────────────────────────


def _populate_mapper_from_duckdb(
    db: duckdb.DuckDBPyConnection, mapper: IdMapper
) -> None:
    """Populate IdMapper with shipment and goods_receipt IDs from DuckDB tables."""
    # Shipments
    try:
        shp_rows = db.execute(
            "SELECT shipment_number, id FROM erp_shipments"
        ).fetchall()
        for sim_id, pk in shp_rows:
            mapper.get("shipments", sim_id)
    except Exception:
        pass

    # Goods receipts
    try:
        gr_rows = db.execute(
            "SELECT gr_number, id FROM erp_goods_receipts"
        ).fetchall()
        for sim_id, pk in gr_rows:
            mapper.get("goods_receipts", sim_id)
    except Exception:
        pass


def _write_csv(path: Path, headers: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
