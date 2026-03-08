"""CLI entry point for ERP data generation.

Usage:
    poetry run python -m scripts.erp --input-dir data/output --output-dir data/output/erp
    poetry run python -m scripts.erp --input-dir data/output --output-dir data/output/erp --format parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .config import load_erp_config
from .id_mapper import IdMapper
from .schema import TABLE_MAP, get_column_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("erp")


# ── Export Map: DuckDB table → (subdir, schema_name) ──────────
# Tables that live in DuckDB and need to be exported.
# Master tables are written by Python (csv.writer) and are NOT in this map.
DUCKDB_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # Transactional (Phase 2)
    "erp_orders": ("transactional", "orders"),
    "erp_order_lines": ("transactional", "order_lines"),
    "erp_pos": ("transactional", "purchase_orders"),
    "erp_po_lines": ("transactional", "purchase_order_lines"),
    "erp_shipments": ("transactional", "shipments"),
    "erp_shipment_lines": ("transactional", "shipment_lines"),
    "erp_goods_receipts": ("transactional", "goods_receipts"),
    "erp_gr_lines": ("transactional", "goods_receipt_lines"),
    "erp_inventory": ("transactional", "inventory"),
    "erp_work_orders": ("transactional", "work_orders"),
    "erp_batches": ("transactional", "batches"),
    "erp_batch_ingredients": ("transactional", "batch_ingredients"),
    "erp_returns": ("transactional", "returns"),
    "erp_return_lines": ("transactional", "return_lines"),
    "erp_disposition_logs": ("transactional", "disposition_logs"),
    "erp_demand_forecasts": ("transactional", "demand_forecasts"),
    # Financial (Phase 3)
    "erp_gl_journal": ("transactional", "gl_journal"),
    "erp_ap_invoices": ("transactional", "ap_invoices"),
    "erp_ap_invoice_lines": ("transactional", "ap_invoice_lines"),
    "erp_ar_invoices": ("transactional", "ar_invoices"),
    "erp_ar_invoice_lines": ("transactional", "ar_invoice_lines"),
    # Friction (Phase 3.5) — may not exist if friction disabled
    "erp_invoice_variances": ("transactional", "invoice_variances"),
    "erp_ap_payments": ("transactional", "ap_payments"),
    "erp_ar_receipts": ("transactional", "ar_receipts"),
}

# Tables friction expects with short names (rename erp_ → short before friction)
_FRICTION_RENAMES: dict[str, str] = {
    "erp_ap_invoices": "ap_invoices",
    "erp_ap_invoice_lines": "ap_invoice_lines",
    "erp_ar_invoices": "ar_invoices",
    "erp_gl_journal": "gl_journal",
    "erp_goods_receipts": "goods_receipts",
    "erp_shipments": "shipments",
    "erp_suppliers": "suppliers",
    "erp_skus": "skus",
}


def main() -> None:
    """Orchestrate the full ERP generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate ERP tables from Prism Sim output"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/output"),
        help="Sim output directory (contains parquet + static_world/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/erp"),
        help="ERP output directory",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("src/prism_sim/config"),
        help="Config directory (cost_master.json, etc.)",
    )
    parser.add_argument(
        "--reporting-date",
        type=int,
        default=None,
        help="Reporting snapshot day (default: max_day - 25). "
             "Documents after this day are excluded; recent ones keep intermediate statuses.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format (default: parquet). CSV kept for backwards compat.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    config_dir: Path = args.config_dir
    output_format: str = args.format

    # Resolve static world + parquet dirs
    static_dir = input_dir / "static_world"
    if not static_dir.exists():
        static_dir = input_dir  # flat layout fallback

    t0 = time.perf_counter()
    logger.info("ERP generation starting: %s → %s (format=%s)",
                input_dir, output_dir, output_format)

    # Create output dirs
    for subdir in ("master", "transactional", "reference"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = load_erp_config(config_dir)

    # Resolve reporting_date: CLI > auto-compute from max(day) in orders.parquet
    if args.reporting_date is not None:
        cfg.reporting_date = args.reporting_date
    else:
        orders_path = input_dir / "orders.parquet"
        if orders_path.exists():
            md = pq.read_metadata(orders_path)
            # Quick max(day) via row group statistics
            max_day = 0
            for rg_idx in range(md.num_row_groups):
                rg = md.row_group(rg_idx)
                for col_idx in range(rg.num_columns):
                    col = rg.column(col_idx)
                    if col.path_in_schema == "day" and col.statistics is not None:
                        max_day = max(max_day, col.statistics.max)
            if max_day == 0:
                # Fallback: read the column
                tbl = pq.read_table(orders_path, columns=["day"])
                max_day = tbl.column("day").to_pylist()[-1]
            cfg.reporting_date = max_day - 25
        else:
            cfg.reporting_date = 999999  # no filter
    logger.info("Reporting date: day %d", cfg.reporting_date)

    # Initialize ID mapper
    mapper = IdMapper()

    # DuckDB in-memory connection (single session for entire pipeline)
    db = duckdb.connect()
    db.execute("SET memory_limit='10GB'")

    # ── Phase 1: Master Tables ────────────────────────────────
    from .master_tables import generate_master_tables

    logger.info("Phase 1: Master tables")
    generate_master_tables(db, static_dir, output_dir, mapper, cfg)
    t1 = time.perf_counter()
    logger.info("Phase 1 done in %.1fs", t1 - t0)

    # ── Phase 2: Transactional Tables ─────────────────────────
    from .transactional import generate_transactional_tables

    logger.info("Phase 2: Transactional tables")
    generate_transactional_tables(db, input_dir, output_dir, mapper, cfg)
    t2 = time.perf_counter()
    logger.info("Phase 2 done in %.1fs", t2 - t1)

    # ── Phase 3: GL Journal + Invoices ────────────────────────
    from .gl_journal import generate_gl_journal
    from .invoices import generate_invoices

    logger.info("Phase 3: Financial layer")
    generate_gl_journal(db, output_dir, mapper, cfg)
    generate_invoices(db, output_dir, mapper, cfg)
    t3 = time.perf_counter()
    logger.info("Phase 3 done in %.1fs", t3 - t2)

    # ── Free memory: drop consumed intermediates ────────────────
    # Hot parquets + costed_shipments were consumed by Phases 2-3.
    # Dropping them frees several GB for friction's GL re-sort.
    # Drop VIEWs (pq_* parquet references)
    for v in (
        "pq_shipments", "pq_orders", "pq_batches",
        "pq_batch_ingredients", "pq_production_orders",
        "pq_forecasts", "pq_returns", "pq_inventory",
    ):
        db.execute(f"DROP VIEW IF EXISTS {v}")
    # Drop TABLEs (materialized intermediates)
    for t in (
        "costed_shipments",
        "raw_products", "raw_locations", "raw_links", "raw_recipes",
        "loc_map", "prod_map", "route_cost_map",
        "sku_cost_tbl", "sku_price_tbl", "ing_cost_tbl",
        "dso_tbl",
    ):
        db.execute(f"DROP TABLE IF EXISTS {t}")

    # ── Phase 3.5: Friction Layer (optional) ──────────────────
    if cfg.friction.enabled:
        from .friction import generate_friction

        logger.info("Phase 3.5: Friction layer")

        # Rename erp_ tables to short names for friction
        for erp_name, short_name in _FRICTION_RENAMES.items():
            try:
                db.execute(f"ALTER TABLE {erp_name} RENAME TO {short_name}")
            except duckdb.CatalogException:
                logger.warning("  Table %s not found for friction rename", erp_name)

        friction_stats = generate_friction(output_dir, cfg, main_db=db)

        # Rename back: short → erp_
        for erp_name, short_name in _FRICTION_RENAMES.items():
            try:
                db.execute(f"ALTER TABLE {short_name} RENAME TO {erp_name}")
            except duckdb.CatalogException:
                pass

        # Pick up new friction tables
        for short_name in ("invoice_variances", "ap_payments", "ar_receipts"):
            try:
                db.execute(f"SELECT 1 FROM {short_name} LIMIT 1")
                db.execute(f"ALTER TABLE {short_name} RENAME TO erp_{short_name}")
            except duckdb.CatalogException:
                pass

        t35 = time.perf_counter()
        logger.info("Phase 3.5 done in %.1fs — %s", t35 - t3, friction_stats)
    else:
        logger.info("Phase 3.5: Friction disabled, skipping")

    # ── Phase 4: Verify + Export + Artifacts ──────────────────
    from .verify import run_verification

    logger.info("Running verification checks")
    run_verification(output_dir, db=db)

    # Save ID mapping
    mapper.save(output_dir / "reference" / "id_mapping.json")

    # Generate Neo4j headers + PostgreSQL DDL
    from .neo4j_headers import generate_neo4j_headers
    from .schema import generate_ddl

    generate_neo4j_headers(output_dir)

    schema_path = output_dir / "erp_schema.sql"
    schema_path.write_text(generate_ddl())
    logger.info("Schema DDL written to %s", schema_path)

    # ── Export all DuckDB tables ──────────────────────────────
    t_export = time.perf_counter()
    _export_all(db, output_dir, output_format)
    t4 = time.perf_counter()
    logger.info("Export done in %.1fs (%s)", t4 - t_export, output_format)

    db.close()

    total = time.perf_counter() - t0
    logger.info("ERP generation complete in %.1fs", total)


def _export_all(
    db: duckdb.DuckDBPyConnection,
    output_dir: Path,
    fmt: str,
) -> None:
    """Export all DuckDB tables to output_dir in the chosen format.

    Uses schema.py column definitions to select only schema-defined columns,
    stripping internal columns like source_sim_id.
    """
    ext = "parquet" if fmt == "parquet" else "csv"
    exported = 0

    for ddb_name, (subdir, schema_name) in DUCKDB_EXPORT_MAP.items():
        # Check table exists
        try:
            db.execute(f"SELECT 1 FROM {ddb_name} LIMIT 1")
        except duckdb.CatalogException:
            continue

        # Get schema columns for column selection
        if schema_name in TABLE_MAP:
            cols = get_column_names(schema_name)
            # Verify all schema columns exist in the DuckDB table
            ddb_cols = {
                r[0] for r in db.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{ddb_name}'"
                ).fetchall()
            }
            # Only select columns that exist in both schema and DuckDB
            select_cols = [c for c in cols if c in ddb_cols]
            col_list = ", ".join(select_cols)
        else:
            col_list = "*"

        out_path = output_dir / subdir / f"{schema_name}.{ext}"

        if fmt == "parquet":
            db.execute(f"""
                COPY (SELECT {col_list} FROM {ddb_name})
                TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            # ap_invoices needs ORDER BY for friction duplicate monotonicity
            order_clause = ""
            if schema_name == "ap_invoices":
                order_clause = " ORDER BY transaction_sequence_id, id"
            db.execute(f"""
                COPY (SELECT {col_list} FROM {ddb_name}{order_clause})
                TO '{out_path}' (HEADER, DELIMITER ',')
            """)

        count = db.execute(f"SELECT COUNT(*) FROM {ddb_name}").fetchone()[0]
        logger.info("  %-30s → %-45s %10s rows",
                     schema_name, str(out_path.name), f"{count:,}")
        exported += 1

    # Master tables: already written as CSV by master_tables.py
    # If format is parquet, convert master CSVs to parquet
    if fmt == "parquet":
        master_dir = output_dir / "master"
        for csv_file in sorted(master_dir.glob("*.csv")):
            schema_name = csv_file.stem
            if schema_name not in TABLE_MAP:
                continue
            pq_path = master_dir / f"{schema_name}.parquet"
            try:
                db.execute(f"""
                    COPY (SELECT * FROM read_csv_auto('{csv_file}'))
                    TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
                exported += 1
            except Exception as e:
                logger.warning("  Could not convert %s to parquet: %s",
                               csv_file.name, e)

    logger.info("Exported %d tables as %s", exported, fmt)


if __name__ == "__main__":
    main()
