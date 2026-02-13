"""CLI entry point for ERP data generation.

Usage:
    poetry run python -m scripts.erp --input-dir data/output --output-dir data/output/erp
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import duckdb

from .config import load_erp_config
from .id_mapper import IdMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("erp")


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
        help="ERP CSV output directory",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("src/prism_sim/config"),
        help="Config directory (cost_master.json, etc.)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    config_dir: Path = args.config_dir

    # Resolve static world + parquet dirs
    static_dir = input_dir / "static_world"
    if not static_dir.exists():
        static_dir = input_dir  # flat layout fallback

    t0 = time.perf_counter()
    logger.info("ERP generation starting: %s → %s", input_dir, output_dir)

    # Create output dirs
    for subdir in ("master", "transactional", "reference"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = load_erp_config(config_dir)

    # Initialize ID mapper
    mapper = IdMapper()

    # DuckDB in-memory connection
    db = duckdb.connect()

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

    # ── Phase 4: Save artifacts ───────────────────────────────
    mapper.save(output_dir / "reference" / "id_mapping.json")

    from .neo4j_headers import generate_neo4j_headers

    generate_neo4j_headers(output_dir)
    t4 = time.perf_counter()
    logger.info("Phase 4 done in %.1fs", t4 - t3)

    # ── Verify ────────────────────────────────────────────────
    from .verify import run_verification

    logger.info("Running verification checks")
    run_verification(output_dir)

    db.close()

    total = time.perf_counter() - t0
    logger.info("ERP generation complete in %.1fs", total)


if __name__ == "__main__":
    main()
