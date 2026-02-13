"""Enterprise Data Generator for Prism Sim.

Transforms simulation parquet output into normalized ERP tables (CSV)
loadable into PostgreSQL and Neo4j.

Usage:
    poetry run python -m scripts.erp --input-dir data/output --output-dir data/output/erp
"""
