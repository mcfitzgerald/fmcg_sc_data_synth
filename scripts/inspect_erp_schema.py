"""Inspect erp.duckdb schema metadata: tables, FKs, domain labels, column comments.

Usage:
    poetry run python scripts/inspect_erp_schema.py
    poetry run python scripts/inspect_erp_schema.py --db path/to/erp.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect erp.duckdb schema metadata")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/output/erp/erp.duckdb"),
        help="Path to erp.duckdb file",
    )
    args = parser.parse_args()

    db = duckdb.connect(str(args.db), read_only=True)

    # ── Tables by domain ─────────────────────────────────────
    print("=" * 80)
    print("TABLES BY DOMAIN")
    print("=" * 80)
    rows = db.execute("""
        SELECT table_name, comment AS domain, estimated_size AS est_rows
        FROM duckdb_tables()
        WHERE comment IS NOT NULL
        ORDER BY comment, table_name
    """).fetchall()

    current_domain = ""
    for table_name, domain, est_rows in rows:
        if domain != current_domain:
            current_domain = domain
            print(f"\n  {domain}")
            print(f"  {'-' * len(domain)}")
        row_str = f"{est_rows:,}" if est_rows else "?"
        print(f"    {table_name:<35s} {row_str:>15s} rows")
    print(f"\n  Total: {len(rows)} tables")

    # ── Primary keys ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PRIMARY KEYS")
    print("=" * 80)
    rows = db.execute("""
        SELECT table_name, constraint_column_names
        FROM duckdb_constraints()
        WHERE constraint_type = 'PRIMARY KEY'
        ORDER BY table_name
    """).fetchall()
    for table_name, cols in rows:
        print(f"  {table_name:<35s} {cols}")
    print(f"\n  Total: {len(rows)} PKs")

    # ── Foreign keys ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FOREIGN KEYS")
    print("=" * 80)
    rows = db.execute("""
        SELECT table_name, constraint_column_names
        FROM duckdb_constraints()
        WHERE constraint_type = 'FOREIGN KEY'
        ORDER BY table_name
    """).fetchall()
    for table_name, cols in rows:
        print(f"  {table_name:<35s} {cols}")
    print(f"\n  Total: {len(rows)} FKs")

    # ── Column comments ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("COLUMN COMMENTS (semantic hints)")
    print("=" * 80)
    rows = db.execute("""
        SELECT table_name, column_name, comment
        FROM duckdb_columns()
        WHERE comment IS NOT NULL
        ORDER BY table_name, column_name
    """).fetchall()
    for table_name, col_name, comment in rows:
        print(f"  {table_name}.{col_name}: {comment}")
    print(f"\n  Total: {len(rows)} annotated columns")

    # ── Indexes ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("INDEXES")
    print("=" * 80)
    rows = db.execute("""
        SELECT index_name, table_name, is_unique
        FROM duckdb_indexes()
        ORDER BY table_name, index_name
    """).fetchall()
    for idx_name, table_name, is_unique in rows:
        unique_str = " (UNIQUE)" if is_unique else ""
        print(f"  {idx_name:<45s} ON {table_name}{unique_str}")
    print(f"\n  Total: {len(rows)} indexes")

    db.close()


if __name__ == "__main__":
    main()
