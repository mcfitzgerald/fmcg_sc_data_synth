#!/usr/bin/env python3
"""Create a small data subset for fast diagnostic iteration.

Filters all parquet files to the first N days, copies small files as-is.

Usage:
    poetry run python scripts/analysis/slice_data.py
    poetry run python scripts/analysis/slice_data.py --days 60
    poetry run python scripts/analysis/slice_data.py --data-dir data/output --out-dir data/output_small
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


# Map: filename -> day column name
DAY_COLUMNS = {
    "shipments.parquet": "creation_day",
    "orders.parquet": "day",
    "inventory.parquet": "day",
    "batches.parquet": "day_produced",
    "forecasts.parquet": "day",
    "production_orders.parquet": "creation_day",
    "returns.parquet": "day",
}


def slice_parquet(src: Path, dst: Path, day_col: str, max_day: int) -> None:
    """Filter a parquet file to rows where day_col <= max_day."""
    pf = pq.ParquetFile(src)
    meta = pf.metadata
    print(f"  {src.name}: {meta.num_rows:,} rows, {meta.num_row_groups} RGs -> ", end="", flush=True)

    tables = []
    kept = 0
    for rg_idx in range(meta.num_row_groups):
        table = pf.read_row_group(rg_idx)
        col = table.column(day_col)
        mask = pc.less_equal(col, max_day)
        filtered = table.filter(mask)
        if filtered.num_rows > 0:
            tables.append(filtered)
            kept += filtered.num_rows

    if tables:
        import pyarrow as pa
        combined = pa.concat_tables(tables)
        pq.write_table(combined, dst, compression="snappy")
    else:
        # Write empty file with same schema
        pq.write_table(pf.read_row_group(0).slice(0, 0), dst)

    print(f"{kept:,} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="Slice simulation data to first N days")
    parser.add_argument("--days", type=int, default=30, help="Max day to keep (default: 30)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/output_small"))
    args = parser.parse_args()

    src_dir: Path = args.data_dir
    dst_dir: Path = args.out_dir
    max_day: int = args.days

    if not src_dir.exists():
        print(f"ERROR: source dir not found: {src_dir}")
        return 1

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Slicing {src_dir} -> {dst_dir} (days <= {max_day})\n")

    t0 = time.time()

    # Parquet files
    for filename, day_col in DAY_COLUMNS.items():
        src = src_dir / filename
        if src.exists():
            slice_parquet(src, dst_dir / filename, day_col, max_day)
        else:
            print(f"  {filename}: MISSING, skipping")

    # Copy small files as-is
    for name in ["metrics.json", "batch_ingredients.parquet"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            print(f"  {name}: copied")

    # Copy static_world directory
    static_src = src_dir / "static_world"
    static_dst = dst_dir / "static_world"
    if static_src.exists():
        if static_dst.exists():
            shutil.rmtree(static_dst)
        shutil.copytree(static_src, static_dst)
        print(f"  static_world/: copied")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Show sizes
    total = 0
    for f in sorted(dst_dir.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
    print(f"Output size: {total / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
