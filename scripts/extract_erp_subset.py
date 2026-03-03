"""Extract a referentially-consistent ERP subset (~50-60 MB).

Strategy: Stratified sample of ~5% stores (~200 of 3,817), keep all upstream
infrastructure (plants, DCs, RDCs) in full, cascade FKs through transactional
tables, and filter GL journal by matching reference_ids.

Usage:
    poetry run python scripts/extract_erp_subset.py
    poetry run python scripts/extract_erp_subset.py --pct 10  # 10% of stores
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

SRC = Path("data/output/erp")
DST = Path("data/output/erp_test_30d")
MAX_SIZE_MB = 65  # WARN threshold for total subset size

# ── Tables kept in full (upstream, small) ──
UPSTREAM_TABLES = [
    "purchase_orders",
    "purchase_order_lines",
    "goods_receipts",
    "goods_receipt_lines",
    "work_orders",
    "batches",
    "batch_ingredients",
    "demand_forecasts",
    "ap_invoices",
    "ap_invoice_lines",
    "ap_payments",
    "invoice_variances",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ERP subset")
    parser.add_argument(
        "--pct",
        type=float,
        default=5.0,
        help="Percent of stores to sample per channel (default 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "master").mkdir(exist_ok=True)
    (DST / "transactional").mkdir(exist_ok=True)

    con = duckdb.connect()

    # Helper paths
    def src_master(name: str) -> str:
        return str(SRC / "master" / f"{name}.csv")

    def src_txn(name: str) -> str:
        return str(SRC / "transactional" / f"{name}.csv")

    def dst_master(name: str) -> str:
        return str(DST / "master" / f"{name}.csv")

    def dst_txn(name: str) -> str:
        return str(DST / "transactional" / f"{name}.csv")

    def write_csv(path: str, query: str) -> int:
        """Execute query → CSV, return row count."""
        con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")
        cnt: int = con.execute(
            f"SELECT count(*) FROM read_csv_auto('{path}')"
        ).fetchone()[0]  # type: ignore[index]
        return cnt

    row_counts: dict[str, int] = {}

    # ================================================================
    # Step 1: Stratified store sample
    # ================================================================
    print("Step 1: Stratified store sample")

    con.execute(f"SELECT setseed({args.seed / 1000.0})")
    con.execute(f"""
        CREATE TEMP TABLE sampled_stores AS
        WITH ranked AS (
            SELECT id, channel,
                   ROW_NUMBER() OVER (
                       PARTITION BY channel ORDER BY RANDOM()
                   ) AS rn,
                   COUNT(*) OVER (PARTITION BY channel) AS channel_total
            FROM read_csv_auto('{src_master("retail_locations")}')
        )
        SELECT id
        FROM ranked
        WHERE rn <= GREATEST(2, CEIL(channel_total * {args.pct} / 100.0))
    """)

    store_count: int = con.execute(
        "SELECT count(*) FROM sampled_stores"
    ).fetchone()[0]  # type: ignore[index]

    # Report per-channel counts
    channel_counts = con.execute(f"""
        SELECT rl.channel, count(*) AS n
        FROM read_csv_auto('{src_master("retail_locations")}') rl
        JOIN sampled_stores ss ON rl.id = ss.id
        GROUP BY rl.channel ORDER BY n DESC
    """).fetchall()
    print(f"  Sampled {store_count} stores:")
    for ch, n in channel_counts:
        short = ch.replace("CustomerChannel.", "")
        print(f"    {short}: {n}")

    # ================================================================
    # Step 2: Master tables — copy all as-is
    # ================================================================
    print("\nStep 2: Master tables (copy as-is)")
    for f in sorted((SRC / "master").glob("*.csv")):
        name = f.stem
        cnt = write_csv(
            dst_master(name),
            f"SELECT * FROM read_csv_auto('{f}')",
        )
        row_counts[f"master/{name}"] = cnt
        print(f"  {name}: {cnt:,}")

    # ================================================================
    # Step 3: Upstream transactional — copy in full
    # ================================================================
    print("\nStep 3: Upstream transactional (copy in full)")
    for name in UPSTREAM_TABLES:
        cnt = write_csv(
            dst_txn(name),
            f"SELECT * FROM read_csv_auto('{src_txn(name)}')",
        )
        row_counts[f"transactional/{name}"] = cnt
        print(f"  {name}: {cnt:,}")

    # ================================================================
    # Step 4: Store-filtered transactional — FK cascade
    # ================================================================
    print("\nStep 4: Store-filtered transactional (FK cascade)")

    # ── 4a: Orders ──
    con.execute(f"""
        CREATE TEMP TABLE filt_orders AS
        SELECT * FROM read_csv_auto('{src_txn("orders")}')
        WHERE retail_location_id IN (SELECT id FROM sampled_stores)
    """)
    cnt = write_csv(dst_txn("orders"), "SELECT * FROM filt_orders")
    row_counts["transactional/orders"] = cnt
    print(f"  orders: {cnt:,}")

    # ── 4b: Order lines ──
    cnt = write_csv(
        dst_txn("order_lines"),
        f"""SELECT ol.* FROM read_csv_auto('{src_txn("order_lines")}') ol
            WHERE ol.order_id IN (SELECT id FROM filt_orders)""",
    )
    row_counts["transactional/order_lines"] = cnt
    print(f"  order_lines: {cnt:,}")

    # ── 4c: Shipments — supplier_to_plant (full) + dc_to_store (filtered) ──
    # Internal movements (plant_to_rdc, rdc_to_dc, plant_to_dc) are dropped:
    # no other table FKs to them, and their GL entries are already excluded.
    # goods_receipts.shipment_id only references supplier_to_plant.
    # ar_invoices.shipment_id only references dc_to_store (for filtered stores).
    con.execute(f"""
        CREATE TEMP TABLE filt_shipments AS
        SELECT * FROM read_csv_auto('{src_txn("shipments")}')
        WHERE route_type = 'supplier_to_plant'
           OR (route_type = 'dc_to_store'
               AND destination_id IN (SELECT id FROM sampled_stores))
    """)
    cnt = write_csv(dst_txn("shipments"), "SELECT * FROM filt_shipments")
    row_counts["transactional/shipments"] = cnt
    print(f"  shipments: {cnt:,}")

    # ── 4d: Shipment lines ──
    cnt = write_csv(
        dst_txn("shipment_lines"),
        f"""SELECT sl.* FROM read_csv_auto('{src_txn("shipment_lines")}') sl
            WHERE sl.shipment_id IN (SELECT id FROM filt_shipments)""",
    )
    row_counts["transactional/shipment_lines"] = cnt
    print(f"  shipment_lines: {cnt:,}")

    # ── 4e: Returns ──
    con.execute(f"""
        CREATE TEMP TABLE filt_returns AS
        SELECT * FROM read_csv_auto('{src_txn("returns")}')
        WHERE source_id IN (SELECT id FROM sampled_stores)
    """)
    cnt = write_csv(dst_txn("returns"), "SELECT * FROM filt_returns")
    row_counts["transactional/returns"] = cnt
    print(f"  returns: {cnt:,}")

    # ── 4f: Return lines + disposition logs ──
    cnt = write_csv(
        dst_txn("return_lines"),
        f"""SELECT rl.* FROM read_csv_auto('{src_txn("return_lines")}') rl
            WHERE rl.return_id IN (SELECT id FROM filt_returns)""",
    )
    row_counts["transactional/return_lines"] = cnt
    print(f"  return_lines: {cnt:,}")

    cnt = write_csv(
        dst_txn("disposition_logs"),
        f"""SELECT dl.* FROM read_csv_auto('{src_txn("disposition_logs")}') dl
            WHERE dl.return_id IN (SELECT id FROM filt_returns)""",
    )
    row_counts["transactional/disposition_logs"] = cnt
    print(f"  disposition_logs: {cnt:,}")

    # ── 4g: AR invoices ──
    con.execute(f"""
        CREATE TEMP TABLE filt_ar_invoices AS
        SELECT * FROM read_csv_auto('{src_txn("ar_invoices")}')
        WHERE customer_location_id IN (SELECT id FROM sampled_stores)
    """)
    cnt = write_csv(dst_txn("ar_invoices"), "SELECT * FROM filt_ar_invoices")
    row_counts["transactional/ar_invoices"] = cnt
    print(f"  ar_invoices: {cnt:,}")

    # ── 4h: AR invoice lines ──
    cnt = write_csv(
        dst_txn("ar_invoice_lines"),
        f"""SELECT al.* FROM read_csv_auto('{src_txn("ar_invoice_lines")}') al
            WHERE al.invoice_id IN (SELECT id FROM filt_ar_invoices)""",
    )
    row_counts["transactional/ar_invoice_lines"] = cnt
    print(f"  ar_invoice_lines: {cnt:,}")

    # ── 4i: AR receipts ──
    cnt = write_csv(
        dst_txn("ar_receipts"),
        f"""SELECT ar.* FROM read_csv_auto('{src_txn("ar_receipts")}') ar
            WHERE ar.invoice_id IN (SELECT id FROM filt_ar_invoices)""",
    )
    row_counts["transactional/ar_receipts"] = cnt
    print(f"  ar_receipts: {cnt:,}")

    # ── 4j: Inventory — sampled stores + serving DCs + plants ──
    # Only keep DCs that are origins for filtered dc_to_store shipments,
    # plus all plants. Non-serving DCs/RDCs are dropped (saves ~2 MB).
    con.execute(f"""
        CREATE TEMP TABLE inventory_locs AS
        SELECT id FROM sampled_stores
        UNION
        SELECT DISTINCT origin_id FROM filt_shipments
            WHERE route_type = 'dc_to_store'
        UNION
        SELECT id FROM read_csv_auto('{src_master("plants")}')
    """)
    cnt = write_csv(
        dst_txn("inventory"),
        f"""SELECT * FROM read_csv_auto('{src_txn("inventory")}')
            WHERE location_id IN (SELECT id FROM inventory_locs)""",
    )
    row_counts["transactional/inventory"] = cnt
    print(f"  inventory: {cnt:,}")

    # ── 4k: GL Journal — mixed upstream/store logic ──
    # Upstream types (production, goods_receipt, payment, variance): keep all.
    # Store-facing types (shipment, sale, freight): only dc_to_store for sampled
    # stores — upstream route GL entries are omitted for size, but the underlying
    # shipments/shipment_lines are still preserved in full.
    cnt = write_csv(
        dst_txn("gl_journal"),
        f"""SELECT * FROM read_csv_auto('{src_txn("gl_journal")}')
            WHERE
                -- Upstream: keep all production, goods_receipt, payment, variance
                reference_type IN (
                    'production', 'goods_receipt', 'payment',
                    'price_variance', 'qty_variance', 'bad_debt'
                )
                -- Shipment/sale/freight: only dc_to_store for sampled stores
                OR (reference_type IN ('shipment', 'sale', 'freight')
                    AND reference_id IN (
                        SELECT shipment_number FROM filt_shipments
                        WHERE route_type = 'dc_to_store'
                    ))
                -- Return: match to filtered return_numbers
                OR (reference_type = 'return'
                    AND reference_id IN (
                        SELECT return_number FROM filt_returns
                    ))
                -- Receipt: match to filtered AR invoice_numbers
                OR (reference_type = 'receipt'
                    AND reference_id IN (
                        SELECT invoice_number FROM filt_ar_invoices
                    ))""",
    )
    row_counts["transactional/gl_journal"] = cnt
    print(f"  gl_journal: {cnt:,}")

    # ================================================================
    # Step 5: Verification
    # ================================================================
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    errors: list[str] = []

    # 5a: FK spot-checks
    checks = [
        (
            "order_lines → orders",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("order_lines")}') ol
                WHERE ol.order_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("orders")}'))""",
        ),
        (
            "shipment_lines → shipments",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("shipment_lines")}') sl
                WHERE sl.shipment_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("shipments")}'))""",
        ),
        (
            "ar_invoice_lines → ar_invoices",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("ar_invoice_lines")}') al
                WHERE al.invoice_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("ar_invoices")}'))""",
        ),
        (
            "ar_receipts → ar_invoices",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("ar_receipts")}') ar
                WHERE ar.invoice_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("ar_invoices")}'))""",
        ),
        (
            "return_lines → returns",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("return_lines")}') rl
                WHERE rl.return_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("returns")}'))""",
        ),
        (
            "disposition_logs → returns",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("disposition_logs")}') dl
                WHERE dl.return_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("returns")}'))""",
        ),
        (
            "batch_ingredients → batches",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("batch_ingredients")}') bi
                WHERE bi.batch_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("batches")}'))""",
        ),
        (
            "goods_receipts → shipments",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("goods_receipts")}') gr
                WHERE gr.shipment_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("shipments")}'))""",
        ),
        (
            "ar_invoices → shipments",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("ar_invoices")}') ar
                WHERE ar.shipment_id NOT IN (
                    SELECT id FROM read_csv_auto('{dst_txn("shipments")}'))""",
        ),
        (
            "gl_journal(shipment) → shipments",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("gl_journal")}') gl
                WHERE gl.reference_type IN ('shipment', 'sale', 'freight')
                  AND gl.reference_id NOT IN (
                    SELECT shipment_number
                    FROM read_csv_auto('{dst_txn("shipments")}'))""",
        ),
        (
            "gl_journal(return) → returns",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("gl_journal")}') gl
                WHERE gl.reference_type = 'return'
                  AND gl.reference_id NOT IN (
                    SELECT return_number
                    FROM read_csv_auto('{dst_txn("returns")}'))""",
        ),
        (
            "gl_journal(receipt) → ar_invoices",
            f"""SELECT count(*) FROM read_csv_auto('{dst_txn("gl_journal")}') gl
                WHERE gl.reference_type = 'receipt'
                  AND gl.reference_id NOT IN (
                    SELECT invoice_number
                    FROM read_csv_auto('{dst_txn("ar_invoices")}'))""",
        ),
    ]

    for label, query in checks:
        orphans: int = con.execute(query).fetchone()[0]  # type: ignore[index]
        status = "PASS" if orphans == 0 else "FAIL"
        if orphans > 0:
            errors.append(f"{label}: {orphans:,} orphan rows")
        print(f"  FK {label}: {status} ({orphans:,} orphans)")

    # 5b: Channel coverage
    channels_present = con.execute(f"""
        SELECT DISTINCT rl.channel
        FROM read_csv_auto('{dst_master("retail_locations")}') rl
        JOIN sampled_stores ss ON rl.id = ss.id
    """).fetchall()
    channels_in_full = con.execute(f"""
        SELECT DISTINCT channel
        FROM read_csv_auto('{src_master("retail_locations")}')
    """).fetchall()
    n_present = len(channels_present)
    n_full = len(channels_in_full)
    ch_status = "PASS" if n_present == n_full else "FAIL"
    print(f"  Channel coverage: {ch_status} ({n_present}/{n_full} channels)")
    if n_present < n_full:
        errors.append(
            f"Missing channels: {n_full - n_present} of {n_full}"
        )

    # 5c: Disk size
    total_bytes = sum(
        f.stat().st_size for f in DST.rglob("*.csv")
    )
    total_mb = total_bytes / (1024 * 1024)
    size_status = "PASS" if total_mb < MAX_SIZE_MB else "WARN"
    print(f"  Total disk size: {size_status} ({total_mb:.1f} MB)")

    if errors:
        print(f"\n  ERRORS: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  All checks passed!")

    # ================================================================
    # Step 6: Write SCHEMA.md
    # ================================================================
    write_schema_md(row_counts, store_count, total_mb, args.pct)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Subset written to {DST}/")

    con.close()


def write_schema_md(
    row_counts: dict[str, int],
    store_count: int,
    total_mb: float,
    pct: float,
) -> None:
    """Write SCHEMA.md with updated row counts."""
    # Build the row-count table for the template
    master_lines: list[str] = []
    txn_lines: list[str] = []
    for key, cnt in sorted(row_counts.items()):
        prefix, name = key.split("/", 1)
        line = f"| {name} | {cnt:,} |"
        if prefix == "master":
            master_lines.append(line)
        else:
            txn_lines.append(line)

    schema = f"""\
# ERP Test Subset — CSV Schema

**Subset strategy:** Stratified {pct:.0f}% store sample ({store_count} of 3,817 stores).
All upstream infrastructure (plants, RDCs, DCs) and upstream transactional tables
kept in full. Store-facing tables filtered by FK cascade. Total size: {total_mb:.1f} MB.

All dates are integer sim-day numbers (not calendar dates).
All IDs are integer surrogates. FK columns reference `id` in the parent table.

## Master Tables (14 tables, copied in full)

| Table | Rows |
|-------|------|
{chr(10).join(master_lines)}

## Transactional Tables (24 tables)

**Upstream (kept in full):** purchase_orders, purchase_order_lines, goods_receipts,
goods_receipt_lines, work_orders, batches, batch_ingredients, demand_forecasts,
ap_invoices, ap_invoice_lines, ap_payments, invoice_variances.

**Store-filtered (FK cascade):** orders, order_lines, shipments, shipment_lines,
returns, return_lines, disposition_logs, ar_invoices, ar_invoice_lines, ar_receipts,
inventory, gl_journal.

| Table | Rows |
|-------|------|
{chr(10).join(txn_lines)}

## Filtering Logic

```
sampled_stores (~{pct:.0f}% per channel, stratified)
    |
    +-- orders (WHERE retail_location_id IN stores)
    |     +-- order_lines (FK cascade)
    |
    +-- shipments (supplier_to_plant full + dc_to_store filtered)
    |     +-- shipment_lines (FK cascade)
    |     NOTE: internal movements (plant_to_rdc, rdc_to_dc, plant_to_dc) are dropped
    |           — no other table FKs to them
    |
    +-- returns (WHERE source_id IN stores)
    |     +-- return_lines (FK cascade)
    |     +-- disposition_logs (FK cascade)
    |
    +-- ar_invoices (WHERE customer_location_id IN stores)
    |     +-- ar_invoice_lines (FK cascade)
    |     +-- ar_receipts (FK cascade)
    |
    +-- inventory (sampled stores + serving DCs + all plants)
    |
    +-- gl_journal:
          upstream (production, goods_receipt, *_variance, payment, bad_debt) -> all
          shipment/sale/freight -> dc_to_store only (filtered shipment_numbers)
          return -> WHERE reference_id IN filtered return_numbers
          receipt -> WHERE reference_id IN filtered AR invoice_numbers
```

## Key Relationships (ER diagram, text)

```
suppliers --< supplier_ingredients >-- ingredients
    |                                      |
    +--< purchase_orders --< purchase_order_lines >-- ingredients
              |
              +-- goods_receipts --< goods_receipt_lines
                        |
                        +-- ap_invoices --< ap_invoice_lines
                                |              |
                                +-- invoice_variances
                                +-- ap_payments

plants --< production_lines
   |
   +--< work_orders --< batches --< batch_ingredients >-- ingredients
              |            |
              +-- formulas -+
                     |
                     +--< formula_ingredients

skus --< order_lines --< orders >-- retail_locations
  |                                      |
  |                                      +-- channels
  |
  +--< shipment_lines --< shipments
  |                          |
  |                          +-- distribution_centers (origin/destination)
  |                          +-- ar_invoices --< ar_invoice_lines
  |                                    |
  |                                    +-- ar_receipts
  |
  +--< inventory
  +--< demand_forecasts
  +--< return_lines --< returns --< disposition_logs

gl_journal >-- chart_of_accounts (via account_code)
```

## Notes

- `shipments.total_weight_kg` = bin capacity (FTL = 20,000 kg), not actual weight
- `transaction_sequence_id` = deterministic: `day * 10M + category * 1M + counter`
- `ap_invoices.status = 'duplicate'` = intentional friction (~0.5% of invoices)
- `gl_journal.is_reversal` = duplicate posting reversals (friction layer)
- For fill rate metrics, filter `shipments` to `route_type = 'dc_to_store'`
- `inventory` is sampled (may not have every day x location x sku combination)
"""
    (DST / "SCHEMA.md").write_text(schema)
    print(f"\n  Wrote {DST / 'SCHEMA.md'}")


if __name__ == "__main__":
    main()
