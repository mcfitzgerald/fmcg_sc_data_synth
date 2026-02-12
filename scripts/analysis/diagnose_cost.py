#!/usr/bin/env python3
"""
COST ANALYTICS — Post-Simulation Enrichment Diagnostic.

Computes SKU profitability, OTIF, working capital, and cost-to-serve
from simulation parquet output + cost_master.json config.

No simulation physics changes — pure post-hoc analysis.

Sections:
  1. COGS by ABC / Echelon
  2. Logistics Cost by Route
  3. Inventory Carrying Cost by Echelon
  4. OTIF (On-Time In-Full)
  5. Cost-to-Serve by Channel
  6. Cash-to-Cash (DIO + DSO - DPO)

v0.69.0

Usage:
    poetry run python scripts/analysis/diagnose_cost.py
    poetry run python scripts/analysis/diagnose_cost.py --data-dir data/output_converged
"""

# ruff: noqa: E501, PLR0915, RUF001
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WIDTH = 78

# Demand endpoint prefixes (ECOM-FC, DTC-FC, STORE- are POS consumers)
DEMAND_PREFIXES = ("STORE-", "ECOM-FC-", "DTC-FC-")

# Columns needed from each parquet file (memory optimization)
SHIP_COLS = ["creation_day", "arrival_day", "source_id", "target_id", "product_id", "quantity"]
ORDER_COLS = ["day", "source_id", "target_id", "product_id", "quantity", "requested_date"]
INV_COLS = ["day", "node_id", "product_id", "actual_inventory"]


def classify_echelon(node_id: str) -> str:
    if node_id.startswith("PLANT-"):
        return "Plant"
    if node_id.startswith("RDC-"):
        return "RDC"
    if node_id.startswith("SUP-"):
        return "Supplier"
    if any(node_id.startswith(p) for p in DEMAND_PREFIXES):
        if node_id.startswith("STORE-"):
            return "Store"
        return "Customer DC"
    # Everything else is a Customer DC (CLUB-DC, DIST-DC, GRO-DC, etc.)
    if "-DC-" in node_id or node_id.startswith("PHARM-DC"):
        return "Customer DC"
    return "Store"


def classify_channel(node_id: str) -> str:
    for prefix, chan in [
        ("CLUB-DC-", "CLUB"), ("DIST-DC-", "DISTRIBUTOR"),
        ("GRO-DC-", "GROCERY"), ("PHARM-DC-", "PHARMACY"),
        ("ECOM-FC-", "ECOMMERCE"), ("DTC-FC-", "DTC"),
    ]:
        if node_id.startswith(prefix):
            return chan
    return "OTHER"


def load_cost_master(data_dir: Path) -> dict:
    """Load cost_master.json from config directory."""
    candidates = [
        Path(__file__).parent.parent.parent
        / "src" / "prism_sim" / "config" / "cost_master.json",
        data_dir / "cost_master.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"cost_master.json not found. Searched: {[str(c) for c in candidates]}"
    )


def build_abc_map(products: pd.DataFrame, shipments: pd.DataFrame) -> dict[str, str]:
    """Classify products into ABC by shipped volume."""
    vol = shipments.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
    total = vol.sum()
    if total == 0:
        return {}
    cumvol = vol.cumsum() / total
    abc = {}
    a_thresh, b_thresh = 0.80, 0.95
    for pid, cum in cumvol.items():
        if cum <= a_thresh:
            abc[str(pid)] = "A"
        elif cum <= b_thresh:
            abc[str(pid)] = "B"
        else:
            abc[str(pid)] = "C"
    return abc


def _build_echelon_map(node_ids: pd.Series) -> dict[str, str]:
    """Build echelon lookup dict from unique node IDs (vectorized)."""
    return {nid: classify_echelon(nid) for nid in node_ids.unique()}


def _stream_inventory_agg(
    inv_path: Path, cat_cost_map: dict[str, float],
) -> tuple[pd.DataFrame, int] | None:
    """Stream inventory parquet row-groups to compute echelon averages.

    Returns (avg_inv DataFrame, n_days) or None if file missing.
    Memory: O(one row-group) instead of O(entire file).
    """
    if not inv_path.exists():
        return None

    pf = pq.ParquetFile(inv_path)
    # Accumulators: echelon -> [total_cases, total_value, row_count]
    accum: dict[str, list[float]] = {}
    all_days: set[int] = set()

    for rg_idx in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=INV_COLS)
        df = tbl.to_pandas()
        del tbl
        # Filter FG only
        df = df[df["product_id"].str.startswith("SKU-")]
        if df.empty:
            continue

        all_days.update(df["day"].unique().tolist())
        df["echelon"] = df["node_id"].map(
            {nid: classify_echelon(nid) for nid in df["node_id"].unique()}
        )
        df["cost"] = df["product_id"].map(cat_cost_map).fillna(9.0)
        df["value"] = df["actual_inventory"] * df["cost"]

        for ech, grp in df.groupby("echelon"):
            if ech not in accum:
                accum[ech] = [0.0, 0.0, 0]
            accum[ech][0] += grp["actual_inventory"].sum()
            accum[ech][1] += grp["value"].sum()
            accum[ech][2] += len(grp)
        del df

    if not accum:
        return None

    n_days = len(all_days)
    rows = []
    for ech, (total_cases, total_value, count) in accum.items():
        rows.append({
            "echelon": ech,
            "avg_cases": total_cases / count if count > 0 else 0,
            "total_value": total_value,
            "avg_value": total_value / n_days if n_days > 0 else 0,
        })
    return pd.DataFrame(rows).set_index("echelon"), n_days


def print_header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}".center(WIDTH))
    print("=" * WIDTH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost Analytics Diagnostic")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/output",
        help="Directory containing simulation parquet output",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Tee output to file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_dir = data_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    report_path = diag_dir / f"diagnose_cost_{ts}.txt"
    tee = io.StringIO()

    class TeeWriter:
        def write(self, s: str) -> int:
            sys.__stdout__.write(s)
            tee.write(s)
            return len(s)

        def flush(self) -> None:
            sys.__stdout__.flush()

    sys.stdout = TeeWriter()  # type: ignore[assignment]

    t0 = time.time()

    print_header("COST ANALYTICS DIAGNOSTIC")
    print(f"\n  Data directory: {data_dir}")

    # Load data
    print("\nLoading data...")
    cost_master = load_cost_master(data_dir)
    product_costs = cost_master.get("product_costs", {})
    logistics = cost_master.get("logistics_costs", {})
    working_cap = cost_master.get("working_capital", {})

    products = pd.read_csv(
        data_dir / "static_world" / "products.csv",
        usecols=["id", "name", "category"],
    )
    products["category"] = products["category"].str.replace(
        "ProductCategory.", "", regex=False
    )

    # Build category->cost lookup
    cat_cost_map: dict[str, float] = {}
    for _, row in products.iterrows():
        cat = row["category"]
        default_cost = product_costs.get("default", 9.0)
        cat_cost_map[row["id"]] = product_costs.get(cat, default_cost)

    # Load shipments (selected columns only — ~3GB savings vs loading all)
    ships = pd.read_parquet(data_dir / "shipments.parquet", columns=SHIP_COLS)
    ships = ships[ships["product_id"].str.startswith("SKU-")]
    sim_days = int(ships["creation_day"].max() - ships["creation_day"].min()) + 1
    print(f"  Shipments: {len(ships):,} FG rows, {sim_days} sim days")

    # Enrich shipments with vectorized dict lookups (no .apply())
    ech_map = _build_echelon_map(pd.concat([ships["source_id"], ships["target_id"]]))
    ships["src_ech"] = ships["source_id"].map(ech_map)
    ships["tgt_ech"] = ships["target_id"].map(ech_map)
    ships["cost_per_case"] = ships["product_id"].map(cat_cost_map).fillna(9.0)

    # Load orders (selected columns only)
    orders = pd.read_parquet(data_dir / "orders.parquet", columns=ORDER_COLS)
    orders = orders[orders["product_id"].str.startswith("SKU-")]
    print(f"  Orders: {len(orders):,} FG rows")

    # Load batches (small — ~115K rows, safe to load fully)
    batches = pd.read_parquet(data_dir / "batches.parquet")
    batches = batches[batches["product_id"].str.startswith("SKU-")]
    print(f"  Batches: {len(batches):,} FG rows")

    # ABC classification
    abc_map = build_abc_map(products, ships)
    ships["abc"] = ships["product_id"].map(abc_map).fillna("C")

    # ═══════════════════════════════════════════════════════════════════
    # 1. COGS by ABC / Echelon
    # ═══════════════════════════════════════════════════════════════════
    print_header("1. COGS BY ABC CLASS & ECHELON")

    ships["cogs"] = ships["quantity"] * ships["cost_per_case"]
    total_cogs = ships["cogs"].sum()

    # By ABC
    cogs_abc = ships.groupby("abc")["cogs"].sum().sort_index()
    qty_abc = ships.groupby("abc")["quantity"].sum().sort_index()
    print(f"\n  Total COGS (shipped value): ${total_cogs:,.0f}")
    print(f"\n  {'ABC':>4s}  {'COGS':>14s}  {'Share%':>7s}  {'Cases':>14s}  {'$/Case':>7s}")
    print(f"  {'────':>4s}  {'──────────────':>14s}  {'───────':>7s}  {'──────────────':>14s}  {'───────':>7s}")
    for abc_cls in ["A", "B", "C"]:
        cogs = cogs_abc.get(abc_cls, 0)
        qty = qty_abc.get(abc_cls, 0)
        avg = cogs / qty if qty > 0 else 0
        print(f"  {abc_cls:>4s}  ${cogs:>13,.0f}  {cogs / total_cogs:>6.1%}  {qty:>14,.0f}  ${avg:>6.2f}")

    # By route (source echelon -> target echelon)
    ships["route"] = ships["src_ech"] + " → " + ships["tgt_ech"]
    cogs_route = ships.groupby("route")["cogs"].sum().sort_values(ascending=False)
    print(f"\n  {'Route':<30s}  {'COGS':>14s}  {'Share%':>7s}")
    print(f"  {'──────────────────────────────':<30s}  {'──────────────':>14s}  {'───────':>7s}")
    for route, cogs in cogs_route.head(10).items():
        print(f"  {route:<30s}  ${cogs:>13,.0f}  {cogs / total_cogs:>6.1%}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. LOGISTICS COST BY ROUTE
    # ═══════════════════════════════════════════════════════════════════
    print_header("2. LOGISTICS COST BY ROUTE")

    ltl_per_case = logistics.get("ltl_cost_per_case", 0.75)
    handling_per_case = logistics.get("handling_cost_per_case", 0.25)

    # Store deliveries are LTL (per-case); upstream routes use handling only
    ships["is_ltl"] = ships["tgt_ech"] == "Store"
    ships["transport_cost"] = np.where(
        ships["is_ltl"],
        ships["quantity"] * ltl_per_case,
        0.0,
    )
    ships["handling_cost"] = ships["quantity"] * handling_per_case
    ships["logistics_cost"] = ships["transport_cost"] + ships["handling_cost"]

    total_logistics = ships["logistics_cost"].sum()
    total_transport = ships["transport_cost"].sum()
    total_handling = ships["handling_cost"].sum()

    print(f"\n  Total logistics cost: ${total_logistics:,.0f}")
    print(f"    Transport (LTL): ${total_transport:,.0f}")
    print(f"    Handling:  ${total_handling:,.0f}")

    # By route
    log_route = ships.groupby("route").agg(
        logistics=("logistics_cost", "sum"),
        cases=("quantity", "sum"),
    ).sort_values("logistics", ascending=False)
    print(f"\n  {'Route':<30s}  {'Cost':>14s}  {'Cases':>14s}  {'$/Case':>7s}")
    print(f"  {'──────────────────────────────':<30s}  {'──────────────':>14s}  {'──────────────':>14s}  {'───────':>7s}")
    for route, row in log_route.head(10).iterrows():
        avg = row["logistics"] / row["cases"] if row["cases"] > 0 else 0
        print(f"  {route:<30s}  ${row['logistics']:>13,.0f}  {row['cases']:>14,.0f}  ${avg:>6.2f}")

    # Drop temp columns to reduce memory before next sections
    ships.drop(columns=["is_ltl", "transport_cost", "handling_cost"], inplace=True, errors="ignore")

    # ═══════════════════════════════════════════════════════════════════
    # 3. INVENTORY CARRYING COST BY ECHELON (streamed — O(1 row-group))
    # ═══════════════════════════════════════════════════════════════════
    print_header("3. INVENTORY CARRYING COST BY ECHELON")

    wh_per_case_day = logistics.get("warehouse_cost_per_case_per_day", 0.02)
    carrying_pct = working_cap.get("annual_carrying_cost_pct", 0.25)

    inv_path = data_dir / "inventory.parquet"
    inv_result = _stream_inventory_agg(inv_path, cat_cost_map)

    total_carrying = 0.0
    total_warehouse = 0.0
    total_inv_value = 0.0

    if inv_result is not None:
        avg_inv, n_inv_days = inv_result
        avg_inv["warehouse_cost"] = avg_inv["avg_cases"] * wh_per_case_day * 365
        avg_inv["carrying_cost"] = avg_inv["avg_value"] * carrying_pct

        total_carrying = avg_inv["carrying_cost"].sum()
        total_warehouse = avg_inv["warehouse_cost"].sum()
        total_inv_value = avg_inv["avg_value"].sum()

        print(f"\n  Inventory sampled on {n_inv_days} days")
        print(f"  Annual carrying cost (inventory value × {carrying_pct:.0%}): ${total_carrying:,.0f}")
        print(f"  Annual warehouse cost (${wh_per_case_day}/case/day): ${total_warehouse:,.0f}")
        print(f"\n  {'Echelon':<15s}  {'Avg Cases':>14s}  {'Avg Value':>14s}  {'Carry Cost':>14s}  {'WH Cost':>14s}")
        print(f"  {'───────────────':<15s}  {'──────────────':>14s}  {'──────────────':>14s}  {'──────────────':>14s}  {'──────────────':>14s}")
        for ech in ["Plant", "RDC", "Customer DC", "Store"]:
            if ech in avg_inv.index:
                r = avg_inv.loc[ech]
                print(
                    f"  {ech:<15s}  {r['avg_cases']:>14,.0f}  ${r['avg_value']:>13,.0f}"
                    f"  ${r['carrying_cost']:>13,.0f}  ${r['warehouse_cost']:>13,.0f}"
                )
    else:
        print("\n  WARNING: inventory.parquet not found — skipping carrying cost")

    # ═══════════════════════════════════════════════════════════════════
    # 4. OTIF (ON-TIME IN-FULL)
    # ═══════════════════════════════════════════════════════════════════
    print_header("4. OTIF (ON-TIME IN-FULL)")

    has_requested_date = "requested_date" in orders.columns and orders["requested_date"].notna().any()
    otif_pct = 0.0

    if has_requested_date:
        # Aggregate to (day, src, tgt, product) level to reduce row count before merge
        ord_agg = orders.groupby(["day", "source_id", "target_id", "product_id"]).agg(
            ordered_qty=("quantity", "sum"),
            requested_date=("requested_date", "first"),
        ).reset_index()

        ship_agg = ships.groupby(
            ["creation_day", "source_id", "target_id", "product_id"]
        ).agg(
            shipped_qty=("quantity", "sum"),
            arrival_day=("arrival_day", "max"),
        ).reset_index()

        merged = ord_agg.merge(
            ship_agg,
            left_on=["day", "source_id", "target_id", "product_id"],
            right_on=["creation_day", "source_id", "target_id", "product_id"],
            how="left",
        )
        del ord_agg, ship_agg

        merged["shipped_qty"] = merged["shipped_qty"].fillna(0)
        merged["arrival_day"] = merged["arrival_day"].fillna(9999)

        merged["in_full"] = merged["shipped_qty"] >= merged["ordered_qty"] * 0.99
        merged["on_time"] = merged["arrival_day"] <= merged["requested_date"] + 1  # 1-day tolerance
        merged["otif"] = merged["in_full"] & merged["on_time"]

        total_lines = len(merged)
        in_full_pct = merged["in_full"].mean()
        on_time_pct = merged["on_time"].mean()
        otif_pct = merged["otif"].mean()

        print(f"\n  Order lines evaluated: {total_lines:,}")
        print(f"  In-Full:  {in_full_pct:.1%}")
        print(f"  On-Time:  {on_time_pct:.1%}")
        print(f"  OTIF:     {otif_pct:.1%}")

        # By ABC
        merged["abc"] = merged["product_id"].map(abc_map).fillna("C")
        otif_abc = merged.groupby("abc").agg(
            in_full=("in_full", "mean"),
            on_time=("on_time", "mean"),
            otif=("otif", "mean"),
            count=("otif", "count"),
        )
        print(f"\n  {'ABC':>4s}  {'In-Full':>8s}  {'On-Time':>8s}  {'OTIF':>8s}  {'Lines':>10s}")
        print(f"  {'────':>4s}  {'────────':>8s}  {'────────':>8s}  {'────────':>8s}  {'──────────':>10s}")
        for abc_cls in ["A", "B", "C"]:
            if abc_cls in otif_abc.index:
                r = otif_abc.loc[abc_cls]
                print(
                    f"  {abc_cls:>4s}  {r['in_full']:>7.1%}  {r['on_time']:>7.1%}"
                    f"  {r['otif']:>7.1%}  {r['count']:>10,.0f}"
                )
        del merged
    else:
        print("\n  WARNING: requested_date not found in orders.parquet")
        print("  Run sim with v0.69.0+ to populate requested_date.")
        print("  Falling back to shipment-based fill rate estimate...")

        ord_daily = orders.groupby("day")["quantity"].sum()
        ship_daily = ships[ships["tgt_ech"] == "Store"].groupby("creation_day")["quantity"].sum()
        common_days = ord_daily.index.intersection(ship_daily.index)
        if len(common_days) > 0:
            ordered = ord_daily.loc[common_days].sum()
            shipped = ship_daily.loc[common_days].sum()
            print(f"  Estimated fill rate: {shipped / ordered:.1%} (shipped/ordered)")

    # Free orders — no longer needed
    del orders

    # ═══════════════════════════════════════════════════════════════════
    # 5. COST-TO-SERVE BY CHANNEL
    # ═══════════════════════════════════════════════════════════════════
    print_header("5. COST-TO-SERVE BY CHANNEL")

    store_ships = ships[ships["tgt_ech"].isin(["Store", "Customer DC"])].copy()

    # Build channel map
    locations_path = data_dir / "static_world" / "locations.csv"
    if locations_path.exists():
        locations = pd.read_csv(locations_path)
        if "channel" in locations.columns:
            chan_map = dict(zip(locations["id"], locations["channel"], strict=True))
            store_ships["channel"] = store_ships["target_id"].map(chan_map).fillna("OTHER")
        else:
            store_ships["channel"] = store_ships["source_id"].map(classify_channel)
    else:
        store_ships["channel"] = store_ships["source_id"].map(classify_channel)

    chan_agg = store_ships.groupby("channel").agg(
        cases=("quantity", "sum"),
        cogs=("cogs", "sum"),
        logistics=("logistics_cost", "sum"),
    )
    chan_agg["total_cost"] = chan_agg["cogs"] + chan_agg["logistics"]
    chan_agg["cost_per_case"] = chan_agg["total_cost"] / chan_agg["cases"].clip(lower=1)
    chan_agg = chan_agg.sort_values("total_cost", ascending=False)

    total_served_cost = chan_agg["total_cost"].sum()
    print(f"\n  {'Channel':<15s}  {'Cases':>14s}  {'Total Cost':>14s}  {'$/Case':>7s}  {'Share%':>7s}")
    print(f"  {'───────────────':<15s}  {'──────────────':>14s}  {'──────────────':>14s}  {'───────':>7s}  {'───────':>7s}")
    for chan, row in chan_agg.iterrows():
        print(
            f"  {chan:<15s}  {row['cases']:>14,.0f}  ${row['total_cost']:>13,.0f}"
            f"  ${row['cost_per_case']:>6.2f}  {row['total_cost'] / total_served_cost:>6.1%}"
        )
    del store_ships

    # ═══════════════════════════════════════════════════════════════════
    # 6. CASH-TO-CASH CYCLE
    # ═══════════════════════════════════════════════════════════════════
    print_header("6. CASH-TO-CASH CYCLE")

    dso = working_cap.get("dso_days", 30.0)
    dpo = working_cap.get("dpo_days", 45.0)

    daily_cogs = total_cogs / sim_days if sim_days > 0 else 1
    if inv_result is not None:
        dio = total_inv_value / daily_cogs if daily_cogs > 0 else 0
    else:
        turns = 10.0
        dio = 365 / turns

    c2c = dio + dso - dpo

    print(f"\n  DIO (Days Inventory Outstanding): {dio:.1f} days")
    print(f"  DSO (Days Sales Outstanding):     {dso:.1f} days (config)")
    print(f"  DPO (Days Payables Outstanding):   {dpo:.1f} days (config)")
    print("  ────────────────────────────────────────")
    print(f"  Cash-to-Cash Cycle:                {c2c:.1f} days")
    print("  (C2C = DIO + DSO - DPO)")

    if inv_result is not None:
        turns = total_cogs / total_inv_value if total_inv_value > 0 else 0
        print(f"\n  Inventory Turns (COGS/Avg Inv):    {turns:.1f}x")
        print(f"  DIO cross-check (365/Turns):       {365 / turns if turns > 0 else 0:.1f} days")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print_header("COST ANALYTICS SUMMARY")

    total_cost = total_cogs + total_logistics
    if inv_result is not None:
        total_cost += total_carrying + total_warehouse

    daily_demand_cases = ships[
        ships["target_id"].str.startswith(DEMAND_PREFIXES[0])
        | ships["target_id"].str.startswith(DEMAND_PREFIXES[1])
        | ships["target_id"].str.startswith(DEMAND_PREFIXES[2])
    ]["quantity"].sum() / sim_days

    print(f"\n  Simulation: {sim_days} days, {daily_demand_cases:,.0f} cases/day demand")
    print("\n  Cost Component           Annual Estimate")
    print("  ────────────────────── ──────────────────")
    print(f"  COGS (shipped value)   ${total_cogs:>17,.0f}")
    print(f"  Logistics (transport)  ${total_logistics:>17,.0f}")
    if inv_result is not None:
        print(f"  Carrying cost          ${total_carrying:>17,.0f}")
        print(f"  Warehouse cost         ${total_warehouse:>17,.0f}")
    print("  ────────────────────── ──────────────────")
    print(f"  TOTAL                  ${total_cost:>17,.0f}")
    if has_requested_date:
        print(f"\n  OTIF: {otif_pct:.1%}")
    print(f"  Cash-to-Cash: {c2c:.1f} days")

    elapsed = time.time() - t0
    print(f"\n{'=' * WIDTH}")
    print(f"  DIAGNOSTIC COMPLETE — {elapsed:.0f}s elapsed")
    print(f"{'=' * WIDTH}")

    # Save report
    sys.stdout = sys.__stdout__  # type: ignore[assignment]
    with open(report_path, "w") as f:
        f.write(tee.getvalue())
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
