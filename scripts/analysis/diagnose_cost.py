#!/usr/bin/env python3
"""
DEPRECATED: Use diagnose_supply_chain.py instead (v0.72.0).

This script is superseded by the unified diagnostic which covers all 35
questions from a consultant's checklist. Run:
    poetry run python scripts/analysis/diagnose_supply_chain.py

---

COST ANALYTICS — Post-Simulation Enrichment Diagnostic.

Computes SKU profitability, OTIF, working capital, and cost-to-serve
from simulation parquet output + cost_master.json config.

No simulation physics changes — pure post-hoc analysis.

Sections:
  1. COGS by ABC / Echelon (per-SKU costs from products.csv)
  2. Logistics Cost by Route (per-echelon FTL/LTL rates)
  3. Inventory Carrying Cost by Echelon (echelon-specific warehouse rates)
  4. OTIF (On-Time In-Full)
  5. Bottom-Up Manufacturing COGS (batch_ingredients → material + labor + overhead)
  6. Revenue & Margin by Channel
  7. Cost-to-Serve by Channel
  8. Cash-to-Cash (DIO + channel-weighted DSO - DPO)

v0.71.0

Usage:
    poetry run python scripts/analysis/diagnose_cost.py
    poetry run python scripts/analysis/diagnose_cost.py --data-dir data/output_converged
"""

# ruff: noqa: E501, PLR0915, PLR0912
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

# Route key mapping: (src_echelon, tgt_echelon) → config route key
ROUTE_KEY_MAP: dict[tuple[str, str], str] = {
    ("Supplier", "Plant"): "supplier_to_plant",
    ("Plant", "RDC"): "plant_to_rdc",
    ("Plant", "Customer DC"): "plant_to_dc",
    ("Plant", "Store"): "plant_to_dc",
    ("RDC", "Customer DC"): "rdc_to_dc",
    ("RDC", "Store"): "rdc_to_dc",
    ("Customer DC", "Store"): "dc_to_store",
}


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


def load_world_definition() -> dict:
    """Load world_definition.json for channel economics."""
    path = (
        Path(__file__).parent.parent.parent
        / "src" / "prism_sim" / "config" / "world_definition.json"
    )
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


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


def _build_distance_map(data_dir: Path) -> dict[tuple[str, str], float]:
    """Build (source_id, target_id) → distance_km lookup from links.csv."""
    links_path = data_dir / "static_world" / "links.csv"
    if not links_path.exists():
        return {}
    links = pd.read_csv(links_path, usecols=["source_id", "target_id", "distance_km"])
    return {
        (row["source_id"], row["target_id"]): row["distance_km"]
        for _, row in links.iterrows()
    }


def _build_channel_map(data_dir: Path) -> dict[str, str]:
    """Build node_id → channel lookup from locations.csv."""
    loc_path = data_dir / "static_world" / "locations.csv"
    if not loc_path.exists():
        return {}
    loc = pd.read_csv(loc_path, usecols=["id", "channel"])
    loc["channel"] = loc["channel"].str.replace("CustomerChannel.", "", regex=False)
    return dict(zip(loc["id"], loc["channel"], strict=False))


def _stream_inventory_agg(
    inv_path: Path,
    sku_cost_map: dict[str, float],
    wh_rates: dict[str, float],
) -> tuple[pd.DataFrame, int] | None:
    """Stream inventory parquet row-groups to compute echelon averages.

    Returns (avg_inv DataFrame, n_days) or None if file missing.
    Memory: O(one row-group) instead of O(entire file).
    """
    if not inv_path.exists():
        return None

    default_wh = 0.02
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
        df["cost"] = df["product_id"].map(sku_cost_map).fillna(9.0)
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
        avg_cases = total_cases / count if count > 0 else 0
        wh_rate = wh_rates.get(ech, default_wh)
        rows.append({
            "echelon": ech,
            "avg_cases": avg_cases,
            "total_value": total_value,
            "avg_value": total_value / n_days if n_days > 0 else 0,
            "wh_rate": wh_rate,
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

    print_header("COST ANALYTICS DIAGNOSTIC (v0.71.0)")
    print(f"\n  Data directory: {data_dir}")

    # ───────────────────────────────────────────────────────────────────
    # LOAD DATA
    # ───────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    cost_master = load_cost_master(data_dir)
    logistics_cfg = cost_master.get("logistics_costs", {})
    route_cfg = logistics_cfg.get("routes", {})
    wh_rates = logistics_cfg.get("warehouse_cost_per_case_per_day", {})
    mfg_costs = cost_master.get("manufacturing_costs", {})
    working_cap = cost_master.get("working_capital", {})
    product_costs = cost_master.get("product_costs", {})

    world_def = load_world_definition()
    channel_econ = world_def.get("channel_economics", {})

    # Products — per-SKU costs + category fallback
    products = pd.read_csv(
        data_dir / "static_world" / "products.csv",
        usecols=["id", "name", "category", "cost_per_case", "price_per_case"],
    )
    products["category"] = products["category"].str.replace(
        "ProductCategory.", "", regex=False
    )

    # Per-SKU cost & price maps (preferred)
    sku_cost_map: dict[str, float] = {}
    sku_price_map: dict[str, float] = {}
    cat_cost_map: dict[str, float] = {}  # fallback
    ing_cost_map: dict[str, float] = {}  # ingredients + bulk intermediates
    sku_cat_map: dict[str, str] = {}  # product_id → category

    for _, row in products.iterrows():
        pid = row["id"]
        cat = row["category"]
        cost = row.get("cost_per_case", 0)
        price = row.get("price_per_case", 0)

        if pid.startswith("SKU-"):
            sku_cost_map[pid] = float(cost) if pd.notna(cost) and cost > 0 else product_costs.get(cat, product_costs.get("default", 9.0))
            sku_price_map[pid] = float(price) if pd.notna(price) and price > 0 else 0.0
            sku_cat_map[pid] = cat
            # Category fallback
            default_cost = product_costs.get("default", 9.0)
            cat_cost_map[pid] = product_costs.get(cat, default_cost)
        elif pd.notna(cost) and cost > 0:
            # Ingredients and bulk intermediates
            ing_cost_map[pid] = float(cost)

    # Links → distance map
    dist_map = _build_distance_map(data_dir)

    # Channel map from locations
    chan_map = _build_channel_map(data_dir)

    # Load shipments (selected columns only)
    ships = pd.read_parquet(data_dir / "shipments.parquet", columns=SHIP_COLS)
    ships = ships[ships["product_id"].str.startswith("SKU-")]
    sim_days = int(ships["creation_day"].max() - ships["creation_day"].min()) + 1
    print(f"  Shipments: {len(ships):,} FG rows, {sim_days} sim days")

    # Enrich shipments with vectorized dict lookups
    ech_map = _build_echelon_map(pd.concat([ships["source_id"], ships["target_id"]]))
    ships["src_ech"] = ships["source_id"].map(ech_map)
    ships["tgt_ech"] = ships["target_id"].map(ech_map)

    # Per-SKU COGS (with category fallback)
    ships["cost_per_case"] = ships["product_id"].map(sku_cost_map)
    ships["cost_per_case"] = ships["cost_per_case"].fillna(
        ships["product_id"].map(cat_cost_map)
    ).fillna(9.0)

    # Load orders
    orders = pd.read_parquet(data_dir / "orders.parquet", columns=ORDER_COLS)
    orders = orders[orders["product_id"].str.startswith("SKU-")]
    print(f"  Orders: {len(orders):,} FG rows")

    # Load batches
    batches = pd.read_parquet(data_dir / "batches.parquet")
    batches_fg = batches[batches["product_id"].str.startswith("SKU-")]
    print(f"  Batches: {len(batches_fg):,} FG rows ({len(batches):,} total)")

    # Load batch ingredients (1.1M rows — small enough to load fully)
    bi_path = data_dir / "batch_ingredients.parquet"
    has_batch_ingredients = bi_path.exists()
    if has_batch_ingredients:
        batch_ings = pd.read_parquet(bi_path)
        print(f"  Batch ingredients: {len(batch_ings):,} rows")
    else:
        batch_ings = pd.DataFrame()
        print("  Batch ingredients: not found (skipping bottom-up mfg COGS)")

    # ABC classification
    abc_map = build_abc_map(products, ships)
    ships["abc"] = ships["product_id"].map(abc_map).fillna("C")

    # ═══════════════════════════════════════════════════════════════════
    # 1. COGS by ABC / Echelon (per-SKU costs)
    # ═══════════════════════════════════════════════════════════════════
    print_header("1. COGS BY ABC CLASS & ECHELON")

    ships["cogs"] = ships["quantity"] * ships["cost_per_case"]
    total_cogs = ships["cogs"].sum()

    # Compare per-SKU vs old flat-category approach
    ships["flat_cost"] = ships["product_id"].map(cat_cost_map).fillna(9.0)
    flat_cogs = (ships["quantity"] * ships["flat_cost"]).sum()
    delta_pct = (total_cogs - flat_cogs) / flat_cogs * 100 if flat_cogs > 0 else 0
    ships.drop(columns=["flat_cost"], inplace=True)

    print(f"\n  Total COGS (per-SKU):      ${total_cogs:,.0f}")
    print(f"  Old flat-category COGS:    ${flat_cogs:,.0f}")
    print(f"  Delta (per-SKU vs flat):   {delta_pct:+.1f}%")

    # By ABC
    cogs_abc = ships.groupby("abc")["cogs"].sum().sort_index()
    qty_abc = ships.groupby("abc")["quantity"].sum().sort_index()
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
    # 2. LOGISTICS COST BY ROUTE (per-echelon FTL/LTL)
    # ═══════════════════════════════════════════════════════════════════
    print_header("2. LOGISTICS COST BY ROUTE")

    # Map each shipment to its config route key
    ships["route_key"] = [
        ROUTE_KEY_MAP.get((src, tgt), "")
        for src, tgt in zip(ships["src_ech"], ships["tgt_ech"], strict=True)
    ]

    # Distance lookup: (source_id, target_id) → km
    ships["distance_km"] = [
        dist_map.get((src, tgt), np.nan)
        for src, tgt in zip(ships["source_id"], ships["target_id"], strict=True)
    ]

    # Compute average distance per route key for fallback
    route_avg_dist = ships.groupby("route_key")["distance_km"].mean()

    # Fill missing distances with route-key average
    for rk in ships["route_key"].unique():
        if rk and rk in route_avg_dist.index:
            mask = (ships["route_key"] == rk) & ships["distance_km"].isna()
            ships.loc[mask, "distance_km"] = route_avg_dist[rk]
    ships["distance_km"] = ships["distance_km"].fillna(0)

    # Compute average shipment size per route for FTL per-case allocation
    route_avg_size = ships.groupby("route_key")["quantity"].mean()

    # Compute transport + handling per shipment
    transport_costs = np.zeros(len(ships), dtype=np.float64)
    handling_costs = np.zeros(len(ships), dtype=np.float64)

    for rk, rcfg in route_cfg.items():
        mask = ships["route_key"] == rk
        if not mask.any():
            continue

        handling_rate = rcfg.get("handling_cost_per_case", 0.20)
        handling_costs[mask.values] = ships.loc[mask, "quantity"].values * handling_rate

        if rcfg.get("mode") == "FTL":
            # FTL: cost_per_km * distance, allocated across avg shipment size
            cost_per_km = rcfg.get("cost_per_km", 1.85)
            avg_size = route_avg_size.get(rk, 1000)
            if avg_size <= 0:
                avg_size = 1000
            # Truck cost = cost_per_km * distance; per-case = truck_cost / avg_shipment_size
            transport_costs[mask.values] = (
                cost_per_km * ships.loc[mask, "distance_km"].values
                / avg_size
                * ships.loc[mask, "quantity"].values
            )
        elif rcfg.get("mode") == "LTL":
            # LTL: flat cost per case
            cost_per_case = rcfg.get("cost_per_case", 0.75)
            transport_costs[mask.values] = ships.loc[mask, "quantity"].values * cost_per_case

    ships["transport_cost"] = transport_costs
    ships["handling_cost"] = handling_costs
    ships["logistics_cost"] = ships["transport_cost"] + ships["handling_cost"]

    total_logistics = ships["logistics_cost"].sum()
    total_transport = ships["transport_cost"].sum()
    total_handling = ships["handling_cost"].sum()

    print(f"\n  Total logistics cost: ${total_logistics:,.0f}")
    print(f"    Transport:  ${total_transport:,.0f}")
    print(f"    Handling:   ${total_handling:,.0f}")

    # 5-route breakdown
    log_route = ships.groupby("route_key").agg(
        transport=("transport_cost", "sum"),
        handling=("handling_cost", "sum"),
        logistics=("logistics_cost", "sum"),
        cases=("quantity", "sum"),
        avg_dist=("distance_km", "mean"),
    ).sort_values("logistics", ascending=False)

    print(f"\n  {'Route Key':<20s}  {'Mode':>4s}  {'Avg km':>7s}  {'Transport':>12s}  {'Handling':>12s}  {'$/Case':>7s}")
    print(f"  {'────────────────────':<20s}  {'────':>4s}  {'───────':>7s}  {'────────────':>12s}  {'────────────':>12s}  {'───────':>7s}")
    for rk, row in log_route.iterrows():
        if not rk:
            continue
        mode = route_cfg.get(rk, {}).get("mode", "?")
        avg_per = row["logistics"] / row["cases"] if row["cases"] > 0 else 0
        print(
            f"  {rk:<20s}  {mode:>4s}  {row['avg_dist']:>7,.0f}  ${row['transport']:>11,.0f}  ${row['handling']:>11,.0f}  ${avg_per:>6.2f}"
        )

    # Drop temp columns
    ships.drop(columns=["route_key", "distance_km", "transport_cost", "handling_cost"], inplace=True, errors="ignore")

    # ═══════════════════════════════════════════════════════════════════
    # 3. INVENTORY CARRYING COST BY ECHELON (echelon-specific rates)
    # ═══════════════════════════════════════════════════════════════════
    print_header("3. INVENTORY CARRYING COST BY ECHELON")

    carrying_pct = working_cap.get("annual_carrying_cost_pct", 0.25)

    inv_path = data_dir / "inventory.parquet"
    inv_result = _stream_inventory_agg(inv_path, sku_cost_map, wh_rates)

    total_carrying = 0.0
    total_warehouse = 0.0
    total_inv_value = 0.0

    if inv_result is not None:
        avg_inv, n_inv_days = inv_result
        avg_inv["warehouse_cost"] = avg_inv["avg_cases"] * avg_inv["wh_rate"] * 365
        avg_inv["carrying_cost"] = avg_inv["avg_value"] * carrying_pct

        total_carrying = avg_inv["carrying_cost"].sum()
        total_warehouse = avg_inv["warehouse_cost"].sum()
        total_inv_value = avg_inv["avg_value"].sum()

        print(f"\n  Inventory sampled on {n_inv_days} days")
        print(f"  Annual carrying cost (inventory value x {carrying_pct:.0%}): ${total_carrying:,.0f}")
        print(f"  Annual warehouse cost (echelon-specific rates): ${total_warehouse:,.0f}")
        print(f"\n  {'Echelon':<15s}  {'Avg Cases':>14s}  {'Avg Value':>14s}  {'WH $/c/d':>8s}  {'Carry Cost':>14s}  {'WH Cost':>14s}")
        print(f"  {'───────────────':<15s}  {'──────────────':>14s}  {'──────────────':>14s}  {'────────':>8s}  {'──────────────':>14s}  {'──────────────':>14s}")
        for ech in ["Plant", "RDC", "Customer DC", "Store"]:
            if ech in avg_inv.index:
                r = avg_inv.loc[ech]
                print(
                    f"  {ech:<15s}  {r['avg_cases']:>14,.0f}  ${r['avg_value']:>13,.0f}"
                    f"  ${r['wh_rate']:>7.3f}  ${r['carrying_cost']:>13,.0f}  ${r['warehouse_cost']:>13,.0f}"
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
        merged["on_time"] = merged["arrival_day"] <= merged["requested_date"] + 1
        merged["otif"] = merged["in_full"] & merged["on_time"]

        total_lines = len(merged)
        in_full_pct = merged["in_full"].mean()
        on_time_pct = merged["on_time"].mean()
        otif_pct = merged["otif"].mean()

        print(f"\n  Order lines evaluated: {total_lines:,}")
        print(f"  In-Full:  {in_full_pct:.1%}")
        print(f"  On-Time:  {on_time_pct:.1%}")
        print(f"  OTIF:     {otif_pct:.1%}")

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

    del orders

    # ═══════════════════════════════════════════════════════════════════
    # 5. BOTTOM-UP MANUFACTURING COGS
    # ═══════════════════════════════════════════════════════════════════
    print_header("5. BOTTOM-UP MANUFACTURING COGS")

    labor_pcts = mfg_costs.get("labor_pct_of_material", {})
    overhead_pcts = mfg_costs.get("overhead_pct_of_material", {})
    default_labor = labor_pcts.get("default", 0.25)
    default_overhead = overhead_pcts.get("default", 0.22)

    if has_batch_ingredients and not batch_ings.empty:
        # Material cost per ingredient consumption
        # ing_cost_map has $/case but batch_ingredients has quantity_kg
        # Since ingredients are bulk (1 case = weight_kg from products.csv),
        # cost_per_case = cost per unit weight. quantity_kg is kg consumed.
        # We need to normalize: cost_per_kg = cost_per_case / weight_kg
        ing_weights = products[~products["id"].str.startswith("SKU-")].set_index("id")

        # Build cost_per_kg map
        ing_cost_per_kg: dict[str, float] = {}
        for pid, cost in ing_cost_map.items():
            if pid in ing_weights.index:
                wt = ing_weights.loc[pid].get("weight_kg" if "weight_kg" in ing_weights.columns else "_dummy", 1.0)
                # ingredients have weight_kg in products.csv
                ing_cost_per_kg[pid] = cost / max(wt, 0.01) if pd.notna(wt) and wt > 0 else cost
            else:
                ing_cost_per_kg[pid] = cost

        # Need weight_kg for proper normalization — reload with that column
        prod_full = pd.read_csv(
            data_dir / "static_world" / "products.csv",
            usecols=["id", "weight_kg", "cost_per_case"],
        )
        wt_map = dict(zip(prod_full["id"], prod_full["weight_kg"], strict=False))
        # Rebuild per-kg cost: cost_per_case / weight_kg
        for pid, cost in ing_cost_map.items():
            wt = wt_map.get(pid, 1.0)
            if pd.notna(wt) and wt > 0:
                ing_cost_per_kg[pid] = cost / wt
            else:
                ing_cost_per_kg[pid] = cost

        batch_ings["unit_cost_per_kg"] = batch_ings["ingredient_id"].map(ing_cost_per_kg).fillna(0)
        batch_ings["material_cost"] = batch_ings["quantity_kg"] * batch_ings["unit_cost_per_kg"]

        # Aggregate to batch level
        batch_material = batch_ings.groupby("batch_id")["material_cost"].sum()

        # Join with FG batches
        fg = batches_fg[["batch_id", "product_id", "quantity"]].copy()
        fg["material_cost"] = fg["batch_id"].map(batch_material).fillna(0)
        fg["material_per_case"] = fg["material_cost"] / fg["quantity"].clip(lower=1)

        # Category for each product
        fg["category"] = fg["product_id"].map(sku_cat_map).fillna("default")

        # Apply labor + overhead multipliers
        fg["labor_pct"] = fg["category"].map(labor_pcts).fillna(default_labor)
        fg["overhead_pct"] = fg["category"].map(overhead_pcts).fillna(default_overhead)
        fg["labor_cost"] = fg["material_cost"] * fg["labor_pct"]
        fg["overhead_cost"] = fg["material_cost"] * fg["overhead_pct"]
        fg["full_mfg_cost"] = fg["material_cost"] + fg["labor_cost"] + fg["overhead_cost"]
        fg["full_per_case"] = fg["full_mfg_cost"] / fg["quantity"].clip(lower=1)

        # products.csv reference cost per case
        fg["ref_cost"] = fg["product_id"].map(sku_cost_map).fillna(9.0)

        # Report by category
        cat_agg = fg.groupby("category").agg(
            total_qty=("quantity", "sum"),
            material_cost=("material_cost", "sum"),
            labor_cost=("labor_cost", "sum"),
            overhead_cost=("overhead_cost", "sum"),
            full_mfg_cost=("full_mfg_cost", "sum"),
            ref_cost_total=("ref_cost", lambda x: (x * fg.loc[x.index, "quantity"]).sum()),
        )
        cat_agg["mat_per_case"] = cat_agg["material_cost"] / cat_agg["total_qty"].clip(lower=1)
        cat_agg["lab_per_case"] = cat_agg["labor_cost"] / cat_agg["total_qty"].clip(lower=1)
        cat_agg["oh_per_case"] = cat_agg["overhead_cost"] / cat_agg["total_qty"].clip(lower=1)
        cat_agg["full_per_case"] = cat_agg["full_mfg_cost"] / cat_agg["total_qty"].clip(lower=1)
        cat_agg["ref_per_case"] = cat_agg["ref_cost_total"] / cat_agg["total_qty"].clip(lower=1)
        cat_agg["delta_pct"] = (cat_agg["full_per_case"] - cat_agg["ref_per_case"]) / cat_agg["ref_per_case"].clip(lower=0.01) * 100

        total_mat = cat_agg["material_cost"].sum()
        total_full = cat_agg["full_mfg_cost"].sum()
        total_ref = cat_agg["ref_cost_total"].sum()

        print(f"\n  Total material cost (bottom-up):   ${total_mat:,.0f}")
        print(f"  Total fully-loaded mfg cost:       ${total_full:,.0f}")
        print(f"  Total products.csv reference cost:  ${total_ref:,.0f}")
        print(f"  Material share of full cost:        {total_mat / total_full:.1%}" if total_full > 0 else "")

        print(f"\n  {'Category':<15s}  {'Mat$/c':>7s}  {'Lab$/c':>7s}  {'OH$/c':>7s}  {'Full$/c':>8s}  {'Ref$/c':>7s}  {'Delta%':>7s}")
        print(f"  {'───────────────':<15s}  {'───────':>7s}  {'───────':>7s}  {'───────':>7s}  {'────────':>8s}  {'───────':>7s}  {'───────':>7s}")
        for cat in ["ORAL_CARE", "PERSONAL_WASH", "HOME_CARE"]:
            if cat in cat_agg.index:
                r = cat_agg.loc[cat]
                print(
                    f"  {cat:<15s}  ${r['mat_per_case']:>6.2f}  ${r['lab_per_case']:>6.2f}  ${r['oh_per_case']:>6.2f}"
                    f"  ${r['full_per_case']:>7.2f}  ${r['ref_per_case']:>6.2f}  {r['delta_pct']:>+6.1f}%"
                )

        del fg, batch_material
    else:
        print("\n  Skipped — batch_ingredients.parquet not available.")
        print("  Run simulation v0.70.0+ with 3-level BOM to generate batch ingredient data.")

    # ═══════════════════════════════════════════════════════════════════
    # 6. REVENUE & MARGIN BY CHANNEL
    # ═══════════════════════════════════════════════════════════════════
    print_header("6. REVENUE & MARGIN BY CHANNEL")

    # Revenue applies to demand-endpoint shipments (store/ecom/dtc deliveries)
    demand_ships = ships[
        ships["target_id"].str.startswith(DEMAND_PREFIXES[0])
        | ships["target_id"].str.startswith(DEMAND_PREFIXES[1])
        | ships["target_id"].str.startswith(DEMAND_PREFIXES[2])
    ].copy()

    if chan_map:
        demand_ships["channel"] = demand_ships["target_id"].map(chan_map).fillna("OTHER")
    else:
        demand_ships["channel"] = demand_ships["source_id"].map(classify_channel)

    demand_ships["price_per_case"] = demand_ships["product_id"].map(sku_price_map).fillna(0)
    demand_ships["revenue"] = demand_ships["quantity"] * demand_ships["price_per_case"]

    # Channel aggregation
    chan_rev = demand_ships.groupby("channel").agg(
        cases=("quantity", "sum"),
        revenue=("revenue", "sum"),
        cogs=("cogs", "sum"),
        logistics=("logistics_cost", "sum"),
    )
    chan_rev["margin"] = chan_rev["revenue"] - chan_rev["cogs"] - chan_rev["logistics"]
    chan_rev["margin_pct"] = chan_rev["margin"] / chan_rev["revenue"].clip(lower=1) * 100
    chan_rev = chan_rev.sort_values("revenue", ascending=False)

    # Target margins from world_definition.json
    target_margins = {ch: info.get("margin_pct", 0) for ch, info in channel_econ.items()}

    total_revenue = chan_rev["revenue"].sum()
    total_margin = chan_rev["margin"].sum()
    overall_margin_pct = total_margin / total_revenue * 100 if total_revenue > 0 else 0

    print(f"\n  Total revenue:  ${total_revenue:,.0f}")
    print(f"  Total margin:   ${total_margin:,.0f}  ({overall_margin_pct:.1f}%)")

    print(f"\n  {'Channel':<15s}  {'Revenue':>14s}  {'COGS':>12s}  {'Logistics':>11s}  {'Margin':>12s}  {'Margin%':>8s}  {'Target%':>8s}")
    print(f"  {'───────────────':<15s}  {'──────────────':>14s}  {'────────────':>12s}  {'───────────':>11s}  {'────────────':>12s}  {'────────':>8s}  {'────────':>8s}")
    for chan, row in chan_rev.iterrows():
        target = target_margins.get(str(chan), 0)
        print(
            f"  {chan:<15s}  ${row['revenue']:>13,.0f}  ${row['cogs']:>11,.0f}"
            f"  ${row['logistics']:>10,.0f}  ${row['margin']:>11,.0f}"
            f"  {row['margin_pct']:>7.1f}%  {target:>7d}%"
        )

    # Margin by ABC class
    demand_ships["abc"] = demand_ships["product_id"].map(abc_map).fillna("C")
    demand_ships["margin"] = demand_ships["revenue"] - demand_ships["cogs"] - demand_ships["logistics_cost"]
    abc_margin = demand_ships.groupby("abc").agg(
        revenue=("revenue", "sum"),
        margin=("margin", "sum"),
        cases=("quantity", "sum"),
    )
    abc_margin["margin_pct"] = abc_margin["margin"] / abc_margin["revenue"].clip(lower=1) * 100
    abc_margin["margin_per_case"] = abc_margin["margin"] / abc_margin["cases"].clip(lower=1)

    print(f"\n  {'ABC':>4s}  {'Revenue':>14s}  {'Margin':>12s}  {'Margin%':>8s}  {'$/Case':>8s}")
    print(f"  {'────':>4s}  {'──────────────':>14s}  {'────────────':>12s}  {'────────':>8s}  {'────────':>8s}")
    for abc_cls in ["A", "B", "C"]:
        if abc_cls in abc_margin.index:
            r = abc_margin.loc[abc_cls]
            print(
                f"  {abc_cls:>4s}  ${r['revenue']:>13,.0f}  ${r['margin']:>11,.0f}"
                f"  {r['margin_pct']:>7.1f}%  ${r['margin_per_case']:>7.2f}"
            )

    del demand_ships

    # ═══════════════════════════════════════════════════════════════════
    # 7. COST-TO-SERVE BY CHANNEL
    # ═══════════════════════════════════════════════════════════════════
    print_header("7. COST-TO-SERVE BY CHANNEL")

    store_ships = ships[ships["tgt_ech"].isin(["Store", "Customer DC"])].copy()

    if chan_map:
        store_ships["channel"] = store_ships["target_id"].map(chan_map).fillna("OTHER")
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
    # 8. CASH-TO-CASH CYCLE (channel-weighted DSO)
    # ═══════════════════════════════════════════════════════════════════
    print_header("8. CASH-TO-CASH CYCLE")

    dpo = working_cap.get("dpo_days", 45.0)
    dso_by_channel = working_cap.get("dso_days_by_channel", {})

    # Channel-weighted DSO from demand shipment volumes
    demand_filter = (
        ships["target_id"].str.startswith(DEMAND_PREFIXES[0])
        | ships["target_id"].str.startswith(DEMAND_PREFIXES[1])
        | ships["target_id"].str.startswith(DEMAND_PREFIXES[2])
    )
    demand_vols = ships[demand_filter].copy()
    if chan_map:
        demand_vols["channel"] = demand_vols["target_id"].map(chan_map).fillna("OTHER")
    else:
        demand_vols["channel"] = demand_vols["source_id"].map(classify_channel)

    chan_vol = demand_vols.groupby("channel")["quantity"].sum()
    total_vol = chan_vol.sum()

    weighted_dso = 0.0
    if total_vol > 0:
        print("\n  Channel-weighted DSO:")
        print(f"  {'Channel':<15s}  {'Volume%':>8s}  {'DSO':>5s}  {'Weighted':>8s}")
        print(f"  {'───────────────':<15s}  {'────────':>8s}  {'─────':>5s}  {'────────':>8s}")
        for chan, vol in chan_vol.sort_values(ascending=False).items():
            pct = vol / total_vol
            dso_val = dso_by_channel.get(str(chan), 30)
            contrib = pct * dso_val
            weighted_dso += contrib
            print(f"  {chan:<15s}  {pct:>7.1%}  {dso_val:>5d}  {contrib:>8.1f}")
        print(f"  {'':─<15s}  {'':─>8s}  {'':─>5s}  {'':─>8s}")
        print(f"  {'Weighted DSO':<15s}  {'':>8s}  {'':>5s}  {weighted_dso:>8.1f}")
    else:
        weighted_dso = 30.0  # fallback
        print(f"\n  DSO (fallback uniform): {weighted_dso:.1f} days")

    del demand_vols

    daily_cogs = total_cogs / sim_days if sim_days > 0 else 1
    if inv_result is not None:
        dio = total_inv_value / daily_cogs if daily_cogs > 0 else 0
    else:
        turns = 10.0
        dio = 365 / turns

    c2c = dio + weighted_dso - dpo

    print(f"\n  DIO (Days Inventory Outstanding): {dio:.1f} days")
    print(f"  DSO (channel-weighted):           {weighted_dso:.1f} days")
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

    daily_demand_cases = ships[demand_filter]["quantity"].sum() / sim_days

    print(f"\n  Simulation: {sim_days} days, {daily_demand_cases:,.0f} cases/day demand")
    print("\n  Cost Component             Annual Estimate")
    print("  ──────────────────────── ──────────────────")
    print(f"  COGS (per-SKU shipped)   ${total_cogs:>17,.0f}")
    print(f"  Logistics (FTL/LTL)      ${total_logistics:>17,.0f}")
    if inv_result is not None:
        print(f"  Carrying cost            ${total_carrying:>17,.0f}")
        print(f"  Warehouse cost           ${total_warehouse:>17,.0f}")
    print("  ──────────────────────── ──────────────────")
    print(f"  TOTAL                    ${total_cost:>17,.0f}")
    if total_revenue > 0:
        print(f"\n  Revenue:       ${total_revenue:>17,.0f}")
        print(f"  Gross margin:  ${total_margin:>17,.0f}  ({overall_margin_pct:.1f}%)")
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
