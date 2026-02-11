#!/usr/bin/env python3
"""
Deep Flow Diagnostic — 20 Questions to Diagnose Structural Issues.

Seven themes, 20 precise questions answered from simulation Parquet data:
  Theme 1: Where Is the Inventory? (Q1–Q3)
  Theme 2: What's Flowing Where? (Q4–Q6)
  Theme 3: Demand Signal Fidelity (Q7–Q9)
  Theme 4: Production & MRP (Q10–Q12)
  Theme 5: Deployment Mechanics (Q13–Q15)
  Theme 6: Push & Pull Interaction (Q16–Q18)
  Theme 7: System Dynamics (Q19–Q20)

Usage:
    poetry run python scripts/analysis/diagnose_flow_deep.py
    poetry run python scripts/analysis/diagnose_flow_deep.py --data-dir data/output
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the diagnostics package is importable
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics.loader import (
    DataBundle,
    classify_node,
    is_demand_endpoint,
    load_all_data,
)

WIDTH = 78


# ═══════════════════════════════════════════════════════════════════════════════
# Precomputed columns (avoids O(62M) .map() per question)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnrichedData:
    """DataBundle with precomputed echelon/channel columns on shipments & orders."""

    bundle: DataBundle

    # Precomputed on shipments (62M rows — built ONCE)
    ship_src_ech: pd.Series    # source echelon
    ship_tgt_ech: pd.Series    # target echelon
    ship_tgt_chan: pd.Series   # target channel (for DCs)
    ship_src_str: pd.Series    # source_id as str
    ship_tgt_str: pd.Series    # target_id as str
    ship_abc: pd.Series        # ABC class
    ship_is_demand: pd.Series  # target is demand endpoint

    # Precomputed on orders (58M rows — built ONCE)
    ord_src_ech: pd.Series
    ord_tgt_ech: pd.Series
    ord_src_str: pd.Series
    ord_tgt_str: pd.Series

    # Static lookups
    plant_direct_dcs: set[str]
    store_counts: dict[str, int]

    @property
    def ships(self) -> pd.DataFrame:
        return self.bundle.shipments

    @property
    def orders(self) -> pd.DataFrame:
        return self.bundle.orders

    @property
    def sim_days(self) -> int:
        return self.bundle.sim_days

    @property
    def inv(self) -> pd.DataFrame:
        return self.bundle.inv_by_echelon

    @property
    def batches(self) -> pd.DataFrame:
        return self.bundle.batches

    @property
    def forecasts(self) -> pd.DataFrame:
        return self.bundle.forecasts

    @property
    def abc_map(self) -> dict[str, str]:
        return self.bundle.abc_map

    @property
    def links(self) -> pd.DataFrame:
        return self.bundle.links


def _ensure_str(series: pd.Series) -> pd.Series:
    """Convert categorical to str if needed."""
    if hasattr(series, "cat"):
        return series.astype(str)
    return series


def _channel_from_node(node_id: str) -> str:
    """Extract channel from a Customer DC node ID."""
    if node_id.startswith("RET-DC-"):
        return "MASS_RETAIL"
    if node_id.startswith("GRO-DC-"):
        return "GROCERY"
    if node_id.startswith("CLUB-DC-"):
        return "CLUB"
    if node_id.startswith("PHARM-DC-"):
        return "PHARMACY"
    if node_id.startswith("DIST-DC-"):
        return "DISTRIBUTOR"
    if node_id.startswith("ECOM-FC-"):
        return "ECOMMERCE"
    if node_id.startswith("DTC-FC-"):
        return "DTC"
    return "OTHER"


def _cat_unique(series: pd.Series) -> set[str]:
    """Get unique string values from a Series (fast path for categoricals)."""
    if hasattr(series, "cat"):
        return set(str(x) for x in series.cat.categories)
    return set(str(x) for x in series.unique())


def enrich_data(data: DataBundle) -> EnrichedData:
    """Build precomputed columns from unique node IDs (O(4200)), then map vectorized.

    Uses categorical .map() which only maps the ~4200 unique categories,
    not all 62M rows individually.
    """
    t0 = time.time()
    print("  Precomputing echelon/channel/ABC columns...")

    ships = data.shipments
    orders = data.orders

    # Use columns directly — they're already categorical from the loader
    src_col = ships["source_id"]
    tgt_col = ships["target_id"]
    ord_src_col = orders["source_id"]
    ord_tgt_col = orders["target_id"]

    # 1. Build lookup dicts from UNIQUE node IDs (~4200 unique)
    all_node_ids = _cat_unique(src_col) | _cat_unique(tgt_col) | _cat_unique(ord_src_col) | _cat_unique(ord_tgt_col)
    node_ech = {nid: classify_node(nid) for nid in all_node_ids}
    node_chan = {nid: _channel_from_node(nid) for nid in all_node_ids}
    node_demand = {nid: is_demand_endpoint(nid) for nid in all_node_ids}

    # 2. Map on categoricals — pandas maps only the unique categories (~4200),
    #    not every row (62M), making this O(categories) not O(rows)
    ship_src_ech = src_col.map(node_ech).astype("category")
    ship_tgt_ech = tgt_col.map(node_ech).astype("category")
    ship_tgt_chan = tgt_col.map(node_chan).astype("category")
    ship_is_demand = tgt_col.map(node_demand)
    ship_abc = ships["product_id"].map(data.abc_map).astype("category")

    ord_src_ech = ord_src_col.map(node_ech).astype("category")
    ord_tgt_ech = ord_tgt_col.map(node_ech).astype("category")

    # 3. Static lookups from links
    plant_direct_dcs: set[str] = set()
    store_counts: dict[str, int] = {}
    for _, row in data.links.iterrows():
        s = str(row["source_id"])
        t = str(row["target_id"])
        if s.startswith("PLANT-") and node_ech.get(t) == "Customer DC":
            plant_direct_dcs.add(t)
        if t.startswith("STORE-") and node_ech.get(s) == "Customer DC":
            store_counts[s] = store_counts.get(s, 0) + 1

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s ({len(all_node_ids)} unique nodes)")

    return EnrichedData(
        bundle=data,
        ship_src_ech=ship_src_ech,
        ship_tgt_ech=ship_tgt_ech,
        ship_tgt_chan=ship_tgt_chan,
        ship_src_str=src_col,
        ship_tgt_str=tgt_col,
        ship_abc=ship_abc,
        ship_is_demand=ship_is_demand,
        ord_src_ech=ord_src_ech,
        ord_tgt_ech=ord_tgt_ech,
        ord_src_str=ord_src_col,
        ord_tgt_str=ord_tgt_col,
        plant_direct_dcs=plant_direct_dcs,
        store_counts=store_counts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _header(q_num: int, title: str) -> str:
    label = f"Q{q_num}. {title}"
    return f"\n{'─' * WIDTH}\n  {label}\n{'─' * WIDTH}"


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 1: Where Is the Inventory? (Q1–Q3)
# ═══════════════════════════════════════════════════════════════════════════════

def q1_dc_accumulation_by_channel(ed: EnrichedData) -> None:
    """Q1: DC Accumulation by Channel — Is the 26% imbalance uniform or concentrated?"""
    print(_header(1, "DC Accumulation by Channel"))

    ships = ed.ships
    sim_days = ed.sim_days

    # Inflow to Customer DCs
    dc_in_mask = ed.ship_tgt_ech == "Customer DC"
    inflow = ships[dc_in_mask]
    in_channels = ed.ship_tgt_chan[dc_in_mask]

    # Outflow from Customer DCs
    dc_out_mask = ed.ship_src_ech == "Customer DC"
    outflow = ships[dc_out_mask]
    out_channels = ed.ship_src_str[dc_out_mask].map(_channel_from_node)

    print(f"\n  {'Channel':<14} {'DCs':>4} {'Inflow/d':>10} {'Outflow/d':>10} "
          f"{'Net/d':>10} {'Imbal%':>8} {'PlantDir':>8}")
    print(f"  {'─'*14} {'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    total_in = total_out = 0.0
    channels = sorted(in_channels.unique())
    for ch in channels:
        ch_in = inflow[in_channels == ch]["quantity"].sum() / sim_days
        ch_out = outflow[out_channels == ch]["quantity"].sum() / sim_days
        net = ch_in - ch_out
        imbal = (ch_in - ch_out) / ch_in * 100 if ch_in > 0 else 0
        total_in += ch_in
        total_out += ch_out

        ch_dcs = list(set(str(x) for x in ed.ship_tgt_str[dc_in_mask & (in_channels == ch)].unique()))
        n_dcs = len(ch_dcs)
        n_pd = sum(1 for d in ch_dcs if d in ed.plant_direct_dcs)

        print(f"  {ch:<14} {n_dcs:>4} {ch_in:>10,.0f} {ch_out:>10,.0f} "
              f"{net:>+10,.0f} {imbal:>7.1f}% {n_pd:>4}/{n_dcs:<3}")

    net_total = total_in - total_out
    imbal_total = (total_in - total_out) / total_in * 100 if total_in > 0 else 0
    print(f"  {'─'*14} {'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'TOTAL':<14} {'':>4} {total_in:>10,.0f} {total_out:>10,.0f} "
          f"{net_total:>+10,.0f} {imbal_total:>7.1f}%")

    # Annotate: ECOM/DTC are demand endpoints — their 100% imbalance is expected
    ecom_dtc_net = 0.0
    for ch in ("ECOMMERCE", "DTC"):
        ch_in_val = inflow[in_channels == ch]["quantity"].sum() / sim_days
        ch_out_val = outflow[out_channels == ch]["quantity"].sum() / sim_days
        ecom_dtc_net += (ch_in_val - ch_out_val)
    if ecom_dtc_net > 0:
        print(f"\n  NOTE: ECOM-FC + DTC-FC are demand endpoints (no downstream stores).")
        print(f"  Their net inflow ({ecom_dtc_net:,.0f}/d) = POS consumption, not accumulation.")
        adjusted = net_total - ecom_dtc_net
        adj_pct = adjusted / (total_in - ecom_dtc_net) * 100 if (total_in - ecom_dtc_net) > 0 else 0
        print(f"  Adjusted DC imbalance (excl ECOM/DTC): {adjusted:+,.0f}/d ({adj_pct:+.1f}%)")


def q2_top_dc_accumulators(ed: EnrichedData) -> None:
    """Q2: Top 10 DC Accumulators — Which specific DCs drive the imbalance?"""
    print(_header(2, "Top 10 DC Accumulators"))

    ships = ed.ships
    sim_days = ed.sim_days

    # Unique DC nodes
    dc_nodes = set()
    for nid in ed.ship_tgt_str[ed.ship_tgt_ech == "Customer DC"].unique():
        dc_nodes.add(str(nid))
    for nid in ed.ship_src_str[ed.ship_src_ech == "Customer DC"].unique():
        dc_nodes.add(str(nid))

    # Vectorized per-DC inflow/outflow using groupby
    in_by_dc = ships[ed.ship_tgt_ech == "Customer DC"].groupby(
        ed.ship_tgt_str[ed.ship_tgt_ech == "Customer DC"]
    )["quantity"].sum()
    out_by_dc = ships[ed.ship_src_ech == "Customer DC"].groupby(
        ed.ship_src_str[ed.ship_src_ech == "Customer DC"]
    )["quantity"].sum()

    rows = []
    for dc in dc_nodes:
        dc_in = in_by_dc.get(dc, 0)
        dc_out = out_by_dc.get(dc, 0)
        net = dc_in - dc_out
        ch = _channel_from_node(dc)
        upstream = "Plant-Dir" if dc in ed.plant_direct_dcs else "RDC-Route"
        stores = ed.store_counts.get(dc, 0)
        rows.append({
            "dc": dc, "channel": ch, "upstream": upstream,
            "stores": stores, "inflow": dc_in, "outflow": dc_out,
            "net_accum": net, "net_per_day": net / sim_days,
        })

    df = pd.DataFrame(rows).sort_values("net_accum", ascending=False)

    print(f"\n  {'DC':<18} {'Channel':<12} {'Source':<10} {'Stores':>6} "
          f"{'Net/day':>10} {'In/d':>10} {'Out/d':>10}")
    print(f"  {'─'*18} {'─'*12} {'─'*10} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")

    for _, r in df.head(10).iterrows():
        print(f"  {r['dc']:<18} {r['channel']:<12} {r['upstream']:<10} "
              f"{r['stores']:>6} {r['net_per_day']:>+10,.0f} "
              f"{r['inflow']/sim_days:>10,.0f} {r['outflow']/sim_days:>10,.0f}")

    n_positive = (df["net_accum"] > 0).sum()
    n_total = len(df)
    total_net = df["net_accum"].sum()
    top10_share = df.head(10)["net_accum"].sum() / total_net * 100 if total_net > 0 else 0
    print(f"\n  Summary: {n_positive}/{n_total} DCs accumulating. "
          f"Top 10 = {top10_share:.0f}% of total net accumulation.")


def q3_inventory_age_profile(ed: EnrichedData) -> None:
    """Q3: Inventory Age Profile — Is inventory aging or cycling?"""
    print(_header(3, "Inventory Age Profile by Echelon"))

    inv = ed.inv
    if inv.empty:
        print("\n  No inventory data available.")
        return

    ships = ed.ships
    sim_days = ed.sim_days

    # Throughput per echelon using precomputed columns
    # Stores are terminal nodes (no outflow) — use inflow as throughput
    ech_throughput: dict[str, float] = {}
    for ech in ["Plant", "RDC", "Customer DC"]:
        qty = ships[ed.ship_src_ech == ech]["quantity"].sum()
        ech_throughput[ech] = qty / sim_days
    store_inflow = ships[ed.ship_tgt_str.str.startswith("STORE-")]["quantity"].sum()
    ech_throughput["Store"] = store_inflow / sim_days

    max_day = int(inv["day"].max())
    early = inv[(inv["day"] >= 30) & (inv["day"] <= 60)]
    late = inv[(inv["day"] >= max_day - 30)]

    print(f"\n  {'Echelon':<14} {'Early DOS':>10} {'Late DOS':>10} "
          f"{'Throughput/d':>12} {'Cycling?':>10}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")

    for ech in ["Plant", "RDC", "Customer DC", "Store"]:
        early_inv = early[early["echelon"] == ech]["total"].mean()
        late_inv = late[late["echelon"] == ech]["total"].mean()
        thru = ech_throughput.get(ech, 0)
        early_dos = early_inv / thru if thru > 0 else 0
        late_dos = late_inv / thru if thru > 0 else 0
        verdict = "Cycling" if thru > 0 and late_dos < 20 else "Stagnating"
        if late_dos > early_dos * 1.2:
            verdict = "Growing"
        elif late_dos < early_dos * 0.8:
            verdict = "Draining"

        print(f"  {ech:<14} {early_dos:>10.1f} {late_dos:>10.1f} "
              f"{thru:>12,.0f} {verdict:>10}")


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 2: What's Flowing Where? (Q4–Q6)
# ═══════════════════════════════════════════════════════════════════════════════

def q4_dc_inflow_decomposition(ed: EnrichedData) -> None:
    """Q4: DC Inflow Decomposition — Plant deployment vs RDC push vs other."""
    print(_header(4, "DC Inflow Decomposition"))

    ships = ed.ships
    sim_days = ed.sim_days

    dc_mask = ed.ship_tgt_ech == "Customer DC"
    dc_inflow = ships[dc_mask]
    dc_src_ech = ed.ship_src_ech[dc_mask]

    total_qty = dc_inflow["quantity"].sum()
    total_daily = total_qty / sim_days

    print(f"\n  Total DC inflow: {total_daily:,.0f}/day ({total_qty/1e6:.1f}M total)\n")
    print(f"  {'Source':<22} {'Qty/day':>10} {'Share%':>8} {'Total':>14}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*14}")

    for src_ech in ["Plant", "RDC", "Customer DC", "Store", "Other", "Supplier"]:
        mask = dc_src_ech == src_ech
        qty = dc_inflow[mask]["quantity"].sum()
        if qty == 0:
            continue
        share = qty / total_qty * 100
        daily = qty / sim_days
        print(f"  {src_ech:<22} {daily:>10,.0f} {share:>7.1f}% {qty:>14,.0f}")

    # Time evolution
    max_day = int(dc_inflow["creation_day"].max())
    print(f"\n  Time evolution (daily averages):")
    for label, d0, d1 in [("Days 1-30", 0, 30), ("Days 31-180", 31, 180),
                           ("Days 181-365", 181, max_day + 1)]:
        period_mask = dc_mask & (ships["creation_day"] >= d0) & (ships["creation_day"] < d1)
        n_days = min(d1, max_day + 1) - d0
        if n_days <= 0:
            continue
        period = ships[period_mask]
        period_src = ed.ship_src_ech[period_mask]
        plant_q = period[period_src == "Plant"]["quantity"].sum() / n_days
        rdc_q = period[period_src == "RDC"]["quantity"].sum() / n_days
        print(f"    {label:<16}: Plant={plant_q:,.0f}/d, RDC={rdc_q:,.0f}/d, "
              f"Total={period['quantity'].sum()/n_days:,.0f}/d")


def q5_other_to_store_route(ed: EnrichedData) -> None:
    """Q5: The 'Other → Store' Route — What actually feeds stores outside DCs?"""
    print(_header(5, "Store Supply Sources (Beyond Customer DC)"))

    ships = ed.ships
    sim_days = ed.sim_days

    store_mask = ed.ship_tgt_str.str.startswith("STORE-")
    store_inflow = ships[store_mask]
    store_src_ech = ed.ship_src_ech[store_mask]

    total_qty = store_inflow["quantity"].sum()
    total_daily = total_qty / sim_days

    print(f"\n  Total store inflow: {total_daily:,.0f}/day\n")
    print(f"  {'Source Echelon':<22} {'Qty/day':>10} {'Share%':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*8}")

    for src_ech in ["Customer DC", "RDC", "Plant", "Store", "Club", "Other", "Supplier"]:
        qty = store_inflow[store_src_ech == src_ech]["quantity"].sum()
        if qty == 0:
            continue
        share = qty / total_qty * 100
        daily = qty / sim_days
        print(f"  {src_ech:<22} {daily:>10,.0f} {share:>7.1f}%")

    # ECOM/DTC fulfillment centers — they ARE demand endpoints, not stores
    ecom_mask = ed.ship_tgt_str.str.startswith("ECOM-FC-") | ed.ship_tgt_str.str.startswith("DTC-FC-")
    ecom_inflow = ships[ecom_mask]
    if len(ecom_inflow) > 0:
        ecom_daily = ecom_inflow["quantity"].sum() / sim_days
        ecom_src_ech = ed.ship_src_ech[ecom_mask]
        print(f"\n  ECOM/DTC FC inflow (demand endpoints): {ecom_daily:,.0f}/day")
        for src_ech in sorted(ecom_src_ech.unique()):
            q = ecom_inflow[ecom_src_ech == src_ech]["quantity"].sum() / sim_days
            print(f"    From {src_ech}: {q:,.0f}/day")


def q6_plant_deployment_split(ed: EnrichedData) -> None:
    """Q6: Plant Deployment Split — RDC vs Direct-DC by day and ABC."""
    print(_header(6, "Plant Deployment Split"))

    ships = ed.ships
    sim_days = ed.sim_days

    plant_mask = ed.ship_src_ech == "Plant"
    plant_ships = ships[plant_mask]
    plant_tgt_ech = ed.ship_tgt_ech[plant_mask]
    plant_abc = ed.ship_abc[plant_mask]

    total = plant_ships["quantity"].sum()

    print(f"\n  Total plant deployment: {total/sim_days:,.0f}/day\n")
    print(f"  {'Target':<18} {'Qty/day':>10} {'Share%':>8} {'A%':>6} {'B%':>6} {'C%':>6}")
    print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")

    for tgt_ech in ["RDC", "Customer DC", "Store"]:
        ech_mask = plant_tgt_ech == tgt_ech
        qty = plant_ships[ech_mask]["quantity"].sum()
        if qty == 0:
            continue
        share = qty / total * 100
        daily = qty / sim_days
        a_pct = plant_ships[ech_mask & (plant_abc == "A")]["quantity"].sum() / qty * 100
        b_pct = plant_ships[ech_mask & (plant_abc == "B")]["quantity"].sum() / qty * 100
        c_pct = plant_ships[ech_mask & (plant_abc == "C")]["quantity"].sum() / qty * 100
        print(f"  {tgt_ech:<18} {daily:>10,.0f} {share:>7.1f}% {a_pct:>5.1f}% {b_pct:>5.1f}% {c_pct:>5.1f}%")

    # Time evolution of RDC vs DC split
    max_day = int(plant_ships["creation_day"].max())
    print(f"\n  Deployment split over time:")
    for label, d0, d1 in [("Days 1-30", 0, 30), ("Days 31-90", 31, 90),
                           ("Days 91-180", 91, 180), ("Days 181-365", 181, max_day + 1)]:
        period_mask = plant_mask & (ships["creation_day"] >= d0) & (ships["creation_day"] < d1)
        n_days = min(d1, max_day + 1) - d0
        if n_days <= 0:
            continue
        period = ships[period_mask]
        p_tgt = ed.ship_tgt_ech[period_mask]
        rdc_q = period[p_tgt == "RDC"]["quantity"].sum()
        dc_q = period[p_tgt == "Customer DC"]["quantity"].sum()
        total_p = rdc_q + dc_q
        if total_p == 0:
            continue
        rdc_pct = rdc_q / total_p * 100
        dc_pct = dc_q / total_p * 100
        print(f"    {label:<16}: RDC {rdc_pct:.1f}% / DC {dc_pct:.1f}%  "
              f"(Total {total_p/n_days:,.0f}/d)")


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 3: Demand Signal Fidelity (Q7–Q9)
# ═══════════════════════════════════════════════════════════════════════════════

def q7_store_pos_vs_orders(ed: EnrichedData) -> None:
    """Q7: Store POS Demand vs Store Orders — Do stores order what they sell?"""
    print(_header(7, "Store POS Demand vs Store Orders"))

    orders = ed.orders
    ships = ed.ships
    sim_days = ed.sim_days

    # Store replenishment orders: orders WHERE target is a store
    # (In this sim, DCs create replenishment orders targeting stores — stores
    #  don't place their own orders. source=DC, target=store.)
    store_ord_mask = ed.ord_tgt_str.str.startswith("STORE-")
    store_orders = orders[store_ord_mask]
    store_order_daily = store_orders["quantity"].sum() / sim_days

    # Shipments TO stores
    store_ship_mask = ed.ship_tgt_str.str.startswith("STORE-")
    store_ships = ships[store_ship_mask]
    store_ship_daily = store_ships["quantity"].sum() / sim_days

    # Demand endpoint shipments as POS proxy
    demand_daily = ships[ed.ship_is_demand]["quantity"].sum() / sim_days

    print(f"\n  Daily volumes:")
    print(f"    POS demand (endpoint ships): {demand_daily:>12,.0f}")
    print(f"    Replenishment orders→stores: {store_order_daily:>12,.0f}")
    print(f"    Shipments received at store:  {store_ship_daily:>12,.0f}")
    if demand_daily > 0:
        print(f"    Order/Demand ratio:          {store_order_daily/demand_daily:>12.3f}")
    if store_order_daily > 0:
        print(f"    Ship/Order ratio:            {store_ship_daily/store_order_daily:>12.3f}")

    # By ABC
    ord_abc = _ensure_str(store_orders["product_id"]).map(ed.abc_map)
    ship_abc_store = ed.ship_abc[store_ship_mask]
    print(f"\n  {'ABC':>4} {'Orders/d':>10} {'Ships/d':>10} {'Ship/Ord':>10}")
    print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*10}")
    for abc in ["A", "B", "C"]:
        o_d = store_orders[ord_abc == abc]["quantity"].sum() / sim_days
        s_d = store_ships[ship_abc_store == abc]["quantity"].sum() / sim_days
        ratio = s_d / o_d if o_d > 0 else 0
        print(f"  {abc:>4} {o_d:>10,.0f} {s_d:>10,.0f} {ratio:>10.3f}")


def q8_dc_store_order_fill(ed: EnrichedData) -> None:
    """Q8: DC→Store Shipment vs Store Orders — Are orders being shorted?"""
    print(_header(8, "Store Order Fill Analysis"))

    orders = ed.orders
    ships = ed.ships
    sim_days = ed.sim_days

    # Store orders: orders where target is a store
    store_ord_mask = ed.ord_tgt_str.str.startswith("STORE-")
    store_orders = orders[store_ord_mask]

    # Shipments to stores
    store_ship_mask = ed.ship_tgt_str.str.startswith("STORE-")
    store_ships = ships[store_ship_mask]

    # By status
    status_col = _ensure_str(store_orders["status"])
    print(f"\n  Store order status distribution:")
    for st in sorted(status_col.unique()):
        n = (status_col == st).sum()
        pct = n / len(store_orders) * 100
        print(f"    {str(st):<14}: {n:>10,} ({pct:.1f}%)")

    closed = store_orders[status_col == "CLOSED"]
    total_ordered = store_orders["quantity"].sum()
    total_closed = closed["quantity"].sum()
    total_shipped = store_ships["quantity"].sum()

    fill_rate = total_closed / total_ordered * 100 if total_ordered > 0 else 0
    ship_vs_order = total_shipped / total_ordered * 100 if total_ordered > 0 else 0

    print(f"\n  Total ordered:  {total_ordered/sim_days:>10,.0f}/day")
    print(f"  Total CLOSED:   {total_closed/sim_days:>10,.0f}/day")
    print(f"  Total shipped:  {total_shipped/sim_days:>10,.0f}/day")
    print(f"  Fill rate (CLOSED/ordered):  {fill_rate:.1f}%")
    print(f"  Ship/Order ratio:            {ship_vs_order:.1f}%")


def q9_forecast_mape(ed: EnrichedData) -> None:
    """Q9: Forecast vs Actual Demand — Where is MAPE concentrated?"""
    print(_header(9, "Forecast MAPE Decomposition"))

    forecasts = ed.forecasts
    ships = ed.ships
    sim_days = ed.sim_days

    # Actual demand = shipments to demand endpoints, grouped by (day, product)
    demand_ships = ships[ed.ship_is_demand]
    actual_daily = demand_ships.groupby(
        [demand_ships["creation_day"], _ensure_str(demand_ships["product_id"])]
    )["quantity"].sum().reset_index()
    actual_daily.columns = ["day", "product_id", "actual_daily"]

    fg_fc = forecasts[_ensure_str(forecasts["product_id"]).str.startswith("SKU-")].copy()
    fg_fc["product_id"] = _ensure_str(fg_fc["product_id"])

    # forecast_quantity is a 14-day aggregated forecast → daily equivalent
    FORECAST_HORIZON = 14
    fg_fc["forecast_qty"] = fg_fc["forecast_quantity"] / FORECAST_HORIZON

    merged = fg_fc.merge(actual_daily[["day", "product_id", "actual_daily"]],
                         on=["day", "product_id"], how="left")
    merged["actual_daily"] = merged["actual_daily"].fillna(0)
    merged["abc"] = merged["product_id"].map(ed.abc_map)

    nonzero = merged[(merged["forecast_qty"] > 0) | (merged["actual_daily"] > 0)]
    if len(nonzero) == 0:
        print("\n  No forecast data to compare.")
        return

    print(f"\n  NOTE: Forecasts are {FORECAST_HORIZON}-day aggregated; converted to daily equivalent.")
    print(f"  Using WMAPE (weighted) — robust to zero-demand days.\n")
    print(f"  {'ABC':>4} {'Count':>8} {'Fcst/d':>10} {'Actual/d':>12} "
          f"{'WMAPE%':>8} {'Bias%':>8}")
    print(f"  {'─'*4} {'─'*8} {'─'*10} {'─'*12} {'─'*8} {'─'*8}")

    for abc in ["A", "B", "C"]:
        subset = nonzero[nonzero["abc"] == abc]
        if len(subset) == 0:
            continue
        avg_fc = subset["forecast_qty"].mean()
        avg_act = subset["actual_daily"].mean()
        # WMAPE = sum(|error|) / sum(actual) — industry standard, immune to zero-demand
        abs_error = (subset["forecast_qty"] - subset["actual_daily"]).abs().sum()
        actual_sum = subset["actual_daily"].sum()
        wmape = abs_error / actual_sum * 100 if actual_sum > 0 else 0
        bias = (subset["forecast_qty"].sum() - actual_sum) / actual_sum * 100 if actual_sum > 0 else 0
        print(f"  {abc:>4} {len(subset):>8,} {avg_fc:>10.1f} {avg_act:>12.1f} "
              f"{wmape:>7.1f}% {bias:>+7.1f}%")

    max_day = int(nonzero["day"].max())
    print(f"\n  WMAPE by time period:")
    for label, d0, d1 in [("Days 1-90", 0, 90), ("Days 91-180", 91, 180),
                           ("Days 181-270", 181, 270), ("Days 271-365", 271, max_day + 1)]:
        period = nonzero[(nonzero["day"] >= d0) & (nonzero["day"] < d1)]
        if len(period) == 0:
            continue
        abs_err = (period["forecast_qty"] - period["actual_daily"]).abs().sum()
        act_sum = period["actual_daily"].sum()
        wmape = abs_err / act_sum * 100 if act_sum > 0 else 0
        print(f"    {label:<16}: WMAPE={wmape:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 4: Production & MRP (Q10–Q12)
# ═══════════════════════════════════════════════════════════════════════════════

def q10_bc_production_deficit(ed: EnrichedData) -> None:
    """Q10: B/C Production Deficit — Why -1.2% / -3.9% under demand?"""
    print(_header(10, "B/C Production Deficit Analysis"))

    batches = ed.batches
    ships = ed.ships
    sim_days = ed.sim_days

    batch_prod = _ensure_str(batches["product_id"])
    fg_batches = batches[batch_prod.str.startswith("SKU-")].copy()
    fg_batches["abc"] = _ensure_str(fg_batches["product_id"]).map(ed.abc_map)
    fg_batches["plant"] = _ensure_str(fg_batches["plant_id"])

    # Demand = shipments to demand endpoints
    demand_ships = ships[ed.ship_is_demand]
    demand_abc = ed.ship_abc[ed.ship_is_demand]

    print(f"\n  {'ABC':>4} {'Prod/d':>10} {'Demand/d':>10} {'P/D Ratio':>10} "
          f"{'Batches':>8} {'Avg Batch':>10}")
    print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

    for abc in ["A", "B", "C"]:
        prod_q = fg_batches[fg_batches["abc"] == abc]["quantity"].sum()
        dem_q = demand_ships[demand_abc == abc]["quantity"].sum()
        n_batches = len(fg_batches[fg_batches["abc"] == abc])
        avg_batch = prod_q / n_batches if n_batches > 0 else 0
        ratio = prod_q / dem_q if dem_q > 0 else 0
        print(f"  {abc:>4} {prod_q/sim_days:>10,.0f} {dem_q/sim_days:>10,.0f} "
              f"{ratio:>10.4f} {n_batches:>8,} {avg_batch:>10,.0f}")

    # Changeover analysis
    print(f"\n  SKU changeover frequency per plant:")
    for plant in sorted(fg_batches["plant"].unique()):
        plant_b = fg_batches[fg_batches["plant"] == plant]
        skus_per_day = plant_b.groupby("day_produced")["product_id"].nunique()
        print(f"    {plant}: avg {skus_per_day.mean():.1f} SKUs/day, "
              f"max {skus_per_day.max()}")

    # B/C production by quarter
    print(f"\n  B/C prod/demand ratio by quarter:")
    max_day = int(fg_batches["day_produced"].max())
    for label, d0, d1 in [("Q1 (1-90)", 0, 90), ("Q2 (91-180)", 91, 180),
                           ("Q3 (181-270)", 181, 270), ("Q4 (271-365)", 271, max_day + 1)]:
        for abc in ["B", "C"]:
            prod = fg_batches[(fg_batches["abc"] == abc) &
                              (fg_batches["day_produced"] >= d0) &
                              (fg_batches["day_produced"] < d1)]["quantity"].sum()
            day_mask = ed.ship_is_demand & (ships["creation_day"] >= d0) & (ships["creation_day"] < d1)
            dem = ships[day_mask & (ed.ship_abc == abc)]["quantity"].sum()
            ratio = prod / dem if dem > 0 else 0
            if abc == "B":
                print(f"    {label:<16} B: {ratio:.4f}", end="")
            else:
                print(f"  C: {ratio:.4f}")


def q11_plant_fg_timeline(ed: EnrichedData) -> None:
    """Q11: Plant FG Buildup Timeline — When and where does it accumulate?"""
    print(_header(11, "Plant FG Buildup Timeline"))

    inv = ed.inv
    if inv.empty:
        print("\n  No inventory data available.")
        return

    plant_inv = inv[inv["echelon"] == "Plant"].sort_values("day")
    if plant_inv.empty:
        print("\n  No plant inventory data.")
        return

    min_day = int(plant_inv["day"].min())
    max_day = int(plant_inv["day"].max())
    print(f"\n  Plant FG inventory trajectory (days {min_day}-{max_day}):")
    snapshots = [(0, "Day 0"), (7, "Day 7"), (14, "Day 14"), (30, "Day 30"),
                 (60, "Day 60"), (90, "Day 90"), (180, "Day 180"),
                 (270, "Day 270"), (max_day, f"Day {max_day}")]

    for day, label in snapshots:
        if day > max_day + 1:
            continue  # Skip snapshots beyond data range
        exact = plant_inv[plant_inv["day"] == day]
        if len(exact) > 0:
            val = exact.iloc[0]["total"]
        else:
            closest = plant_inv.iloc[(plant_inv["day"] - day).abs().argsort()[:1]]
            val = closest.iloc[0]["total"]
        print(f"    {label:<12}: {val:>12,.0f} ({val/1e6:.1f}M)")

    day_0 = plant_inv.iloc[0]["total"]
    day_30_data = plant_inv[plant_inv["day"] == min(30, max_day)]
    day_30 = day_30_data.iloc[0]["total"] if len(day_30_data) > 0 else plant_inv.iloc[-1]["total"]
    day_end = plant_inv.iloc[-1]["total"]

    growth_30 = (day_30 - day_0) / day_0 * 100 if day_0 > 0 else 0
    if max_day > 30:
        growth_rest = (day_end - day_30) / day_30 * 100 if day_30 > 0 else 0
        print(f"\n  First 30d growth: {growth_30:+.1f}%")
        print(f"  Day 30→{max_day} growth: {growth_rest:+.1f}%")
    else:
        print(f"\n  Growth over {max_day}d: {growth_30:+.1f}%")

    # Per-plant production vs deployment
    batches = ed.batches
    ships = ed.ships
    sim_days = ed.sim_days

    fg_batches = batches[_ensure_str(batches["product_id"]).str.startswith("SKU-")].copy()
    fg_batches["plant"] = _ensure_str(fg_batches["plant_id"])

    plant_mask = ed.ship_src_ech == "Plant"
    plant_deploy = ships[plant_mask]
    plant_deploy_src = ed.ship_src_str[plant_mask]

    # Vectorized per-plant production and deployment
    prod_by_plant = fg_batches.groupby("plant")["quantity"].sum() / sim_days
    deploy_by_plant = plant_deploy.groupby(plant_deploy_src)["quantity"].sum() / sim_days

    print(f"\n  {'Plant':<12} {'Prod/d':>10} {'Deploy/d':>10} {'Net/d':>10} {'Accum?':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for plant in sorted(prod_by_plant.index):
        prod = prod_by_plant.get(plant, 0)
        deploy = deploy_by_plant.get(plant, 0)
        net = prod - deploy
        verdict = "YES" if net > prod * 0.05 else "no"
        print(f"  {plant:<12} {prod:>10,.0f} {deploy:>10,.0f} {net:>+10,.0f} {verdict:>8}")


def q12_mrp_backpressure(ed: EnrichedData) -> None:
    """Q12: MRP Backpressure Effectiveness — Do DOS caps actually bite?"""
    print(_header(12, "MRP Backpressure Effectiveness"))

    inv = ed.inv
    batches = ed.batches
    ships = ed.ships
    sim_days = ed.sim_days

    if inv.empty:
        print("\n  No inventory data for backpressure analysis.")
        return

    plant_inv = inv[inv["echelon"] == "Plant"].sort_values("day")

    fg_batches = batches[_ensure_str(batches["product_id"]).str.startswith("SKU-")].copy()
    daily_prod = fg_batches.groupby("day_produced")["quantity"].sum()

    caps = {"A": 22, "B": 25, "C": 25}

    # Total daily demand
    demand_ships = ships[ed.ship_is_demand]
    total_demand = demand_ships["quantity"].sum() / sim_days

    # Plant DOS per day (vectorized)
    dos_df = plant_inv[["day", "total"]].copy()
    dos_df.columns = ["day", "inv"]
    dos_df["dos"] = dos_df["inv"] / total_demand if total_demand > 0 else 0.0
    dos_df = dos_df.reset_index(drop=True)

    if len(dos_df) == 0:
        print("\n  Insufficient data.")
        return

    above_a = (dos_df["dos"] > caps["A"]).sum()
    total_days = len(dos_df)
    avg_dos = dos_df["dos"].mean()

    print(f"\n  Plant aggregate DOS:")
    print(f"    Average:       {avg_dos:.1f} days")
    print(f"    Days > {caps['A']}d:   {above_a}/{total_days} ({above_a/total_days*100:.0f}%)")
    print(f"    Max DOS:       {dos_df['dos'].max():.1f} days")
    print(f"    Min DOS:       {dos_df['dos'].min():.1f} days")

    # Backpressure correlation
    if len(dos_df) > 10 and len(daily_prod) > 10:
        prod_series = daily_prod.reindex(dos_df["day"]).fillna(0)
        inv_series = dos_df.set_index("day")["inv"]
        common_days = sorted(set(inv_series.index) & set(prod_series.index))
        if len(common_days) > 20:
            inv_vals = np.array([inv_series.loc[d] for d in common_days[:-1]])
            prod_vals = np.array([prod_series.loc[d] for d in common_days[1:]])
            if inv_vals.std() > 0 and prod_vals.std() > 0:
                corr = np.corrcoef(inv_vals, prod_vals)[0, 1]
                print(f"\n  Backpressure correlation (inv[t] vs prod[t+1]): {corr:+.3f}")
                if corr > 0:
                    print(f"    WARNING: Positive — higher inventory → MORE production")
                    print(f"    Expected: Negative (backpressure should reduce production)")
                else:
                    print(f"    Good: Negative — backpressure is working")


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 5: Deployment Mechanics (Q13–Q15)
# ═══════════════════════════════════════════════════════════════════════════════

def q13_deployment_need_vs_actual(ed: EnrichedData) -> None:
    """Q13: Deployment Need vs Actual — Are targets achievable?"""
    print(_header(13, "Deployment Need vs Actual"))

    ships = ed.ships
    sim_days = ed.sim_days

    plant_mask = ed.ship_src_ech == "Plant"
    plant_ships = ships[plant_mask]
    plant_tgt_ech = ed.ship_tgt_ech[plant_mask]

    deploy_daily = plant_ships["quantity"].sum() / sim_days

    demand_ships = ships[ed.ship_is_demand]
    demand_daily = demand_ships["quantity"].sum() / sim_days

    fg_batches = ed.batches[_ensure_str(ed.batches["product_id"]).str.startswith("SKU-")]
    prod_daily = fg_batches["quantity"].sum() / sim_days

    print(f"\n  Daily flow rates:")
    print(f"    Production:          {prod_daily:>12,.0f}")
    print(f"    Plant deployment:    {deploy_daily:>12,.0f}")
    print(f"    Demand (endpoint):   {demand_daily:>12,.0f}")
    if prod_daily > 0:
        print(f"    Deploy/Prod:         {deploy_daily/prod_daily:>12.3f}")
    if demand_daily > 0:
        print(f"    Deploy/Demand:       {deploy_daily/demand_daily:>12.3f}")

    rdc_deploy = plant_ships[plant_tgt_ech == "RDC"]
    dc_deploy = plant_ships[plant_tgt_ech == "Customer DC"]

    rdc_targets = rdc_deploy.groupby(
        ed.ship_tgt_str[plant_mask][plant_tgt_ech == "RDC"]
    )["quantity"].sum() / sim_days if len(rdc_deploy) > 0 else pd.Series(dtype=float)

    dc_targets = dc_deploy.groupby(
        ed.ship_tgt_str[plant_mask][plant_tgt_ech == "Customer DC"]
    )["quantity"].sum() / sim_days if len(dc_deploy) > 0 else pd.Series(dtype=float)

    print(f"\n  Deployment targets: {len(rdc_targets)} RDCs + {len(dc_targets)} DCs = "
          f"{len(rdc_targets) + len(dc_targets)} total")

    if len(rdc_targets) > 0:
        print(f"\n  RDC deployment per target (target DOS=15):")
        print(f"    Mean:  {rdc_targets.mean():>10,.0f}/day")
        print(f"    Min:   {rdc_targets.min():>10,.0f}/day")
        print(f"    Max:   {rdc_targets.max():>10,.0f}/day")

    if len(dc_targets) > 0:
        print(f"\n  DC deployment per target (targets vary by ABC):")
        print(f"    Mean:  {dc_targets.mean():>10,.0f}/day")
        print(f"    Min:   {dc_targets.min():>10,.0f}/day")
        print(f"    Max:   {dc_targets.max():>10,.0f}/day")


def q14_seasonal_alignment(ed: EnrichedData) -> None:
    """Q14: Seasonal Factor Alignment — Does deployment track POS seasonality?"""
    print(_header(14, "Seasonal Factor Alignment"))

    ships = ed.ships
    sim_days = ed.sim_days

    # POS demand by day
    demand_ships = ships[ed.ship_is_demand]
    daily_demand = demand_ships.groupby("creation_day")["quantity"].sum()

    # Plant deployment by day
    plant_ships = ships[ed.ship_src_ech == "Plant"]
    daily_deploy = plant_ships.groupby("creation_day")["quantity"].sum()

    # Production by day
    fg_batches = ed.batches[_ensure_str(ed.batches["product_id"]).str.startswith("SKU-")]
    daily_prod = fg_batches.groupby("day_produced")["quantity"].sum()

    print(f"\n  Monthly flow comparison (30-day averages):")
    print(f"  {'Month':>6} {'Demand/d':>10} {'Deploy/d':>10} {'Prod/d':>10} "
          f"{'Dep/Dem':>8} {'Prd/Dem':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    max_day = max(daily_demand.index.max(), daily_deploy.index.max())
    for month in range(1, 13):
        d0 = (month - 1) * 30
        d1 = month * 30
        if d0 > max_day:
            break
        dem = daily_demand[(daily_demand.index >= d0) & (daily_demand.index < d1)].mean()
        dep = daily_deploy[(daily_deploy.index >= d0) & (daily_deploy.index < d1)].mean()
        prd = daily_prod[(daily_prod.index >= d0) & (daily_prod.index < d1)].mean()
        dep_dem = dep / dem if dem > 0 else 0
        prd_dem = prd / dem if dem > 0 else 0
        print(f"  {month:>6} {dem:>10,.0f} {dep:>10,.0f} {prd:>10,.0f} "
              f"{dep_dem:>8.3f} {prd_dem:>8.3f}")

    amplitude = 0.12
    phase = 150
    cycle = 365
    print(f"\n  Expected seasonal pattern (amplitude={amplitude}, phase={phase}d):")
    for month in [1, 4, 7, 10]:
        day = (month - 1) * 30 + 15
        factor = 1.0 + amplitude * np.sin(2 * np.pi * (day - phase) / cycle)
        print(f"    Month {month}: factor={factor:.3f}")


def q15_over_deployment(ed: EnrichedData) -> None:
    """Q15: Over-Deployment Detection — Do plants deploy to targets already above target?"""
    print(_header(15, "Over-Deployment Detection"))

    ships = ed.ships
    sim_days = ed.sim_days

    if ed.inv.empty:
        print("\n  No inventory data for over-deployment analysis.")
        return

    # DC-level throughput and inflow using precomputed echelons
    dc_out_mask = ed.ship_src_ech == "Customer DC"
    dc_throughput = ships[dc_out_mask].groupby(
        ed.ship_src_str[dc_out_mask]
    )["quantity"].sum() / sim_days

    dc_in_mask = ed.ship_tgt_ech == "Customer DC"
    dc_inflow_daily = ships[dc_in_mask].groupby(
        ed.ship_tgt_str[dc_in_mask]
    )["quantity"].sum() / sim_days

    plant_dc_mask = (ed.ship_src_ech == "Plant") & (ed.ship_tgt_ech == "Customer DC")
    pl_deploy_daily = ships[plant_dc_mask].groupby(
        ed.ship_tgt_str[plant_dc_mask]
    )["quantity"].sum() / sim_days

    # Exclude demand endpoints (ECOM/DTC have zero outflow by design)
    dc_list = sorted(set(str(x) for x in dc_throughput.index) | set(str(x) for x in dc_inflow_daily.index))
    flow_dcs = [dc for dc in dc_list if not (dc.startswith("ECOM-FC-") or dc.startswith("DTC-FC-"))]

    print(f"\n  {'DC Target':<18} {'Inflow/d':>10} {'Outflow/d':>10} "
          f"{'In/Out':>8} {'Net/d':>10} {'PlDeploy/d':>12}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*12}")

    over_deployed = balanced = under_deployed = 0
    for dc in flow_dcs[:20]:
        inf_d = dc_inflow_daily.get(dc, 0)
        out_d = dc_throughput.get(dc, 0)
        ratio = inf_d / out_d if out_d > 0 else float("inf")
        net = inf_d - out_d
        pl_dep = pl_deploy_daily.get(dc, 0)
        if ratio > 1.1:
            over_deployed += 1
        elif ratio < 0.9:
            under_deployed += 1
        else:
            balanced += 1
        print(f"  {dc:<18} {inf_d:>10,.0f} {out_d:>10,.0f} "
              f"{ratio:>8.2f} {net:>+10,.0f} {pl_dep:>12,.0f}")

    n_flow = len(flow_dcs)
    print(f"\n  Flow DCs (excl ECOM/DTC): {n_flow}")
    print(f"  Over-deployed (in/out > 1.1):  {over_deployed}")
    print(f"  Balanced (0.9-1.1):            {balanced}")
    print(f"  Under-deployed (in/out < 0.9): {under_deployed}")


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 6: Push & Pull Interaction (Q16–Q18)
# ═══════════════════════════════════════════════════════════════════════════════

def q16_rdc_push_analysis(ed: EnrichedData) -> None:
    """Q16: RDC Push Frequency & Volume — How bursty is the push?"""
    print(_header(16, "RDC Push Frequency & Volume"))

    ships = ed.ships
    sim_days = ed.sim_days

    rdc_dc_mask = (ed.ship_src_ech == "RDC") & (ed.ship_tgt_ech == "Customer DC")
    rdc_to_dc = ships[rdc_dc_mask]

    total_daily = rdc_to_dc["quantity"].sum() / sim_days

    dc_in_mask = ed.ship_tgt_ech == "Customer DC"
    dc_inflow_daily = ships[dc_in_mask]["quantity"].sum() / sim_days
    rdc_share = total_daily / dc_inflow_daily * 100 if dc_inflow_daily > 0 else 0

    print(f"\n  RDC → DC volume:  {total_daily:,.0f}/day ({rdc_share:.1f}% of total DC inflow)")

    rdc_src = ed.ship_src_str[rdc_dc_mask]
    print(f"\n  {'RDC':<12} {'Qty/day':>10} {'Days Active':>12} {'Avg/Active Day':>15}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*15}")

    for rdc_id in sorted(rdc_src.unique()):
        grp = rdc_to_dc[rdc_src == rdc_id]
        daily_vol = grp.groupby("creation_day")["quantity"].sum()
        days_active = len(daily_vol)
        avg_active = daily_vol.mean()
        total = grp["quantity"].sum() / sim_days
        print(f"  {str(rdc_id):<12} {total:>10,.0f} {days_active:>12} {avg_active:>15,.0f}")

    daily_rdc_vol = rdc_to_dc.groupby("creation_day")["quantity"].sum()
    if len(daily_rdc_vol) > 10:
        cv = daily_rdc_vol.std() / daily_rdc_vol.mean() if daily_rdc_vol.mean() > 0 else 0
        print(f"\n  Burstiness (CV of daily volume): {cv:.2f}")
        print(f"    (CV < 0.3 = smooth, 0.3-0.7 = moderate, >0.7 = bursty)")


def q17_dual_inflow_collision(ed: EnrichedData) -> None:
    """Q17: Dual-Inflow Collision — Do DCs receive deployment AND push on same day?"""
    print(_header(17, "Dual-Inflow Collision Detection"))

    ships = ed.ships
    sim_days = ed.sim_days

    plant_dc_mask = (ed.ship_src_ech == "Plant") & (ed.ship_tgt_ech == "Customer DC")
    rdc_dc_mask = (ed.ship_src_ech == "RDC") & (ed.ship_tgt_ech == "Customer DC")

    plant_to_dc = ships[plant_dc_mask]
    rdc_to_dc = ships[rdc_dc_mask]

    if len(plant_to_dc) == 0 or len(rdc_to_dc) == 0:
        print("\n  Insufficient data for collision analysis.")
        return

    plant_arrivals = set(
        zip(plant_to_dc["arrival_day"].values,
            _ensure_str(ed.ship_tgt_str[plant_dc_mask]).values)
    )
    rdc_arrivals = set(
        zip(rdc_to_dc["arrival_day"].values,
            _ensure_str(ed.ship_tgt_str[rdc_dc_mask]).values)
    )

    collisions = plant_arrivals & rdc_arrivals
    n_collisions = len(collisions)
    all_dc_days = plant_arrivals | rdc_arrivals
    collision_pct = n_collisions / len(all_dc_days) * 100 if len(all_dc_days) > 0 else 0

    print(f"\n  DC-days with plant deployment only:  {len(plant_arrivals - rdc_arrivals):>8,}")
    print(f"  DC-days with RDC push only:          {len(rdc_arrivals - plant_arrivals):>8,}")
    print(f"  DC-days with BOTH (collision):       {n_collisions:>8,}")
    print(f"  Collision rate:                       {collision_pct:.1f}%")


def q18_push_pull_interference(ed: EnrichedData) -> None:
    """Q18: Push vs Pull Interference — DCs receiving push while orders suppressed?"""
    print(_header(18, "Push vs Pull Interference"))

    orders = ed.orders
    ships = ed.ships
    sim_days = ed.sim_days

    # DC orders (DCs as source)
    dc_ord_mask = ed.ord_src_ech == "Customer DC"
    dc_orders = orders[dc_ord_mask]

    # DC inbound shipments
    dc_in_mask = ed.ship_tgt_ech == "Customer DC"
    dc_inbound = ships[dc_in_mask]

    dc_order_daily = dc_orders.groupby("day")["quantity"].sum()
    dc_inbound_daily = dc_inbound.groupby("arrival_day")["quantity"].sum()

    all_days = sorted(set(dc_order_daily.index) | set(dc_inbound_daily.index))
    if not all_days:
        print("\n  Insufficient data.")
        return

    order_series = dc_order_daily.reindex(all_days, fill_value=0)
    inbound_series = dc_inbound_daily.reindex(all_days, fill_value=0)

    order_median = order_series.median()
    inbound_median = inbound_series.median()

    low_order_days = order_series < order_median * 0.5
    high_inbound_days = inbound_series > inbound_median * 1.2
    interference_days = (low_order_days & high_inbound_days).sum()
    total_days = len(all_days)

    print(f"\n  DC aggregate daily volumes:")
    print(f"    Order median:    {order_median:>12,.0f}")
    print(f"    Inbound median:  {inbound_median:>12,.0f}")
    if order_median > 0:
        print(f"    Ratio (In/Ord):  {inbound_median/order_median:.3f}")

    print(f"\n  Interference detection:")
    print(f"    Days with low orders (<50% median):       {low_order_days.sum():>6}")
    print(f"    Days with high inbound (>120% median):    {high_inbound_days.sum():>6}")
    print(f"    Days with BOTH (interference):            {interference_days:>6} ({interference_days/total_days*100:.1f}%)")

    # DC-level asymmetry
    dc_order_by_dc = dc_orders.groupby(ed.ord_src_str[dc_ord_mask])["quantity"].sum()
    dc_inbound_by_dc = dc_inbound.groupby(ed.ship_tgt_str[dc_in_mask])["quantity"].sum()
    all_dcs = sorted(set(dc_order_by_dc.index) | set(dc_inbound_by_dc.index))
    n_more_inbound = 0
    n_dc_total = 0
    for dc in all_dcs:
        if classify_node(str(dc)) != "Customer DC":
            continue
        n_dc_total += 1
        ordered = dc_order_by_dc.get(dc, 0)
        received = dc_inbound_by_dc.get(dc, 0)
        if received > ordered * 1.1:
            n_more_inbound += 1

    print(f"\n  DCs receiving >10% more than they ordered: {n_more_inbound}/{n_dc_total}")


# ═══════════════════════════════════════════════════════════════════════════════
# Theme 7: System Dynamics (Q19–Q20)
# ═══════════════════════════════════════════════════════════════════════════════

def q19_e2e_cycle_time(ed: EnrichedData) -> None:
    """Q19: End-to-End Cycle Time — How long from production to POS?"""
    print(_header(19, "End-to-End Cycle Time"))

    ships = ed.ships
    sim_days = ed.sim_days

    print(f"\n  Transit time by route (arrival_day - creation_day):")
    routes = [
        ("Plant → RDC", (ed.ship_src_ech == "Plant") & (ed.ship_tgt_ech == "RDC")),
        ("Plant → DC", (ed.ship_src_ech == "Plant") & (ed.ship_tgt_ech == "Customer DC")),
        ("RDC → DC", (ed.ship_src_ech == "RDC") & (ed.ship_tgt_ech == "Customer DC")),
        ("DC → Store", (ed.ship_src_ech == "Customer DC") & (ed.ship_tgt_str.str.startswith("STORE-"))),
    ]

    print(f"  {'Route':<18} {'Mean':>6} {'Median':>8} {'P95':>6} {'Count':>10}")
    print(f"  {'─'*18} {'─'*6} {'─'*8} {'─'*6} {'─'*10}")

    transit_means: dict[str, float] = {}
    for name, mask in routes:
        subset = ships[mask]
        if len(subset) == 0:
            continue
        transit = subset["arrival_day"].astype(float) - subset["creation_day"].astype(float)
        transit = transit[transit >= 0]
        if len(transit) == 0:
            continue
        print(f"  {name:<18} {transit.mean():>6.1f} {transit.median():>8.1f} "
              f"{np.percentile(transit, 95):>6.1f} {len(transit):>10,}")
        transit_means[name] = transit.mean()

    path1 = sum(transit_means.get(r, 0) for r in ["Plant → RDC", "RDC → DC", "DC → Store"])
    path2 = sum(transit_means.get(r, 0) for r in ["Plant → DC", "DC → Store"])

    print(f"\n  Estimated end-to-end transit time:")
    print(f"    Path 1 (Plant→RDC→DC→Store): {path1:.1f} days")
    print(f"    Path 2 (Plant→DC→Store):     {path2:.1f} days")

    # Dwell time via Little's Law
    inv = ed.inv
    if not inv.empty:
        print(f"\n  Dwell time (inventory / throughput = Little's Law):")
        ech_throughput: dict[str, float] = {}
        for ech in ["Plant", "RDC", "Customer DC"]:
            outflow = ships[ed.ship_src_ech == ech]["quantity"].sum() / sim_days
            ech_throughput[ech] = outflow
        # Stores are terminal — use inflow (POS consumption) as throughput
        store_inflow = ships[ed.ship_tgt_str.str.startswith("STORE-")]["quantity"].sum()
        ech_throughput["Store"] = store_inflow / sim_days

        max_day = int(inv["day"].max())
        late_inv = inv[inv["day"] >= max_day - 90]

        print(f"  {'Echelon':<14} {'Avg Inv':>12} {'Throughput/d':>13} {'Dwell (d)':>10}")
        print(f"  {'─'*14} {'─'*12} {'─'*13} {'─'*10}")

        total_dwell_time = 0.0
        for ech in ["Plant", "RDC", "Customer DC", "Store"]:
            avg_inv = late_inv[late_inv["echelon"] == ech]["total"].mean()
            thru = ech_throughput.get(ech, 0)
            dwell = avg_inv / thru if thru > 0 else 0
            total_dwell_time += dwell
            print(f"  {ech:<14} {avg_inv:>12,.0f} {thru:>13,.0f} {dwell:>10.1f}")

        print(f"\n  Total system dwell time: {total_dwell_time:.1f} days")
        print(f"  Total E2E (transit + dwell): ~{total_dwell_time + path1:.0f} days (RDC path)")


def q20_convergence_analysis(ed: EnrichedData) -> None:
    """Q20: Steady-State Convergence — Converging, oscillating, or diverging?"""
    print(_header(20, "Steady-State Convergence Analysis"))

    inv = ed.inv
    if inv.empty:
        print("\n  No inventory data for convergence analysis.")
        return

    max_day = int(inv["day"].max())
    late_inv = inv[inv["day"] >= max_day - 180]

    print(f"\n  Trend analysis (last 180 days, linear regression):")
    print(f"  {'Echelon':<14} {'Slope/day':>12} {'R²':>8} {'Trend':>12} "
          f"{'Start':>12} {'End':>12}")
    print(f"  {'─'*14} {'─'*12} {'─'*8} {'─'*12} {'─'*12} {'─'*12}")

    for ech in ["Plant", "RDC", "Customer DC", "Store"]:
        ech_data = late_inv[late_inv["echelon"] == ech].sort_values("day")
        if len(ech_data) < 5:
            continue

        x = ech_data["day"].values.astype(float)
        y = ech_data["total"].values.astype(float)

        if len(x) > 1 and y.std() > 0:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if abs(slope) < y.mean() * 0.0001:
                trend = "STABLE"
            elif slope > 0:
                trend = "GROWING"
            else:
                trend = "DRAINING"

            print(f"  {ech:<14} {slope:>+12,.0f} {r_sq:>8.3f} {trend:>12} "
                  f"{y[0]:>12,.0f} {y[-1]:>12,.0f}")

    # Oscillation detection
    print(f"\n  Oscillation check (detrended autocorrelation):")
    for ech in ["Plant", "RDC", "Customer DC"]:
        ech_data = late_inv[late_inv["echelon"] == ech].sort_values("day")
        if len(ech_data) < 20:
            continue

        y = ech_data["total"].values.astype(float)
        x = np.arange(len(y), dtype=float)
        coeffs = np.polyfit(x, y, 1)
        detrended = y - np.polyval(coeffs, x)

        if detrended.std() > 0:
            n = len(detrended)
            best_lag = 0
            best_corr = 0.0
            for lag in range(5, min(31, n // 3)):
                a = detrended[:-lag]
                b = detrended[lag:]
                if a.std() > 0 and b.std() > 0:
                    corr = np.corrcoef(a, b)[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_lag = lag

            if best_corr > 0.3:
                day_vals = ech_data["day"].values
                if len(day_vals) > best_lag:
                    period_days = (day_vals[-1] - day_vals[0]) / len(day_vals) * best_lag
                else:
                    period_days = best_lag * 7
                print(f"    {ech:<14}: Period ≈ {period_days:.0f} days "
                      f"(autocorr={best_corr:.2f} at lag {best_lag})")
            else:
                print(f"    {ech:<14}: No clear oscillation (max autocorr={best_corr:.2f})")

    # Convergence projection
    print(f"\n  Convergence projection:")
    for ech in ["Plant", "RDC"]:
        ech_data = late_inv[late_inv["echelon"] == ech].sort_values("day")
        if len(ech_data) < 5:
            continue

        x = ech_data["day"].values.astype(float)
        y = ech_data["total"].values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        current = y[-1]

        if abs(slope) > 0 and slope > 0:
            growth_per_year = slope * 365
            growth_pct = growth_per_year / current * 100 if current > 0 else 0
            print(f"    {ech}: Growing at {growth_pct:+.1f}%/year — no convergence without intervention")
        elif slope < 0:
            print(f"    {ech}: Draining — converging toward lower steady state")
        else:
            print(f"    {ech}: Stable")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep Flow Diagnostic — 20 Questions"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/output"),
        help="Simulation output directory (default: data/output)",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    print("=" * WIDTH)
    print("  DEEP FLOW DIAGNOSTIC — 20 QUESTIONS".center(WIDTH))
    print("=" * WIDTH)
    print(f"\n  Data directory: {data_dir}\n")

    t0 = time.time()

    # Load data
    data = load_all_data(data_dir)

    t_load = time.time() - t0
    print(f"\n  Data loaded in {t_load:.1f}s")
    print(f"  Shipments: {len(data.shipments):,}, Orders: {len(data.orders):,}")
    print(f"  Batches: {len(data.batches):,}, Sim days: {data.sim_days}")

    # Precompute echelon/channel/ABC columns ONCE
    ed = enrich_data(data)

    # ── Theme 1: Where Is the Inventory? ──────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 1: WHERE IS THE INVENTORY?".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q1_dc_accumulation_by_channel(ed)
    q2_top_dc_accumulators(ed)
    q3_inventory_age_profile(ed)

    # ── Theme 2: What's Flowing Where? ────────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 2: WHAT'S FLOWING WHERE?".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q4_dc_inflow_decomposition(ed)
    q5_other_to_store_route(ed)
    q6_plant_deployment_split(ed)

    # ── Theme 3: Demand Signal Fidelity ───────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 3: DEMAND SIGNAL FIDELITY".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q7_store_pos_vs_orders(ed)
    q8_dc_store_order_fill(ed)
    q9_forecast_mape(ed)

    # ── Theme 4: Production & MRP ─────────────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 4: PRODUCTION & MRP".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q10_bc_production_deficit(ed)
    q11_plant_fg_timeline(ed)
    q12_mrp_backpressure(ed)

    # ── Theme 5: Deployment Mechanics ─────────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 5: DEPLOYMENT MECHANICS".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q13_deployment_need_vs_actual(ed)
    q14_seasonal_alignment(ed)
    q15_over_deployment(ed)

    # ── Theme 6: Push & Pull Interaction ──────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 6: PUSH & PULL INTERACTION".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q16_rdc_push_analysis(ed)
    q17_dual_inflow_collision(ed)
    q18_push_pull_interference(ed)

    # ── Theme 7: System Dynamics ──────────────────────────────────────────
    print(f"\n{'═' * WIDTH}")
    print("  THEME 7: SYSTEM DYNAMICS".center(WIDTH))
    print(f"{'═' * WIDTH}")

    q19_e2e_cycle_time(ed)
    q20_convergence_analysis(ed)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═' * WIDTH}")
    print(f"  DIAGNOSTIC COMPLETE — {elapsed:.0f}s elapsed".center(WIDTH))
    print(f"{'═' * WIDTH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
