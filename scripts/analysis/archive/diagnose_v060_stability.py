#!/usr/bin/env python3
"""
v0.60.0 Stability Diagnostic — Targeted Investigation.

Tests five hypotheses about plant FG accumulation and RDC DOS readings:

  H1: Diagnostic DOS denominator is wrong for non-store echelons
  H2: Static _target_expected_demand causes seasonal deployment mismatch
  H3: All deployment targets saturated -> need=0 -> FG stays at plants
  H4: Flow conservation counting includes priming period artifacts
  H5: MRP backpressure not working (production doesn't slow when FG high)

Usage:
    poetry run python scripts/analysis/diagnose_v060_stability.py
    poetry run python scripts/analysis/diagnose_v060_stability.py \
        --data-dir data/output
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Ensure diagnostics package importable
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics.loader import (
    DataBundle,
    classify_node,
    is_demand_endpoint,
    is_finished_good,
    load_all_data,
)

WIDTH = 78

# Thresholds
_SEASONAL_SWING_BUG = 20  # % swing -> REAL BUG
_SEASONAL_SWING_MINOR = 10  # % swing -> MINOR
_SATURATION_PCT = 0.6  # fraction of targets saturated -> ARTIFACT
_UNDER_TARGET_PCT = 0.3  # fraction under target -> REAL BUG
_ACTIVE_DAY_PCT = 50  # % days with shipments -> "active"
_OVER_TARGET_RATIO = 1.1
_NEAR_TARGET_RATIO = 0.8
_BACKPRESSURE_STRONG = 0.90  # ratio < this -> WORKING
_BACKPRESSURE_WEAK = 0.97  # ratio < this -> WEAK
_DOS_ARTIFACT_THRESH = 10.0  # corrected DOS >= this -> artifact
_DOS_BUG_THRESH = 8.0  # corrected DOS < this -> real bug
_DIAG_BUG_THRESH = 8.0
_FLOW_STABLE_PCT = 5  # delta% below this -> stable
_FLOW_PERSISTENT_PCT = 10  # delta% above this -> persistent issue
_MIN_COMMON_DAYS = 10
_ROLLING_WINDOW = 7
_CORRELATION_MIN_DAYS = 30
_EXCLUDE_PRIMING_DAYS = 30
_ARTIFACT_MAJORITY = 3
_BUG_THRESHOLD = 2
_MAX_DC_DISPLAY = 5  # max DCs shown in per-target DOS table
_SEASONALITY_AMPLITUDE = 0.12  # from simulation_config.json
_RDC_TARGET_DOS = 15.0
_DC_BUFFER_DAYS = 7.0
_DC_MULT_A = 1.5
_DC_MULT_B = 2.0
_DC_MULT_C = 2.5
_MRP_CAP_A = 30
_MRP_CAP_B = 35
_MRP_CAP_C = 35


# ===================================================================
# Investigation 1: Per-Target DOS (H1)
# ===================================================================

def investigate_per_target_dos(data: DataBundle) -> dict:
    """Compute DOS per deployment target using target-specific demand.

    The existing diagnostic divides total echelon inventory by total
    network POS demand. But RDCs only serve RDC-routed stores, and
    plant-direct DCs bypass RDCs entirely. This investigation computes
    DOS using only the downstream demand each target actually serves.
    """
    print(
        f"\n{'[H1] Per-Target DOS — Denominator Test':=^{WIDTH}}"
    )

    links = data.links
    locations = data.locations
    shipments = data.shipments
    inv = data.inv_by_echelon

    # Build downstream map from links
    downstream_map: dict[str, list[str]] = defaultdict(list)
    upstream_map: dict[str, str] = {}
    for _, row in links.iterrows():
        src = str(row["source_id"])
        tgt = str(row["target_id"])
        downstream_map[src].append(tgt)
        upstream_map[tgt] = src

    # Identify deployment targets (RDCs + plant-direct DCs)
    rdc_ids = [
        str(row["id"])
        for _, row in locations.iterrows()
        if str(row["id"]).startswith("RDC-")
    ]
    dc_ids = [
        str(row["id"])
        for _, row in locations.iterrows()
        if classify_node(str(row["id"])) == "Customer DC"
    ]
    plant_direct_dc_ids = [
        dc
        for dc in dc_ids
        if upstream_map.get(dc, "").startswith("PLANT-")
    ]

    # Compute per-target downstream POS demand via DFS
    sim_days = data.sim_days

    # Get daily POS by store
    demand_ships = shipments[
        shipments["target_id"].apply(is_demand_endpoint)
        & shipments["product_id"].apply(is_finished_good)
    ]
    store_daily_demand: dict[str, float] = {}
    if len(demand_ships) > 0:
        store_totals = (
            demand_ships.groupby("target_id", observed=True)["quantity"].sum()
        )
        for store_id, total in store_totals.items():
            store_daily_demand[str(store_id)] = (
                float(total) / sim_days
            )

    def _get_downstream_stores(root_id: str) -> set[str]:
        """DFS to find demand-endpoint stores downstream."""
        visited: set[str] = set()
        stack = [root_id]
        stores: set[str] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if is_demand_endpoint(node):
                stores.add(node)
            for child in downstream_map.get(node, []):
                stack.append(child)
        return stores

    # Compute per-target daily demand
    target_demand: dict[str, float] = {}
    target_stores: dict[str, set[str]] = {}
    all_targets = rdc_ids + plant_direct_dc_ids
    for target_id in all_targets:
        stores = _get_downstream_stores(target_id)
        target_stores[target_id] = stores
        daily = sum(
            store_daily_demand.get(s, 0.0) for s in stores
        )
        target_demand[target_id] = daily

    total_network_demand = sum(store_daily_demand.values())

    # Get latest-day inventory by target echelon
    if len(inv) == 0:
        print("  WARNING: No inventory data")
        return {"verdict": "NO_DATA"}

    max_day = int(inv["day"].max())

    # Stream inventory.parquet for latest day
    inv_path = data.data_dir / "inventory.parquet"
    per_node_inv = _stream_latest_day_inventory(inv_path, max_day)

    # Compute per-target DOS
    rdc_results: list[dict] = []
    dc_results: list[dict] = []

    for target_id in all_targets:
        daily_dem = target_demand.get(target_id, 0.0)
        node_inv = per_node_inv.get(target_id, 0.0)
        dos = node_inv / daily_dem if daily_dem > 0 else np.nan

        n_stores = len(target_stores.get(target_id, set()))
        echelon = classify_node(target_id)

        share = (
            daily_dem / total_network_demand * 100
            if total_network_demand > 0
            else 0
        )
        entry = {
            "target_id": target_id,
            "echelon": echelon,
            "inventory": node_inv,
            "daily_demand": daily_dem,
            "demand_share": share,
            "dos": dos,
            "n_downstream_stores": n_stores,
        }
        if target_id.startswith("RDC-"):
            rdc_results.append(entry)
        else:
            dc_results.append(entry)

    # Echelon-level comparison
    rdc_total_inv = sum(r["inventory"] for r in rdc_results)
    rdc_total_dem = sum(r["daily_demand"] for r in rdc_results)
    dc_total_inv = sum(r["inventory"] for r in dc_results)
    dc_total_dem = sum(r["daily_demand"] for r in dc_results)

    def _safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else np.nan

    rdc_dos_corr = _safe_div(rdc_total_inv, rdc_total_dem)
    rdc_dos_diag = _safe_div(rdc_total_inv, total_network_demand)
    dc_dos_corr = _safe_div(dc_total_inv, dc_total_dem)
    dc_dos_diag = _safe_div(dc_total_inv, total_network_demand)

    rdc_share = (
        rdc_total_dem / total_network_demand * 100
        if total_network_demand > 0
        else 0
    )
    dc_share = (
        dc_total_dem / total_network_demand * 100
        if total_network_demand > 0
        else 0
    )

    print("\n  Network demand allocation:")
    print(
        f"    Total POS demand/day:       "
        f"    {total_network_demand:>12,.0f}"
    )
    print(
        f"    RDC-routed demand/day:      "
        f"    {rdc_total_dem:>12,.0f}  ({rdc_share:.1f}%)"
    )
    print(
        f"    Plant-direct DC demand/day: "
        f"    {dc_total_dem:>12,.0f}  ({dc_share:.1f}%)"
    )
    overlap = rdc_total_dem + dc_total_dem
    cov_pct = overlap / total_network_demand * 100
    print(
        f"    Sum (coverage check):       "
        f"    {overlap:>12,.0f}  ({cov_pct:.1f}%)"
    )

    print(f"\n  RDC DOS comparison (day {max_day}):")
    print(
        f"    Diagnostic (/ total POS):   "
        f"    {rdc_dos_diag:.1f} DOS"
    )
    print(
        f"    Corrected (/ RDC demand):   "
        f"    {rdc_dos_corr:.1f} DOS"
    )
    print("    Target:                         15.0 DOS")
    correction_factor = np.nan
    if not np.isnan(rdc_dos_corr) and rdc_dos_diag > 0:
        correction_factor = rdc_dos_corr / rdc_dos_diag
        print(
            f"    Correction factor:          "
            f"    {correction_factor:.2f}x"
        )

    print(f"\n  Plant-direct DC DOS (day {max_day}):")
    print(
        f"    Diagnostic (/ total POS):   "
        f"    {dc_dos_diag:.1f} DOS"
    )
    print(
        f"    Corrected (/ DC demand):    "
        f"    {dc_dos_corr:.1f} DOS"
    )
    print("    Target (A/B/C):                 10.5 / 14.0 / 17.5")

    # Per-target detail
    rdc_results.sort(
        key=lambda r: r["dos"] if not np.isnan(r["dos"]) else 999
    )
    print(f"\n  Per-RDC DOS ({len(rdc_results)} targets):")
    _print_target_table(rdc_results)

    if dc_results:
        dc_results.sort(
            key=lambda r: r["dos"]
            if not np.isnan(r["dos"])
            else 999
        )
        dc_dos_vals = [
            r["dos"] for r in dc_results if not np.isnan(r["dos"])
        ]
        print(
            f"\n  Plant-direct DC DOS "
            f"({len(dc_results)} targets):"
        )
        if dc_dos_vals:
            print(
                f"    Range: {min(dc_dos_vals):.1f} - "
                f"{max(dc_dos_vals):.1f}  "
                f"Median: {np.median(dc_dos_vals):.1f}"
            )
        _print_target_table(dc_results[:_MAX_DC_DISPLAY])
        if len(dc_results) > _MAX_DC_DISPLAY:
            print(
                f"    ... ({len(dc_results) - _MAX_DC_DISPLAY} more)"
            )

    # Verdict
    is_artifact = False
    if not np.isnan(rdc_dos_corr) and not np.isnan(rdc_dos_diag):
        if (
            rdc_dos_corr >= _DOS_ARTIFACT_THRESH
            and rdc_dos_diag < _DIAG_BUG_THRESH
        ):
            is_artifact = True
            print(
                f"\n  VERDICT: ARTIFACT — RDC DOS underreported "
                f"by {correction_factor:.1f}x"
            )
        elif rdc_dos_corr < _DOS_BUG_THRESH:
            print(
                f"\n  VERDICT: REAL BUG — RDC DOS still low "
                f"({rdc_dos_corr:.1f}) even corrected"
            )
        else:
            print(
                f"\n  VERDICT: ARTIFACT — corrected "
                f"DOS ({rdc_dos_corr:.1f}) near target"
            )
            is_artifact = True

    return {
        "rdc_dos_diagnostic": rdc_dos_diag,
        "rdc_dos_corrected": rdc_dos_corr,
        "dc_dos_diagnostic": dc_dos_diag,
        "dc_dos_corrected": dc_dos_corr,
        "rdc_demand_share": rdc_share,
        "dc_demand_share": dc_share,
        "verdict": "ARTIFACT" if is_artifact else "REAL BUG",
    }


def _print_target_table(rows: list[dict]) -> None:
    """Print a target DOS table."""
    hdr = (
        f"    {'Target':<14}  {'Inv':>12}  "
        f"{'Demand/d':>10}  {'DOS':>8}  {'Stores':>6}"
    )
    sep = (
        f"    {'-'*14}  {'-'*12}  "
        f"{'-'*10}  {'-'*8}  {'-'*6}"
    )
    print(hdr)
    print(sep)
    for r in rows:
        dos_s = (
            f"{r['dos']:.1f}"
            if not np.isnan(r["dos"])
            else "N/A"
        )
        print(
            f"    {r['target_id']:<14}  "
            f"{r['inventory']:>12,.0f}  "
            f"{r['daily_demand']:>10,.0f}  "
            f"{dos_s:>8}  "
            f"{r['n_downstream_stores']:>6}"
        )


def _stream_latest_day_inventory(
    inv_path: Path, target_day: int
) -> dict[str, float]:
    """Stream inventory.parquet for per-node FG inventory on target_day.

    Returns {node_id: total_fg_inventory}.
    """
    if not inv_path.exists():
        return {}

    pf = pq.ParquetFile(inv_path)
    columns = ["day", "node_id", "product_id", "actual_inventory"]
    per_node: dict[str, float] = defaultdict(float)

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=columns)

        # Filter to target day
        day_col = table.column("day")
        day_mask = pc.equal(day_col, target_day)
        filtered = table.filter(day_mask)
        if filtered.num_rows == 0:
            continue

        # Filter to FG products
        prod_col = filtered.column("product_id")
        if hasattr(prod_col.type, "value_type"):
            prod_col = prod_col.cast(prod_col.type.value_type)
        fg_mask = pc.starts_with(prod_col, pattern="SKU-")
        fg_table = filtered.filter(fg_mask)
        if fg_table.num_rows == 0:
            continue

        chunk = fg_table.to_pandas()
        for col in ("node_id", "product_id"):
            if hasattr(chunk[col], "cat"):
                chunk[col] = chunk[col].astype(str)

        grp = chunk.groupby("node_id")["actual_inventory"].sum()
        for node_id, inv_val in grp.items():
            per_node[str(node_id)] += float(inv_val)

    return dict(per_node)


# ===================================================================
# Investigation 2: Seasonal Demand Gap (H2)
# ===================================================================

def investigate_seasonal_gap(data: DataBundle) -> dict:
    """Compare static base demand (used by deployment) vs actual POS.

    _precompute_deployment_targets() uses base_demand_matrix (static).
    If POS has seasonal patterns, deployment under-ships during peaks
    and over-ships during troughs, causing plant FG accumulation.
    """
    print(
        f"\n{'[H2] Seasonal Demand Gap — Static vs Actual':=^{WIDTH}}"
    )

    shipments = data.shipments
    sim_days = data.sim_days

    # Actual POS demand by day (30-day rolling average)
    demand_ships = shipments[
        shipments["target_id"].apply(is_demand_endpoint)
        & shipments["product_id"].apply(is_finished_good)
    ]
    if len(demand_ships) == 0:
        print("  WARNING: No demand shipments found")
        return {"verdict": "NO_DATA"}

    daily_pos = (
        demand_ships.groupby("creation_day")["quantity"].sum()
    )
    all_days = range(
        int(daily_pos.index.min()),
        int(daily_pos.index.max()) + 1,
    )
    daily_pos = daily_pos.reindex(all_days, fill_value=0)
    pos_rolling = daily_pos.rolling(30, min_periods=1).mean()

    # Base demand = average daily POS (static expected)
    base_daily = float(daily_pos.sum()) / sim_days

    print(f"\n  Base daily demand (static):  {base_daily:>12,.0f}")
    print(f"  Simulation days:             {sim_days:>12}")

    print("\n  Seasonal gap (30d rolling vs static):")
    print(
        f"  {'Period':>12}  {'Actual/d':>12}  "
        f"{'Base/d':>12}  {'Gap':>12}  {'Gap%':>8}"
    )
    print(
        f"  {'-'*12}  {'-'*12}  "
        f"{'-'*12}  {'-'*12}  {'-'*8}"
    )

    gaps: list[float] = []
    snapshots: list[dict] = []
    for snap_day in range(30, sim_days + 1, 30):
        actual = (
            float(pos_rolling.loc[snap_day])
            if snap_day in pos_rolling.index
            else base_daily
        )
        gap = actual - base_daily
        gap_pct = gap / base_daily * 100 if base_daily > 0 else 0
        gaps.append(gap_pct)
        snapshots.append({
            "day": snap_day,
            "actual": actual,
            "base": base_daily,
            "gap": gap,
            "gap_pct": gap_pct,
        })
        print(
            f"  {snap_day:>8}d    {actual:>12,.0f}  "
            f"{base_daily:>12,.0f}  {gap:>+12,.0f}  "
            f"{gap_pct:>+7.1f}%"
        )

    max_gap = max(gaps) if gaps else 0
    min_gap = min(gaps) if gaps else 0
    swing = max_gap - min_gap

    print(f"\n  Peak gap:   {max_gap:>+.1f}%")
    print(f"  Trough gap: {min_gap:>+.1f}%")
    print(f"  Swing:      {swing:.1f}%")

    amp_pct = _SEASONALITY_AMPLITUDE * 100
    print(f"\n  Config seasonality amplitude: +/-{amp_pct:.0f}%")
    print(f"  Measured POS swing:           {swing:.1f}%")

    if swing > _SEASONAL_SWING_BUG:
        verdict = "REAL BUG"
        print(
            f"\n  VERDICT: REAL BUG — {swing:.0f}% swing "
            f"vs static deployment"
        )
        print(
            "  Deployment under-ships during peaks, "
            "over-ships during troughs."
        )
    elif swing > _SEASONAL_SWING_MINOR:
        verdict = "MINOR"
        print(
            f"\n  VERDICT: MINOR — {swing:.1f}% swing "
            f"is noticeable but moderate"
        )
    else:
        verdict = "ARTIFACT"
        print(
            f"\n  VERDICT: ARTIFACT — {swing:.1f}% swing "
            f"is within normal noise"
        )

    return {
        "base_daily": base_daily,
        "max_gap_pct": max_gap,
        "min_gap_pct": min_gap,
        "swing_pct": swing,
        "snapshots": snapshots,
        "verdict": verdict,
    }


# ===================================================================
# Investigation 3: Deployment Need Saturation (H3)
# ===================================================================

def investigate_need_saturation(data: DataBundle) -> dict:
    """Check if deployment targets are saturated (need ~ 0).

    If on_hand + in_transit >= target_position for most targets,
    need = 0 and FG stays at plants even though production continues.
    """
    print(
        f"\n{'[H3] Deployment Need Saturation':=^{WIDTH}}"
    )

    shipments = data.shipments
    links = data.links
    locations = data.locations
    sim_days = data.sim_days

    # Config targets
    rdc_target_dos = _RDC_TARGET_DOS
    dc_dos_b = _DC_BUFFER_DAYS * _DC_MULT_B  # 14.0

    # Build upstream map
    upstream_map: dict[str, str] = {}
    downstream_map: dict[str, list[str]] = defaultdict(list)
    for _, row in links.iterrows():
        src = str(row["source_id"])
        tgt = str(row["target_id"])
        upstream_map[tgt] = src
        downstream_map[src].append(tgt)

    # Identify deployment targets
    rdc_ids = [
        str(row["id"])
        for _, row in locations.iterrows()
        if str(row["id"]).startswith("RDC-")
    ]
    dc_ids = [
        str(row["id"])
        for _, row in locations.iterrows()
        if classify_node(str(row["id"])) == "Customer DC"
    ]
    plant_direct_dc_ids = [
        dc
        for dc in dc_ids
        if upstream_map.get(dc, "").startswith("PLANT-")
    ]
    all_targets = rdc_ids + plant_direct_dc_ids

    # Compute daily demand per target (from POS)
    demand_ships = shipments[
        shipments["target_id"].apply(is_demand_endpoint)
        & shipments["product_id"].apply(is_finished_good)
    ]
    store_daily_demand: dict[str, float] = {}
    if len(demand_ships) > 0:
        store_totals = (
            demand_ships.groupby("target_id", observed=True)["quantity"].sum()
        )
        for store_id, total in store_totals.items():
            store_daily_demand[str(store_id)] = (
                float(total) / sim_days
            )

    def _get_downstream_demand(root_id: str) -> float:
        visited: set[str] = set()
        stack = [root_id]
        total = 0.0
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if is_demand_endpoint(node):
                total += store_daily_demand.get(node, 0.0)
            for child in downstream_map.get(node, []):
                stack.append(child)
        return total

    target_daily_demand = {
        tid: _get_downstream_demand(tid) for tid in all_targets
    }

    # Analyze plant shipments to targets over time
    fg_ships = shipments[
        shipments["product_id"].apply(is_finished_good)
    ].copy()
    fg_ships["source_echelon"] = (
        fg_ships["source_id"].map(classify_node)
    )

    plant_ships = fg_ships[
        fg_ships["source_echelon"] == "Plant"
    ]

    # Daily deployment volume by target
    deploy_by_td = (
        plant_ships.groupby(["target_id", "creation_day"], observed=True)[
            "quantity"
        ]
        .sum()
        .reset_index()
    )

    # What fraction of days did each target receive shipments?
    target_active_days: dict[str, int] = {}
    target_total_received: dict[str, float] = {}
    if len(deploy_by_td) > 0:
        for target_id in all_targets:
            t_ships = deploy_by_td[
                deploy_by_td["target_id"] == target_id
            ]
            target_active_days[target_id] = len(t_ships)
            target_total_received[target_id] = float(
                t_ships["quantity"].sum()
            )

    inv_path = data.data_dir / "inventory.parquet"

    # Compute target positions
    target_pos_map: dict[str, float] = {}
    for tid in all_targets:
        daily = target_daily_demand.get(tid, 0.0)
        if tid.startswith("RDC-"):
            target_pos_map[tid] = rdc_target_dos * daily
        else:
            target_pos_map[tid] = dc_dos_b * daily

    # Report
    n_rdc = len(rdc_ids)
    n_dc = len(plant_direct_dc_ids)
    print(
        f"\n  Deployment targets: {len(all_targets)} "
        f"({n_rdc} RDCs + {n_dc} plant-direct DCs)"
    )
    print(
        "\n  Deployment activity "
        "(days with shipments / total sim days):"
    )
    print(
        f"  {'Target':<14}  {'Days Active':>12}  "
        f"{'Active%':>8}  {'Total Recv':>14}  "
        f"{'Dem/day':>10}  {'Target Pos':>12}"
    )
    print(
        f"  {'-'*14}  {'-'*12}  {'-'*8}  "
        f"{'-'*14}  {'-'*10}  {'-'*12}"
    )

    saturated_count = 0
    active_count = 0
    for tid in sorted(all_targets):
        days_active = target_active_days.get(tid, 0)
        active_pct = (
            days_active / sim_days * 100 if sim_days > 0 else 0
        )
        total_recv = target_total_received.get(tid, 0.0)
        daily = target_daily_demand.get(tid, 0.0)
        target_pos = target_pos_map.get(tid, 0.0)

        if active_pct < _ACTIVE_DAY_PCT:
            saturated_count += 1
        else:
            active_count += 1

        print(
            f"  {tid:<14}  {days_active:>12}  "
            f"{active_pct:>7.1f}%  {total_recv:>14,.0f}  "
            f"{daily:>10,.0f}  {target_pos:>12,.0f}"
        )

    # Per-target inventory vs target at end of sim
    max_day = (
        int(data.inv_by_echelon["day"].max())
        if len(data.inv_by_echelon) > 0
        else sim_days
    )
    per_node_inv = _stream_latest_day_inventory(inv_path, max_day)

    print(f"\n  End-of-sim position vs target (day {max_day}):")
    print(
        f"  {'Target':<14}  {'On-Hand':>12}  "
        f"{'Target Pos':>12}  {'Ratio':>8}  {'Status':>10}"
    )
    print(
        f"  {'-'*14}  {'-'*12}  "
        f"{'-'*12}  {'-'*8}  {'-'*10}"
    )

    over_target = 0
    under_target = 0
    near_target = 0
    for tid in sorted(all_targets):
        on_hand = per_node_inv.get(tid, 0.0)
        target_pos = target_pos_map.get(tid, 0.0)
        ratio = on_hand / target_pos if target_pos > 0 else np.nan

        if np.isnan(ratio):
            status = "NO_DATA"
        elif ratio >= _OVER_TARGET_RATIO:
            status = "OVER"
            over_target += 1
        elif ratio >= _NEAR_TARGET_RATIO:
            status = "NEAR"
            near_target += 1
        else:
            status = "UNDER"
            under_target += 1

        ratio_s = (
            f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
        )
        print(
            f"  {tid:<14}  {on_hand:>12,.0f}  "
            f"{target_pos:>12,.0f}  {ratio_s:>8}  {status:>10}"
        )

    total_targets = over_target + under_target + near_target
    print(
        f"\n  Summary: {over_target} OVER, "
        f"{near_target} NEAR, {under_target} UNDER "
        f"(of {total_targets})"
    )
    print(
        f"  Days active < {_ACTIVE_DAY_PCT}%: "
        f"{saturated_count} targets (need ~ 0)"
    )

    n_all = len(all_targets)
    if saturated_count > n_all * _SATURATION_PCT:
        verdict = "ARTIFACT"
        print(
            f"\n  VERDICT: ARTIFACT — "
            f"{saturated_count}/{n_all} targets saturated"
        )
        print(
            "  FG accumulates at plants because "
            "downstream doesn't need more."
        )
        print(
            f"  MRP DOS caps ({_MRP_CAP_A}/{_MRP_CAP_B}/"
            f"{_MRP_CAP_C}) allow continued production."
        )
    elif under_target > n_all * _UNDER_TARGET_PCT:
        verdict = "REAL BUG"
        print(
            f"\n  VERDICT: REAL BUG — "
            f"{under_target}/{total_targets} targets under"
        )
        print(
            "  Targets need inventory but aren't receiving it."
        )
    else:
        verdict = "MIXED"
        print(
            "\n  VERDICT: MIXED — partially saturated, "
            "some under target"
        )

    return {
        "n_targets": n_all,
        "n_saturated": saturated_count,
        "n_over": over_target,
        "n_near": near_target,
        "n_under": under_target,
        "verdict": verdict,
    }


# ===================================================================
# Investigation 4: Flow Conservation Excluding Priming (H4)
# ===================================================================

def investigate_flow_excl_priming(data: DataBundle) -> dict:
    """Recompute flow conservation excluding first 30 days.

    The existing diagnostic counts ALL days including priming (0-10)
    and stabilization. Priming creates large synthetic inflows with
    no corresponding outflows.
    """
    print(
        f"\n{'[H4] Flow Conservation excl. Priming':=^{WIDTH}}"
    )

    exclude_days = _EXCLUDE_PRIMING_DAYS
    ships = data.shipments[
        data.shipments["product_id"].apply(is_finished_good)
    ].copy()
    ships["source_echelon"] = (
        ships["source_id"].map(classify_node)
    )
    ships["target_echelon"] = (
        ships["target_id"].map(classify_node)
    )

    fg_batches = data.batches[
        data.batches["product_id"].apply(is_finished_good)
    ]

    sim_days = data.sim_days
    post_stab_days = sim_days - exclude_days

    ships_post = ships[ships["creation_day"] >= exclude_days]
    batches_post = fg_batches[
        fg_batches["day_produced"] >= exclude_days
    ]

    echelon_order = ["Plant", "RDC", "Customer DC", "Store"]

    def _compute_flows(
        s: pd.DataFrame,
        b: pd.DataFrame,
        n_days: int,
    ) -> dict[str, dict]:
        demand_s = s[s["target_id"].apply(is_demand_endpoint)]
        flows: dict[str, dict] = {}
        for ech in echelon_order:
            if ech == "Plant":
                inflow = float(b["quantity"].sum())
                outflow = float(
                    s[s["source_echelon"] == "Plant"][
                        "quantity"
                    ].sum()
                )
            elif ech == "Store":
                inflow = float(
                    s[s["target_echelon"] == ech][
                        "quantity"
                    ].sum()
                )
                outflow = float(
                    demand_s[
                        demand_s["target_echelon"] == ech
                    ]["quantity"].sum()
                )
            else:
                inflow = float(
                    s[s["target_echelon"] == ech][
                        "quantity"
                    ].sum()
                )
                outflow = float(
                    s[s["source_echelon"] == ech][
                        "quantity"
                    ].sum()
                )

            daily_in = inflow / n_days if n_days > 0 else 0
            daily_out = outflow / n_days if n_days > 0 else 0
            delta = daily_in - daily_out
            throughput = max(daily_in, daily_out)
            delta_pct = (
                abs(delta) / throughput * 100
                if throughput > 0
                else 0
            )
            flows[ech] = {
                "daily_inflow": daily_in,
                "daily_outflow": daily_out,
                "daily_delta": delta,
                "delta_pct": delta_pct,
            }
        return flows

    flows_all = _compute_flows(ships, fg_batches, sim_days)
    flows_post = _compute_flows(
        ships_post, batches_post, post_stab_days
    )

    print(
        f"\n  Full sim vs Post-stabilization "
        f"(day {exclude_days}+)"
    )
    print(
        f"\n  {'Echelon':<14}  {'Full Delta%':>12}  "
        f"{'Post Delta%':>14}  {'Dir':>6}  {'Change':>8}"
    )
    print(
        f"  {'-'*14}  {'-'*12}  "
        f"{'-'*14}  {'-'*6}  {'-'*8}"
    )

    improvements = 0
    for ech in echelon_order:
        fa = flows_all.get(ech, {})
        fp = flows_post.get(ech, {})
        full_pct = fa.get("delta_pct", 0)
        post_pct = fp.get("delta_pct", 0)
        post_dir = (
            "ACCUM"
            if fp.get("daily_delta", 0) > 0
            else "DRAIN"
        )
        change = full_pct - post_pct

        if post_pct < full_pct:
            improvements += 1

        print(
            f"  {ech:<14}  {full_pct:>11.1f}%  "
            f"{post_pct:>13.1f}%  {post_dir:>6}  "
            f"{change:>+7.1f}%"
        )

    # Detailed post-stabilization flows
    print(
        f"\n  Post-stabilization daily flows "
        f"(day {exclude_days}+):"
    )
    print(
        f"  {'Echelon':<14}  {'Inflow/day':>14}  "
        f"{'Outflow/day':>14}  {'Delta/day':>12}  {'%':>6}"
    )
    print(
        f"  {'-'*14}  {'-'*14}  "
        f"{'-'*14}  {'-'*12}  {'-'*6}"
    )
    for ech in echelon_order:
        fp = flows_post.get(ech, {})
        print(
            f"  {ech:<14}  "
            f"{fp['daily_inflow']:>14,.0f}  "
            f"{fp['daily_outflow']:>14,.0f}  "
            f"{fp['daily_delta']:>+12,.0f}  "
            f"{fp['delta_pct']:>5.1f}%"
        )

    # Check if Customer DC issue resolves
    dc_full = flows_all.get("Customer DC", {}).get("delta_pct", 0)
    dc_post = flows_post.get("Customer DC", {}).get("delta_pct", 0)
    plant_post = flows_post.get("Plant", {}).get("delta_pct", 0)

    if dc_post < _FLOW_STABLE_PCT and dc_full > _FLOW_PERSISTENT_PCT:
        print(
            f"\n  Customer DC: {dc_full:.1f}% -> "
            f"{dc_post:.1f}% (RESOLVED)"
        )
    elif dc_post < dc_full:
        print(
            f"\n  Customer DC: {dc_full:.1f}% -> "
            f"{dc_post:.1f}% (improved, still elevated)"
        )

    if improvements >= len(echelon_order) - 1:
        verdict = "ARTIFACT"
        print(
            "\n  VERDICT: ARTIFACT — Most echelons improve "
            "when excluding priming"
        )
    elif plant_post > _FLOW_PERSISTENT_PCT:
        verdict = "REAL BUG"
        print(
            f"\n  VERDICT: REAL BUG — Plant accumulation "
            f"persists ({plant_post:.1f}%) post-stabilization"
        )
    else:
        verdict = "MIXED"
        print(
            "\n  VERDICT: MIXED — Some improvement, "
            "some persistent imbalance"
        )

    return {
        "flows_full": flows_all,
        "flows_post": flows_post,
        "improvements": improvements,
        "verdict": verdict,
    }


# ===================================================================
# Investigation 5: MRP Backpressure Check (H5)
# ===================================================================

def investigate_mrp_backpressure(data: DataBundle) -> dict:
    """Check if production slows when plant FG inventory is high.

    MRP DOS caps should throttle production when inventory position
    exceeds thresholds. If production doesn't slow when plant FG is
    high, caps may be too loose.
    """
    print(
        f"\n{'[H5] MRP Backpressure Check':=^{WIDTH}}"
    )

    batches = data.batches[
        data.batches["product_id"].apply(is_finished_good)
    ]
    inv = data.inv_by_echelon
    sim_days = data.sim_days

    if len(inv) == 0 or len(batches) == 0:
        print("  WARNING: Insufficient data")
        return {"verdict": "NO_DATA"}

    daily_prod = batches.groupby("day_produced")["quantity"].sum()

    plant_inv = inv[inv["echelon"] == "Plant"].sort_values("day")
    if len(plant_inv) == 0:
        print("  WARNING: No plant inventory data")
        return {"verdict": "NO_DATA"}

    plant_daily = plant_inv.groupby("day")["total"].sum()

    # Align on common days (inventory sampled weekly)
    common_days = sorted(
        set(plant_daily.index) & set(daily_prod.index)
    )
    if len(common_days) < _MIN_COMMON_DAYS:
        prod_series = daily_prod.reindex(
            range(
                int(daily_prod.index.min()),
                int(daily_prod.index.max()) + 1,
            ),
            fill_value=0,
        )
        inv_all_days = plant_daily.reindex(
            range(
                int(plant_daily.index.min()),
                int(plant_daily.index.max()) + 1,
            ),
        ).ffill()
        common_days = sorted(
            set(prod_series.index) & set(inv_all_days.index)
        )
        if len(common_days) < _MIN_COMMON_DAYS:
            print("  WARNING: Too few common days")
            return {"verdict": "NO_DATA"}

        prod_aligned = prod_series.loc[common_days]
        inv_aligned = inv_all_days.loc[common_days]
    else:
        prod_aligned = daily_prod.loc[common_days]
        inv_aligned = plant_daily.loc[common_days]

    # 7-day rolling average to smooth batch cycles
    prod_smooth = prod_aligned.rolling(
        _ROLLING_WINDOW, min_periods=1
    ).mean()

    # Split into high vs low FG halves
    median_fg = float(inv_aligned.median())
    high_fg_days = inv_aligned[inv_aligned >= median_fg].index
    low_fg_days = inv_aligned[inv_aligned < median_fg].index

    prod_high = float(
        prod_smooth.loc[
            prod_smooth.index.isin(high_fg_days)
        ].mean()
    )
    prod_low = float(
        prod_smooth.loc[
            prod_smooth.index.isin(low_fg_days)
        ].mean()
    )
    ratio = prod_high / prod_low if prod_low > 0 else np.nan

    print("\n  Plant FG inventory statistics:")
    print(f"    Median:    {median_fg:>14,.0f}")
    print(f"    Min:       {float(inv_aligned.min()):>14,.0f}")
    print(f"    Max:       {float(inv_aligned.max()):>14,.0f}")
    p25 = float(inv_aligned.quantile(0.25))
    p75 = float(inv_aligned.quantile(0.75))
    print(f"    P25:       {p25:>14,.0f}")
    print(f"    P75:       {p75:>14,.0f}")

    print("\n  Production (7d avg) vs Plant FG level:")
    print(
        f"    When FG < median ({median_fg:,.0f}): "
        f" {prod_low:>12,.0f}/day"
    )
    print(
        f"    When FG >= median:          "
        f" {prod_high:>12,.0f}/day"
    )
    ratio_s = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"
    print(f"    Ratio (high/low):            {ratio_s}")

    # Quartile analysis
    q1_days = inv_aligned[inv_aligned <= p25].index
    q4_days = inv_aligned[inv_aligned >= p75].index

    prod_q1 = (
        float(
            prod_smooth.loc[
                prod_smooth.index.isin(q1_days)
            ].mean()
        )
        if len(q1_days) > 0
        else 0
    )
    prod_q4 = (
        float(
            prod_smooth.loc[
                prod_smooth.index.isin(q4_days)
            ].mean()
        )
        if len(q4_days) > 0
        else 0
    )
    q_ratio = prod_q4 / prod_q1 if prod_q1 > 0 else np.nan

    print("\n  Quartile analysis:")
    print(
        f"    Q1 (lowest FG, <={p25:,.0f}): "
        f" {prod_q1:>12,.0f}/day  ({len(q1_days)} days)"
    )
    print(
        f"    Q4 (highest FG, >={p75:,.0f}):"
        f" {prod_q4:>12,.0f}/day  ({len(q4_days)} days)"
    )
    qr_s = f"{q_ratio:.3f}" if not np.isnan(q_ratio) else "N/A"
    print(f"    Q4/Q1 ratio:                 {qr_s}")

    # Trend snapshots
    print("\n  Production vs Plant FG (60d snapshots):")
    print(
        f"  {'Day':>6}  {'Prod/day (7d avg)':>18}  "
        f"{'Plant FG':>14}"
    )
    print(f"  {'-'*6}  {'-'*18}  {'-'*14}")

    for snap_day in range(60, sim_days + 1, 60):
        if snap_day in prod_smooth.index:
            pv = float(prod_smooth.loc[snap_day])
        else:
            closest = min(
                prod_smooth.index,
                key=lambda d: abs(d - snap_day),
            )
            pv = float(prod_smooth.loc[closest])

        if snap_day in inv_aligned.index:
            iv = float(inv_aligned.loc[snap_day])
        else:
            closest = min(
                inv_aligned.index,
                key=lambda d: abs(d - snap_day),
            )
            iv = float(inv_aligned.loc[closest])

        print(f"  {snap_day:>6}  {pv:>18,.0f}  {iv:>14,.0f}")

    # Correlation
    corr = np.nan
    if len(common_days) > _CORRELATION_MIN_DAYS:
        corr = float(
            np.corrcoef(
                inv_aligned.values.astype(float),
                prod_smooth.values.astype(float),
            )[0, 1]
        )
        print(
            f"\n  Correlation (Plant FG vs Production): "
            f"{corr:+.3f}"
        )
        print(
            "  (Negative = backpressure working, "
            "Positive = no backpressure)"
        )

    # Verdict
    if not np.isnan(ratio) and ratio < _BACKPRESSURE_STRONG:
        verdict = "WORKING"
        drop_pct = (1 - ratio) * 100
        print(
            f"\n  VERDICT: WORKING — Production drops "
            f"{drop_pct:.0f}% when FG is high"
        )
    elif not np.isnan(ratio) and ratio < _BACKPRESSURE_WEAK:
        verdict = "WEAK"
        drop_pct = (1 - ratio) * 100
        print(
            f"\n  VERDICT: WEAK — Production drops only "
            f"{drop_pct:.0f}% when FG high"
        )
    else:
        verdict = "TOO LOOSE"
        print(
            "\n  VERDICT: TOO LOOSE — Production "
            "unchanged regardless of FG level"
        )
        print(
            f"  MRP DOS caps ({_MRP_CAP_A}/{_MRP_CAP_B}/"
            f"{_MRP_CAP_C}) may be too generous."
        )

    return {
        "median_fg": median_fg,
        "prod_when_high": prod_high,
        "prod_when_low": prod_low,
        "high_low_ratio": ratio,
        "q4_q1_ratio": q_ratio,
        "correlation": corr if not np.isnan(corr) else None,
        "verdict": verdict,
    }


# ===================================================================
# Summary
# ===================================================================

def print_summary(results: dict[str, dict]) -> None:
    """Print hypothesis summary table."""
    print(f"\n{'=' * WIDTH}")
    print(
        "  v0.60.0 Stability Diagnostic Summary".center(WIDTH)
    )
    print(f"{'=' * WIDTH}")

    hypotheses = [
        ("H1", "DOS denominator", results.get("h1", {})),
        ("H2", "Seasonal gap", results.get("h2", {})),
        ("H3", "Need saturation", results.get("h3", {})),
        ("H4", "Priming artifacts", results.get("h4", {})),
        ("H5", "MRP backpressure", results.get("h5", {})),
    ]

    for code, label, res in hypotheses:
        verdict = res.get("verdict", "NOT RUN")
        print(f"  {code} ({label + '):':<22}  {verdict}")

    # Interpretation
    print(f"\n{'INTERPRETATION':=^{WIDTH}}")

    h1_v = results.get("h1", {}).get("verdict", "")
    h2_v = results.get("h2", {}).get("verdict", "")
    h3_v = results.get("h3", {}).get("verdict", "")
    h4_v = results.get("h4", {}).get("verdict", "")
    h5_v = results.get("h5", {}).get("verdict", "")

    verdicts = [h1_v, h2_v, h3_v, h4_v]
    artifact_count = sum(1 for v in verdicts if v == "ARTIFACT")
    real_bug_count = sum(1 for v in verdicts if v == "REAL BUG")

    if artifact_count >= _ARTIFACT_MAJORITY:
        print(
            "  CONCLUSION: Stability readings are "
            "predominantly MEASUREMENT ARTIFACTS"
        )
        print(
            "  The simulation is likely operating "
            "correctly. Diagnostic measurements"
        )
        print(
            "  overstate the problems due to denominator "
            "mismatch, priming period"
        )
        print("  inclusion, and target saturation effects.")
    elif real_bug_count >= _BUG_THRESHOLD:
        print(
            "  CONCLUSION: Multiple REAL BUGS detected."
        )
        print(
            "  Priority fixes needed — see "
            "individual hypothesis details."
        )
    else:
        print(
            "  CONCLUSION: Mixed results — "
            "some artifacts, some potential issues."
        )
        print(
            "  Review each hypothesis for "
            "specific action items."
        )

    if h1_v == "ARTIFACT":
        h1_data = results.get("h1", {})
        corr_dos = h1_data.get("rdc_dos_corrected", 0)
        diag_dos = h1_data.get("rdc_dos_diagnostic", 0)
        share = h1_data.get("rdc_demand_share", 0)
        print(
            f"\n  H1 detail: RDC DOS {diag_dos:.1f} "
            f"(diagnostic) -> {corr_dos:.1f} (corrected)"
        )
        print(
            f"    RDCs serve {share:.0f}% of "
            f"network demand, not 100%"
        )

    if h3_v == "ARTIFACT":
        h3_data = results.get("h3", {})
        n_sat = h3_data.get("n_saturated", 0)
        n_tot = h3_data.get("n_targets", 0)
        print(
            f"\n  H3 detail: {n_sat}/{n_tot} "
            f"targets saturated"
        )
        print(
            "    Plant FG accumulation is expected "
            "when downstream is stocked"
        )

    if h5_v == "TOO LOOSE":
        print(
            "\n  H5 action: Consider tightening MRP DOS caps"
            f" (currently A={_MRP_CAP_A}, B={_MRP_CAP_B},"
            f" C={_MRP_CAP_C})"
        )
        print(
            "    to reduce unnecessary plant FG buildup"
        )

    if h2_v in ("REAL BUG", "MINOR"):
        h2_data = results.get("h2", {})
        swing = h2_data.get("swing_pct", 0)
        print(
            f"\n  H2 detail: {swing:.1f}% seasonal swing "
            f"in POS vs static deployment"
        )
        if h2_v == "REAL BUG":
            print(
                "    FIX: Make _target_expected_demand "
                "dynamic (update from rolling POS)"
            )

    print()


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    """Run all five stability investigations."""
    parser = argparse.ArgumentParser(
        description="v0.60.0 Stability Diagnostic"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/output"),
        help="Simulation output directory",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    print(f"{'=' * WIDTH}")
    print(
        "  v0.60.0 STABILITY DIAGNOSTIC".center(WIDTH)
    )
    print(
        "  Targeted Investigation".center(WIDTH)
    )
    print(f"{'=' * WIDTH}")
    print(f"\n  Data directory: {data_dir}\n")

    # Load data
    data = load_all_data(data_dir)

    # Run investigations
    results: dict[str, dict] = {}

    print("\n" + "=" * WIDTH)
    print("  INVESTIGATION 1: Per-Target DOS".center(WIDTH))
    print("=" * WIDTH)
    results["h1"] = investigate_per_target_dos(data)

    print("\n" + "=" * WIDTH)
    print(
        "  INVESTIGATION 2: Seasonal Demand Gap".center(WIDTH)
    )
    print("=" * WIDTH)
    results["h2"] = investigate_seasonal_gap(data)

    print("\n" + "=" * WIDTH)
    print(
        "  INVESTIGATION 3: Need Saturation".center(WIDTH)
    )
    print("=" * WIDTH)
    results["h3"] = investigate_need_saturation(data)

    print("\n" + "=" * WIDTH)
    print(
        "  INVESTIGATION 4: Flow excl. Priming".center(WIDTH)
    )
    print("=" * WIDTH)
    results["h4"] = investigate_flow_excl_priming(data)

    print("\n" + "=" * WIDTH)
    print(
        "  INVESTIGATION 5: MRP Backpressure".center(WIDTH)
    )
    print("=" * WIDTH)
    results["h5"] = investigate_mrp_backpressure(data)

    # Summary
    print_summary(results)

    print(f"{'=' * WIDTH}")
    print(
        "  END OF v0.60.0 STABILITY DIAGNOSTIC".center(WIDTH)
    )
    print(f"{'=' * WIDTH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
