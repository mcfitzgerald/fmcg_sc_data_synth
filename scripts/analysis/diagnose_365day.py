#!/usr/bin/env python3
"""
365-Day Drift Diagnostic: Root Cause Analysis + Data Validation.

Proves the MRP demand floor bug causes monotonic inventory drift over a
365-day simulation.  Six analyses run from the collected parquet data:

  1. Production vs Demand time series  — shows gap widening during troughs
  2. Inventory by echelon over time    — shows WHERE excess accumulates
  3. Cumulative excess (SLOB proxy)    — shows monotonic inventory growth
  4. MRP signal floor proof            — shows PO qty never drops below annual avg
  5. Seasonal correlation              — production flat while demand is seasonal
  6. ABC class breakdown               — which class drives SLOB

Memory safety: The 6.2 GB inventory.parquet (652M rows, 730 RGs) is streamed
via PyArrow row groups.  Only lightweight per-day aggregates are accumulated
(~365 days x 7 echelons x 3 ABC classes ≈ 7.7 K entries).

Usage:
    poetry run python scripts/analysis/diagnose_365day.py \
        --data-dir data/output --window 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Thresholds for reporting
_LOW_CV_THRESH = 0.1  # PO qty CV below which floor is "proven"
_FLATTENED_CV_RATIO = 0.5  # prod_cv/demand_cv below which → "flattened"
_DAMPENED_CV_RATIO = 0.8  # prod_cv/demand_cv below which → "dampened"

# v0.46.0: All demand-generating endpoint node prefixes (7-channel model)
_DEMAND_PREFIXES = ("STORE-", "CLUB-", "ECOM-FC-", "DTC-FC-")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_node(node_id: str) -> str:
    """Classify node ID into echelon tier."""
    if node_id.startswith("PLANT-"):
        return "Plant"
    if node_id.startswith("RDC-"):
        return "RDC"
    if node_id.startswith(("RET-DC-", "DIST-DC-", "ECOM-FC-", "DTC-FC-",
                           "PHARM-DC-", "CLUB-DC-")):
        return "Customer DC"
    if node_id.startswith("STORE-"):
        return "Store"
    if node_id.startswith("CLUB-"):
        return "Club"
    if node_id.startswith("SUP-"):
        return "Supplier"
    return "Other"


def is_finished_good(product_id: str) -> bool:
    """True for SKU- products (not ingredients/packaging)."""
    return product_id.startswith("SKU-")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_products(data_dir: Path) -> pd.DataFrame:
    """Load product catalog from static world."""
    df = pd.read_csv(
        data_dir / "static_world" / "products.csv",
        usecols=["id", "name", "category"],
    )
    df["category"] = df["category"].str.replace("ProductCategory.", "", regex=False)
    return df


def load_shipments(data_dir: Path) -> pd.DataFrame:
    """Load shipments parquet (selective columns)."""
    print("  Loading shipments.parquet...")
    df = pd.read_parquet(
        data_dir / "shipments.parquet",
        columns=[
            "shipment_id", "creation_day", "arrival_day",
            "source_id", "target_id", "product_id", "quantity",
        ],
    )
    print(f"    {len(df):,} rows")
    return df


def load_batches(data_dir: Path) -> pd.DataFrame:
    """Load batches parquet."""
    print("  Loading batches.parquet...")
    df = pd.read_parquet(data_dir / "batches.parquet")
    print(f"    {len(df):,} rows")
    return df


def load_production_orders(data_dir: Path) -> pd.DataFrame:
    """Load production orders parquet."""
    print("  Loading production_orders.parquet...")
    df = pd.read_parquet(data_dir / "production_orders.parquet")
    print(f"    {len(df):,} rows")
    return df


def load_forecasts(data_dir: Path) -> pd.DataFrame:
    """Load forecasts parquet."""
    print("  Loading forecasts.parquet...")
    df = pd.read_parquet(data_dir / "forecasts.parquet")
    print(f"    {len(df):,} rows")
    return df


def load_metrics(data_dir: Path) -> dict:
    """Load simulation metrics JSON."""
    with open(data_dir / "metrics.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ABC Classification (per-category Pareto, matches MRP logic)
# ---------------------------------------------------------------------------

def classify_abc(
    shipments: pd.DataFrame,
    products: pd.DataFrame,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> pd.DataFrame:
    """Per-category Pareto ABC classification from shipment volumes."""
    fg_products = set(products[products["category"] != "INGREDIENT"]["id"])
    fg_ships = shipments[shipments["product_id"].isin(fg_products)]

    product_volume = fg_ships.groupby("product_id")["quantity"].sum().reset_index()
    product_volume.columns = ["product_id", "total_volume"]
    product_volume = product_volume.merge(
        products[["id", "category"]].rename(columns={"id": "product_id"}),
        on="product_id",
        how="left",
    )

    abc_labels = []
    for category in product_volume["category"].dropna().unique():
        cat_df = product_volume[product_volume["category"] == category].copy()
        cat_df = cat_df.sort_values("total_volume", ascending=False)
        cat_total = cat_df["total_volume"].sum()

        if cat_total == 0:
            cat_df["abc_class"] = "C"
        else:
            cum = cat_df["total_volume"].cumsum()
            cat_df["abc_class"] = "C"
            cat_df.loc[cum <= cat_total * a_threshold, "abc_class"] = "A"
            cat_df.loc[
                (cum > cat_total * a_threshold) & (cum <= cat_total * b_threshold),
                "abc_class",
            ] = "B"
            cat_df.iloc[0, cat_df.columns.get_loc("abc_class")] = "A"

        abc_labels.append(cat_df[["product_id", "abc_class"]])

    abc_df = pd.concat(abc_labels, ignore_index=True)
    counts = abc_df["abc_class"].value_counts()
    a, b, c = counts.get("A", 0), counts.get("B", 0), counts.get("C", 0)
    print(f"  ABC: A={a}, B={b}, C={c} SKUs")
    return abc_df


# ---------------------------------------------------------------------------
# Analysis 1: Production vs Demand Time Series
# ---------------------------------------------------------------------------

def analysis_1_production_vs_demand(
    batches: pd.DataFrame,
    shipments: pd.DataFrame,
    abc_map: dict[str, str],
    window: int,
) -> dict:
    """
    Daily production (batches) vs daily demand (store-bound shipments).

    Returns rolling-window averages for smooth comparison.
    """
    # Demand proxy: all demand-endpoint shipments by creation_day
    demand_ships = shipments[
        shipments["target_id"].str.startswith(_DEMAND_PREFIXES)
        & shipments["product_id"].apply(is_finished_good)
    ]
    demand_daily = demand_ships.groupby("creation_day")["quantity"].sum()

    # Production: finished-good batches by day_produced
    fg_batches = batches[batches["product_id"].apply(is_finished_good)]
    prod_daily = fg_batches.groupby("day_produced")["quantity"].sum()

    # Align indices
    all_days = sorted(set(demand_daily.index) | set(prod_daily.index))
    demand_series = demand_daily.reindex(all_days, fill_value=0)
    prod_series = prod_daily.reindex(all_days, fill_value=0)

    # Rolling averages
    demand_rolling = demand_series.rolling(window, min_periods=1).mean()
    prod_rolling = prod_series.rolling(window, min_periods=1).mean()

    # Compute gap ratio (production / demand)
    gap_ratio = prod_rolling / demand_rolling.replace(0, np.nan)

    # 30-day snapshot table
    snapshot_days = list(range(30, max(all_days) + 1, 30))
    snapshots = []
    for d in snapshot_days:
        if d in demand_rolling.index and d in prod_rolling.index:
            dem = demand_rolling.loc[d]
            pro = prod_rolling.loc[d]
            ratio = pro / dem if dem > 0 else np.nan
            snapshots.append({
                "day": d,
                "demand_avg": dem,
                "production_avg": pro,
                "ratio": ratio,
                "gap_pct": (ratio - 1) * 100 if not np.isnan(ratio) else np.nan,
            })

    # By ABC class
    abc_breakdown = {}
    for cls in ("A", "B", "C"):
        cls_prods = {p for p, c in abc_map.items() if c == cls}
        cls_ship = demand_ships[demand_ships["product_id"].isin(cls_prods)]
        cls_dem = cls_ship.groupby("creation_day")["quantity"].sum()
        cls_bat = fg_batches[fg_batches["product_id"].isin(cls_prods)]
        cls_prod = cls_bat.groupby("day_produced")["quantity"].sum()
        abc_breakdown[cls] = {
            "total_demand": cls_dem.sum(),
            "total_production": cls_prod.sum(),
            "ratio": cls_prod.sum() / cls_dem.sum() if cls_dem.sum() > 0 else np.nan,
        }

    return {
        "snapshots": snapshots,
        "demand_rolling": demand_rolling,
        "prod_rolling": prod_rolling,
        "gap_ratio": gap_ratio,
        "abc_breakdown": abc_breakdown,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Inventory by Echelon Over Time (streamed)
# ---------------------------------------------------------------------------

def stream_inventory_by_echelon_day(
    data_dir: Path,
    abc_map: dict[str, str],
) -> pd.DataFrame:
    """
    Memory-safe streaming of inventory.parquet.

    Returns a DataFrame indexed by (day, echelon) with total FG inventory.
    Also returns per-ABC-class columns.
    """
    pf = pq.ParquetFile(data_dir / "inventory.parquet")
    n_rg = pf.metadata.num_row_groups
    n_rows = pf.metadata.num_rows
    print(f"  Streaming inventory.parquet ({n_rows:,} rows, {n_rg} RGs)...")

    columns = ["day", "node_id", "product_id", "actual_inventory"]

    # Accumulate: (day, echelon) -> {total, A, B, C}
    accum: dict[tuple[int, str], dict[str, float]] = {}
    rg_loaded = 0

    for rg_idx in range(n_rg):
        chunk = pf.read_row_group(rg_idx, columns=columns).to_pandas()
        rg_loaded += 1

        # Convert dictionary-encoded categoricals
        for col in ("node_id", "product_id"):
            if hasattr(chunk[col], "cat"):
                chunk[col] = chunk[col].astype(str)

        # Keep only finished goods
        fg = chunk[chunk["product_id"].apply(is_finished_good)]
        if len(fg) == 0:
            continue

        fg = fg.copy()
        fg["echelon"] = fg["node_id"].map(classify_node)
        fg["abc"] = fg["product_id"].map(abc_map)

        # Aggregate by (day, echelon)
        grp = fg.groupby(["day", "echelon"])
        totals = grp["actual_inventory"].sum()

        for (day, ech), inv_total in totals.items():
            key = (day, ech)
            if key not in accum:
                accum[key] = {"total": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
            accum[key]["total"] += inv_total

        # Per-ABC within each (day, echelon)
        for cls in ("A", "B", "C"):
            cls_fg = fg[fg["abc"] == cls]
            if len(cls_fg) == 0:
                continue
            cls_totals = cls_fg.groupby(["day", "echelon"])["actual_inventory"].sum()
            for (day, ech), inv_total in cls_totals.items():
                key = (day, ech)
                if key not in accum:
                    accum[key] = {"total": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
                accum[key][cls] += inv_total

        if rg_loaded % 100 == 0:
            print(f"    ... {rg_loaded}/{n_rg} row groups processed")

    print(f"    Done: {rg_loaded} row groups, {len(accum)} (day, echelon) entries")

    # Build DataFrame
    rows = []
    for (day, ech), vals in accum.items():
        rows.append({"day": day, "echelon": ech, **vals})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["day", "echelon", "total", "A", "B", "C"])

    return df.sort_values(["day", "echelon"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 3: Cumulative Excess (SLOB Proxy)
# ---------------------------------------------------------------------------

def analysis_3_cumulative_excess(
    batches: pd.DataFrame,
    shipments: pd.DataFrame,
) -> dict:
    """
    Cumulative production minus cumulative demand over time.

    A monotonically growing curve proves inventory is never drawn down.
    """
    demand_ships = shipments[
        shipments["target_id"].str.startswith(_DEMAND_PREFIXES)
        & shipments["product_id"].apply(is_finished_good)
    ]
    demand_daily = demand_ships.groupby("creation_day")["quantity"].sum()

    fg_batches = batches[batches["product_id"].apply(is_finished_good)]
    prod_daily = fg_batches.groupby("day_produced")["quantity"].sum()

    all_days = sorted(set(demand_daily.index) | set(prod_daily.index))
    demand_cum = demand_daily.reindex(all_days, fill_value=0).cumsum()
    prod_cum = prod_daily.reindex(all_days, fill_value=0).cumsum()

    excess_cum = prod_cum - demand_cum

    # Check monotonicity (is excess always increasing?)
    diffs = excess_cum.diff().dropna()
    monotonic_pct = (diffs >= 0).mean() * 100

    # Snapshot every 30 days
    snapshots = []
    for d in range(30, max(all_days) + 1, 30):
        if d in excess_cum.index:
            snapshots.append({
                "day": d,
                "cum_production": prod_cum.loc[d],
                "cum_demand": demand_cum.loc[d],
                "cum_excess": excess_cum.loc[d],
                "excess_pct": (
                    excess_cum.loc[d] / demand_cum.loc[d] * 100
                    if demand_cum.loc[d] > 0
                    else 0
                ),
            })

    return {
        "excess_cum": excess_cum,
        "monotonic_pct": monotonic_pct,
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Analysis 4: MRP Signal Floor Proof
# ---------------------------------------------------------------------------

def analysis_4_mrp_signal_floor(
    production_orders: pd.DataFrame,
    forecasts: pd.DataFrame,
    abc_map: dict[str, str],
) -> dict:
    """
    Compare PO quantities with expected daily demand (from forecasts).

    If demand floor is active, PO quantities should never drop below
    the annual-average-based batch size, even during seasonal troughs.
    """
    # Get average forecast per product (proxy for expected_daily_demand)
    avg_forecast = forecasts.groupby("product_id")["forecast_quantity"].mean()

    # PO quantities per product per day
    fg_pos = production_orders[
        production_orders["product_id"].apply(is_finished_good)
    ].copy()
    fg_pos["abc"] = fg_pos["product_id"].map(abc_map)

    # For each product, check if PO qty ever drops below expected
    product_stats = []
    for product_id in fg_pos["product_id"].unique():
        product_pos = fg_pos[fg_pos["product_id"] == product_id]
        avg_po_qty = product_pos["quantity"].mean()
        min_po_qty = product_pos["quantity"].min()
        max_po_qty = product_pos["quantity"].max()
        n_pos = len(product_pos)

        expected = avg_forecast.get(product_id, np.nan)
        abc = abc_map.get(product_id, "?")

        # Ratio of min PO to expected daily demand
        # If floor is active, min_po / expected should be >= abc_horizon
        min_ratio = min_po_qty / expected if expected and expected > 0 else np.nan

        product_stats.append({
            "product_id": product_id,
            "abc": abc,
            "n_pos": n_pos,
            "avg_po_qty": avg_po_qty,
            "min_po_qty": min_po_qty,
            "max_po_qty": max_po_qty,
            "expected_daily": expected,
            "min_po_to_expected": min_ratio,
            "qty_cv": (
                product_pos["quantity"].std() / avg_po_qty
                if avg_po_qty > 0
                else np.nan
            ),
        })

    stats_df = pd.DataFrame(product_stats)

    # Key metric: how many products have min_PO < expected * horizon?
    # With demand floor, this should be near zero.
    # Without floor, trough POs would be ~0.88 * expected * horizon

    # Summary by ABC
    abc_summary = {}
    for cls in ("A", "B", "C"):
        cls_stats = stats_df[stats_df["abc"] == cls]
        if len(cls_stats) == 0:
            continue
        abc_summary[cls] = {
            "n_products": len(cls_stats),
            "mean_qty_cv": cls_stats["qty_cv"].mean(),
            "mean_min_ratio": cls_stats["min_po_to_expected"].mean(),
            "pct_low_cv": (cls_stats["qty_cv"] < _LOW_CV_THRESH).mean() * 100,
        }

    return {
        "stats_df": stats_df,
        "abc_summary": abc_summary,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Seasonal Correlation
# ---------------------------------------------------------------------------

def analysis_5_seasonal_correlation(
    batches: pd.DataFrame,
    shipments: pd.DataFrame,
    window: int,
    amplitude: float = 0.12,
) -> dict:
    """
    Compare production and demand rhythms with the configured sine-wave
    seasonality.  If MRP floor is active, production should show much less
    seasonal variation than demand.
    """
    demand_ships = shipments[
        shipments["target_id"].str.startswith(_DEMAND_PREFIXES)
        & shipments["product_id"].apply(is_finished_good)
    ]
    demand_daily = demand_ships.groupby("creation_day")["quantity"].sum()

    fg_batches = batches[batches["product_id"].apply(is_finished_good)]
    prod_daily = fg_batches.groupby("day_produced")["quantity"].sum()

    all_days = sorted(set(demand_daily.index) | set(prod_daily.index))
    demand_s = (
        demand_daily.reindex(all_days, fill_value=0)
        .rolling(window, min_periods=1)
        .mean()
    )
    prod_s = (
        prod_daily.reindex(all_days, fill_value=0)
        .rolling(window, min_periods=1)
        .mean()
    )

    # Normalize to mean=1
    demand_norm = demand_s / demand_s.mean() if demand_s.mean() > 0 else demand_s
    prod_norm = prod_s / prod_s.mean() if prod_s.mean() > 0 else prod_s

    # Expected sine wave: 1 + amplitude * sin(2π * (day - peak_offset) / 365)
    # peak_offset estimated from demand data
    days_arr = np.array(all_days, dtype=float)
    sine_wave = 1.0 + amplitude * np.sin(2 * np.pi * days_arr / 365)

    # Coefficient of variation (how much does each signal vary?)
    demand_cv = demand_norm.std() / demand_norm.mean() if demand_norm.mean() > 0 else 0
    prod_cv = prod_norm.std() / prod_norm.mean() if prod_norm.mean() > 0 else 0
    sine_cv = np.std(sine_wave) / np.mean(sine_wave)

    # Correlation with sine wave
    _min_corr_pts = 3
    demand_corr = (
        np.corrcoef(demand_norm.values, sine_wave)[0, 1]
        if len(demand_norm) >= _min_corr_pts
        else np.nan
    )
    prod_corr = (
        np.corrcoef(prod_norm.values, sine_wave)[0, 1]
        if len(prod_norm) >= _min_corr_pts
        else np.nan
    )

    # Seasonal peak/trough ratios
    half_year = len(all_days) // 2
    if half_year > 0:
        first_half_demand = demand_s.iloc[:half_year].mean()
        second_half_demand = demand_s.iloc[half_year:].mean()
        first_half_prod = prod_s.iloc[:half_year].mean()
        second_half_prod = prod_s.iloc[half_year:].mean()
    else:
        first_half_demand = second_half_demand = 0
        first_half_prod = second_half_prod = 0

    return {
        "demand_cv": demand_cv,
        "prod_cv": prod_cv,
        "sine_cv": sine_cv,
        "demand_sine_corr": demand_corr,
        "prod_sine_corr": prod_corr,
        "demand_half_ratio": (
            first_half_demand / second_half_demand
            if second_half_demand > 0
            else np.nan
        ),
        "prod_half_ratio": (
            first_half_prod / second_half_prod
            if second_half_prod > 0
            else np.nan
        ),
        "cv_ratio": prod_cv / demand_cv if demand_cv > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Analysis 6: ABC Class Breakdown
# ---------------------------------------------------------------------------

def analysis_6_abc_breakdown(
    batches: pd.DataFrame,
    shipments: pd.DataFrame,
    inv_echelon_day: pd.DataFrame,
    abc_map: dict[str, str],
    metrics: dict,
) -> dict:
    """Show which ABC class drives SLOB accumulation."""
    demand_ships = shipments[
        shipments["target_id"].str.startswith(_DEMAND_PREFIXES)
        & shipments["product_id"].apply(is_finished_good)
    ]
    fg_batches = batches[batches["product_id"].apply(is_finished_good)]

    results = {}
    for cls in ("A", "B", "C"):
        cls_prods = {p for p, c in abc_map.items() if c == cls}

        cls_ship = demand_ships[demand_ships["product_id"].isin(cls_prods)]
        cls_demand = cls_ship["quantity"].sum()
        cls_bat = fg_batches[fg_batches["product_id"].isin(cls_prods)]
        cls_prod = cls_bat["quantity"].sum()

        # Inventory from streamed data — latest day total by echelon
        max_day = inv_echelon_day["day"].max() if len(inv_echelon_day) > 0 else 0
        latest_inv = inv_echelon_day[inv_echelon_day["day"] == max_day]
        cls_inv_total = latest_inv[cls].sum() if cls in latest_inv.columns else 0

        # Inventory trend: first available day vs last
        min_day = inv_echelon_day["day"].min() if len(inv_echelon_day) > 0 else 0
        first_inv = inv_echelon_day[inv_echelon_day["day"] == min_day]
        cls_inv_first = first_inv[cls].sum() if cls in first_inv.columns else 0

        results[cls] = {
            "total_demand": cls_demand,
            "total_production": cls_prod,
            "excess": cls_prod - cls_demand,
            "excess_pct": (
                (cls_prod - cls_demand) / cls_demand * 100
                if cls_demand > 0
                else 0
            ),
            "inv_first_day": cls_inv_first,
            "inv_last_day": cls_inv_total,
            "inv_growth": cls_inv_total - cls_inv_first,
            "inv_growth_pct": (
                (cls_inv_total - cls_inv_first) / cls_inv_first * 100
                if cls_inv_first > 0
                else 0
            ),
        }

    # Echelon-level inventory at end
    echelon_inv = {}
    if len(inv_echelon_day) > 0 and max_day > 0:
        latest = inv_echelon_day[inv_echelon_day["day"] == max_day]
        for _, row in latest.iterrows():
            echelon_inv[row["echelon"]] = {
                "total": row["total"],
                "A": row.get("A", 0),
                "B": row.get("B", 0),
                "C": row.get("C", 0),
            }

    return {
        "by_class": results,
        "echelon_inv": echelon_inv,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(  # noqa: PLR0915
    a1: dict,
    inv_echelon: pd.DataFrame,
    a3: dict,
    a4: dict,
    a5: dict,
    a6: dict,
    window: int,
    metrics: dict,
) -> None:
    """Print structured diagnostic report."""
    w = 78
    print()
    print("=" * w)
    print("  365-DAY DRIFT DIAGNOSTIC REPORT".center(w))
    print("  Root Cause: MRP Demand Floor Bug".center(w))
    print("=" * w)

    # --- Headline Metrics ---
    print(f"\n{'HEADLINE METRICS':=^{w}}")
    slob_info = metrics.get("slob", {})
    svc = metrics.get("service_level_by_abc", {})
    turns = metrics.get("inventory_turns", {})
    print(f"  SLOB:              {slob_info.get('mean', 0):.1%}  (target: <30%)")
    print(f"  Inventory Turns:   {turns.get('mean', 0):.1f}    (target: 6-14)")
    print(f"  A-item Fill:       {svc.get('A', 0):.1%}  (target: >85%)")
    print(f"  B-item Fill:       {svc.get('B', 0):.1%}")
    print(f"  C-item Fill:       {svc.get('C', 0):.1%}")
    store_svc = metrics.get("store_service_level", {}).get("mean", 0)
    print(f"  Store Service:     {store_svc:.1%}")

    # --- Analysis 1: Production vs Demand ---
    print(f"\n{'ANALYSIS 1: PRODUCTION vs DEMAND TIME SERIES':=^{w}}")
    print(f"  ({window}-day rolling average)\n")
    hdr = f"  {'Day':>6}  {'Demand':>14}  {'Production':>14}"
    print(hdr + f"  {'Ratio':>8}  {'Gap%':>8}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*8}")
    for s in a1["snapshots"]:
        gp = s.get("gap_pct", np.nan)
        gap_str = f"{gp:+.1f}%" if not np.isnan(gp) else "N/A"
        rt = s.get("ratio", np.nan)
        ratio_str = f"{rt:.3f}" if not np.isnan(rt) else "N/A"
        print(
            f"  {s['day']:>6}  {s['demand_avg']:>14,.0f}  {s['production_avg']:>14,.0f}"
            f"  {ratio_str:>8}  {gap_str:>8}"
        )

    print("\n  Production vs Demand by ABC class (full sim):")
    h2 = f"  {'Class':>6}  {'Demand':>14}  {'Production':>14}"
    print(h2 + f"  {'Ratio':>8}  {'Excess%':>8}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*8}")
    for cls in ("A", "B", "C"):
        b = a1["abc_breakdown"].get(cls, {})
        dem = b.get("total_demand", 0)
        pro = b.get("total_production", 0)
        ratio = b.get("ratio", np.nan)
        excess = (ratio - 1) * 100 if not np.isnan(ratio) else np.nan
        print(
            f"  {cls:>6}  {dem:>14,.0f}  {pro:>14,.0f}"
            f"  {(f'{ratio:.3f}' if not np.isnan(ratio) else 'N/A'):>8}"
            f"  {(f'{excess:+.1f}%' if not np.isnan(excess) else 'N/A'):>8}"
        )

    # --- Analysis 2: Inventory by Echelon ---
    print(f"\n{'ANALYSIS 2: INVENTORY BY ECHELON OVER TIME':=^{w}}")
    if len(inv_echelon) > 0:
        max_day = inv_echelon["day"].max()
        min_day = inv_echelon["day"].min()

        # Snapshot every 60 days
        print("\n  Total FG inventory by echelon (selected days):\n")
        echelons = ["Plant", "RDC", "Customer DC", "Store", "Club"]
        header = f"  {'Day':>6}"
        for e in echelons:
            header += f"  {e:>14}"
        header += f"  {'TOTAL':>14}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*14}" * (len(echelons) + 1))

        for d in range(min_day, max_day + 1, 60):
            day_data = inv_echelon[inv_echelon["day"] == d]
            if len(day_data) == 0:
                # Find closest day
                diffs = (inv_echelon["day"] - d).abs()
                closest_idx = diffs.argsort().iloc[0]
                closest_day = int(inv_echelon.iloc[closest_idx]["day"])
                day_data = inv_echelon[inv_echelon["day"] == closest_day]
                d = closest_day  # noqa: PLW2901
            row = f"  {d:>6}"
            day_total = 0
            for e in echelons:
                e_data = day_data[day_data["echelon"] == e]
                val = e_data["total"].sum() if len(e_data) > 0 else 0
                day_total += val
                row += f"  {val:>14,.0f}"
            row += f"  {day_total:>14,.0f}"
            print(row)

        # Growth rates
        first_data = inv_echelon[inv_echelon["day"] == min_day]
        last_data = inv_echelon[inv_echelon["day"] == max_day]
        print(f"\n  Inventory growth (day {min_day} → day {max_day}):")
        h2g = f"  {'Echelon':<14}  {'Start':>14}  {'End':>14}"
        print(h2g + f"  {'Change':>10}  {'Growth%':>8}")
        print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*8}")
        for e in echelons:
            f_val = first_data[first_data["echelon"] == e]["total"].sum()
            l_val = last_data[last_data["echelon"] == e]["total"].sum()
            change = l_val - f_val
            growth = change / f_val * 100 if f_val > 0 else np.nan
            print(
                f"  {e:<14}  {f_val:>14,.0f}  {l_val:>14,.0f}  {change:>+10,.0f}"
                f"  {(f'{growth:+.0f}%' if not np.isnan(growth) else 'N/A'):>8}"
            )

    # --- Analysis 3: Cumulative Excess ---
    print(f"\n{'ANALYSIS 3: CUMULATIVE PRODUCTION - DEMAND (SLOB PROXY)':=^{w}}")
    mono = a3["monotonic_pct"]
    print(f"\n  Monotonicity: {mono:.1f}% of days excess grew"
          " (100% = pure drift)\n")
    h3 = f"  {'Day':>6}  {'Cum Production':>16}  {'Cum Demand':>16}"
    print(h3 + f"  {'Cum Excess':>14}  {'Excess%':>8}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*16}  {'-'*14}  {'-'*8}")
    for s in a3["snapshots"]:
        print(
            f"  {s['day']:>6}  {s['cum_production']:>16,.0f}  {s['cum_demand']:>16,.0f}"
            f"  {s['cum_excess']:>14,.0f}  {s['excess_pct']:>7.1f}%"
        )

    # --- Analysis 4: MRP Signal Floor ---
    print(f"\n{'ANALYSIS 4: MRP SIGNAL FLOOR PROOF':=^{w}}")
    print("\n  PO quantity variation by ABC class:")
    print("  If demand floor is active, PO qty CV should be"
          " very low (floor clips troughs)\n")
    h4 = f"  {'Class':>6}  {'#Products':>10}  {'Mean CV':>10}"
    print(h4 + f"  {'Mean MinRatio':>14}  {'%LowCV':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*12}")
    for cls in ("A", "B", "C"):
        s = a4["abc_summary"].get(cls, {})
        if not s:
            continue
        print(
            f"  {cls:>6}  {s['n_products']:>10}  {s['mean_qty_cv']:>10.3f}"
            f"  {s['mean_min_ratio']:>14.1f}  {s['pct_low_cv']:>11.1f}%"
        )

    # Show sample A-items
    stats = a4["stats_df"]
    a_stats = stats[stats["abc"] == "A"].sort_values("qty_cv")
    if len(a_stats) > 0:
        print("\n  Sample A-items (lowest PO qty variation — floor evidence):")
        h4b = f"  {'Product':>22}  {'#POs':>6}  {'AvgQty':>12}"
        print(h4b + f"  {'MinQty':>12}  {'MaxQty':>12}  {'CV':>8}")
        print(f"  {'-'*22}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for _, r in a_stats.head(10).iterrows():
            left = f"  {r['product_id']:>22}  {r['n_pos']:>6}"
            mid = f"  {r['avg_po_qty']:>12,.0f}  {r['min_po_qty']:>12,.0f}"
            right = f"  {r['max_po_qty']:>12,.0f}  {r['qty_cv']:>8.3f}"
            print(left + mid + right)

    # --- Analysis 5: Seasonal Correlation ---
    print(f"\n{'ANALYSIS 5: SEASONAL CORRELATION':=^{w}}")
    print("\n  If demand floor is active, production should"
          " NOT track demand seasonality.\n")
    h5 = f"  {'Signal':<14}  {'CV':>8}  {'Sine Corr':>10}"
    print(h5 + f"  {'Half Ratio':>11}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*11}")
    print(
        f"  {'Demand':<14}  {a5['demand_cv']:>8.4f}  {a5['demand_sine_corr']:>10.3f}"
        f"  {a5['demand_half_ratio']:>11.3f}"
    )
    print(
        f"  {'Production':<14}  {a5['prod_cv']:>8.4f}  {a5['prod_sine_corr']:>10.3f}"
        f"  {a5['prod_half_ratio']:>11.3f}"
    )
    print(
        f"  {'Sine (config)':<14}  {a5['sine_cv']:>8.4f}  {'1.000':>10}  {'—':>11}"
    )
    print(f"\n  Production CV / Demand CV = {a5['cv_ratio']:.3f}")
    if a5["cv_ratio"] < _FLATTENED_CV_RATIO:
        print("  → Production is FLATTENED relative to demand"
              " (demand floor active)")
    elif a5["cv_ratio"] < _DAMPENED_CV_RATIO:
        print("  → Production is DAMPENED relative to demand")
    else:
        print("  → Production tracks demand seasonality well")

    # --- Analysis 6: ABC Breakdown ---
    print(f"\n{'ANALYSIS 6: ABC CLASS BREAKDOWN':=^{w}}")

    by_class = a6["by_class"]
    print("\n  Production excess and inventory growth by ABC class:\n")
    h6 = f"  {'Class':>6}  {'TotalDemand':>14}  {'TotalProd':>14}"
    print(h6 + f"  {'Excess%':>8}  {'InvGrowth':>12}  {'Grw%':>8}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*12}  {'-'*8}")
    for cls in ("A", "B", "C"):
        c = by_class.get(cls, {})
        dem = c.get("total_demand", 0)
        pro = c.get("total_production", 0)
        exc = c.get("excess_pct", 0)
        grw = c.get("inv_growth", 0)
        grw_p = c.get("inv_growth_pct", 0)
        left = f"  {cls:>6}  {dem:>14,.0f}  {pro:>14,.0f}"
        print(left + f"  {exc:>+7.1f}%  {grw:>12,.0f}  {grw_p:>+7.1f}%")

    # Echelon inventory at end of sim
    echelon_inv = a6["echelon_inv"]
    if echelon_inv:
        print("\n  End-of-sim inventory by echelon and ABC class:\n")
        h6e = f"  {'Echelon':<14}  {'Total':>14}  {'A':>12}"
        print(h6e + f"  {'B':>12}  {'C':>12}  {'%Total':>8}")
        print(f"  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        grand_total = sum(v["total"] for v in echelon_inv.values())
        for ech in ["Plant", "RDC", "Customer DC", "Store", "Club"]:
            v = echelon_inv.get(ech, {"total": 0, "A": 0, "B": 0, "C": 0})
            pct = v["total"] / grand_total * 100 if grand_total > 0 else 0
            print(
                f"  {ech:<14}  {v['total']:>14,.0f}  {v['A']:>12,.0f}"
                f"  {v['B']:>12,.0f}  {v['C']:>12,.0f}  {pct:>7.1f}%"
            )
        print(f"  {'TOTAL':<14}  {grand_total:>14,.0f}")

    # --- Root Cause Summary ---
    print(f"\n{'ROOT CAUSE SUMMARY':=^{w}}")
    print("""
  BUG LOCATION: src/prism_sim/simulation/mrp.py:996-1004

  MECHANISM:
    demand_for_dos = max(expected, actual*0.35 + expected*0.65)

    - `expected` is annual average (static, from base_demand_matrix)
    - During seasonal troughs (demand = 0.88x annual avg):
      blended = 0.88E * 0.35 + E * 0.65 = 0.958E
      max(E, 0.958E) = E  ← ALWAYS floors at annual average
    - Batch sizing: E * abc_horizon → produces 14 days of annual-avg demand
    - Real demand only needs 0.88 * 14 = 12.3 days worth
    - Result: ~12% systematic overproduction for ~6 months of trough

  COMPOUNDING FACTORS:
    1. seasonal_factor IS calculated (line 930) but only used in fallback
       branch (line 1020) that's never reached
    2. inventory_cap_dos=30.0 is configured but DEAD CODE — initialized at
       line 126, never referenced in batch/trigger logic
    3. A-item target inventory (line 1058) also uses floored demand_for_dos
    4. No SLOB clearance mechanism — aged inventory flagged but never removed

  EVIDENCE FROM THIS DIAGNOSTIC:
    - Analysis 1: Production/demand ratio should widen during troughs
    - Analysis 3: Cumulative excess should be monotonically increasing
    - Analysis 4: PO qty CV should be very low (floor clips variation)
    - Analysis 5: Production CV < Demand CV (production doesn't track season)
""")

    print(f"{'=' * w}")
    print("  END OF DIAGNOSTIC REPORT".center(w))
    print(f"{'=' * w}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="365-Day Drift Diagnostic: Root Cause Analysis"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/output"),
        help="Simulation output directory (default: data/output)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window size in days (default: 30)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    window: int = args.window

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    print(f"Data directory:  {data_dir}")
    print(f"Rolling window:  {window} days")
    print()

    # --- Load lightweight files ---
    print("Loading data...")
    products = load_products(data_dir)
    metrics = load_metrics(data_dir)
    shipments = load_shipments(data_dir)
    batches = load_batches(data_dir)
    production_orders = load_production_orders(data_dir)
    forecasts = load_forecasts(data_dir)

    # --- ABC classification ---
    print("\nClassifying ABC...")
    abc_df = classify_abc(shipments, products)
    abc_map: dict[str, str] = dict(
        zip(abc_df["product_id"], abc_df["abc_class"], strict=False)
    )

    # --- Analysis 1 ---
    print("\nAnalysis 1: Production vs Demand time series...")
    a1 = analysis_1_production_vs_demand(batches, shipments, abc_map, window)

    # --- Analysis 2: Stream inventory (memory-safe) ---
    print("\nAnalysis 2: Inventory by echelon over time (streaming)...")
    inv_echelon_day = stream_inventory_by_echelon_day(data_dir, abc_map)

    # --- Analysis 3 ---
    print("\nAnalysis 3: Cumulative excess...")
    a3 = analysis_3_cumulative_excess(batches, shipments)

    # --- Analysis 4 ---
    print("\nAnalysis 4: MRP signal floor proof...")
    a4 = analysis_4_mrp_signal_floor(production_orders, forecasts, abc_map)

    # --- Analysis 5 ---
    print("\nAnalysis 5: Seasonal correlation...")
    a5 = analysis_5_seasonal_correlation(batches, shipments, window)

    # --- Analysis 6 ---
    print("\nAnalysis 6: ABC class breakdown...")
    a6 = analysis_6_abc_breakdown(batches, shipments, inv_echelon_day, abc_map, metrics)

    # --- Report ---
    print_report(a1, inv_echelon_day, a3, a4, a5, a6, window, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
