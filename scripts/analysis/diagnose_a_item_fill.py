#!/usr/bin/env python3
"""
A-Item Fill Rate Root Cause Diagnostic.

Analyzes the ~10pp gap between A-item fill rate (90.4%) and B/C items (98%+)
across four diagnostic layers:
  0. Fill rate measurement validation
  1. Stockout location analysis (WHERE)
  2. Root cause identification (WHY)
  3. Root cause ranking by estimated impact

Memory-safe: streams through the 9.7GB inventory.parquet one row group at a
time, accumulating only summary aggregates (~100KB peak overhead).

Usage:
    poetry run python scripts/analysis/diagnose_a_item_fill.py --data-dir data/output
    poetry run python scripts/analysis/diagnose_a_item_fill.py --data-dir data/output --steady-state-start-day 365
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_node(node_id: str) -> str:
    """Classify node ID into echelon tier."""
    if node_id.startswith("PLANT-"):
        return "Plant"
    if node_id.startswith("RDC-"):
        return "RDC"
    if node_id.startswith(("RET-DC-", "DIST-DC-", "ECOM-FC-")):
        return "Customer DC"
    if node_id.startswith("STORE-"):
        return "Store"
    if node_id.startswith("SUP-"):
        return "Supplier"
    return "Other"


def classify_echelon_link(source_id: str, target_id: str) -> str:
    """Classify a shipment/order link into an echelon lane."""
    return f"{classify_node(source_id)}->{classify_node(target_id)}"


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


def load_locations(data_dir: Path) -> pd.DataFrame:
    """Load node/location catalog from static world."""
    return pd.read_csv(
        data_dir / "static_world" / "locations.csv",
        usecols=["id", "name", "type"],
    )


def load_orders(data_dir: Path) -> pd.DataFrame:
    """Load full orders parquet."""
    print("  Loading orders.parquet...")
    df = pd.read_parquet(data_dir / "orders.parquet")
    print(f"    {len(df):,} rows")
    return df


def load_shipments(data_dir: Path) -> pd.DataFrame:
    """Load full shipments parquet."""
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
    """Load full batches parquet."""
    print("  Loading batches.parquet...")
    df = pd.read_parquet(data_dir / "batches.parquet")
    print(f"    {len(df):,} rows")
    return df


def load_metrics(data_dir: Path) -> dict:
    """Load simulation metrics JSON."""
    with open(data_dir / "metrics.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ABC Classification (matches MRP logic)
# ---------------------------------------------------------------------------

def classify_abc(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> pd.DataFrame:
    """Per-category Pareto ABC classification matching MRPEngine logic."""
    fg_products = set(products[products["category"] != "INGREDIENT"]["id"])
    fg_orders = orders[orders["product_id"].isin(fg_products)]

    product_volume = fg_orders.groupby("product_id")["quantity"].sum().reset_index()
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
            # Ensure top item is always A
            cat_df.iloc[0, cat_df.columns.get_loc("abc_class")] = "A"

        abc_labels.append(cat_df[["product_id", "abc_class"]])

    abc_df = pd.concat(abc_labels, ignore_index=True)
    counts = abc_df["abc_class"].value_counts()
    print(f"  ABC: A={counts.get('A', 0)}, B={counts.get('B', 0)}, C={counts.get('C', 0)} SKUs")
    return abc_df


# ---------------------------------------------------------------------------
# Streaming Inventory Aggregation (memory-safe)
# ---------------------------------------------------------------------------

def _add_or_init(
    running: pd.DataFrame | None, chunk: pd.DataFrame
) -> pd.DataFrame:
    """Accumulate a chunk into a running DataFrame using index-aligned add."""
    if running is None:
        return chunk
    return running.add(chunk, fill_value=0)


def stream_inventory_aggregates(
    data_dir: Path,
    start_day: int,
    abc_map: dict[str, str],
    node_echelon: dict[str, str],
    low_inv_threshold: float = 1.0,
) -> dict:
    """
    Single streaming pass through inventory.parquet.

    Accumulates lightweight summary DataFrames per row group.
    Peak memory: ~one row group (~120MB) + small accumulators.

    Returns dict of summary DataFrames ready for Layer 1.
    """
    pf = pq.ParquetFile(data_dir / "inventory.parquet")
    n_rg = pf.metadata.num_row_groups
    print(f"  Streaming inventory.parquet ({pf.metadata.num_rows:,} rows, {n_rg} RGs)...")
    print(f"    day >= {start_day}, low-inv threshold < {low_inv_threshold} cases")

    columns = ["day", "node_id", "product_id", "actual_inventory"]

    # Running accumulators
    dos_agg: pd.DataFrame | None = None        # (echelon, abc, product_id) -> inv_sum, obs_count
    store_abc_agg: pd.DataFrame | None = None   # (abc) -> total, low
    a_sku_agg: pd.DataFrame | None = None       # (product_id) -> total, low
    a_store_agg: pd.DataFrame | None = None     # (node_id) -> total, low
    n_days_seen: set[int] = set()
    rg_loaded = 0

    for rg_idx in range(n_rg):
        # Quick day check (single column, fast)
        day_col = pf.read_row_group(rg_idx, columns=["day"])
        rg_days = day_col["day"].unique().to_pylist()
        if not rg_days or max(rg_days) < start_day:
            continue

        chunk = pf.read_row_group(rg_idx, columns=columns).to_pandas()
        chunk = chunk[chunk["day"] >= start_day]
        if len(chunk) == 0:
            continue
        rg_loaded += 1
        n_days_seen.update(chunk["day"].unique())

        # Convert dictionary-encoded categoricals to plain strings
        for col in ("node_id", "product_id"):
            if hasattr(chunk[col], "cat"):
                chunk[col] = chunk[col].astype(str)

        # Map echelon and ABC
        chunk["echelon"] = chunk["node_id"].map(node_echelon).fillna("Other")
        chunk["abc"] = chunk["product_id"].map(abc_map)

        # Keep only FG products (those with an ABC class)
        fg = chunk.dropna(subset=["abc"])
        if len(fg) == 0:
            continue

        # --- 1. DOS accumulation (all echelons) ---
        grp = fg.groupby(["echelon", "abc", "product_id"], observed=True)
        chunk_dos = pd.DataFrame({
            "inv_sum": grp["actual_inventory"].sum(),
            "obs_count": grp.size(),
        })
        dos_agg = _add_or_init(dos_agg, chunk_dos)

        # --- 2-4. Store-level low-inventory analysis ---
        stores = fg[fg["echelon"] == "Store"]
        if len(stores) == 0:
            continue

        low_mask = stores["actual_inventory"] < low_inv_threshold
        stores_low = stores[low_mask]

        # 2. Low-inv by ABC class
        chunk_abc = pd.DataFrame({
            "total": stores.groupby("abc", observed=True).size(),
            "low": stores_low.groupby("abc", observed=True).size(),
        }).fillna(0)
        store_abc_agg = _add_or_init(store_abc_agg, chunk_abc)

        # 3-4. A-item breakdown
        a_all = stores[stores["abc"] == "A"]
        a_low = stores_low[stores_low["abc"] == "A"]

        if len(a_all) > 0:
            chunk_sku = pd.DataFrame({
                "total": a_all.groupby("product_id", observed=True).size(),
                "low": a_low.groupby("product_id", observed=True).size(),
            }).fillna(0)
            a_sku_agg = _add_or_init(a_sku_agg, chunk_sku)

            chunk_st = pd.DataFrame({
                "total": a_all.groupby("node_id", observed=True).size(),
                "low": a_low.groupby("node_id", observed=True).size(),
            }).fillna(0)
            a_store_agg = _add_or_init(a_store_agg, chunk_st)

        if rg_loaded % 100 == 0:
            print(f"    ... {rg_loaded} row groups processed")

    print(f"    Done: {rg_loaded} row groups, {len(n_days_seen)} unique days")
    return {
        "dos_agg": dos_agg,
        "store_abc_agg": store_abc_agg,
        "a_sku_agg": a_sku_agg,
        "a_store_agg": a_store_agg,
        "n_days": len(n_days_seen),
    }


# ---------------------------------------------------------------------------
# Layer 0: Measurement Validation
# ---------------------------------------------------------------------------

def layer_0_measurement(
    orders: pd.DataFrame,
    shipments: pd.DataFrame,
    abc_df: pd.DataFrame,
    metrics: dict,
    steady_start: int,
) -> dict:
    """
    Cross-check reported fill rate against DC->Store order fill rate.

    The orchestrator computes store fill rate at end-of-day (after arrivals
    and production), not from the pre-sales snapshot at step 2.  We compare
    with a shipment-based fill rate to quantify any gap.
    """
    abc_map = abc_df.set_index("product_id")["abc_class"]

    # --- DC->Store order fill rate (same-echelon matching) ---
    # Filter to store-destination only
    ss_orders = orders[
        (orders["day"] >= steady_start)
        & (orders["target_id"].str.startswith("STORE-"))
    ].copy()
    ss_ships = shipments[
        (shipments["creation_day"] >= steady_start)
        & (shipments["target_id"].str.startswith("STORE-"))
    ].copy()

    # Aggregate per (source, target, product) over full steady state
    ord_vol = ss_orders.groupby(["source_id", "target_id", "product_id"])["quantity"].sum()
    shp_vol = ss_ships.groupby(["source_id", "target_id", "product_id"])["quantity"].sum()

    merged = pd.DataFrame({"ordered": ord_vol, "shipped": shp_vol}).fillna(0)
    # Cap at 1.0 to handle over-shipment (MOQs, batch sizes)
    merged["fill"] = np.minimum(merged["shipped"], merged["ordered"]) / merged["ordered"]
    merged["fill"] = merged["fill"].clip(0, 1)
    merged["abc"] = merged.index.get_level_values("product_id").map(abc_map)

    result: dict = {}
    for cls in ("A", "B", "C"):
        m = merged[merged["abc"] == cls]
        total_ord = m["ordered"].sum()
        total_filled = np.minimum(m["shipped"], m["ordered"]).sum()
        result[f"{cls}_order_fill"] = total_filled / total_ord if total_ord > 0 else 1.0

    total_ord = merged["ordered"].sum()
    total_filled = np.minimum(merged["shipped"], merged["ordered"]).sum()
    result["overall_order_fill"] = total_filled / total_ord if total_ord > 0 else 1.0

    # Arrival-vs-demand ratio (does same-day arrival mask stockouts?)
    store_arrivals_per_day = (
        ss_ships.groupby("arrival_day")["quantity"].sum().mean()
    )
    store_orders_per_day = (
        ss_orders.groupby("day")["quantity"].sum().mean()
    )
    result["arrival_demand_ratio"] = (
        store_arrivals_per_day / store_orders_per_day
        if store_orders_per_day > 0
        else np.nan
    )

    # Reported fill rates from metrics.json
    abc_reported = metrics.get("service_level_by_abc", {})
    result["A_reported_fill"] = abc_reported.get("A", np.nan)
    result["B_reported_fill"] = abc_reported.get("B", np.nan)
    result["C_reported_fill"] = abc_reported.get("C", np.nan)
    result["overall_reported_fill"] = metrics.get("store_service_level", {}).get(
        "mean", np.nan
    )

    return result


# ---------------------------------------------------------------------------
# Layer 1: Stockout Location (from streamed aggregates)
# ---------------------------------------------------------------------------

def layer_1_stockout_location(
    inv_agg: dict,
    orders: pd.DataFrame,
    abc_df: pd.DataFrame,
    steady_start: int,
) -> dict:
    """Build Layer 1 tables from pre-computed streaming aggregates."""
    dos_agg = inv_agg["dos_agg"]
    store_abc_agg = inv_agg["store_abc_agg"]
    a_sku_agg = inv_agg["a_sku_agg"]
    a_store_agg = inv_agg["a_store_agg"]
    n_inv_days = inv_agg["n_days"]

    # --- Average DOS by echelon + ABC ---
    # Compute avg daily demand per product from store orders
    ss_orders = orders[
        (orders["day"] >= steady_start)
        & (orders["target_id"].str.startswith("STORE-"))
    ]
    n_order_days = ss_orders["day"].nunique()
    if n_order_days == 0:
        n_order_days = 1

    avg_daily_demand = (
        ss_orders.groupby("product_id")["quantity"].sum() / n_order_days
    )

    # Mean inventory per (echelon, abc, product) = inv_sum / obs_count
    dos_df = dos_agg.copy()
    dos_df["mean_inv"] = dos_df["inv_sum"] / dos_df["obs_count"]

    # Map demand onto product level
    product_ids = dos_df.index.get_level_values("product_id")
    dos_df["avg_demand"] = product_ids.map(avg_daily_demand).fillna(0).values
    dos_df["dos"] = np.where(
        dos_df["avg_demand"] > 0,
        dos_df["mean_inv"] / dos_df["avg_demand"],
        np.nan,
    )

    dos_by_echelon_abc = (
        dos_df.groupby(level=["echelon", "abc"])["dos"]
        .median()
        .unstack(fill_value=np.nan)
    )

    # --- Low-inventory frequency at stores ---
    low_pct = (store_abc_agg["low"] / store_abc_agg["total"] * 100).fillna(0)

    # --- Worst A-item SKUs ---
    worst_a_skus = pd.Series(dtype=float)
    if a_sku_agg is not None and len(a_sku_agg) > 0:
        sku_rate = (a_sku_agg["low"] / a_sku_agg["total"] * 100).fillna(0)
        worst_a_skus = sku_rate.nlargest(20)

    # --- Worst stores for A-items ---
    worst_stores = pd.Series(dtype=float)
    if a_store_agg is not None and len(a_store_agg) > 0:
        store_rate = (a_store_agg["low"] / a_store_agg["total"] * 100).fillna(0)
        worst_stores = store_rate.nlargest(20)

    return {
        "dos_by_echelon_abc": dos_by_echelon_abc,
        "low_count_by_abc": store_abc_agg["low"] if store_abc_agg is not None else pd.Series(dtype=float),
        "total_count_by_abc": store_abc_agg["total"] if store_abc_agg is not None else pd.Series(dtype=float),
        "low_pct_by_abc": low_pct,
        "worst_a_skus": worst_a_skus,
        "worst_stores": worst_stores,
        "n_inv_days": n_inv_days,
    }


# ---------------------------------------------------------------------------
# Layer 2: Root Cause Analysis
# ---------------------------------------------------------------------------

def layer_2a_allocation(
    orders: pd.DataFrame,
    shipments: pd.DataFrame,
    abc_df: pd.DataFrame,
    steady_start: int,
) -> dict:
    """Analyze allocation fairness: are A-items disproportionately rationed?"""
    ss_orders = orders[orders["day"] >= steady_start]
    ss_ships = shipments[shipments["creation_day"] >= steady_start]

    ordered_by_sp = (
        ss_orders.groupby(["source_id", "product_id"])["quantity"].sum().reset_index()
    )
    ordered_by_sp.columns = ["source_id", "product_id", "ordered"]

    shipped_by_sp = (
        ss_ships.groupby(["source_id", "product_id"])["quantity"].sum().reset_index()
    )
    shipped_by_sp.columns = ["source_id", "product_id", "shipped"]

    merged = ordered_by_sp.merge(
        shipped_by_sp, on=["source_id", "product_id"], how="left"
    )
    merged["shipped"] = merged["shipped"].fillna(0)
    merged["fill_ratio"] = np.where(
        merged["ordered"] > 0,
        np.minimum(merged["shipped"], merged["ordered"]) / merged["ordered"],
        1.0,
    )
    merged = merged.merge(abc_df, on="product_id", how="left")

    # Constrained sources: any product fill < 0.99
    source_min = merged.groupby("source_id")["fill_ratio"].min()
    constrained_ids = source_min[source_min < 0.99].index
    constrained = merged[merged["source_id"].isin(constrained_ids)]

    def _abc_fill(df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("abc_class").agg(
            total_ordered=("ordered", "sum"),
            total_shipped=("shipped", "sum"),
        )
        agg["fill_ratio"] = np.minimum(agg["total_shipped"], agg["total_ordered"]) / agg["total_ordered"]
        return agg

    return {
        "constrained_fill_by_abc": _abc_fill(constrained) if len(constrained) > 0 else pd.DataFrame(),
        "all_fill_by_abc": _abc_fill(merged),
        "n_constrained_sources": len(constrained_ids),
        "n_total_sources": merged["source_id"].nunique(),
    }


def layer_2b_production(
    batches: pd.DataFrame,
    orders: pd.DataFrame,
    abc_df: pd.DataFrame,
    steady_start: int,
) -> dict:
    """Check if production capacity is adequate for A-items."""
    a_products = set(abc_df[abc_df["abc_class"] == "A"]["product_id"])

    a_batches = batches[batches["product_id"].isin(a_products)]
    a_orders = orders[orders["product_id"].isin(a_products)]

    total_a_production = a_batches["quantity"].sum()
    total_a_demand = a_orders["quantity"].sum()

    prod_freq = a_batches.groupby("product_id")["day_produced"].nunique()
    total_days = (
        batches["day_produced"].max() - batches["day_produced"].min() + 1
        if len(batches) > 0
        else 365
    )

    ss_a_batches = a_batches[a_batches["day_produced"] >= steady_start]
    ss_a_orders = a_orders[a_orders["day"] >= steady_start]
    ss_production = ss_a_batches["quantity"].sum()
    ss_demand = ss_a_orders["quantity"].sum()

    return {
        "total_a_production": total_a_production,
        "total_a_demand": total_a_demand,
        "prod_demand_ratio": total_a_production / total_a_demand if total_a_demand > 0 else np.nan,
        "ss_a_production": ss_production,
        "ss_a_demand": ss_demand,
        "ss_prod_demand_ratio": ss_production / ss_demand if ss_demand > 0 else np.nan,
        "a_sku_prod_frequency": prod_freq,
        "total_days": total_days,
    }


def layer_2c_lead_times(
    shipments: pd.DataFrame,
    steady_start: int,
) -> dict:
    """Compute actual echelon lead times vs config assumptions."""
    ss = shipments[shipments["creation_day"] >= steady_start].copy()
    ss["lead_time"] = ss["arrival_day"] - ss["creation_day"]

    # Vectorised lane classification (avoid row-wise apply)
    src_ech = ss["source_id"].apply(classify_node)
    tgt_ech = ss["target_id"].apply(classify_node)
    ss["lane"] = src_ech + "->" + tgt_ech

    config_lt = {
        "Plant->RDC": 5.0,
        "RDC->Customer DC": 3.0,
        "Customer DC->Store": 1.0,
    }

    lane_stats = ss.groupby("lane")["lead_time"].agg(
        ["mean", "median", "std", "count"]
    ).sort_values("count", ascending=False)

    total_echelon = 0.0
    for lane in ("Plant->RDC", "RDC->Customer DC", "Customer DC->Store"):
        if lane in lane_stats.index:
            total_echelon += lane_stats.loc[lane, "mean"]

    return {
        "lane_stats": lane_stats,
        "config_lt": config_lt,
        "total_echelon_lt": total_echelon,
    }


def layer_2d_replenishment_gaps(
    orders: pd.DataFrame,
    abc_df: pd.DataFrame,
    steady_start: int,
) -> dict:
    """Analyze replenishment frequency for store-level A/B/C items."""
    result: dict = {}

    for cls in ("A", "B", "C"):
        cls_products = set(abc_df[abc_df["abc_class"] == cls]["product_id"])
        cls_orders = orders[
            (orders["day"] >= steady_start)
            & (orders["target_id"].str.startswith("STORE-"))
            & (orders["product_id"].isin(cls_products))
        ].sort_values(["target_id", "product_id", "day"])

        cls_orders = cls_orders.copy()
        cls_orders["prev_day"] = cls_orders.groupby(
            ["target_id", "product_id"]
        )["day"].shift(1)
        gaps = (cls_orders["day"] - cls_orders["prev_day"]).dropna()

        prefix = f"{cls}_" if cls != "A" else ""
        result[f"{prefix}mean_gap"] = gaps.mean() if len(gaps) > 0 else np.nan
        result[f"{prefix}median_gap"] = gaps.median() if len(gaps) > 0 else np.nan
        if cls == "A":
            result["p95_gap"] = gaps.quantile(0.95) if len(gaps) > 0 else np.nan
            result["max_gap"] = gaps.max() if len(gaps) > 0 else np.nan
            result["pct_gap_gt_5"] = (gaps > 5).mean() * 100 if len(gaps) > 0 else 0
            result["pct_gap_gt_10"] = (gaps > 10).mean() * 100 if len(gaps) > 0 else 0

    return result


# ---------------------------------------------------------------------------
# Layer 3: Root Cause Ranking
# ---------------------------------------------------------------------------

def layer_3_isolation(
    l0: dict, l1: dict, l2a: dict, l2b: dict, l2c: dict, l2d: dict,
) -> list[dict]:
    """Rank root causes by estimated impact."""
    causes: list[dict] = []

    # 1. Measurement timing
    reported = l0.get("A_reported_fill", np.nan)
    order_based = l0.get("A_order_fill", np.nan)
    if not (np.isnan(reported) or np.isnan(order_based)):
        gap = abs(reported - order_based) * 100
        causes.append({
            "cause": "Measurement Timing (end-of-day vs pre-sales)",
            "impact_pp": gap,
            "evidence": (
                f"Reported A-fill={reported:.1%}, "
                f"DC->Store order fill={order_based:.1%}, "
                f"gap={gap:.1f}pp, "
                f"arrival/demand ratio={l0.get('arrival_demand_ratio', 0):.2f}"
            ),
            "verdict": "Material" if gap > 1.0 else "Negligible",
        })

    # 2. Allocation fairness
    alloc = l2a["constrained_fill_by_abc"]
    if len(alloc) > 0 and "A" in alloc.index:
        a_fill = alloc.loc["A", "fill_ratio"]
        b_fill = alloc.loc["B", "fill_ratio"] if "B" in alloc.index else 1.0
        alloc_gap = max(0, (b_fill - a_fill) * 100)
        causes.append({
            "cause": "Allocation Fairness (uniform ratios, no ABC weighting)",
            "impact_pp": alloc_gap,
            "evidence": (
                f"Constrained sources: A-fill={a_fill:.3f}, B-fill={b_fill:.3f}"
            ),
            "verdict": "Material" if alloc_gap > 1.0 else "Negligible",
        })

    # 3. Production capacity
    pr = l2b["ss_prod_demand_ratio"]
    if not np.isnan(pr):
        prod_gap = max(0, (1.0 - pr) * 100)
        causes.append({
            "cause": "Production Capacity Shortfall",
            "impact_pp": prod_gap,
            "evidence": (
                f"Steady-state prod/demand={pr:.2f}, "
                f"production={l2b['ss_a_production']:,.0f}, "
                f"demand={l2b['ss_a_demand']:,.0f}"
            ),
            "verdict": "Material" if pr < 0.95 else "Negligible",
        })

    # 4. Safety stock / lead time mismatch
    total_lt = l2c["total_echelon_lt"]
    config_assumed = 3.0
    lt_delta = total_lt - config_assumed
    causes.append({
        "cause": "Safety Stock Undersized (lead time mismatch)",
        "impact_pp": max(0, lt_delta * 2),
        "evidence": (
            f"Echelon LT={total_lt:.1f}d, config={config_assumed:.1f}d, "
            f"delta={lt_delta:+.1f}d"
        ),
        "verdict": "Material" if lt_delta > 2.0 else "Negligible",
    })

    # 5. Replenishment frequency
    mean_gap = l2d.get("mean_gap", np.nan)
    if not np.isnan(mean_gap) and mean_gap > 4.0:
        gap_impact = (mean_gap - 2.0) * 1.5
        causes.append({
            "cause": "Replenishment Too Infrequent",
            "impact_pp": gap_impact,
            "evidence": (
                f"A-item mean order gap={mean_gap:.1f}d (config order_cycle=2d), "
                f"p95={l2d.get('p95_gap', 0):.1f}d"
            ),
            "verdict": "Material" if gap_impact > 2.0 else "Minor",
        })

    # 6. Disproportionate low inventory
    low_pct = l1["low_pct_by_abc"]
    a_low = low_pct.get("A", 0) if hasattr(low_pct, "get") else 0
    b_low = low_pct.get("B", 0) if hasattr(low_pct, "get") else 0
    stockout_gap = a_low - b_low
    if stockout_gap > 1.0:
        causes.append({
            "cause": "Disproportionate A-Item Store Low-Inventory",
            "impact_pp": stockout_gap,
            "evidence": f"A-item low-inv rate={a_low:.1f}%, B={b_low:.1f}%",
            "verdict": "Symptom (not root cause)",
        })

    causes.sort(key=lambda c: c["impact_pp"], reverse=True)
    return causes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    l0: dict, l1: dict, l2a: dict, l2b: dict, l2c: dict, l2d: dict,
    causes: list[dict], low_inv_threshold: float,
) -> None:
    """Print structured diagnostic report."""
    w = 70
    print()
    print("=" * w)
    print("  A-ITEM FILL RATE DIAGNOSTIC REPORT".center(w))
    print("=" * w)

    # --- Layer 0 ---
    print(f"\n{'LAYER 0: MEASUREMENT VALIDATION':=^{w}}")
    print(f"  Reported A-item fill rate (on-shelf): {l0['A_reported_fill']:.1%}")
    print(f"  DC->Store order fill rate (capped):   {l0['A_order_fill']:.1%}")
    gap_pp = (l0["A_reported_fill"] - l0["A_order_fill"]) * 100
    print(f"  Gap:                                  {gap_pp:+.1f} pp")
    arr_ratio = l0.get("arrival_demand_ratio", np.nan)
    if not np.isnan(arr_ratio):
        print(f"  Daily store arrivals / orders:         {arr_ratio:.2f}")
    verdict = "MATERIAL (>1pp)" if abs(gap_pp) > 1.0 else "Negligible (<1pp)"
    print(f"  Verdict: {verdict}")

    print(f"\n  {'Class':<8} {'Reported':>10} {'Order-Fill':>11} {'Gap':>8}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 11} {'-' * 8}")
    for cls in ("A", "B", "C"):
        rep = l0.get(f"{cls}_reported_fill", np.nan)
        orb = l0.get(f"{cls}_order_fill", np.nan)
        g = (rep - orb) * 100 if not (np.isnan(rep) or np.isnan(orb)) else np.nan
        print(
            f"  {cls:<8} "
            f"{(f'{rep:.1%}' if not np.isnan(rep) else 'N/A'):>10} "
            f"{(f'{orb:.1%}' if not np.isnan(orb) else 'N/A'):>11} "
            f"{(f'{g:+.1f}pp' if not np.isnan(g) else 'N/A'):>8}"
        )
    print(f"\n  NOTE: Reported = on-shelf availability (min(demand, inv) / demand)")
    print(f"        Order-fill = DC->Store shipped/ordered (capped at 1.0)")

    # --- Layer 1 ---
    print(f"\n{'LAYER 1: STOCKOUT LOCATION':=^{w}}")

    dos = l1["dos_by_echelon_abc"]
    print(f"\n  Median Days of Supply by Echelon:")
    print(f"  {'Echelon':<15} {'A-items':>10} {'B-items':>10} {'C-items':>10}")
    print(f"  {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10}")
    for ech in ("Plant", "RDC", "Customer DC", "Store"):
        if ech in dos.index:
            row = dos.loc[ech]
            vals = []
            for c in ("A", "B", "C"):
                v = row.get(c, np.nan)
                vals.append(f"{v:.1f}d" if not np.isnan(v) else "N/A")
            print(f"  {ech:<15} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print(f"\n  Low-Inventory Events (<{low_inv_threshold} case) at stores:")
    print(f"  {'Class':<8} {'Low-inv':>12} {'Total obs':>12} {'Rate':>8}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 8}")
    for cls in ("A", "B", "C"):
        lo = int(l1["low_count_by_abc"].get(cls, 0))
        tot = int(l1["total_count_by_abc"].get(cls, 0))
        pct = l1["low_pct_by_abc"].get(cls, 0)
        print(f"  {cls:<8} {lo:>12,} {tot:>12,} {pct:>7.1f}%")

    if len(l1["worst_a_skus"]) > 0:
        print(f"\n  Top-10 Worst A-Item SKUs (low-inv rate at stores):")
        for i, (sku, rate) in enumerate(l1["worst_a_skus"].head(10).items()):
            print(f"    {i + 1:2d}. {sku:<25s} {rate:.1f}%")

    if len(l1["worst_stores"]) > 0:
        print(f"\n  Top-10 Worst Stores for A-Items (low-inv rate):")
        for i, (store, rate) in enumerate(l1["worst_stores"].head(10).items()):
            print(f"    {i + 1:2d}. {store:<30s} {rate:.1f}%")

    # --- Layer 2 ---
    print(f"\n{'LAYER 2: ROOT CAUSES':=^{w}}")

    # 2a
    print("\n  2a. Allocation Fill Ratios")
    print(f"      Constrained sources: {l2a['n_constrained_sources']}/{l2a['n_total_sources']}")
    for label, df in [("ALL sources", l2a["all_fill_by_abc"]),
                       ("CONSTRAINED only", l2a["constrained_fill_by_abc"])]:
        if len(df) == 0:
            continue
        print(f"\n      {label}:")
        print(f"      {'Class':<8} {'Ordered':>14} {'Shipped':>14} {'Fill':>8}")
        print(f"      {'-' * 8} {'-' * 14} {'-' * 14} {'-' * 8}")
        for cls in ("A", "B", "C"):
            if cls in df.index:
                r = df.loc[cls]
                print(
                    f"      {cls:<8} {r['total_ordered']:>14,.0f} "
                    f"{r['total_shipped']:>14,.0f} {r['fill_ratio']:>7.3f}"
                )

    # 2b
    print("\n  2b. Production vs Demand (A-items)")
    print(f"      Overall   production: {l2b['total_a_production']:>14,.0f} cases")
    print(f"      Overall   demand:     {l2b['total_a_demand']:>14,.0f} cases")
    print(f"      Overall   ratio:      {l2b['prod_demand_ratio']:>14.3f}")
    print(f"      Steady-st production: {l2b['ss_a_production']:>14,.0f} cases")
    print(f"      Steady-st demand:     {l2b['ss_a_demand']:>14,.0f} cases")
    print(f"      Steady-st ratio:      {l2b['ss_prod_demand_ratio']:>14.3f}")
    pf = l2b["a_sku_prod_frequency"]
    if len(pf) > 0:
        print(f"\n      Production frequency ({l2b['total_days']}d sim):")
        print(f"        Mean: {pf.mean():.1f}d  Median: {pf.median():.1f}d  "
              f"Min: {pf.min()}d  Max: {pf.max()}d")
        print(f"        SKUs produced <10 days: {(pf < 10).sum()}")

    # 2c
    print("\n  2c. Echelon Lead Times (steady state)")
    ls = l2c["lane_stats"]
    cfg = l2c["config_lt"]
    print(f"      {'Lane':<25s} {'Actual':>8} {'Config':>8} {'Delta':>8} {'Count':>10}")
    print(f"      {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
    for lane in ("Plant->RDC", "RDC->Customer DC", "Customer DC->Store"):
        if lane in ls.index:
            a = ls.loc[lane, "mean"]
            c = cfg.get(lane, np.nan)
            d = a - c if not np.isnan(c) else np.nan
            print(
                f"      {lane:<25s} {a:>7.1f}d "
                f"{(f'{c:.1f}d' if not np.isnan(c) else 'N/A'):>8} "
                f"{(f'{d:+.1f}d' if not np.isnan(d) else 'N/A'):>8} "
                f"{int(ls.loc[lane, 'count']):>10,}"
            )
    print(f"      Total echelon LT: {l2c['total_echelon_lt']:.1f} days")

    other = [l for l in ls.index if l not in cfg]
    if other:
        print("      Other lanes:")
        for lane in other:
            r = ls.loc[lane]
            print(f"      {lane:<25s} {r['mean']:>7.1f}d  (n={int(r['count']):,})")

    # 2d
    print("\n  2d. Replenishment Gap Analysis (store-level)")
    print(f"      {'Metric':<25s} {'A-items':>10} {'B-items':>10} {'C-items':>10}")
    print(f"      {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
    for metric, keys in [
        ("Mean gap (days)", ("mean_gap", "B_mean_gap", "C_mean_gap")),
        ("Median gap (days)", ("median_gap", "B_median_gap", "C_median_gap")),
    ]:
        vals = []
        for k in keys:
            v = l2d.get(k, np.nan)
            vals.append(f"{v:.1f}" if not np.isnan(v) else "N/A")
        print(f"      {metric:<25s} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
    print(f"      A-item P95 gap:      {l2d.get('p95_gap', 0):.1f} days")
    print(f"      A-item max gap:      {l2d.get('max_gap', 0):.0f} days")
    print(f"      A-item orders >5d gap:  {l2d.get('pct_gap_gt_5', 0):.1f}%")
    print(f"      A-item orders >10d gap: {l2d.get('pct_gap_gt_10', 0):.1f}%")

    # --- Layer 3 ---
    print(f"\n{'LAYER 3: ROOT CAUSE RANKING':=^{w}}")
    for i, c in enumerate(causes):
        flag = " ***" if c["verdict"] == "Material" else ""
        print(f"\n  #{i + 1}: {c['cause']}{flag}")
        print(f"      Impact: ~{c['impact_pp']:.1f} pp")
        print(f"      Evidence: {c['evidence']}")
        print(f"      Verdict: {c['verdict']}")

    print(f"\n{'=' * w}")
    print("  END OF DIAGNOSTIC REPORT".center(w))
    print(f"{'=' * w}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose A-item fill rate gap vs B/C items"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/output"),
        help="Simulation output directory (default: data/output)",
    )
    parser.add_argument(
        "--steady-state-start-day",
        type=int,
        default=275,
        help="First day of steady-state window (default: 275)",
    )
    parser.add_argument(
        "--low-inv-threshold",
        type=float,
        default=1.0,
        help="Cases below which inventory is 'near-stockout' (default: 1.0)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    steady_start: int = args.steady_state_start_day
    low_inv: float = args.low_inv_threshold

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    print(f"Data directory:  {data_dir}")
    print(f"Steady state:    day >= {steady_start}")
    print(f"Low-inv thresh:  < {low_inv} cases")
    print()

    # --- Load lightweight files ---
    print("Loading data...")
    products = load_products(data_dir)
    locations = load_locations(data_dir)
    metrics = load_metrics(data_dir)
    orders = load_orders(data_dir)
    shipments = load_shipments(data_dir)
    batches = load_batches(data_dir)

    # --- ABC classification ---
    print("\nClassifying ABC...")
    abc_df = classify_abc(orders, products)

    # Build lookup dicts for streaming
    abc_map: dict[str, str] = dict(zip(abc_df["product_id"], abc_df["abc_class"]))
    node_echelon: dict[str, str] = {
        row["id"]: classify_node(row["id"]) for _, row in locations.iterrows()
    }

    # --- Stream inventory (memory-safe) ---
    print()
    inv_agg = stream_inventory_aggregates(
        data_dir, steady_start, abc_map, node_echelon, low_inv,
    )

    # --- Layer 0 ---
    print("\nLayer 0: Measurement validation...")
    l0 = layer_0_measurement(orders, shipments, abc_df, metrics, steady_start)

    # --- Layer 1 ---
    print("Layer 1: Stockout location...")
    l1 = layer_1_stockout_location(inv_agg, orders, abc_df, steady_start)

    # --- Layer 2 ---
    print("Layer 2a: Allocation...")
    l2a = layer_2a_allocation(orders, shipments, abc_df, steady_start)
    print("Layer 2b: Production...")
    l2b = layer_2b_production(batches, orders, abc_df, steady_start)
    print("Layer 2c: Lead times...")
    l2c = layer_2c_lead_times(shipments, steady_start)
    print("Layer 2d: Replenishment gaps...")
    l2d = layer_2d_replenishment_gaps(orders, abc_df, steady_start)

    # --- Layer 3 ---
    print("Layer 3: Root cause ranking...")
    causes = layer_3_isolation(l0, l1, l2a, l2b, l2c, l2d)

    # --- Report ---
    print_report(l0, l1, l2a, l2b, l2c, l2d, causes, low_inv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
