# ruff: noqa: E501
"""
Commercial Analysis Module — Channel P&L, margins, concentration risk.

Functions:
  - compute_channel_pnl: Revenue, COGS, logistics, margin by channel
  - compute_cost_to_serve: $/case by channel
  - compute_margin_by_abc: Margin $/case by ABC class
  - compute_fill_by_abc_channel: Fill rate cross-tab (NEW Q4)
  - compute_concentration_risk: Pareto analysis (NEW Q12)
  - compute_tail_sku_drag: Bottom-20% cost analysis (NEW Q13)

v0.72.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .loader import DataBundle


def compute_channel_pnl(
    bundle: DataBundle,
    logistics_arr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Revenue, COGS, logistics cost, and margin by channel.

    Args:
        bundle: DataBundle with shipments, cost maps, channel maps.
        logistics_arr: Per-shipment logistics cost array (from cost_analysis).
            If None, logistics is estimated as 0.

    Returns dict with by_channel (DataFrame), total_revenue, total_margin,
    target_margins.
    """
    ships = bundle.shipments
    demand_mask = ships["is_demand_endpoint"]
    demand_ships = ships[demand_mask].copy()

    # Channel assignment
    if bundle.channel_map:
        demand_ships["channel"] = demand_ships["target_id"].map(
            bundle.channel_map
        ).fillna("OTHER")
    else:
        demand_ships["channel"] = "OTHER"

    # COGS
    demand_ships["cost_per_case"] = demand_ships["product_id"].map(
        bundle.sku_cost_map
    ).fillna(9.0)
    demand_ships["cogs"] = demand_ships["quantity"] * demand_ships["cost_per_case"]

    # Revenue
    demand_ships["price_per_case"] = demand_ships["product_id"].map(
        bundle.sku_price_map
    ).fillna(0)
    demand_ships["revenue"] = demand_ships["quantity"] * demand_ships["price_per_case"]

    # Logistics (use provided array if available, only for demand rows)
    if logistics_arr is not None:
        demand_ships["logistics"] = logistics_arr[demand_mask.values]
    else:
        demand_ships["logistics"] = 0.0

    # Channel aggregation
    by_channel = demand_ships.groupby("channel").agg(
        cases=("quantity", "sum"),
        revenue=("revenue", "sum"),
        cogs=("cogs", "sum"),
        logistics=("logistics", "sum"),
    )
    by_channel["margin"] = by_channel["revenue"] - by_channel["cogs"] - by_channel["logistics"]
    by_channel["margin_pct"] = (
        by_channel["margin"] / by_channel["revenue"].clip(lower=1) * 100
    )
    by_channel = by_channel.sort_values("revenue", ascending=False)

    target_margins = {
        ch: info.get("margin_pct", 0)
        for ch, info in bundle.channel_econ.items()
    }

    total_revenue = float(by_channel["revenue"].sum())
    total_margin = float(by_channel["margin"].sum())

    return {
        "by_channel": by_channel,
        "total_revenue": total_revenue,
        "total_margin": total_margin,
        "overall_margin_pct": total_margin / total_revenue * 100 if total_revenue > 0 else 0,
        "target_margins": target_margins,
    }


def compute_cost_to_serve(
    bundle: DataBundle,
    logistics_arr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cost-to-serve by channel: COGS + logistics per case.

    Returns DataFrame indexed by channel with columns:
        cases, cogs, logistics, total_cost, cost_per_case, share
    """
    ships = bundle.shipments
    # Include shipments to stores and customer DCs
    ech_mask = ships["target_echelon"].isin(["Store", "Customer DC"])
    store_ships = ships[ech_mask].copy()

    if bundle.channel_map:
        store_ships["channel"] = store_ships["target_id"].map(
            bundle.channel_map
        ).fillna("OTHER")
    else:
        store_ships["channel"] = "OTHER"

    store_ships["cost_per_case"] = store_ships["product_id"].map(
        bundle.sku_cost_map
    ).fillna(9.0)
    store_ships["cogs"] = store_ships["quantity"] * store_ships["cost_per_case"]

    if logistics_arr is not None:
        store_ships["logistics"] = logistics_arr[ech_mask.values]
    else:
        store_ships["logistics"] = 0.0

    result = store_ships.groupby("channel").agg(
        cases=("quantity", "sum"),
        cogs=("cogs", "sum"),
        logistics=("logistics", "sum"),
    )
    result["total_cost"] = result["cogs"] + result["logistics"]
    result["cost_per_case"] = result["total_cost"] / result["cases"].clip(lower=1)
    total = result["total_cost"].sum()
    result["share"] = result["total_cost"] / total if total > 0 else 0
    return result.sort_values("total_cost", ascending=False)


def compute_margin_by_abc(
    bundle: DataBundle,
    logistics_arr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Margin per case by ABC class from demand-endpoint shipments.

    Returns DataFrame indexed by ABC with columns:
        revenue, margin, cases, margin_pct, margin_per_case
    """
    ships = bundle.shipments
    demand_mask = ships["is_demand_endpoint"]
    demand_ships = ships[demand_mask].copy()

    demand_ships["cost_per_case"] = demand_ships["product_id"].map(
        bundle.sku_cost_map
    ).fillna(9.0)
    demand_ships["cogs"] = demand_ships["quantity"] * demand_ships["cost_per_case"]
    demand_ships["price_per_case"] = demand_ships["product_id"].map(
        bundle.sku_price_map
    ).fillna(0)
    demand_ships["revenue"] = demand_ships["quantity"] * demand_ships["price_per_case"]

    if logistics_arr is not None:
        demand_ships["logistics"] = logistics_arr[demand_mask.values]
    else:
        demand_ships["logistics"] = 0.0

    demand_ships["margin"] = (
        demand_ships["revenue"] - demand_ships["cogs"] - demand_ships["logistics"]
    )
    demand_ships["abc"] = demand_ships["abc_class"]

    result = demand_ships.groupby("abc", observed=True).agg(
        revenue=("revenue", "sum"),
        margin=("margin", "sum"),
        cases=("quantity", "sum"),
    )
    result["margin_pct"] = result["margin"] / result["revenue"].clip(lower=1) * 100
    result["margin_per_case"] = result["margin"] / result["cases"].clip(lower=1)
    return result


def compute_fill_by_abc_channel(bundle: DataBundle) -> pd.DataFrame:
    """Fill rate cross-tab by ABC x Channel (NEW Q4).

    Uses orders vs shipments grouped by ABC and target channel.

    Returns DataFrame with ABC as rows, channels as columns, values = fill %.
    """
    orders = bundle.orders

    # Map orders to channel via target_id
    if bundle.channel_map:
        ord_channel = orders["target_id"].map(bundle.channel_map).fillna("OTHER")
    else:
        ord_channel = pd.Series("OTHER", index=orders.index)
    ord_abc = orders["product_id"].map(bundle.abc_map).fillna("C")

    # Total ordered by (ABC, channel)
    orders_grouped = orders.assign(
        abc=ord_abc, channel=ord_channel
    ).groupby(["abc", "channel"], observed=True)["quantity"].sum()

    # Fulfilled (CLOSED) orders
    closed = orders[orders["status"] == "CLOSED"]
    if bundle.channel_map:
        closed_channel = closed["target_id"].map(bundle.channel_map).fillna("OTHER")
    else:
        closed_channel = pd.Series("OTHER", index=closed.index)
    closed_abc = closed["product_id"].map(bundle.abc_map).fillna("C")

    closed_grouped = closed.assign(
        abc=closed_abc, channel=closed_channel
    ).groupby(["abc", "channel"], observed=True)["quantity"].sum()

    # Fill rate = fulfilled / ordered
    fill_rate = (closed_grouped / orders_grouped).fillna(0)
    fill_matrix = fill_rate.unstack(fill_value=0)

    return fill_matrix


def compute_concentration_risk(bundle: DataBundle) -> dict[str, Any]:
    """Pareto analysis for SKU concentration risk (NEW Q12).

    Returns dict with:
        top20_volume_pct, top20_inv_value_pct, single_source_count
    """
    ships = bundle.shipments

    # Volume Pareto
    vol_by_sku = ships.groupby("product_id", observed=True)["quantity"].sum().sort_values(ascending=False)
    total_vol = vol_by_sku.sum()
    n_skus = len(vol_by_sku)
    top20_n = max(1, int(n_skus * 0.2))
    top20_vol = vol_by_sku.iloc[:top20_n].sum()
    top20_volume_pct = top20_vol / total_vol * 100 if total_vol > 0 else 0

    # Inventory value Pareto (from latest inventory snapshot)
    inv = bundle.inv_by_echelon
    if not inv.empty:
        # We don't have per-SKU inventory in inv_by_echelon, so estimate from
        # average daily shipment value as proxy for inventory value share
        vol_by_sku_value = vol_by_sku.copy()
        vol_by_sku_value.index = vol_by_sku_value.index.astype(str)
        inv_value = vol_by_sku_value * vol_by_sku_value.index.map(
            lambda x: bundle.sku_cost_map.get(str(x), 9.0)
        )
        inv_value = inv_value.sort_values(ascending=False)
        total_inv_val = inv_value.sum()
        top20_inv_val = inv_value.iloc[:top20_n].sum()
        top20_inv_value_pct = (
            top20_inv_val / total_inv_val * 100 if total_inv_val > 0 else 0
        )
    else:
        top20_inv_value_pct = 0.0

    # Single-source products (made at only 1 plant)
    fg_batches = bundle.fg_batches
    plants_per_product = fg_batches.groupby("product_id", observed=True)["plant_id"].nunique()
    single_source = int((plants_per_product == 1).sum())

    return {
        "n_skus": n_skus,
        "top20_n": top20_n,
        "top20_volume_pct": top20_volume_pct,
        "top20_inv_value_pct": top20_inv_value_pct,
        "single_source_count": single_source,
        "total_products": len(plants_per_product),
    }


def compute_tail_sku_drag(bundle: DataBundle) -> dict[str, Any]:
    """Bottom-20% SKU analysis: volume vs carrying cost share (NEW Q13).

    Returns dict with:
        n_tail_skus, tail_volume_pct, tail_inv_value_pct,
        tail_avg_turns, tail_changeovers_per_year
    """
    ships = bundle.shipments
    sim_days = bundle.sim_days

    vol_by_sku = ships.groupby("product_id", observed=True)["quantity"].sum().sort_values(ascending=False)
    total_vol = vol_by_sku.sum()
    n_skus = len(vol_by_sku)
    bottom20_n = max(1, int(n_skus * 0.2))
    tail_skus = set(vol_by_sku.iloc[-bottom20_n:].index.astype(str))

    tail_vol = vol_by_sku.iloc[-bottom20_n:].sum()
    tail_volume_pct = tail_vol / total_vol * 100 if total_vol > 0 else 0

    # Inventory value share (proxy from shipped volume x cost)
    vol_value = vol_by_sku * vol_by_sku.index.map(
        lambda x: bundle.sku_cost_map.get(str(x), 9.0)
    )
    total_value = vol_value.sum()
    tail_value = vol_value.iloc[-bottom20_n:].sum()
    tail_inv_value_pct = tail_value / total_value * 100 if total_value > 0 else 0

    # Tail turns (annualized)
    annual_factor = 365 / sim_days if sim_days > 0 else 1
    # Use average system DOS from metrics as proxy
    avg_turns = bundle.metrics.get("inventory_turns", {}).get("mean", 10)
    # Tail items turn slower — estimate from volume share vs value share
    tail_avg_turns = avg_turns * (tail_volume_pct / tail_inv_value_pct) if tail_inv_value_pct > 0 else 0

    # Changeover frequency for tail SKUs
    fg_batches = bundle.fg_batches
    tail_batches = fg_batches[fg_batches["product_id"].isin(tail_skus)]
    tail_changeovers = len(tail_batches)
    tail_changeovers_year = tail_changeovers * annual_factor

    # Compare with A-item changeovers
    a_skus = {p for p, c in bundle.abc_map.items() if c == "A"}
    a_batches = fg_batches[fg_batches["product_id"].isin(a_skus)]
    a_changeovers_year = len(a_batches) * annual_factor / max(len(a_skus), 1)
    tail_changeovers_per_sku = tail_changeovers_year / max(len(tail_skus), 1)

    return {
        "n_tail_skus": len(tail_skus),
        "tail_volume_pct": tail_volume_pct,
        "tail_inv_value_pct": tail_inv_value_pct,
        "tail_avg_turns": tail_avg_turns,
        "tail_changeovers_per_sku_year": tail_changeovers_per_sku,
        "a_changeovers_per_sku_year": a_changeovers_year,
    }
