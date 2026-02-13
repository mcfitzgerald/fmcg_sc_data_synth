# ruff: noqa: E501
"""
Cost Analysis Module — Extracted from diagnose_cost.py for reuse.

Functions:
  - compute_per_sku_cogs: COGS by ABC/route using per-SKU costs
  - compute_logistics_by_route: FTL/LTL transport + handling by route
  - stream_carrying_cost: Echelon-specific warehouse + carrying cost (streaming)
  - compute_cash_to_cash: DIO + channel-weighted DSO - DPO
  - compute_otif: On-Time In-Full decomposition

v0.72.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .loader import (
    DataBundle,
    classify_node,
)

# Route key mapping: (src_echelon, tgt_echelon) -> config route key
ROUTE_KEY_MAP: dict[tuple[str, str], str] = {
    ("Supplier", "Plant"): "supplier_to_plant",
    ("Plant", "RDC"): "plant_to_rdc",
    ("Plant", "Customer DC"): "plant_to_dc",
    ("Plant", "Store"): "plant_to_dc",
    ("RDC", "Customer DC"): "rdc_to_dc",
    ("RDC", "Store"): "rdc_to_dc",
    ("Customer DC", "Store"): "dc_to_store",
}


def compute_per_sku_cogs(bundle: DataBundle) -> dict[str, Any]:
    """Compute COGS using per-SKU costs from products.csv.

    Returns dict with:
        total_cogs, by_abc (DataFrame), by_route (DataFrame)
    """
    ships = bundle.shipments.copy()
    ships["cost_per_case"] = ships["product_id"].map(bundle.sku_cost_map).fillna(9.0)
    ships["cogs"] = ships["quantity"] * ships["cost_per_case"]
    total_cogs = float(ships["cogs"].sum())

    # By ABC
    by_abc = ships.groupby("abc_class").agg(
        cogs=("cogs", "sum"),
        cases=("quantity", "sum"),
    )
    by_abc["avg_cost"] = by_abc["cogs"] / by_abc["cases"].clip(lower=1)
    by_abc["share"] = by_abc["cogs"] / total_cogs

    # By route
    ships["route"] = (
        ships["source_echelon"].astype(str) + " -> "
        + ships["target_echelon"].astype(str)
    )
    by_route = ships.groupby("route").agg(
        cogs=("cogs", "sum"),
        cases=("quantity", "sum"),
    ).sort_values("cogs", ascending=False)
    by_route["share"] = by_route["cogs"] / total_cogs

    return {
        "total_cogs": total_cogs,
        "by_abc": by_abc,
        "by_route": by_route,
    }


def compute_logistics_by_route(bundle: DataBundle) -> dict[str, Any]:
    """Compute logistics cost by route using cost_master config.

    Returns dict with:
        total_logistics, total_transport, total_handling, by_route (DataFrame)
    """
    route_cfg = bundle.cost_master.get("logistics_costs", {}).get("routes", {})
    if not route_cfg:
        return {"total_logistics": 0, "total_transport": 0, "total_handling": 0,
                "by_route": pd.DataFrame()}

    ships = bundle.shipments
    src_ech = ships["source_echelon"].astype(str)
    tgt_ech = ships["target_echelon"].astype(str)

    # Map route keys
    route_keys = pd.Series(
        [ROUTE_KEY_MAP.get((s, t), "") for s, t in zip(src_ech, tgt_ech, strict=False)],
        index=ships.index,
    )

    # Distance lookup
    distances = pd.Series(
        [bundle.dist_map.get((s, t), np.nan)
         for s, t in zip(ships["source_id"].astype(str),
                         ships["target_id"].astype(str), strict=False)],
        index=ships.index,
    )

    # Fill missing distances with route-key average
    route_avg_dist = pd.DataFrame({"rk": route_keys, "d": distances}).groupby("rk")["d"].mean()
    for rk in route_keys.unique():
        if rk and rk in route_avg_dist.index:
            mask = (route_keys == rk) & distances.isna()
            distances[mask] = route_avg_dist[rk]
    distances = distances.fillna(0)

    # Average shipment size per route for FTL allocation
    route_avg_size = pd.DataFrame(
        {"rk": route_keys, "q": ships["quantity"]}
    ).groupby("rk")["q"].mean()

    transport = np.zeros(len(ships), dtype=np.float64)
    handling = np.zeros(len(ships), dtype=np.float64)

    for rk, rcfg in route_cfg.items():
        mask = (route_keys == rk).values
        if not mask.any():
            continue

        handling_rate = rcfg.get("handling_cost_per_case", 0.20)
        handling[mask] = ships.loc[mask, "quantity"].values * handling_rate

        if rcfg.get("mode") == "FTL":
            cost_per_km = rcfg.get("cost_per_km", 1.85)
            avg_size = route_avg_size.get(rk, 1000)
            if avg_size <= 0:
                avg_size = 1000
            transport[mask] = (
                cost_per_km * distances[mask].values / avg_size
                * ships.loc[mask, "quantity"].values
            )
        elif rcfg.get("mode") == "LTL":
            cost_per_case = rcfg.get("cost_per_case", 0.75)
            transport[mask] = ships.loc[mask, "quantity"].values * cost_per_case

    total_transport = float(transport.sum())
    total_handling = float(handling.sum())

    # Build per-route summary
    df = pd.DataFrame({
        "route_key": route_keys,
        "transport": transport,
        "handling": handling,
        "logistics": transport + handling,
        "cases": ships["quantity"].values,
        "distance_km": distances.values,
    })
    by_route = df[df["route_key"] != ""].groupby("route_key").agg(
        transport=("transport", "sum"),
        handling=("handling", "sum"),
        logistics=("logistics", "sum"),
        cases=("cases", "sum"),
        avg_dist=("distance_km", "mean"),
    ).sort_values("logistics", ascending=False)

    return {
        "total_logistics": total_transport + total_handling,
        "total_transport": total_transport,
        "total_handling": total_handling,
        "by_route": by_route,
        "route_cfg": route_cfg,
        # Return arrays for downstream enrichment (channel P&L needs per-shipment logistics)
        "_transport_arr": transport,
        "_handling_arr": handling,
    }


def stream_carrying_cost(bundle: DataBundle) -> dict[str, Any] | None:
    """Stream inventory.parquet to compute carrying cost by echelon.

    Returns dict with:
        total_carrying, total_warehouse, total_inv_value,
        by_echelon (DataFrame), n_days
    Or None if inventory.parquet missing.
    """
    inv_path = bundle.data_dir / "inventory.parquet"
    if not inv_path.exists():
        return None

    wh_rates = bundle.cost_master.get("logistics_costs", {}).get(
        "warehouse_cost_per_case_per_day", {}
    )
    carrying_pct = bundle.cost_master.get("working_capital", {}).get(
        "annual_carrying_cost_pct", 0.25
    )

    default_wh = 0.02
    pf = pq.ParquetFile(inv_path)
    inv_cols = ["day", "node_id", "product_id", "actual_inventory"]

    accum: dict[str, list[float]] = {}
    all_days: set[int] = set()

    for rg_idx in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=inv_cols)
        df = tbl.to_pandas()
        del tbl
        df = df[df["product_id"].str.startswith("SKU-")]
        if df.empty:
            continue

        all_days.update(df["day"].unique().tolist())
        ech_lookup = {nid: classify_node(nid) for nid in df["node_id"].unique()}
        df["echelon"] = df["node_id"].map(ech_lookup)
        df["cost"] = df["product_id"].map(bundle.sku_cost_map).fillna(9.0)
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
        avg_value = total_value / n_days if n_days > 0 else 0
        wh_rate = wh_rates.get(ech, default_wh)
        warehouse_cost = avg_cases * wh_rate * 365
        carrying_cost = avg_value * carrying_pct
        rows.append({
            "echelon": ech,
            "avg_cases": avg_cases,
            "avg_value": avg_value,
            "wh_rate": wh_rate,
            "warehouse_cost": warehouse_cost,
            "carrying_cost": carrying_cost,
        })

    by_echelon = pd.DataFrame(rows).set_index("echelon")
    total_carrying = by_echelon["carrying_cost"].sum()
    total_warehouse = by_echelon["warehouse_cost"].sum()
    total_inv_value = by_echelon["avg_value"].sum()

    return {
        "total_carrying": total_carrying,
        "total_warehouse": total_warehouse,
        "total_inv_value": total_inv_value,
        "by_echelon": by_echelon,
        "n_days": n_days,
        "carrying_pct": carrying_pct,
    }


def compute_cash_to_cash(bundle: DataBundle, inv_value: float = 0.0,
                         total_cogs: float = 0.0) -> dict[str, Any]:
    """Compute Cash-to-Cash cycle: DIO + channel-weighted DSO - DPO.

    Args:
        bundle: DataBundle with shipments, cost master, channel maps.
        inv_value: Average daily inventory value (from streaming or metrics).
        total_cogs: Total COGS for the simulation period.
    """
    working_cap = bundle.cost_master.get("working_capital", {})
    dpo = working_cap.get("dpo_days", 45.0)
    dso_by_channel = working_cap.get("dso_days_by_channel", {})
    sim_days = bundle.sim_days

    # Channel-weighted DSO from demand shipment volumes
    ships = bundle.shipments
    demand_mask = ships["is_demand_endpoint"]
    demand_vols = ships[demand_mask].copy()

    if bundle.channel_map:
        demand_vols["channel"] = demand_vols["target_id"].map(
            bundle.channel_map
        ).fillna("OTHER")
    else:
        demand_vols["channel"] = "OTHER"

    chan_vol = demand_vols.groupby("channel")["quantity"].sum()
    total_vol = chan_vol.sum()

    weighted_dso = 0.0
    dso_breakdown: list[dict[str, Any]] = []
    if total_vol > 0:
        for chan, vol in chan_vol.sort_values(ascending=False).items():
            pct = vol / total_vol
            dso_val = dso_by_channel.get(str(chan), 30)
            contrib = pct * dso_val
            weighted_dso += contrib
            dso_breakdown.append({
                "channel": str(chan), "volume_pct": pct,
                "dso": dso_val, "weighted": contrib,
            })
    else:
        weighted_dso = 30.0

    # DIO
    daily_cogs = total_cogs / sim_days if sim_days > 0 else 1
    dio = inv_value / daily_cogs if daily_cogs > 0 else 365 / 10

    c2c = dio + weighted_dso - dpo

    return {
        "dio": dio,
        "dso": weighted_dso,
        "dpo": dpo,
        "c2c": c2c,
        "dso_breakdown": dso_breakdown,
    }


def compute_otif(bundle: DataBundle) -> dict[str, Any]:
    """Compute On-Time In-Full decomposition by ABC.

    Returns dict with in_full_pct, on_time_pct, otif_pct, by_abc, or empty if
    requested_date is missing.
    """
    orders = bundle.orders
    ships = bundle.shipments

    # Check for requested_date column
    if "requested_date" not in orders.columns or not orders["requested_date"].notna().any():
        return {"available": False}

    ord_agg = orders.groupby(
        ["day", "source_id", "target_id", "product_id"]
    ).agg(
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
    merged["shipped_qty"] = merged["shipped_qty"].fillna(0)
    merged["arrival_day"] = merged["arrival_day"].fillna(9999)

    merged["in_full"] = merged["shipped_qty"] >= merged["ordered_qty"] * 0.99
    merged["on_time"] = merged["arrival_day"] <= merged["requested_date"] + 1
    merged["otif"] = merged["in_full"] & merged["on_time"]

    in_full_pct = float(merged["in_full"].mean())
    on_time_pct = float(merged["on_time"].mean())
    otif_pct = float(merged["otif"].mean())

    merged["abc"] = merged["product_id"].map(bundle.abc_map).fillna("C")
    by_abc = merged.groupby("abc").agg(
        in_full=("in_full", "mean"),
        on_time=("on_time", "mean"),
        otif=("otif", "mean"),
        count=("otif", "count"),
    )

    return {
        "available": True,
        "total_lines": len(merged),
        "in_full_pct": in_full_pct,
        "on_time_pct": on_time_pct,
        "otif_pct": otif_pct,
        "by_abc": by_abc,
    }
