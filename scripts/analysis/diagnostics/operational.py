"""
Layer 2: Operational Health Assessment.

2.1 Inventory Positioning — DOS by echelon x ABC vs configured targets
2.2 Service Level Decomposition — fill rate by ABC x time window
2.3 Production vs Demand Alignment — ratio trends, ABC breakdown, cumulative gap
2.4 SLOB Decomposition — by ABC class, echelon contributions, risk products

v0.66.0: Uses precomputed echelon/demand/ABC columns from DataBundle.
DOS targets read from config via DataBundle.dos_targets (no hardcodes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .loader import (
    ECHELON_ORDER,
    DataBundle,
)

# Thresholds
_PROD_RATIO_LOW = 0.95
_PROD_RATIO_HIGH = 1.05
_FILL_RATE_THRESH = 0.90
_SLOB_RISK_EXCESS = 0.20

# ---------------------------------------------------------------------------
# 2.1 Inventory Positioning (DOS by Echelon x ABC)
# ---------------------------------------------------------------------------

def analyze_inventory_positioning(
    data: DataBundle, window: int = 30
) -> dict[str, Any]:
    """DOS by echelon x ABC class vs configured targets.

    Returns:
        dict with 'matrix' (echelon x ABC DOS), 'targets', 'trend'.
    """
    inv = data.inv_by_echelon
    ships = data.shipments
    # Shipments already FG-filtered; use precomputed is_demand_endpoint
    demand_ships = ships[ships["is_demand_endpoint"]]
    sim_days = data.sim_days

    # Daily demand rate by ABC using precomputed abc_class (for total_dos)
    demand_abc = demand_ships["abc_class"]
    demand_qty = demand_ships["quantity"]
    demand_by_abc: dict[str, float] = {}
    for cls in ("A", "B", "C"):
        cls_dem = demand_qty[demand_abc == cls].sum()
        demand_by_abc[cls] = cls_dem / sim_days if sim_days > 0 else 0

    total_demand_daily = sum(demand_by_abc.values())

    # Per-echelon throughput by ABC for DOS calculation
    # Plant/RDC/DC: use outflow (shipments FROM); Store/Club: use inflow (shipments TO)
    throughput_by_ech_abc: dict[tuple[str, str], float] = {}
    for ech in ECHELON_ORDER:
        if ech in ("Store", "Club"):
            # Stores consume: use demand-endpoint inflow
            ech_ships = demand_ships[demand_ships["target_echelon"] == ech]
        else:
            # Flow echelons: use outflow
            ech_ships = ships[ships["source_echelon"] == ech]
        ech_abc = ech_ships["abc_class"]
        for cls in ("A", "B", "C"):
            cls_qty = ech_ships.loc[ech_abc == cls, "quantity"].sum()
            rate = cls_qty / sim_days if sim_days > 0 else 0
            throughput_by_ech_abc[(ech, cls)] = rate

    # v0.66.0: Config-derived targets (replaces hardcoded values)
    targets = data.dos_targets.by_echelon

    # Latest-day inventory by echelon x ABC
    if len(inv) == 0:
        return {"matrix": {}, "targets": targets, "trend": []}

    max_day = int(inv["day"].max())
    latest = inv[inv["day"] == max_day]

    matrix: dict[str, dict[str, Any]] = {}
    for ech in ECHELON_ORDER:
        ech_data = latest[latest["echelon"] == ech]
        if len(ech_data) == 0:
            continue
        ech_total = float(ech_data["total"].sum())
        row: dict[str, Any] = {"total_inv": ech_total}
        for cls in ("A", "B", "C"):
            cls_inv = float(ech_data[cls].sum())
            daily_thru = throughput_by_ech_abc.get((ech, cls), 0)
            dos = cls_inv / daily_thru if daily_thru > 0 else np.nan
            target = targets.get(ech, {}).get(cls, np.nan)
            vs_target = dos / target if target > 0 and not np.isnan(dos) else np.nan
            row[cls] = {
                "inventory": cls_inv,
                "dos": dos,
                "target_dos": target,
                "vs_target": vs_target,
            }
        total_dos = ech_total / total_demand_daily if total_demand_daily > 0 else np.nan
        row["total_dos"] = total_dos
        matrix[ech] = row

    # DOS trend at 60-day intervals
    trend: list[dict[str, Any]] = []
    days_available = sorted(inv["day"].unique())
    for target_day in range(0, max_day + 1, 60):
        # Find closest available day
        closest = min(days_available, key=lambda d: abs(d - target_day))
        day_data = inv[inv["day"] == closest]
        snap: dict[str, Any] = {"day": int(closest)}
        for ech in ECHELON_ORDER:
            ech_data = day_data[day_data["echelon"] == ech]
            ech_inv = float(ech_data["total"].sum()) if len(ech_data) > 0 else 0
            dos = ech_inv / total_demand_daily if total_demand_daily > 0 else 0
            snap[ech] = round(dos, 1)
        trend.append(snap)

    return {"matrix": matrix, "targets": targets, "trend": trend}


# ---------------------------------------------------------------------------
# 2.2 Service Level Decomposition
# ---------------------------------------------------------------------------

def analyze_service_levels(data: DataBundle, window: int = 30) -> dict[str, Any]:
    """Fill rate by ABC class x time window.

    Returns:
        dict with 'by_period_abc' (heatmap data), 'underperformers' (product list).
    """
    # Shipments already FG-filtered; use precomputed columns
    demand_ships = data.shipments[data.shipments["is_demand_endpoint"]]
    demand_abc = demand_ships["abc_class"]
    max_day = int(demand_ships["creation_day"].max()) if len(demand_ships) > 0 else 365

    # Per-period x ABC fill rate
    by_period_abc: list[dict[str, Any]] = []
    for start in range(0, max_day + 1, window):
        end = start + window - 1
        period_mask = (
            (demand_ships["creation_day"] >= start)
            & (demand_ships["creation_day"] <= end)
        )
        period_ships = demand_ships[period_mask]
        period_abc = demand_abc[period_mask]
        row: dict[str, Any] = {"start_day": start, "end_day": end}
        for cls in ("A", "B", "C"):
            total_qty = period_ships[period_abc == cls]["quantity"].sum()
            row[f"{cls}_volume"] = total_qty
        row["total_volume"] = period_ships["quantity"].sum()
        by_period_abc.append(row)

    # Per-product fill from orders (if status field available)
    underperformers: list[dict[str, Any]] = []
    orders = data.orders
    if "status" in orders.columns and len(orders) > 0:
        # Orders already FG-filtered by loader
        total_by_product = (
            orders.groupby("product_id", observed=True)["quantity"].sum()
        )
        fulfilled = orders[orders["status"] == "CLOSED"]
        fulfilled_by_product = (
            fulfilled.groupby("product_id", observed=True)["quantity"].sum()
        )

        for pid in total_by_product.index:
            total = total_by_product[pid]
            filled = fulfilled_by_product.get(pid, 0)
            fill_rate = filled / total if total > 0 else 1.0
            if fill_rate < _FILL_RATE_THRESH:
                underperformers.append({
                    "product_id": pid,
                    "abc": data.abc_map.get(pid, "?"),
                    "total_ordered": total,
                    "total_fulfilled": filled,
                    "fill_rate": fill_rate,
                })

    underperformers.sort(key=lambda x: x.get("fill_rate", 1.0))

    # Overall fill from metrics
    svc_by_abc = data.metrics.get("service_level_by_abc", {})
    store_svc = data.metrics.get("store_service_level", {}).get("mean", 0)

    return {
        "by_period_abc": by_period_abc,
        "underperformers": underperformers[:20],  # Top 20 worst
        "fill_by_abc": svc_by_abc,
        "store_fill": store_svc,
    }


# ---------------------------------------------------------------------------
# 2.3 Production vs Demand Alignment
# ---------------------------------------------------------------------------

def analyze_production_alignment(
    data: DataBundle, window: int = 30
) -> dict[str, Any]:
    """Production/demand ratio over time, by ABC, cumulative gap.

    Returns:
        dict with 'snapshots', 'abc_breakdown', 'cumulative_gap', 'verdict'.
    """
    fg_batches = data.fg_batches
    # Shipments already FG-filtered; use precomputed is_demand_endpoint
    demand_ships = data.shipments[data.shipments["is_demand_endpoint"]]

    demand_daily = demand_ships.groupby("creation_day")["quantity"].sum()
    prod_daily = fg_batches.groupby("day_produced")["quantity"].sum()

    all_days = sorted(set(demand_daily.index) | set(prod_daily.index))
    demand_s = demand_daily.reindex(all_days, fill_value=0)
    prod_s = prod_daily.reindex(all_days, fill_value=0)

    demand_rolling = demand_s.rolling(window, min_periods=1).mean()
    prod_rolling = prod_s.rolling(window, min_periods=1).mean()

    # 30-day snapshots
    snapshots: list[dict[str, Any]] = []
    for d in range(window, max(all_days) + 1, window):
        if d in demand_rolling.index and d in prod_rolling.index:
            dem = float(demand_rolling.loc[d])
            pro = float(prod_rolling.loc[d])
            ratio = pro / dem if dem > 0 else np.nan
            snapshots.append({
                "day": d,
                "demand_avg": dem,
                "production_avg": pro,
                "ratio": ratio,
                "gap_pct": (ratio - 1) * 100 if not np.isnan(ratio) else np.nan,
            })

    # By ABC using precomputed abc_class
    demand_abc = demand_ships["abc_class"]
    abc_breakdown: dict[str, dict[str, Any]] = {}
    for cls in ("A", "B", "C"):
        cls_prods = {p for p, c in data.abc_map.items() if c == cls}
        cls_dem = demand_ships[demand_abc == cls]["quantity"].sum()
        cls_prod = fg_batches[fg_batches["product_id"].isin(cls_prods)][
            "quantity"
        ].sum()
        ratio = cls_prod / cls_dem if cls_dem > 0 else np.nan
        abc_breakdown[cls] = {
            "total_demand": cls_dem,
            "total_production": cls_prod,
            "ratio": ratio,
            "excess_pct": (ratio - 1) * 100 if not np.isnan(ratio) else np.nan,
        }

    # Cumulative gap
    demand_cum = demand_s.cumsum()
    prod_cum = prod_s.cumsum()
    excess_cum = prod_cum - demand_cum

    # Monotonicity — what % of days does gap widen?
    diffs = excess_cum.diff().dropna()
    monotonic_pct = float((diffs >= 0).mean() * 100)

    cum_snapshots: list[dict[str, Any]] = []
    for d in range(window, max(all_days) + 1, window):
        if d in excess_cum.index:
            cum_snapshots.append({
                "day": d,
                "cum_production": float(prod_cum.loc[d]),
                "cum_demand": float(demand_cum.loc[d]),
                "cum_excess": float(excess_cum.loc[d]),
                "excess_pct": (
                    float(excess_cum.loc[d]) / float(demand_cum.loc[d]) * 100
                    if demand_cum.loc[d] > 0
                    else 0
                ),
            })

    # Overall ratio
    total_prod = prod_s.sum()
    total_dem = demand_s.sum()
    overall_ratio = total_prod / total_dem if total_dem > 0 else np.nan

    if np.isnan(overall_ratio):
        verdict = "NO_DATA"
    elif _PROD_RATIO_LOW <= overall_ratio <= _PROD_RATIO_HIGH:
        verdict = "ALIGNED"
    elif overall_ratio < _PROD_RATIO_LOW:
        verdict = "UNDERPRODUCING"
    else:
        verdict = "OVERPRODUCING"

    return {
        "snapshots": snapshots,
        "abc_breakdown": abc_breakdown,
        "cumulative_gap": cum_snapshots,
        "monotonic_pct": monotonic_pct,
        "overall_ratio": overall_ratio,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 2.4 SLOB Decomposition
# ---------------------------------------------------------------------------

def analyze_slob(data: DataBundle) -> dict[str, Any]:
    """Decompose SLOB by ABC class and echelon.

    Returns:
        dict with 'headline', 'by_abc', 'by_echelon', 'risk_products'.
    """
    headline = data.metrics.get("slob", {}).get("mean", 0)
    inv = data.inv_by_echelon
    fg_batches = data.fg_batches
    # Shipments already FG-filtered; use precomputed columns
    demand_ships = data.shipments[data.shipments["is_demand_endpoint"]]
    demand_abc = demand_ships["abc_class"]

    # SLOB proxy by ABC: cumulative excess production
    by_abc: dict[str, dict[str, Any]] = {}
    for cls in ("A", "B", "C"):
        cls_prods = {p for p, c in data.abc_map.items() if c == cls}
        cls_dem = demand_ships[demand_abc == cls]["quantity"].sum()
        cls_prod = fg_batches[fg_batches["product_id"].isin(cls_prods)][
            "quantity"
        ].sum()
        excess = cls_prod - cls_dem

        # Inventory growth by ABC
        if len(inv) > 0:
            min_day = int(inv["day"].min())
            max_day = int(inv["day"].max())
            first_inv = float(
                inv[inv["day"] == min_day][cls].sum()
            )
            last_inv = float(
                inv[inv["day"] == max_day][cls].sum()
            )
            inv_growth = last_inv - first_inv
            inv_growth_pct = (
                inv_growth / first_inv * 100 if first_inv > 0 else np.nan
            )
        else:
            first_inv = last_inv = inv_growth = 0.0
            inv_growth_pct = np.nan

        by_abc[cls] = {
            "total_demand": cls_dem,
            "total_production": cls_prod,
            "cum_excess": excess,
            "excess_pct": excess / cls_dem * 100 if cls_dem > 0 else 0,
            "inv_start": first_inv,
            "inv_end": last_inv,
            "inv_growth": inv_growth,
            "inv_growth_pct": inv_growth_pct,
        }

    # SLOB contribution by echelon (inventory growth)
    by_echelon: dict[str, dict[str, Any]] = {}
    if len(inv) > 0:
        min_day = int(inv["day"].min())
        max_day = int(inv["day"].max())
        for ech in ECHELON_ORDER:
            ech_first = inv[(inv["day"] == min_day) & (inv["echelon"] == ech)]
            ech_last = inv[(inv["day"] == max_day) & (inv["echelon"] == ech)]
            start_val = float(ech_first["total"].sum()) if len(ech_first) > 0 else 0
            end_val = float(ech_last["total"].sum()) if len(ech_last) > 0 else 0
            growth = end_val - start_val
            growth_pct = growth / start_val * 100 if start_val > 0 else np.nan
            by_echelon[ech] = {
                "inv_start": start_val,
                "inv_end": end_val,
                "inv_growth": growth,
                "inv_growth_pct": growth_pct,
            }

    # Top SLOB-risk products: highest inventory growth
    risk_products: list[dict[str, Any]] = []
    if len(inv) > 0:
        # Use per-product demand vs production
        prod_demand = demand_ships.groupby(
            "product_id", observed=True,
        )["quantity"].sum()
        prod_production = fg_batches.groupby(
            "product_id", observed=True,
        )["quantity"].sum()

        for pid in prod_production.index:
            dem = prod_demand.get(pid, 0)
            pro = prod_production[pid]
            if dem > 0 and (pro - dem) / dem > _SLOB_RISK_EXCESS:
                risk_products.append({
                    "product_id": pid,
                    "abc": data.abc_map.get(pid, "?"),
                    "demand": dem,
                    "production": pro,
                    "excess_pct": (pro - dem) / dem * 100,
                })

    risk_products.sort(key=lambda x: x["excess_pct"], reverse=True)

    min_day = int(inv["day"].min()) if len(inv) > 0 else 0

    return {
        "headline": headline,
        "by_abc": by_abc,
        "by_echelon": by_echelon,
        "risk_products": risk_products[:15],
        "min_day": min_day,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_inventory_positioning(results: dict[str, Any], width: int = 78) -> str:
    """Format inventory positioning matrix as text."""
    lines = [
        f"\n{'[2.1] Inventory Positioning (DOS by Echelon x ABC)':=^{width}}",
        "",
        f"  {'Echelon':<14}  {'Total Inv':>14}  {'Total DOS':>10}"
        f"  {'A DOS':>8}  {'B DOS':>8}  {'C DOS':>8}",
        f"  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}",
    ]
    matrix = results["matrix"]
    for ech in ECHELON_ORDER:
        if ech not in matrix:
            continue
        m = matrix[ech]
        a_dos = m.get("A", {}).get("dos", np.nan)
        b_dos = m.get("B", {}).get("dos", np.nan)
        c_dos = m.get("C", {}).get("dos", np.nan)
        t_dos = m.get("total_dos", np.nan)
        lines.append(
            f"  {ech:<14}  {m['total_inv']:>14,.0f}"
            f"  {(f'{t_dos:.1f}' if not np.isnan(t_dos) else 'N/A'):>10}"
            f"  {(f'{a_dos:.1f}' if not np.isnan(a_dos) else 'N/A'):>8}"
            f"  {(f'{b_dos:.1f}' if not np.isnan(b_dos) else 'N/A'):>8}"
            f"  {(f'{c_dos:.1f}' if not np.isnan(c_dos) else 'N/A'):>8}"
        )

    # Target comparison
    lines.append("\n  Configured targets (DOS) — from simulation_config.json:")
    targets = results["targets"]
    for ech in ECHELON_ORDER:
        if ech in targets:
            t = targets[ech]
            lines.append(
                f"    {ech:<14}  A={t['A']:.1f}"
                f"  B={t['B']:.1f}  C={t['C']:.1f}"
            )

    # DOS trend
    if results["trend"]:
        lines.append("\n  DOS trend by echelon (total inventory / total demand rate):")
        header = f"  {'Day':>6}"
        for ech in ECHELON_ORDER:
            header += f"  {ech:>12}"
        lines.append(header)
        lines.append(f"  {'-'*6}" + f"  {'-'*12}" * len(ECHELON_ORDER))
        for snap in results["trend"]:
            row = f"  {snap['day']:>6}"
            for ech in ECHELON_ORDER:
                val = snap.get(ech, 0)
                row += f"  {val:>12.1f}"
            lines.append(row)

    return "\n".join(lines)


def format_service_levels(results: dict[str, Any], width: int = 78) -> str:
    """Format service level decomposition as text."""
    lines = [
        f"\n{'[2.2] Service Level Decomposition':=^{width}}",
        "",
        f"  Store fill rate: {results['store_fill']:.2%}",
    ]

    fill = results["fill_by_abc"]
    if fill:
        lines.append("  Fill by ABC class (from metrics):")
        for cls in ("A", "B", "C"):
            val = fill.get(cls, fill.get(cls, 0))
            lines.append(f"    {cls}: {val:.2%}")

    # Period volumes
    if results["by_period_abc"]:
        lines.append("\n  Demand volume by period and ABC class:")
        lines.append(
            f"  {'Period':>12}  {'A Volume':>14}  {'B Volume':>14}"
            f"  {'C Volume':>14}  {'Total':>14}"
        )
        lines.append(
            f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}"
        )
        for p in results["by_period_abc"]:
            lines.append(
                f"  {p['start_day']:>3}-{p['end_day']:>3}d"
                f"      {p.get('A_volume', 0):>14,.0f}"
                f"  {p.get('B_volume', 0):>14,.0f}"
                f"  {p.get('C_volume', 0):>14,.0f}"
                f"  {p.get('total_volume', 0):>14,.0f}"
            )

    # Underperformers
    if results["underperformers"]:
        lines.append("\n  Underperforming products (fill < 90%):")
        lines.append(
            f"  {'Product':>20}  {'ABC':>4}  {'Ordered':>12}"
            f"  {'Fulfilled':>12}  {'Fill%':>8}"
        )
        lines.append(f"  {'-'*20}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*8}")
        for p in results["underperformers"][:10]:
            lines.append(
                f"  {p['product_id']:>20}  {p['abc']:>4}"
                f"  {p['total_ordered']:>12,.0f}  {p['total_fulfilled']:>12,.0f}"
                f"  {p['fill_rate']:>7.1%}"
            )
    else:
        lines.append("\n  No products below 90% fill rate.")

    return "\n".join(lines)


def format_production_alignment(results: dict[str, Any], width: int = 78) -> str:
    """Format production vs demand alignment as text."""
    lines = [
        f"\n{'[2.3] Production vs Demand Alignment':=^{width}}",
        f"  Overall ratio: {results['overall_ratio']:.3f}"
        f"  Verdict: {results['verdict']}",
        "",
        f"  {'Day':>6}  {'Demand/day':>14}  {'Production/day':>14}"
        f"  {'Ratio':>8}  {'Gap%':>8}",
        f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*8}",
    ]
    for s in results["snapshots"]:
        r = s.get("ratio", np.nan)
        g = s.get("gap_pct", np.nan)
        lines.append(
            f"  {s['day']:>6}  {s['demand_avg']:>14,.0f}  {s['production_avg']:>14,.0f}"
            f"  {(f'{r:.3f}' if not np.isnan(r) else 'N/A'):>8}"
            f"  {(f'{g:+.1f}%' if not np.isnan(g) else 'N/A'):>8}"
        )

    # ABC breakdown
    lines.append("\n  By ABC class (full simulation):")
    lines.append(
        f"  {'Class':>6}  {'Demand':>14}  {'Production':>14}"
        f"  {'Ratio':>8}  {'Excess%':>8}"
    )
    lines.append(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*8}")
    for cls in ("A", "B", "C"):
        b = results["abc_breakdown"].get(cls, {})
        r = b.get("ratio", np.nan)
        e = b.get("excess_pct", np.nan)
        lines.append(
            f"  {cls:>6}  {b.get('total_demand', 0):>14,.0f}"
            f"  {b.get('total_production', 0):>14,.0f}"
            f"  {(f'{r:.3f}' if not np.isnan(r) else 'N/A'):>8}"
            f"  {(f'{e:+.1f}%' if not np.isnan(e) else 'N/A'):>8}"
        )

    # Cumulative gap
    lines.append("\n  Cumulative production - demand:")
    lines.append(f"  Monotonicity: {results['monotonic_pct']:.1f}% of days gap widened")
    lines.append(
        f"\n  {'Day':>6}  {'Cum Prod':>16}  {'Cum Demand':>16}"
        f"  {'Cum Excess':>14}  {'Excess%':>8}"
    )
    lines.append(f"  {'-'*6}  {'-'*16}  {'-'*16}  {'-'*14}  {'-'*8}")
    for s in results["cumulative_gap"]:
        lines.append(
            f"  {s['day']:>6}  {s['cum_production']:>16,.0f}"
            f"  {s['cum_demand']:>16,.0f}  {s['cum_excess']:>+14,.0f}"
            f"  {s['excess_pct']:>7.1f}%"
        )

    return "\n".join(lines)


def format_slob(results: dict[str, Any], width: int = 78) -> str:
    """Format SLOB decomposition as text."""
    lines = [
        f"\n{'[2.4] SLOB Decomposition':=^{width}}",
        f"  Headline SLOB: {results['headline']:.1%}",
    ]
    min_day = results.get("min_day", 0)
    if min_day > 0:
        lines.append(
            f"  Note: warm-start baseline (first inv snapshot at day {min_day})"
        )
    lines.append("")

    # By ABC
    lines.append("  SLOB drivers by ABC class:")
    lines.append(
        f"  {'Class':>6}  {'Demand':>14}  {'Production':>14}"
        f"  {'Excess%':>8}  {'Inv Growth':>12}  {'Grw%':>8}"
    )
    lines.append(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*12}  {'-'*8}")
    for cls in ("A", "B", "C"):
        b = results["by_abc"].get(cls, {})
        ep = b.get("excess_pct", 0)
        gp = b.get("inv_growth_pct", np.nan)
        lines.append(
            f"  {cls:>6}  {b.get('total_demand', 0):>14,.0f}"
            f"  {b.get('total_production', 0):>14,.0f}"
            f"  {ep:>+7.1f}%"
            f"  {b.get('inv_growth', 0):>+12,.0f}"
            f"  {(f'{gp:+.1f}%' if not np.isnan(gp) else 'N/A'):>8}"
        )

    # By echelon
    if results["by_echelon"]:
        lines.append("\n  Inventory growth by echelon:")
        lines.append(
            f"  {'Echelon':<14}  {'Start':>14}  {'End':>14}"
            f"  {'Growth':>12}  {'Growth%':>8}"
        )
        lines.append(
            f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*8}"
        )
        for ech in ECHELON_ORDER:
            e = results["by_echelon"].get(ech, {})
            if not e:
                continue
            gp = e.get("inv_growth_pct", np.nan)
            lines.append(
                f"  {ech:<14}  {e.get('inv_start', 0):>14,.0f}"
                f"  {e.get('inv_end', 0):>14,.0f}"
                f"  {e.get('inv_growth', 0):>+12,.0f}"
                f"  {(f'{gp:+.1f}%' if not np.isnan(gp) else 'N/A'):>8}"
            )

    # Risk products
    if results["risk_products"]:
        lines.append("\n  Top SLOB-risk products (production > demand + 20%):")
        lines.append(
            f"  {'Product':>20}  {'ABC':>4}  {'Demand':>12}"
            f"  {'Production':>12}  {'Excess%':>8}"
        )
        lines.append(f"  {'-'*20}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*8}")
        for p in results["risk_products"][:10]:
            lines.append(
                f"  {p['product_id']:>20}  {p['abc']:>4}"
                f"  {p['demand']:>12,.0f}  {p['production']:>12,.0f}"
                f"  {p['excess_pct']:>+7.1f}%"
            )

    return "\n".join(lines)
