"""
Layer 1: First Principles Physics Validation.

1.1 System Mass Balance — production = consumption + delta_inventory + shrinkage
1.2 Echelon Flow Conservation — inflow vs outflow per echelon (waterfall)
1.3 Little's Law — implied cycle time vs configured lead time

v0.66.0: Precomputed echelon/demand columns from DataBundle.
Flow conservation separates demand-endpoint DCs from flow DCs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .loader import (
    ECHELON_ORDER,
    DataBundle,
    classify_node,
    is_finished_good,
)

# Thresholds
_IMBALANCE_MINOR = 5.0
_IMBALANCE_VIOLATION = 15.0
_FLOW_STABLE_PCT = 5
_LITTLE_ALIGNED = 2.0
_LITTLE_ELEVATED = 3.0
_DEFAULT_LEAD_TIME = 3.0

# ---------------------------------------------------------------------------
# 1.1 System Mass Balance
# ---------------------------------------------------------------------------

def analyze_mass_balance(data: DataBundle, window: int = 30) -> dict[str, Any]:
    """Check production = consumption + delta_inventory per period.

    Returns:
        dict with 'periods' (list of period dicts), 'verdict'.
    """
    fg_batches = data.fg_batches
    # Shipments are already FG-filtered by loader; use precomputed is_demand_endpoint
    demand_ships = data.shipments[data.shipments["is_demand_endpoint"]]
    fg_returns = data.returns[data.returns["product_id"].apply(is_finished_good)]

    inv = data.inv_by_echelon
    max_day = max(
        int(fg_batches["day_produced"].max()) if len(fg_batches) > 0 else 0,
        int(demand_ships["creation_day"].max()) if len(demand_ships) > 0 else 0,
    )

    periods: list[dict[str, Any]] = []
    worst_imbalance = 0.0

    for start in range(0, max_day + 1, window):
        end = start + window - 1

        # Production in period
        period_prod = fg_batches[
            (fg_batches["day_produced"] >= start)
            & (fg_batches["day_produced"] <= end)
        ]["quantity"].sum()

        # Consumption (demand-endpoint shipments by creation_day)
        period_cons = demand_ships[
            (demand_ships["creation_day"] >= start)
            & (demand_ships["creation_day"] <= end)
        ]["quantity"].sum()

        # Returns in period
        period_returns = 0.0
        if len(fg_returns) > 0:
            period_returns = fg_returns[
                (fg_returns["day"] >= start) & (fg_returns["day"] <= end)
            ]["quantity"].sum()

        # Inventory delta: find closest available days in streamed data
        inv_start_data = inv[inv["day"] <= start].groupby("day")["total"].sum()
        inv_end_data = inv[inv["day"] <= end].groupby("day")["total"].sum()

        inv_at_start = inv_start_data.iloc[-1] if len(inv_start_data) > 0 else 0.0
        inv_at_end = inv_end_data.iloc[-1] if len(inv_end_data) > 0 else 0.0
        delta_inv = inv_at_end - inv_at_start

        # Imbalance = production - consumption - delta_inv + returns
        # (returns add back to system)
        imbalance = period_prod - period_cons - delta_inv + period_returns
        imbalance_pct = (
            abs(imbalance) / period_prod * 100 if period_prod > 0 else 0.0
        )
        worst_imbalance = max(worst_imbalance, imbalance_pct)

        periods.append({
            "start_day": start,
            "end_day": end,
            "production": period_prod,
            "consumption": period_cons,
            "delta_inv": delta_inv,
            "returns": period_returns,
            "imbalance": imbalance,
            "imbalance_pct": imbalance_pct,
        })

    # Verdict
    if worst_imbalance < _IMBALANCE_MINOR:
        verdict = "BALANCED"
    elif worst_imbalance < _IMBALANCE_VIOLATION:
        verdict = "MINOR_DRIFT"
    else:
        verdict = "VIOLATION"

    return {
        "periods": periods,
        "worst_imbalance_pct": worst_imbalance,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 1.2 Echelon Flow Conservation (Waterfall)
# ---------------------------------------------------------------------------

def analyze_flow_conservation(
    data: DataBundle, snapshot_interval: int = 60
) -> dict[str, Any]:
    """Compute inflow/outflow/accumulation per echelon.

    v0.66.0: Separates demand-endpoint DCs (ECOM-FC, DTC-FC) from flow DCs.
    Reports both raw and adjusted Customer DC imbalance.

    Returns:
        dict with 'echelons' (per-echelon flow stats), 'snapshots', 'verdicts',
        'dc_adjustment' (ECOM/DTC endpoint correction).
    """
    # Shipments already FG-filtered and enriched with echelon columns
    ships = data.shipments

    fg_batches = data.fg_batches
    max_day = int(ships["creation_day"].max()) if len(ships) > 0 else 365
    sim_days = data.sim_days

    # Build daily inflow/outflow per echelon using precomputed columns
    echelon_flows: dict[str, dict[str, float]] = {}

    # Demand-endpoint shipments represent POS consumption
    demand_ships = ships[ships["is_demand_endpoint"]]

    for ech in ECHELON_ORDER:
        if ech == "Plant":
            # Inflow = production
            inflow = fg_batches["quantity"].sum()
            # Outflow = shipments from plants
            outflow = ships[
                ships["source_echelon"] == "Plant"
            ]["quantity"].sum()
        elif ech in ("Store", "Club"):
            # Inflow = shipments TO stores/clubs
            inflow = ships[
                ships["target_echelon"] == ech
            ]["quantity"].sum()
            # Outflow = POS consumption (demand-endpoint shipments)
            # Stores consume via POS, they don't ship outbound
            ech_demand = demand_ships[
                demand_ships["target_echelon"] == ech
            ]["quantity"].sum()
            outflow = ech_demand
        else:
            # Inflow = shipments TO this echelon
            inflow = ships[
                ships["target_echelon"] == ech
            ]["quantity"].sum()
            # Outflow = shipments FROM this echelon
            outflow = ships[
                ships["source_echelon"] == ech
            ]["quantity"].sum()

        daily_inflow = inflow / sim_days if sim_days > 0 else 0
        daily_outflow = outflow / sim_days if sim_days > 0 else 0
        delta = daily_inflow - daily_outflow
        throughput = max(daily_inflow, daily_outflow)
        delta_pct = abs(delta) / throughput * 100 if throughput > 0 else 0

        echelon_flows[ech] = {
            "total_inflow": inflow,
            "total_outflow": outflow,
            "daily_inflow": daily_inflow,
            "daily_outflow": daily_outflow,
            "daily_delta": delta,
            "delta_pct_of_throughput": delta_pct,
        }

    # v0.66.0: Compute DC adjustment for ECOM-FC/DTC-FC demand endpoints.
    # These nodes are classified as "Customer DC" but are demand endpoints —
    # their inflow is POS consumption, not warehouse accumulation.
    tgt = ships["target_id"].str
    src = ships["source_id"].str
    ecom_dtc_in = (
        tgt.startswith("ECOM-FC-") | tgt.startswith("DTC-FC-")
    )
    ecom_dtc_out = (
        src.startswith("ECOM-FC-") | src.startswith("DTC-FC-")
    )
    ecom_dtc_inflow = ships[ecom_dtc_in]["quantity"].sum()
    ecom_dtc_outflow = ships[ecom_dtc_out]["quantity"].sum()
    ecom_dtc_net = (
        (ecom_dtc_inflow - ecom_dtc_outflow) / sim_days
        if sim_days > 0 else 0
    )

    dc_raw = echelon_flows.get("Customer DC", {})
    dc_raw_delta = dc_raw.get("daily_delta", 0)
    dc_adjusted_delta = dc_raw_delta - ecom_dtc_net
    dc_raw_inflow = dc_raw.get("daily_inflow", 0)
    ecom_daily = (
        ecom_dtc_inflow / sim_days if sim_days > 0 else 0
    )
    dc_adj_inflow = dc_raw_inflow - ecom_daily
    dc_adjusted_pct = (
        abs(dc_adjusted_delta) / dc_adj_inflow * 100
        if dc_adj_inflow > 0 else 0
    )

    dc_adjustment = {
        "ecom_dtc_daily_inflow": ecom_daily,
        "ecom_dtc_daily_consumption": ecom_dtc_net,
        "raw_dc_delta": dc_raw_delta,
        "raw_dc_delta_pct": dc_raw.get("delta_pct_of_throughput", 0),
        "adjusted_dc_delta": dc_adjusted_delta,
        "adjusted_dc_delta_pct": dc_adjusted_pct,
    }

    # Snapshots at intervals
    snapshots: list[dict[str, Any]] = []
    window = 30  # Rolling window for rate calculation

    for snap_day in range(snapshot_interval, max_day + 1, snapshot_interval):
        snap_start = max(0, snap_day - window)
        snap_ships = ships[
            (ships["creation_day"] >= snap_start)
            & (ships["creation_day"] <= snap_day)
        ]
        snap_batches = fg_batches[
            (fg_batches["day_produced"] >= snap_start)
            & (fg_batches["day_produced"] <= snap_day)
        ]
        days_in_window = snap_day - snap_start + 1

        snap_demand = snap_ships[snap_ships["is_demand_endpoint"]]
        row: dict[str, Any] = {"day": snap_day}
        for ech in ECHELON_ORDER:
            if ech == "Plant":
                inf = snap_batches["quantity"].sum() / days_in_window
                plant_qty = snap_ships[
                    snap_ships["source_echelon"] == "Plant"
                ]["quantity"].sum()
                outf = plant_qty / days_in_window
            elif ech in ("Store", "Club"):
                inf = (
                    snap_ships[
                        snap_ships["target_echelon"] == ech
                    ]["quantity"].sum()
                    / days_in_window
                )
                outf = (
                    snap_demand[
                        snap_demand["target_echelon"] == ech
                    ]["quantity"].sum()
                    / days_in_window
                )
            else:
                inf = (
                    snap_ships[
                        snap_ships["target_echelon"] == ech
                    ]["quantity"].sum()
                    / days_in_window
                )
                outf = (
                    snap_ships[
                        snap_ships["source_echelon"] == ech
                    ]["quantity"].sum()
                    / days_in_window
                )
            row[f"{ech}_inflow"] = inf
            row[f"{ech}_outflow"] = outf
            row[f"{ech}_delta"] = inf - outf
        snapshots.append(row)

    # Verdicts per echelon
    verdicts: dict[str, str] = {}
    for ech, flows in echelon_flows.items():
        pct = flows["delta_pct_of_throughput"]
        # For Customer DC, use adjusted percentage
        if ech == "Customer DC":
            pct = dc_adjusted_pct
        if pct < _FLOW_STABLE_PCT:
            verdicts[ech] = "STABLE"
        elif flows["daily_delta"] > 0:
            verdicts[ech] = "ACCUMULATING"
        else:
            verdicts[ech] = "DRAINING"

    return {
        "echelons": echelon_flows,
        "snapshots": snapshots,
        "verdicts": verdicts,
        "dc_adjustment": dc_adjustment,
    }


# ---------------------------------------------------------------------------
# 1.3 Little's Law Validation
# ---------------------------------------------------------------------------

def analyze_littles_law(data: DataBundle) -> dict[str, Any]:
    """Validate L = lambda * W at each echelon.

    Compares implied cycle time (inventory / throughput) to configured lead times.
    """
    inv = data.inv_by_echelon
    # Shipments already enriched with echelon columns
    ships = data.shipments

    fg_batches = data.fg_batches
    links = data.links
    sim_days = data.sim_days

    # Configured lead times by echelon pair
    links = links.copy()
    links["source_echelon"] = links["source_id"].map(classify_node)
    links["target_echelon"] = links["target_id"].map(classify_node)
    avg_lt_by_target = links.groupby("target_echelon")["lead_time_days"].mean()

    results: dict[str, dict[str, Any]] = {}

    for ech in ECHELON_ORDER:
        # Average inventory
        ech_inv = inv[inv["echelon"] == ech]
        if len(ech_inv) == 0:
            continue
        avg_inventory = ech_inv.groupby("day")["total"].sum().mean()

        # Throughput = daily outflow
        if ech == "Plant":
            throughput = fg_batches["quantity"].sum() / sim_days if sim_days > 0 else 0
        else:
            throughput = (
                ships[ships["source_echelon"] == ech]["quantity"].sum() / sim_days
                if sim_days > 0
                else 0
            )

        if throughput <= 0:
            continue

        implied_ct = avg_inventory / throughput
        config_lt = avg_lt_by_target.get(ech, np.nan)

        if np.isnan(config_lt) or config_lt <= 0:
            # Use inbound lead time for plant (production cycle)
            if ech == "Plant":
                config_lt = _DEFAULT_LEAD_TIME
            else:
                config_lt = _DEFAULT_LEAD_TIME

        ratio = implied_ct / config_lt if config_lt > 0 else np.nan

        if np.isnan(ratio):
            verdict = "NO_DATA"
        elif ratio <= _LITTLE_ALIGNED:
            verdict = "ALIGNED"
        elif ratio <= _LITTLE_ELEVATED:
            verdict = "ELEVATED"
        else:
            verdict = "EXCESSIVE"

        results[ech] = {
            "avg_inventory": avg_inventory,
            "daily_throughput": throughput,
            "implied_cycle_time": implied_ct,
            "configured_lead_time": config_lt,
            "ratio": ratio,
            "verdict": verdict,
        }

    return {"echelons": results}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_mass_balance(results: dict[str, Any], width: int = 78) -> str:
    """Format mass balance results as text."""
    lines = [
        f"\n{'[1.1] System Mass Balance':=^{width}}",
        "",
        f"  {'Period':>12}  {'Production':>14}  {'Consumption':>14}"
        f"  {'Delta Inv':>12}  {'Imbalance':>12}  {'Imb%':>6}",
        f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*6}",
    ]
    for p in results["periods"]:
        lines.append(
            f"  {p['start_day']:>3}-{p['end_day']:>3}d"
            f"      {p['production']:>14,.0f}  {p['consumption']:>14,.0f}"
            f"  {p['delta_inv']:>+12,.0f}  {p['imbalance']:>+12,.0f}"
            f"  {p['imbalance_pct']:>5.1f}%"
        )
    lines.append(
        f"\n  Verdict: {results['verdict']}"
        f"  (worst period imbalance: {results['worst_imbalance_pct']:.1f}%)"
    )
    return "\n".join(lines)


def format_flow_conservation(results: dict[str, Any], width: int = 78) -> str:
    """Format flow conservation waterfall as text."""
    lines = [
        f"\n{'[1.2] Echelon Flow Conservation (Waterfall)':=^{width}}",
        "",
        "  Full-simulation daily averages:",
        "",
        f"  {'Echelon':<14}  {'Inflow/day':>14}  {'Outflow/day':>14}"
        f"  {'Delta/day':>12}  {'Delta%':>8}  {'Verdict':>12}",
        f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*8}  {'-'*12}",
    ]
    for ech in ECHELON_ORDER:
        if ech not in results["echelons"]:
            continue
        f = results["echelons"][ech]
        v = results["verdicts"].get(ech, "?")
        lines.append(
            f"  {ech:<14}  {f['daily_inflow']:>14,.0f}  {f['daily_outflow']:>14,.0f}"
            f"  {f['daily_delta']:>+12,.0f}"
            f"  {f['delta_pct_of_throughput']:>7.1f}%"
            f"  {v:>12}"
        )

    # v0.66.0: DC adjustment annotation
    adj = results.get("dc_adjustment")
    if adj and adj["ecom_dtc_daily_inflow"] > 0:
        lines.append("")
        ecom_in = adj["ecom_dtc_daily_inflow"]
        raw_d = adj["raw_dc_delta"]
        raw_p = adj["raw_dc_delta_pct"]
        adj_d = adj["adjusted_dc_delta"]
        adj_p = adj["adjusted_dc_delta_pct"]
        lines.append(
            "  NOTE: Customer DC includes ECOM-FC + DTC-FC"
            " demand endpoints."
        )
        lines.append(
            f"  Their inflow ({ecom_in:,.0f}/d) = POS"
            " consumption, not accumulation."
        )
        lines.append(
            f"  Raw DC imbalance:        "
            f"{raw_d:+,.0f}/d ({raw_p:.1f}%)"
        )
        lines.append(
            f"  Adjusted (excl ECOM/DTC): "
            f"{adj_d:+,.0f}/d ({adj_p:.1f}%)"
        )

    # Snapshot table
    if results["snapshots"]:
        lines.append(f"\n  Flow rates at {len(results['snapshots'])} snapshots:")
        header = f"  {'Day':>6}"
        for ech in ECHELON_ORDER:
            header += f"  {ech+' delta':>14}"
        lines.append(header)
        lines.append(f"  {'-'*6}" + f"  {'-'*14}" * len(ECHELON_ORDER))
        for snap in results["snapshots"]:
            row = f"  {snap['day']:>6}"
            for ech in ECHELON_ORDER:
                d = snap.get(f"{ech}_delta", 0)
                row += f"  {d:>+14,.0f}"
            lines.append(row)

    return "\n".join(lines)


def format_littles_law(results: dict[str, Any], width: int = 78) -> str:
    """Format Little's Law validation as text."""
    lines = [
        f"\n{'[1.3] Littles Law Validation':=^{width}}",
        "  L = lambda * W  =>  implied_cycle_time = inventory / throughput",
        "",
        f"  {'Echelon':<14}  {'Avg Inv':>14}  {'Throughput/d':>14}"
        f"  {'Implied CT':>12}  {'Config LT':>10}  {'Ratio':>8}  {'Verdict':>10}",
        f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}",
    ]
    for ech in ECHELON_ORDER:
        if ech not in results["echelons"]:
            continue
        r = results["echelons"][ech]
        lines.append(
            f"  {ech:<14}  {r['avg_inventory']:>14,.0f}"
            f"  {r['daily_throughput']:>14,.0f}"
            f"  {r['implied_cycle_time']:>12.1f}d  {r['configured_lead_time']:>10.1f}d"
            f"  {r['ratio']:>8.1f}x  {r['verdict']:>10}"
        )
    return "\n".join(lines)
