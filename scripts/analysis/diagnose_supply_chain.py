#!/usr/bin/env python3
"""
UNIFIED SUPPLY CHAIN DIAGNOSTIC — A Consultant's Checklist.

35 questions organized around: "Can we serve the customer? At what cost?
With what risk? Is it getting better or worse?"

Replaces the 3 Tier-1 scripts (diagnose_365day, diagnose_flow_deep,
diagnose_cost) while reusing their proven module infrastructure.

Sections:
  1. Physics Validation (3 questions)    — first_principles.py  [REUSE]
  2. Executive Scorecard (8 KPIs)        — operational.py        [REUSE]
  3. Service Performance (5 questions)   — operational + NEW
  4. Inventory Health (5 questions)      — operational + NEW
  5. Flow Efficiency (6 questions)       — flow_analysis.py      [REUSE/EXTRACT]
  6. Manufacturing Performance (5 qs)    — manufacturing.py      [NEW]
  7. Financial Performance (5 questions) — cost/commercial.py    [EXTRACT]
  8. Inventory Deep-Dive (--full only)   — streaming             [EXTRACT]

Usage:
    poetry run python scripts/analysis/diagnose_supply_chain.py
    poetry run python scripts/analysis/diagnose_supply_chain.py --full
    poetry run python scripts/analysis/diagnose_supply_chain.py --section 6
    poetry run python scripts/analysis/diagnose_supply_chain.py --data-dir data/output_warm

v0.72.0
"""

# ruff: noqa: E501, PLR2004
from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure diagnostics package importable
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics.commercial import (
    compute_channel_pnl,
    compute_concentration_risk,
    compute_cost_to_serve,
    compute_fill_by_abc_channel,
    compute_margin_by_abc,
    compute_tail_sku_drag,
)
from diagnostics.cost_analysis import (
    compute_cash_to_cash,
    compute_logistics_by_route,
    compute_otif,
    compute_per_sku_cogs,
    stream_carrying_cost,
)
from diagnostics.first_principles import (
    analyze_flow_conservation,
    analyze_littles_law,
    analyze_mass_balance,
    format_flow_conservation,
    format_littles_law,
    format_mass_balance,
)
from diagnostics.flow_analysis import (
    analyze_bullwhip,
    analyze_control_stability,
    analyze_deployment_effectiveness,
    analyze_lead_times,
    analyze_throughput_map,
    format_bullwhip,
    format_control_stability,
    format_deployment_effectiveness,
    format_lead_times,
    format_throughput_map,
)
from diagnostics.loader import (
    DataBundle,
    load_all_data,
)
from diagnostics.manufacturing import (
    compute_bom_cost_rollup,
    compute_changeover_analysis,
    compute_forward_cover,
    compute_stockout_waterfall,
    compute_upstream_availability,
)
from diagnostics.operational import (
    analyze_inventory_positioning,
    analyze_production_alignment,
    analyze_service_levels,
    analyze_slob,
    format_inventory_positioning,
    format_production_alignment,
)

WIDTH = 78
WINDOW = 30


# ═══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _section_header(num: int, title: str) -> None:
    print(f"\n{'=' * WIDTH}")
    print(f"  SECTION {num}: {title}".center(WIDTH))
    print(f"{'=' * WIDTH}")


def _q_header(num: int, title: str) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"  Q{num}. {title}")
    print(f"{'─' * WIDTH}")


def _kpi_row(name: str, actual: str, target: str, status: str) -> None:
    light = {"GREEN": "GREEN ", "YELLOW": "YELLOW", "RED": "RED   "}.get(status, status)
    print(f"  {name:<24}  {actual:>12}  {target:>12}  {light:>8}")


def _traffic(val: float, green_lo: float, green_hi: float,
             yellow_lo: float = float("-inf"), yellow_hi: float = float("inf")) -> str:
    if green_lo <= val <= green_hi:
        return "GREEN"
    if yellow_lo <= val <= yellow_hi:
        return "YELLOW"
    return "RED"


# ═══════════════════════════════════════════════════════════════════════════════
# Section runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_section1(bundle: DataBundle) -> dict:
    """SECTION 1: Physics Validation (Q1-Q3)."""
    _section_header(1, "PHYSICS VALIDATION")

    r_mass = analyze_mass_balance(bundle, WINDOW)
    r_flow = analyze_flow_conservation(bundle)
    r_little = analyze_littles_law(bundle)

    print(format_mass_balance(r_mass, WIDTH))
    print(format_flow_conservation(r_flow, WIDTH))
    print(format_littles_law(r_little, WIDTH))

    return {"mass_balance": r_mass, "flow_conservation": r_flow, "littles_law": r_little}


def run_section2(bundle: DataBundle, results: dict) -> dict:
    """SECTION 2: Executive Scorecard (K1-K8)."""
    _section_header(2, "EXECUTIVE SCORECARD")

    r_inv = analyze_inventory_positioning(bundle, WINDOW)
    r_svc = analyze_service_levels(bundle, WINDOW)
    r_prod = analyze_production_alignment(bundle, WINDOW)
    r_slob = analyze_slob(bundle)
    r_thru = analyze_throughput_map(bundle)
    r_bw = analyze_bullwhip(bundle)
    r_stab = analyze_control_stability(bundle, WINDOW)

    metrics = bundle.metrics
    svc = metrics.get("store_service_level", {}).get("mean", 0)
    turns = metrics.get("inventory_turns", {}).get("mean", 0)
    slob_val = metrics.get("slob", {}).get("mean", 0)
    oee = metrics.get("oee", {}).get("mean", 0)
    por = metrics.get("perfect_order_rate", {}).get("mean", 0)
    pd_ratio = r_prod.get("overall_ratio", np.nan)
    max_bw = r_bw.get("max_bullwhip", np.nan)
    stab = r_stab.get("overall_verdict", "?")

    print(f"\n  {'KPI':<24}  {'Actual':>12}  {'Target':>12}  {'Status':>8}")
    print(f"  {'-' * 24}  {'-' * 12}  {'-' * 12}  {'-' * 8}")
    _kpi_row("K1 Fill Rate", f"{svc:.1%}", ">=97%",
             _traffic(svc, 0.97, 1.0, 0.93, 1.0))
    _kpi_row("K2 Inventory Turns", f"{turns:.1f}x", "8.5-12x",
             _traffic(turns, 8.5, 12, 6, 14))
    _kpi_row("K3 SLOB %", f"{slob_val:.1%}", "<15%",
             "GREEN" if slob_val < 0.15 else ("YELLOW" if slob_val < 0.30 else "RED"))
    _kpi_row("K4 Prod/Demand", f"{pd_ratio:.3f}" if not np.isnan(pd_ratio) else "N/A",
             "0.95-1.05",
             _traffic(pd_ratio, 0.95, 1.05, 0.90, 1.10) if not np.isnan(pd_ratio) else "RED")
    _kpi_row("K5 Perfect Order", f"{por:.1%}", ">=92%",
             _traffic(por, 0.92, 1.0, 0.85, 1.0))
    _kpi_row("K6 OEE", f"{oee:.1%}", "55-70%",
             _traffic(oee, 0.55, 0.70, 0.45, 0.80))
    _kpi_row("K7 Bullwhip", f"{max_bw:.2f}x" if not np.isnan(max_bw) else "N/A",
             "<2.0x",
             "GREEN" if (not np.isnan(max_bw) and max_bw < 2.0) else "YELLOW")
    _kpi_row("K8 Stability", stab, "STABLE",
             "GREEN" if stab in ("STABLE", "CONVERGING") else (
                 "YELLOW" if stab == "MIXED" else "RED"))

    results.update({
        "inventory_positioning": r_inv,
        "service_levels": r_svc,
        "production_alignment": r_prod,
        "slob": r_slob,
        "throughput_map": r_thru,
        "bullwhip": r_bw,
        "stability": r_stab,
    })
    return results


def run_section3(bundle: DataBundle) -> None:
    """SECTION 3: Service Performance (Q4-Q8)."""
    _section_header(3, "SERVICE PERFORMANCE")

    # Q4: Fill Rate by ABC x Channel
    _q_header(4, "Fill Rate by ABC x Channel")
    fill_matrix = compute_fill_by_abc_channel(bundle)
    if not fill_matrix.empty:
        channels = [c for c in fill_matrix.columns if c != "OTHER"]
        header = f"  {'':>8}" + "".join(f"  {c!s:>12}" for c in channels)
        print(header)
        print(f"  {'─' * 8}" + "".join(f"  {'─' * 12}" for _ in channels))
        for abc in ["A", "B", "C"]:
            if abc in fill_matrix.index:
                row_str = f"  {abc + '-items':>8}"
                for ch in channels:
                    val = fill_matrix.loc[abc, ch] if ch in fill_matrix.columns else 0
                    row_str += f"  {val:>11.1%}"
                print(row_str)
    else:
        print("  No data for fill rate cross-tab.")

    # Q5: Stockout Root Cause Waterfall
    _q_header(5, "Stockout Root Cause Waterfall")
    waterfall = compute_stockout_waterfall(bundle)
    total = waterfall["total_lines"]
    print(f"\n  {'Stage':<40}  {'Lines':>10}  {'Cum Loss':>14}")
    print(f"  {'─' * 40}  {'─' * 10}  {'─' * 14}")
    for s in waterfall["stages"]:
        loss_str = ""
        if "loss" in s:
            loss = s["loss"]
            reason = s.get("loss_reason", "")
            loss_pct = loss / total * 100 if total > 0 else 0
            loss_str = f"-{loss:,} ({loss_pct:.1f}%) — {reason}"
        print(f"  {s['stage']:<40}  {s['lines']:>10,}  {loss_str}")
    print(f"\n  Perfect order rate: {waterfall['perfect_pct']:.1f}%")

    # Q6: OTIF Decomposition
    _q_header(6, "OTIF Decomposition")
    otif = compute_otif(bundle)
    if otif.get("available"):
        print(f"\n  Order lines evaluated: {otif['total_lines']:,}")
        print(f"  In-Full:  {otif['in_full_pct']:.1%}")
        print(f"  On-Time:  {otif['on_time_pct']:.1%}")
        print(f"  OTIF:     {otif['otif_pct']:.1%}")
        by_abc = otif["by_abc"]
        print(f"\n  {'ABC':>4}  {'In-Full':>8}  {'On-Time':>8}  {'OTIF':>8}  {'Lines':>10}")
        print(f"  {'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 10}")
        for abc_cls in ["A", "B", "C"]:
            if abc_cls in by_abc.index:
                r = by_abc.loc[abc_cls]
                print(f"  {abc_cls:>4}  {r['in_full']:>7.1%}  {r['on_time']:>7.1%}"
                      f"  {r['otif']:>7.1%}  {r['count']:>10,.0f}")
    else:
        print("\n  requested_date not found — run sim with v0.69.0+.")

    # Q7: Demand Forecast Accuracy (from diagnose_flow_deep Q9 logic)
    _q_header(7, "Demand Forecast Accuracy (WMAPE)")
    _run_forecast_mape(bundle)

    # Q8: Replenishment Responsiveness
    _q_header(8, "Replenishment Responsiveness (Lead Times)")
    r_lt = analyze_lead_times(bundle)
    print(format_lead_times(r_lt, WIDTH))


def _run_forecast_mape(bundle: DataBundle) -> None:
    """Forecast MAPE analysis (extracted from diagnose_flow_deep Q9)."""
    forecasts = bundle.forecasts
    ships = bundle.shipments

    demand_ships = ships[ships["is_demand_endpoint"]]
    actual_daily = demand_ships.groupby(
        [demand_ships["creation_day"], demand_ships["product_id"].astype(str)]
    )["quantity"].sum().reset_index()
    actual_daily.columns = ["day", "product_id", "actual_daily"]

    fg_fc = forecasts[forecasts["product_id"].astype(str).str.startswith("SKU-")].copy()
    fg_fc["product_id"] = fg_fc["product_id"].astype(str)
    forecast_horizon = 14
    fg_fc["forecast_qty"] = fg_fc["forecast_quantity"] / forecast_horizon

    merged = fg_fc.merge(actual_daily, on=["day", "product_id"], how="left")
    merged["actual_daily"] = merged["actual_daily"].fillna(0)
    merged["abc"] = merged["product_id"].map(bundle.abc_map)

    nonzero = merged[(merged["forecast_qty"] > 0) | (merged["actual_daily"] > 0)]
    if len(nonzero) == 0:
        print("\n  No forecast data to compare.")
        return

    print(f"\n  {'ABC':>4} {'Count':>8} {'WMAPE%':>8} {'Bias%':>8}")
    print(f"  {'─' * 4} {'─' * 8} {'─' * 8} {'─' * 8}")
    for abc in ["A", "B", "C"]:
        subset = nonzero[nonzero["abc"] == abc]
        if len(subset) == 0:
            continue
        abs_error = (subset["forecast_qty"] - subset["actual_daily"]).abs().sum()
        actual_sum = subset["actual_daily"].sum()
        wmape = abs_error / actual_sum * 100 if actual_sum > 0 else 0
        bias = (subset["forecast_qty"].sum() - actual_sum) / actual_sum * 100 if actual_sum > 0 else 0
        print(f"  {abc:>4} {len(subset):>8,} {wmape:>7.1f}% {bias:>+7.1f}%")


def run_section4(bundle: DataBundle, results: dict) -> None:
    """SECTION 4: Inventory Health (Q9-Q13)."""
    _section_header(4, "INVENTORY HEALTH")

    # Q9: DOS by ABC x Echelon (consolidation of operational module)
    _q_header(9, "DOS by ABC x Echelon vs Target")
    r_inv = results.get("inventory_positioning")
    if r_inv is None:
        r_inv = analyze_inventory_positioning(bundle, WINDOW)
    print(format_inventory_positioning(r_inv, WIDTH))

    # Q10: Inventory Age Profile (from diagnose_flow_deep Q3)
    _q_header(10, "Inventory Age Profile by Echelon")
    _run_inventory_age(bundle)

    # Q11: Forward Cover
    _q_header(11, "Forward Cover (Weeks of Cover)")
    fwd = compute_forward_cover(bundle)
    if fwd["by_echelon"]:
        print(f"\n  {'Echelon':<14}  {'Inventory':>12}  {'WoC':>8}  {'Target WoC':>12}")
        print(f"  {'─' * 14}  {'─' * 12}  {'─' * 8}  {'─' * 12}")
        for row in fwd["by_echelon"]:
            print(f"  {row['echelon']:<14}  {row['inventory']:>12,.0f}"
                  f"  {row['median_woc']:>8.1f}  {row['target_woc']:>12.1f}")
    else:
        print("\n  No inventory data available.")

    # Q12: Concentration Risk
    _q_header(12, "Concentration Risk (Pareto Analysis)")
    conc = compute_concentration_risk(bundle)
    print(f"\n  Top 20% SKUs ({conc['top20_n']}/{conc['n_skus']}) by volume = "
          f"{conc['top20_volume_pct']:.1f}% of shipped volume")
    print(f"  Top 20% SKUs by inv value = {conc['top20_inv_value_pct']:.1f}% of inventory value")
    print(f"  Single-source products: {conc['single_source_count']}"
          f" / {conc['total_products']} total")

    # Q13: Tail SKU Drag
    _q_header(13, "Tail SKU Drag (Bottom 20%)")
    tail = compute_tail_sku_drag(bundle)
    print(f"\n  Bottom 20% by volume: {tail['n_tail_skus']} SKUs")
    print(f"    {tail['tail_volume_pct']:.1f}% of shipped volume")
    print(f"    {tail['tail_inv_value_pct']:.1f}% of inventory carrying cost (est)")
    print(f"    Avg turns: {tail['tail_avg_turns']:.1f}x")
    print(f"    Changeovers/year: {tail['tail_changeovers_per_sku_year']:.0f}"
          f" (vs A-items: {tail['a_changeovers_per_sku_year']:.0f})")


def _run_inventory_age(bundle: DataBundle) -> None:
    """Inventory age profile (extracted from diagnose_flow_deep Q3)."""
    inv = bundle.inv_by_echelon
    if inv.empty:
        print("\n  No inventory data available.")
        return

    ships = bundle.shipments
    sim_days = bundle.sim_days

    ech_throughput: dict[str, float] = {}
    for ech in ["Plant", "RDC", "Customer DC"]:
        qty = ships[ships["source_echelon"] == ech]["quantity"].sum()
        ech_throughput[ech] = qty / sim_days
    store_inflow = ships[ships["target_id"].astype(str).str.startswith("STORE-")]["quantity"].sum()
    ech_throughput["Store"] = store_inflow / sim_days

    max_day = int(inv["day"].max())
    early = inv[(inv["day"] >= 30) & (inv["day"] <= 60)]
    late = inv[inv["day"] >= max_day - 30]

    print(f"\n  {'Echelon':<14} {'Early DOS':>10} {'Late DOS':>10} {'Throughput/d':>12} {'Cycling?':>10}")
    print(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 10}")

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
        print(f"  {ech:<14} {early_dos:>10.1f} {late_dos:>10.1f} {thru:>12,.0f} {verdict:>10}")


def run_section5(bundle: DataBundle, results: dict) -> None:
    """SECTION 5: Flow Efficiency (Q14-Q19)."""
    _section_header(5, "FLOW EFFICIENCY")

    # Q14: Network Throughput Map
    _q_header(14, "Network Throughput Map")
    r_thru = results.get("throughput_map")
    if r_thru is None:
        r_thru = analyze_throughput_map(bundle)
    print(format_throughput_map(r_thru, WIDTH))

    # Q15: Deployment Effectiveness
    _q_header(15, "Deployment Effectiveness")
    r_deploy = analyze_deployment_effectiveness(bundle)
    print(format_deployment_effectiveness(r_deploy, WIDTH))

    # Q16: Push/Pull Interaction (extracted from flow_deep Q16-18)
    _q_header(16, "Push/Pull Interaction")
    _run_push_pull(bundle)

    # Q17: Bullwhip by Echelon
    _q_header(17, "Bullwhip by Echelon")
    r_bw = results.get("bullwhip")
    if r_bw is None:
        r_bw = analyze_bullwhip(bundle)
    print(format_bullwhip(r_bw, WIDTH))

    # Q18: DC Accumulation (extracted from flow_deep Q1-2)
    _q_header(18, "DC Accumulation Analysis")
    _run_dc_accumulation(bundle)

    # Q19: E2E Cycle Time (extracted from flow_deep Q19)
    _q_header(19, "End-to-End Cycle Time")
    _run_e2e_cycle_time(bundle)


def _run_push_pull(bundle: DataBundle) -> None:
    """Push/pull interaction summary (extracted from flow_deep Q16-17)."""
    ships = bundle.shipments
    sim_days = bundle.sim_days

    rdc_dc_mask = (ships["source_echelon"] == "RDC") & (ships["target_echelon"] == "Customer DC")
    rdc_to_dc = ships[rdc_dc_mask]
    rdc_daily = rdc_to_dc["quantity"].sum() / sim_days

    dc_in_mask = ships["target_echelon"] == "Customer DC"
    dc_inflow_daily = ships[dc_in_mask]["quantity"].sum() / sim_days
    rdc_share = rdc_daily / dc_inflow_daily * 100 if dc_inflow_daily > 0 else 0

    print(f"\n  RDC -> DC push volume: {rdc_daily:,.0f}/day ({rdc_share:.1f}% of DC inflow)")

    # Dual-inflow collision
    plant_dc_mask = (ships["source_echelon"] == "Plant") & (ships["target_echelon"] == "Customer DC")
    plant_arrivals = set(zip(ships[plant_dc_mask]["arrival_day"].values,
                             ships[plant_dc_mask]["target_id"].astype(str).values, strict=False))
    rdc_arrivals = set(zip(rdc_to_dc["arrival_day"].values,
                           rdc_to_dc["target_id"].astype(str).values, strict=False))
    collisions = plant_arrivals & rdc_arrivals
    all_dc_days = plant_arrivals | rdc_arrivals
    collision_pct = len(collisions) / len(all_dc_days) * 100 if all_dc_days else 0
    print(f"  Dual-inflow collision rate: {collision_pct:.1f}%"
          f" ({len(collisions):,} DC-days)")

    # Burstiness
    daily_rdc_vol = rdc_to_dc.groupby("creation_day")["quantity"].sum()
    if len(daily_rdc_vol) > 10:
        cv = daily_rdc_vol.std() / daily_rdc_vol.mean() if daily_rdc_vol.mean() > 0 else 0
        print(f"  RDC push burstiness (CV): {cv:.2f}"
              f" ({'smooth' if cv < 0.3 else 'moderate' if cv < 0.7 else 'bursty'})")


def _run_dc_accumulation(bundle: DataBundle) -> None:
    """DC accumulation summary (extracted from flow_deep Q1)."""
    ships = bundle.shipments
    sim_days = bundle.sim_days

    dc_in_daily = ships[ships["target_echelon"] == "Customer DC"]["quantity"].sum() / sim_days
    dc_out_daily = ships[ships["source_echelon"] == "Customer DC"]["quantity"].sum() / sim_days
    net = dc_in_daily - dc_out_daily
    imbal = net / dc_in_daily * 100 if dc_in_daily > 0 else 0

    print(f"\n  DC inflow:  {dc_in_daily:,.0f}/day")
    print(f"  DC outflow: {dc_out_daily:,.0f}/day")
    print(f"  Net accumulation: {net:+,.0f}/day ({imbal:+.1f}%)")

    # ECOM/DTC adjustment
    ecom_mask = (
        ships["target_id"].astype(str).str.startswith("ECOM-FC-")
        | ships["target_id"].astype(str).str.startswith("DTC-FC-")
    )
    ecom_in = ships[ecom_mask]["quantity"].sum() / sim_days
    ecom_out_mask = (
        ships["source_id"].astype(str).str.startswith("ECOM-FC-")
        | ships["source_id"].astype(str).str.startswith("DTC-FC-")
    )
    ecom_out = ships[ecom_out_mask]["quantity"].sum() / sim_days
    ecom_net = ecom_in - ecom_out
    if ecom_net > 0:
        adj_net = net - ecom_net
        adj_pct = adj_net / (dc_in_daily - ecom_in) * 100 if (dc_in_daily - ecom_in) > 0 else 0
        print(f"  Adjusted (excl ECOM/DTC endpoints): {adj_net:+,.0f}/day ({adj_pct:+.1f}%)")


def _run_e2e_cycle_time(bundle: DataBundle) -> None:
    """E2E cycle time (extracted from flow_deep Q19)."""
    ships = bundle.shipments

    routes = [
        ("Plant -> RDC", (ships["source_echelon"] == "Plant") & (ships["target_echelon"] == "RDC")),
        ("Plant -> DC", (ships["source_echelon"] == "Plant") & (ships["target_echelon"] == "Customer DC")),
        ("RDC -> DC", (ships["source_echelon"] == "RDC") & (ships["target_echelon"] == "Customer DC")),
        ("DC -> Store", (ships["source_echelon"] == "Customer DC") & (ships["target_id"].astype(str).str.startswith("STORE-"))),
    ]

    print(f"\n  {'Route':<18} {'Mean':>6} {'Median':>8} {'P95':>6}")
    print(f"  {'─' * 18} {'─' * 6} {'─' * 8} {'─' * 6}")

    transit_means: dict[str, float] = {}
    for name, mask in routes:
        subset = ships[mask]
        if len(subset) == 0:
            continue
        transit = subset["arrival_day"].astype(float) - subset["creation_day"].astype(float)
        transit = transit[transit >= 0]
        if len(transit) == 0:
            continue
        print(f"  {name:<18} {transit.mean():>6.1f} {transit.median():>8.1f} {np.percentile(transit, 95):>6.1f}")
        transit_means[name] = transit.mean()

    path1 = sum(transit_means.get(r, 0) for r in ["Plant -> RDC", "RDC -> DC", "DC -> Store"])
    path2 = sum(transit_means.get(r, 0) for r in ["Plant -> DC", "DC -> Store"])
    print(f"\n  E2E transit: RDC path={path1:.1f}d, Direct path={path2:.1f}d")


def run_section6(bundle: DataBundle, results: dict) -> None:
    """SECTION 6: Manufacturing Performance (Q20-Q24)."""
    _section_header(6, "MANUFACTURING PERFORMANCE")

    # Q20: Production/Demand Alignment by ABC
    _q_header(20, "Production/Demand Alignment by ABC")
    r_prod = results.get("production_alignment")
    if r_prod is None:
        r_prod = analyze_production_alignment(bundle, WINDOW)
    print(format_production_alignment(r_prod, WIDTH))

    # Q21: MRP Backpressure (extracted from flow_deep Q12)
    _q_header(21, "MRP Backpressure Effectiveness")
    _run_mrp_backpressure(bundle)

    # Q22: Changeover Analysis
    _q_header(22, "Changeover Analysis")
    changeover = compute_changeover_analysis(bundle)
    if not changeover.empty:
        print(f"\n  {'Plant':<12} {'SKUs/day':>10} {'Batches/day':>12} {'Setup%':>8} {'Lost hrs':>10}")
        print(f"  {'─' * 12} {'─' * 10} {'─' * 12} {'─' * 8} {'─' * 10}")
        for plant, row in changeover.iterrows():
            print(f"  {plant:<12} {row['skus_per_day']:>10.1f} {row['batches_per_day']:>12.1f}"
                  f" {row['implied_setup_pct']:>7.1f}% {row['lost_hours']:>10.1f}")
    else:
        print("\n  No batch data available.")

    # Q23: BOM Cost Rollup
    _q_header(23, "BOM Cost Rollup")
    bom = compute_bom_cost_rollup(bundle)
    if bom:
        print(f"\n  Total material cost:     ${bom['total_material']:,.0f}")
        print(f"  Total full mfg cost:     ${bom['total_full_mfg']:,.0f}")
        print(f"  Material share:          {bom['material_share']:.1%}")
        print(f"  Labor share:             {bom['labor_share']:.1%}")
        print(f"  Overhead share:          {bom['overhead_share']:.1%}")
        by_cat = bom["by_category"]
        print(f"\n  {'Category':<15} {'Mat$/c':>7} {'Lab$/c':>7} {'OH$/c':>7} {'Full$/c':>8}")
        print(f"  {'─' * 15} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 8}")
        for cat in ["ORAL_CARE", "PERSONAL_WASH", "HOME_CARE"]:
            if cat in by_cat.index:
                r = by_cat.loc[cat]
                print(f"  {cat:<15} ${r['mat_per_case']:>6.2f} ${r['lab_per_case']:>6.2f}"
                      f" ${r['oh_per_case']:>6.2f} ${r['full_per_case']:>7.2f}")
        if bom["by_bom_level"]:
            print("\n  BOM Level Breakdown:")
            for level, info in bom["by_bom_level"].items():
                print(f"    {level:<28} ${info['cost']:>12,.0f}  ({info['pct_of_material']:.1f}%)")
    else:
        print("\n  batch_ingredients.parquet not available (requires v0.70.0+).")

    # Q24: Upstream Material Availability
    _q_header(24, "Upstream Material Availability")
    upstream = compute_upstream_availability(bundle)
    print(f"\n  FG batches:          {upstream['total_fg_batches']:,}")
    print(f"  Bulk batches:        {upstream['total_bulk_batches']:,}")
    print(f"  Timing violations:   {upstream['timing_violations']}")
    print(f"  RM->Bulk lead time:  {upstream['avg_rm_to_bulk_lead']:.1f} days (same-day by design)")
    print(f"  Bulk->FG lead time:  {upstream['avg_bulk_to_fg_lead']:.1f} days (same-day by design)")
    print(f"  Co-occurrence (Bulk+FG on same plant-day): {upstream['co_occurrence_pct']:.1f}%")


def _run_mrp_backpressure(bundle: DataBundle) -> None:
    """MRP backpressure analysis (extracted from flow_deep Q12)."""
    inv = bundle.inv_by_echelon
    ships = bundle.shipments
    sim_days = bundle.sim_days

    if inv.empty:
        print("\n  No inventory data for backpressure analysis.")
        return

    plant_inv = inv[inv["echelon"] == "Plant"].sort_values("day")
    daily_prod = bundle.fg_batches.groupby("day_produced")["quantity"].sum()
    caps = bundle.dos_targets.mrp_caps

    demand_ships = ships[ships["is_demand_endpoint"]]
    total_demand = demand_ships["quantity"].sum() / sim_days

    dos_df = plant_inv[["day", "total"]].copy()
    dos_df.columns = ["day", "inv"]
    dos_df["dos"] = dos_df["inv"] / total_demand if total_demand > 0 else 0.0
    dos_df = dos_df.reset_index(drop=True)

    if len(dos_df) == 0:
        print("\n  Insufficient data.")
        return

    avg_dos = dos_df["dos"].mean()
    above_a = (dos_df["dos"] > caps["A"]).sum()
    total_days = len(dos_df)

    print(f"\n  Plant aggregate DOS: avg={avg_dos:.1f}, max={dos_df['dos'].max():.1f}")
    print(f"  Days > {caps['A']}d cap: {above_a}/{total_days}")

    # Backpressure correlation
    if len(dos_df) > 10 and len(daily_prod) > 10:
        prod_series = daily_prod.reindex(dos_df["day"]).fillna(0)
        inv_series = dos_df.set_index("day")["inv"]
        common = sorted(set(inv_series.index) & set(prod_series.index))
        if len(common) > 20:
            inv_vals = np.array([inv_series.loc[d] for d in common[:-1]])
            prod_vals = np.array([prod_series.loc[d] for d in common[1:]])
            if inv_vals.std() > 0 and prod_vals.std() > 0:
                seasonal = bundle.seasonality
                day_inv = np.array(common[:-1], dtype=float)
                day_prod = np.array(common[1:], dtype=float)
                inv_det = inv_vals / seasonal.factor(day_inv)
                prod_det = prod_vals / seasonal.factor(day_prod)
                if inv_det.std() > 0 and prod_det.std() > 0:
                    corr = np.corrcoef(inv_det, prod_det)[0, 1]
                    status = "WEAK" if corr > 0 else "GOOD"
                    print(f"  Backpressure corr (detrended): {corr:+.3f} ({status})")


def run_section7(bundle: DataBundle) -> dict:
    """SECTION 7: Financial Performance (Q25-Q29)."""
    _section_header(7, "FINANCIAL PERFORMANCE")

    # Compute COGS and logistics (shared data for multiple questions)
    cogs_result = compute_per_sku_cogs(bundle)
    logistics_result = compute_logistics_by_route(bundle)

    # Build per-shipment logistics array for channel P&L
    logistics_arr = logistics_result["_transport_arr"] + logistics_result["_handling_arr"]

    # Q25: Channel P&L
    _q_header(25, "Channel P&L")
    pnl = compute_channel_pnl(bundle, logistics_arr)
    print(f"\n  Total revenue:  ${pnl['total_revenue']:,.0f}")
    print(f"  Total margin:   ${pnl['total_margin']:,.0f}  ({pnl['overall_margin_pct']:.1f}%)")
    by_ch = pnl["by_channel"]
    targets = pnl["target_margins"]
    print(f"\n  {'Channel':<15} {'Revenue':>14} {'COGS':>12} {'Logistics':>11} {'Margin%':>8} {'Target%':>8}")
    print(f"  {'─' * 15} {'─' * 14} {'─' * 12} {'─' * 11} {'─' * 8} {'─' * 8}")
    for chan, row in by_ch.iterrows():
        tgt = targets.get(str(chan), 0)
        print(f"  {chan:<15} ${row['revenue']:>13,.0f} ${row['cogs']:>11,.0f}"
              f" ${row['logistics']:>10,.0f} {row['margin_pct']:>7.1f}% {tgt:>7}%")

    # Q26: Cost-to-Serve
    _q_header(26, "Cost-to-Serve by Channel")
    cts = compute_cost_to_serve(bundle, logistics_arr)
    print(f"\n  {'Channel':<15} {'Cases':>14} {'Total Cost':>14} {'$/Case':>7}")
    print(f"  {'─' * 15} {'─' * 14} {'─' * 14} {'─' * 7}")
    for chan, row in cts.iterrows():
        print(f"  {chan:<15} {row['cases']:>14,.0f} ${row['total_cost']:>13,.0f}"
              f" ${row['cost_per_case']:>6.2f}")

    # Q27: Logistics by Route
    _q_header(27, "Logistics Cost by Route")
    print(f"\n  Total logistics: ${logistics_result['total_logistics']:,.0f}")
    print(f"    Transport: ${logistics_result['total_transport']:,.0f}")
    print(f"    Handling:  ${logistics_result['total_handling']:,.0f}")
    by_route = logistics_result["by_route"]
    route_cfg = logistics_result.get("route_cfg", {})
    if not by_route.empty:
        print(f"\n  {'Route':<20} {'Mode':>4} {'Avg km':>7} {'Transport':>12} {'Handling':>12} {'$/Case':>7}")
        print(f"  {'─' * 20} {'─' * 4} {'─' * 7} {'─' * 12} {'─' * 12} {'─' * 7}")
        for rk, row in by_route.iterrows():
            mode = route_cfg.get(rk, {}).get("mode", "?")
            per_case = row["logistics"] / row["cases"] if row["cases"] > 0 else 0
            print(f"  {rk:<20} {mode:>4} {row['avg_dist']:>7,.0f} ${row['transport']:>11,.0f}"
                  f" ${row['handling']:>11,.0f} ${per_case:>6.2f}")

    # Q28: Cash-to-Cash (needs inventory value — use metrics fallback)
    _q_header(28, "Cash-to-Cash Cycle")
    inv_turns = bundle.metrics.get("inventory_turns", {}).get("mean", 10)
    est_inv_value = cogs_result["total_cogs"] / inv_turns if inv_turns > 0 else 0
    c2c = compute_cash_to_cash(bundle, est_inv_value, cogs_result["total_cogs"])
    print(f"\n  DIO:  {c2c['dio']:.1f} days")
    print(f"  DSO:  {c2c['dso']:.1f} days (channel-weighted)")
    print(f"  DPO:  {c2c['dpo']:.1f} days (config)")
    print("  ────────────────────────")
    print(f"  C2C:  {c2c['c2c']:.1f} days")

    # Q29: Margin by ABC
    _q_header(29, "Margin by ABC Class")
    abc_margin = compute_margin_by_abc(bundle, logistics_arr)
    print(f"\n  {'ABC':>4} {'Revenue':>14} {'Margin':>12} {'Margin%':>8} {'$/Case':>8}")
    print(f"  {'─' * 4} {'─' * 14} {'─' * 12} {'─' * 8} {'─' * 8}")
    for abc_cls in ["A", "B", "C"]:
        if abc_cls in abc_margin.index:
            r = abc_margin.loc[abc_cls]
            print(f"  {abc_cls:>4} ${r['revenue']:>13,.0f} ${r['margin']:>11,.0f}"
                  f" {r['margin_pct']:>7.1f}% ${r['margin_per_case']:>7.2f}")

    return {
        "cogs": cogs_result,
        "logistics": logistics_result,
        "pnl": pnl,
        "c2c": c2c,
    }


def run_section8(bundle: DataBundle, financial: dict) -> None:
    """SECTION 8: Inventory Deep-Dive (--full only, Q30-Q32)."""
    _section_header(8, "INVENTORY DEEP-DIVE (STREAMING)")

    print("\n  Streaming inventory.parquet for detailed cost analysis...")
    carrying = stream_carrying_cost(bundle)

    if carrying is None:
        print("  WARNING: inventory.parquet not found.")
        return

    # Q30: Carrying Cost by Echelon
    _q_header(30, "Carrying Cost by Echelon")
    print(f"\n  Inventory sampled on {carrying['n_days']} days")
    print(f"  Annual carrying cost ({carrying['carrying_pct']:.0%}): ${carrying['total_carrying']:,.0f}")
    print(f"  Annual warehouse cost: ${carrying['total_warehouse']:,.0f}")
    by_ech = carrying["by_echelon"]
    print(f"\n  {'Echelon':<15} {'Avg Cases':>14} {'Avg Value':>14} {'Carry Cost':>14} {'WH Cost':>14}")
    print(f"  {'─' * 15} {'─' * 14} {'─' * 14} {'─' * 14} {'─' * 14}")
    for ech in ["Plant", "RDC", "Customer DC", "Store"]:
        if ech in by_ech.index:
            r = by_ech.loc[ech]
            print(f"  {ech:<15} {r['avg_cases']:>14,.0f} ${r['avg_value']:>13,.0f}"
                  f" ${r['carrying_cost']:>13,.0f} ${r['warehouse_cost']:>13,.0f}")

    # Q31: Convergence Analysis
    _q_header(31, "Convergence Analysis (Last 180 Days)")
    r_stab = analyze_control_stability(bundle, WINDOW)
    print(format_control_stability(r_stab, WIDTH))

    # Q32: Inventory Investment Profile
    _q_header(32, "Inventory Investment Profile")
    total_inv_value = carrying["total_inv_value"]
    if total_inv_value > 0:
        print(f"\n  Total daily inventory value: ${total_inv_value:,.0f}")
        print(f"\n  {'Echelon':<15} {'Value/day':>14} {'% of Total':>12}")
        print(f"  {'─' * 15} {'─' * 14} {'─' * 12}")
        for ech in ["Plant", "RDC", "Customer DC", "Store"]:
            if ech in by_ech.index:
                val = by_ech.loc[ech, "avg_value"]
                pct = val / total_inv_value * 100
                print(f"  {ech:<15} ${val:>13,.0f} {pct:>11.1f}%")

    # Update C2C with actual inventory value
    cogs_total = financial.get("cogs", {}).get("total_cogs", 0)
    if cogs_total > 0 and total_inv_value > 0:
        c2c = compute_cash_to_cash(bundle, total_inv_value, cogs_total)
        print(f"\n  Revised C2C (with streaming inv): {c2c['c2c']:.1f} days"
              f" (DIO={c2c['dio']:.1f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Supply Chain Diagnostic — 35 Questions"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/output"),
        help="Simulation output directory",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include Section 8 (Inventory Deep-Dive, streams inventory.parquet)",
    )
    parser.add_argument(
        "--section", type=int, default=0,
        help="Run only a specific section (1-8) for dev/debug",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    # Tee stdout to file
    diag_dir = data_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = diag_dir / f"diagnose_supply_chain_{stamp}.txt"
    tee_buf = io.StringIO()
    _real_stdout = sys.stdout

    class _Tee:
        def write(self, s: str) -> int:
            _real_stdout.write(s)
            tee_buf.write(s)
            return len(s)
        def flush(self) -> None:
            _real_stdout.flush()

    sys.stdout = _Tee()  # type: ignore[assignment]

    t0 = time.time()

    print("=" * WIDTH)
    print("  UNIFIED SUPPLY CHAIN DIAGNOSTIC (v0.72.0)".center(WIDTH))
    print("  35 Questions — A Consultant's Checklist".center(WIDTH))
    print("=" * WIDTH)
    print(f"\n  Data directory: {data_dir}")
    print(f"  Mode: {'FULL (with inventory streaming)' if args.full else 'DEFAULT (sections 1-7)'}")

    # Load all data via enhanced DataBundle
    bundle = load_all_data(data_dir)

    t_load = time.time() - t0
    print(f"\n  Data loaded in {t_load:.1f}s")
    print(f"  Shipments: {len(bundle.shipments):,}, Orders: {len(bundle.orders):,}")
    print(f"  Batches: {len(bundle.batches):,} ({len(bundle.fg_batches):,} FG)")
    print(f"  Batch ingredients: {len(bundle.batch_ingredients):,}")
    print(f"  Sim days: {bundle.sim_days}")

    # Track results across sections for reuse
    results: dict = {}
    financial: dict = {}

    def should_run(section: int) -> bool:
        return args.section in (0, section)

    if should_run(1):
        s1 = run_section1(bundle)
        results.update(s1)

    if should_run(2):
        results = run_section2(bundle, results)

    if should_run(3):
        run_section3(bundle)

    if should_run(4):
        run_section4(bundle, results)

    if should_run(5):
        run_section5(bundle, results)

    if should_run(6):
        run_section6(bundle, results)

    if should_run(7):
        financial = run_section7(bundle)

    if args.full and should_run(8):
        run_section8(bundle, financial)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * WIDTH}")
    print(f"  DIAGNOSTIC COMPLETE — {elapsed:.0f}s elapsed".center(WIDTH))
    print(f"{'=' * WIDTH}")

    if not args.full:
        print("\n  For inventory deep-dive (Section 8), run with --full flag.")
    print("  For deep-dive analysis, see Tier-2 scripts:")
    print("    diagnose_a_item_fill.py, diagnose_service_level.py,")
    print("    diagnose_slob.py, analyze_bullwhip.py, check_plant_balance.py\n")

    # Flush tee to file
    sys.stdout = _real_stdout
    report_path.write_text(tee_buf.getvalue())
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
