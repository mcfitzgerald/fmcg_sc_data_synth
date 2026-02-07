#!/usr/bin/env python3
"""
Comprehensive Supply Chain Diagnostic Suite.

Three-layer analysis pyramid:
  Layer 1: First Principles (mass balance, flow conservation, Little's Law)
  Layer 2: Operational Health (inventory, service, production, SLOB)
  Layer 3: Flow & Stability (E2E throughput, deployment, bullwhip, convergence)

Executive summary with traffic-light scorecard and ASCII flow diagram.
All findings are data-driven — no hardcoded conclusions.

Usage:
    poetry run python scripts/analysis/diagnose_365day.py
    poetry run python scripts/analysis/diagnose_365day.py \
        --data-dir data/output --window 30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure the diagnostics package is importable
sys.path.insert(0, str(Path(__file__).parent))

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
from diagnostics.loader import load_all_data
from diagnostics.operational import (
    analyze_inventory_positioning,
    analyze_production_alignment,
    analyze_service_levels,
    analyze_slob,
    format_inventory_positioning,
    format_production_alignment,
    format_service_levels,
    format_slob,
)

WIDTH = 78

# Issue detection thresholds
_PROD_RATIO_LOW = 0.95
_PROD_RATIO_HIGH = 1.10
_SLOB_RED = 0.30
_SLOB_YELLOW = 0.15
_INV_GROWTH_RED = 50
_INV_GROWTH_YELLOW = 20
_BULLWHIP_THRESH = 2.0
_RETENTION_THRESH = 10
_EXCESS_PCT_THRESH = 5

_PROD_RATIO_ALIGNED = 1.05  # Scorecard "aligned" upper bound

# Scorecard thresholds
_SVC_GREEN = 0.97
_SVC_YELLOW = 0.93
_TURNS_LOW = 6
_TURNS_HIGH = 14
_OEE_LOW = 0.55
_OEE_HIGH = 0.70
_POR_THRESH = 0.95


# ---------------------------------------------------------------------------
# Automated Issue Detection
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A data-driven diagnostic finding."""

    severity: str  # RED, YELLOW, GREEN
    title: str
    detail: str
    layer: int  # 1, 2, or 3
    section: str  # e.g. "2.3"

    @property
    def sort_key(self) -> int:
        return {"RED": 0, "YELLOW": 1, "GREEN": 2}.get(self.severity, 3)


def detect_issues(results: dict) -> list[Finding]:
    """Detect issues from all analysis results. Returns sorted findings."""
    findings: list[Finding] = []

    # --- Layer 1 ---
    mb = results.get("mass_balance", {})
    if mb.get("verdict") == "VIOLATION":
        findings.append(Finding(
            "RED", "Mass balance violation detected",
            f"Worst period imbalance: {mb['worst_imbalance_pct']:.1f}%",
            1, "1.1",
        ))
    elif mb.get("verdict") == "MINOR_DRIFT":
        findings.append(Finding(
            "YELLOW", "Minor mass balance drift",
            f"Worst period imbalance: {mb['worst_imbalance_pct']:.1f}%",
            1, "1.1",
        ))

    fc = results.get("flow_conservation", {})
    for ech, verdict in fc.get("verdicts", {}).items():
        if verdict == "ACCUMULATING":
            pct = fc["echelons"][ech]["delta_pct_of_throughput"]
            findings.append(Finding(
                "YELLOW", f"{ech} echelon accumulating inventory",
                f"Inflow exceeds outflow by {pct:.1f}% of throughput",
                1, "1.2",
            ))
        elif verdict == "DRAINING":
            pct = fc["echelons"][ech]["delta_pct_of_throughput"]
            findings.append(Finding(
                "YELLOW", f"{ech} echelon draining inventory",
                f"Outflow exceeds inflow by {pct:.1f}% of throughput",
                1, "1.2",
            ))

    # --- Layer 2 ---
    pa = results.get("production_alignment", {})
    ratio = pa.get("overall_ratio", np.nan)
    if not np.isnan(ratio) and ratio < _PROD_RATIO_LOW:
        findings.append(Finding(
            "RED",
            f"Production underperforming demand by {(1 - ratio) * 100:.0f}%",
            f"Overall prod/demand ratio: {ratio:.3f} (declining trend)",
            2, "2.3",
        ))
    elif not np.isnan(ratio) and ratio > _PROD_RATIO_HIGH:
        findings.append(Finding(
            "RED",
            f"Production exceeding demand by {(ratio - 1) * 100:.0f}%",
            f"Overall prod/demand ratio: {ratio:.3f}",
            2, "2.3",
        ))

    slob = results.get("slob", {})
    slob_pct = slob.get("headline", 0)
    if slob_pct > _SLOB_RED:
        findings.append(Finding(
            "RED", f"SLOB at {slob_pct:.1%}",
            _slob_detail(slob),
            2, "2.4",
        ))
    elif slob_pct > _SLOB_YELLOW:
        findings.append(Finding(
            "YELLOW", f"SLOB elevated at {slob_pct:.1%}",
            _slob_detail(slob),
            2, "2.4",
        ))

    # Inventory growth by echelon
    for ech, info in slob.get("by_echelon", {}).items():
        gp = info.get("inv_growth_pct", np.nan)
        if not np.isnan(gp) and gp > _INV_GROWTH_RED:
            findings.append(Finding(
                "RED", f"{ech} inventory growing +{gp:.0f}%",
                f"Start: {info['inv_start']:,.0f} -> End: {info['inv_end']:,.0f}",
                2, "2.4",
            ))
        elif not np.isnan(gp) and gp > _INV_GROWTH_YELLOW:
            findings.append(Finding(
                "YELLOW", f"{ech} inventory growing +{gp:.0f}%",
                f"Start: {info['inv_start']:,.0f} -> End: {info['inv_end']:,.0f}",
                2, "2.4",
            ))

    # --- Layer 3 ---
    bw = results.get("bullwhip", {})
    max_bw = bw.get("max_bullwhip", np.nan)
    if not np.isnan(max_bw) and max_bw > _BULLWHIP_THRESH:
        findings.append(Finding(
            "YELLOW", f"Bullwhip amplification detected ({max_bw:.1f}x)",
            "Order variance exceeds POS variance by >2x",
            3, "3.4",
        ))

    stability = results.get("stability", {})
    if stability.get("overall_verdict") == "DIVERGING":
        findings.append(Finding(
            "RED", "Control system diverging",
            "One or more key indicators trending away from targets",
            3, "3.5",
        ))

    # Deployment retention
    deploy = results.get("deployment", {})
    ret = deploy.get("retention_pct", 0)
    if ret > _RETENTION_THRESH:
        findings.append(Finding(
            "YELLOW",
            f"Plant retaining {ret:.1f}% of production (not deployed)",
            "Possible deployment bottleneck or FG buffer buildup",
            3, "3.2",
        ))

    findings.sort(key=lambda f: f.sort_key)
    return findings


def _slob_detail(slob: dict) -> str:
    """Build detail string for SLOB finding."""
    parts = []
    for cls in ("A", "B", "C"):
        info = slob.get("by_abc", {}).get(cls, {})
        ep = info.get("excess_pct", 0)
        if abs(ep) > _EXCESS_PCT_THRESH:
            parts.append(f"{cls}-items: {ep:+.1f}% excess production")
    # Biggest echelon contributor
    biggest_ech = ""
    biggest_gp = 0.0
    for ech, info in slob.get("by_echelon", {}).items():
        gp = info.get("inv_growth_pct", 0)
        if isinstance(gp, (int, float)) and not np.isnan(gp) and gp > biggest_gp:
            biggest_gp = gp
            biggest_ech = ech
    if biggest_ech:
        parts.append(f"Biggest accumulator: {biggest_ech} (+{biggest_gp:.0f}%)")
    return "; ".join(parts) if parts else "See detailed SLOB analysis"


# ---------------------------------------------------------------------------
# Executive Summary Formatting
# ---------------------------------------------------------------------------

def _traffic_light(status: str) -> str:
    """Map status to traffic light indicator."""
    if status in ("GREEN", "OK", "BALANCED", "ALIGNED", "STABLE", "CONVERGING"):
        return "GREEN "
    if status in ("YELLOW", "MINOR_DRIFT", "ELEVATED", "ACCUMULATING", "MIXED"):
        return "YELLOW"
    return "RED   "


def print_executive_summary(
    metrics: dict,
    results: dict,
    findings: list[Finding],
) -> None:
    """Print the executive scorecard and key findings."""
    print()
    print("=" * WIDTH)
    print("  SUPPLY CHAIN DIAGNOSTIC REPORT".center(WIDTH))
    print("=" * WIDTH)

    # --- Scorecard ---
    print(f"\n{'EXECUTIVE SCORECARD':=^{WIDTH}}")

    svc = metrics.get("store_service_level", {}).get("mean", 0)
    turns = metrics.get("inventory_turns", {}).get("mean", 0)
    slob_val = metrics.get("slob", {}).get("mean", 0)
    oee = metrics.get("oee", {}).get("mean", 0)
    por = metrics.get("perfect_order_rate", {}).get("mean", 0)

    pa = results.get("production_alignment", {})
    pd_ratio = pa.get("overall_ratio", np.nan)

    bw = results.get("bullwhip", {})
    max_bw = bw.get("max_bullwhip", np.nan)

    stability = results.get("stability", {})
    stab_verdict = stability.get("overall_verdict", "?")

    rows = [
        ("Store Fill Rate", f"{svc:.1%}", ">=97%",
         "GREEN" if svc >= _SVC_GREEN else (
             "YELLOW" if svc >= _SVC_YELLOW else "RED")),
        ("Inventory Turns", f"{turns:.2f}x", "6-14x",
         "GREEN" if _TURNS_LOW <= turns <= _TURNS_HIGH else "YELLOW"),
        ("SLOB", f"{slob_val:.1%}", "<15%",
         "GREEN" if slob_val < _SLOB_YELLOW else (
             "YELLOW" if slob_val < _SLOB_RED else "RED")),
        ("Prod/Demand Ratio",
         f"{pd_ratio:.2f}" if not np.isnan(pd_ratio) else "N/A",
         "0.95-1.05",
         "GREEN" if (
             not np.isnan(pd_ratio)
             and _PROD_RATIO_LOW <= pd_ratio <= _PROD_RATIO_ALIGNED
         ) else "RED"),
        ("Perfect Order Rate", f"{por:.1%}", ">=95%",
         "GREEN" if por >= _POR_THRESH else "YELLOW"),
        ("OEE", f"{oee:.1%}", "55-70%",
         "GREEN" if _OEE_LOW <= oee <= _OEE_HIGH else "YELLOW"),
        ("Bullwhip Ratio",
         f"{max_bw:.2f}x" if not np.isnan(max_bw) else "N/A",
         "<2.0x",
         "GREEN" if (
             not np.isnan(max_bw) and max_bw < _BULLWHIP_THRESH
         ) else "YELLOW"),
        ("System Stability", stab_verdict, "STABLE",
         "GREEN" if stab_verdict in ("STABLE", "CONVERGING") else (
             "YELLOW" if stab_verdict == "MIXED" else "RED")),
    ]

    print(
        f"  {'KPI':<22}  {'Actual':>10}  {'Target':>10}  {'Status':>8}"
    )
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*8}")
    for kpi, actual, target, status in rows:
        light = _traffic_light(status)
        print(f"  {kpi:<22}  {actual:>10}  {target:>10}  {light:>8}")

    # --- Key Findings ---
    print(f"\n{'KEY FINDINGS':=^{WIDTH}}")
    for i, f in enumerate(findings, 1):
        light = _traffic_light(f.severity)
        print(f"  {i}. [{light.strip()}] {f.title}")
        print(f"     {f.detail}")

    if not findings:
        print("  No significant issues detected.")


# ---------------------------------------------------------------------------
# Report Sections
# ---------------------------------------------------------------------------

def print_layer1(results: dict) -> None:
    """Print Layer 1: First Principles."""
    print(f"\n{'=' * WIDTH}")
    print("  LAYER 1: FIRST PRINCIPLES".center(WIDTH))
    print(f"{'=' * WIDTH}")
    print(format_mass_balance(results["mass_balance"], WIDTH))
    print(format_flow_conservation(results["flow_conservation"], WIDTH))
    print(format_littles_law(results["littles_law"], WIDTH))


def print_layer2(results: dict) -> None:
    """Print Layer 2: Operational Health."""
    print(f"\n{'=' * WIDTH}")
    print("  LAYER 2: OPERATIONAL HEALTH".center(WIDTH))
    print(f"{'=' * WIDTH}")
    print(format_inventory_positioning(results["inventory_positioning"], WIDTH))
    print(format_service_levels(results["service_levels"], WIDTH))
    print(format_production_alignment(results["production_alignment"], WIDTH))
    print(format_slob(results["slob"], WIDTH))


def print_layer3(results: dict) -> None:
    """Print Layer 3: Flow & Stability Analysis."""
    print(f"\n{'=' * WIDTH}")
    print("  LAYER 3: FLOW & STABILITY ANALYSIS".center(WIDTH))
    print(f"{'=' * WIDTH}")
    print(format_throughput_map(results["throughput_map"], WIDTH))
    print(format_deployment_effectiveness(results["deployment"], WIDTH))
    print(format_lead_times(results["lead_times"], WIDTH))
    print(format_bullwhip(results["bullwhip"], WIDTH))
    print(format_control_stability(results["stability"], WIDTH))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comprehensive Supply Chain Diagnostic Suite"
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
    print(f"Rolling window:  {window} days\n")

    # --- Load all data ---
    data = load_all_data(data_dir)

    # --- Run all analyses ---
    print("\nRunning Layer 1: First Principles...")
    r_mass = analyze_mass_balance(data, window)
    r_flow = analyze_flow_conservation(data)
    r_little = analyze_littles_law(data)

    print("Running Layer 2: Operational Health...")
    r_inv = analyze_inventory_positioning(data, window)
    r_svc = analyze_service_levels(data, window)
    r_prod = analyze_production_alignment(data, window)
    r_slob = analyze_slob(data)

    print("Running Layer 3: Flow & Stability...")
    r_thru = analyze_throughput_map(data)
    r_deploy = analyze_deployment_effectiveness(data)
    r_lt = analyze_lead_times(data)
    r_bw = analyze_bullwhip(data)
    r_stab = analyze_control_stability(data, window)

    results = {
        "mass_balance": r_mass,
        "flow_conservation": r_flow,
        "littles_law": r_little,
        "inventory_positioning": r_inv,
        "service_levels": r_svc,
        "production_alignment": r_prod,
        "slob": r_slob,
        "throughput_map": r_thru,
        "deployment": r_deploy,
        "lead_times": r_lt,
        "bullwhip": r_bw,
        "stability": r_stab,
    }

    # --- Detect issues ---
    findings = detect_issues(results)

    # --- Print report ---
    print_executive_summary(data.metrics, results, findings)
    print_layer1(results)
    print_layer2(results)
    print_layer3(results)

    print(f"\n{'=' * WIDTH}")
    print("  END OF DIAGNOSTIC REPORT".center(WIDTH))
    print(f"{'=' * WIDTH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
