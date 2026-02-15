#!/usr/bin/env python3
"""
K8 Stability Deep-Dive — Visual diagnostic for DIVERGING stability verdicts.

Reproduces the stability analysis from flow_analysis.py:analyze_control_stability
and generates 4 matplotlib plots to understand which echelon(s) drive the
overall DIVERGING verdict and whether the thresholds need recalibration.

Plots saved to {data_dir}/diagnostics/stability_*.png

Usage:
    poetry run python scripts/analysis/stability_deep_dive.py
    poetry run python scripts/analysis/stability_deep_dive.py --data-dir data/output_warm
"""

# ruff: noqa: E501, PLR2004, RUF001
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure diagnostics package importable
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics.loader import ECHELON_ORDER, DataBundle, load_all_data

# Thresholds (mirrored from flow_analysis.py for reference lines)
_INV_SLOPE_STABLE = 0.01
_INV_SLOPE_SLOW = 0.05
_STABILITY_WINDOW_DAYS = 180
_STABILITY_SLOPE_ZERO = 0.0001
_STABILITY_SLOPE_SLOW = 0.001
_MIN_ECHELON_POINTS = 5
_MIN_REGRESSION_POINTS = 10

# Active echelons (Club is typically empty)
_ACTIVE_ECHELONS = [e for e in ECHELON_ORDER if e != "Club"]

VERDICT_COLORS = {
    "STABLE": "#2ecc71",
    "CONVERGING": "#f39c12",
    "DIVERGING": "#e74c3c",
    "DRAINING": "#9b59b6",
}


def _classify_inv_stability(pct_slope: float) -> str:
    """Classify inventory trend (mirrors flow_analysis.py)."""
    if abs(pct_slope) < _INV_SLOPE_STABLE:
        return "STABLE"
    if abs(pct_slope) < _INV_SLOPE_SLOW:
        return "CONVERGING"
    return "DIVERGING" if pct_slope > 0 else "DRAINING"


def _classify_ratio_stability(slope: float, current: float, target: float) -> str:
    """Classify trend toward/away from target (mirrors flow_analysis.py)."""
    if abs(slope) < _STABILITY_SLOPE_ZERO:
        return "STABLE"
    distance = current - target
    if distance > 0 and slope < 0:
        return "CONVERGING"
    if distance < 0 and slope > 0:
        return "CONVERGING"
    if abs(slope) > _STABILITY_SLOPE_SLOW:
        return "DIVERGING"
    return "STABLE"


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def _fit_inventory_echelon(
    daily_inv: pd.Series, last_180_start: int, seasonal: Any,
) -> dict | None:
    """Fit linear+harmonic model to an echelon's inventory over last 180 days.

    Returns dict with all intermediate values, or None if insufficient data.
    """
    last_days = daily_inv[daily_inv.index >= last_180_start]
    if len(last_days) <= _MIN_ECHELON_POINTS:
        return None

    day_vals = np.array(last_days.index, dtype=float)
    y_vals = last_days.values.astype(float)
    omega = 2 * np.pi / seasonal.cycle_days
    phase = day_vals - seasonal.phase_shift_days

    # Design matrix: [1, t, sin(wt'), cos(wt')]
    design = np.column_stack([
        np.ones(len(day_vals)),
        day_vals,
        np.sin(omega * phase),
        np.cos(omega * phase),
    ])
    coeffs, *_ = np.linalg.lstsq(design, y_vals, rcond=None)
    a, b, c, d = coeffs

    fitted = design @ coeffs
    linear_component = a + b * day_vals
    seasonal_component = c * np.sin(omega * phase) + d * np.cos(omega * phase)
    residuals = y_vals - fitted
    detrended = y_vals - linear_component  # actual - (a + bt), keeps seasonal

    mean_inv = float(y_vals.mean())
    pct_slope = b / mean_inv * 100 if mean_inv > 0 else 0.0
    verdict = _classify_inv_stability(pct_slope)

    return {
        "day_vals": day_vals,
        "y_vals": y_vals,
        "fitted": fitted,
        "linear_component": linear_component,
        "seasonal_component": seasonal_component,
        "residuals": residuals,
        "detrended": detrended,
        "coeffs": (a, b, c, d),
        "slope": b,
        "pct_slope": pct_slope,
        "mean_inv": mean_inv,
        "verdict": verdict,
        "all_days": np.array(daily_inv.index, dtype=float),
        "all_vals": daily_inv.values.astype(float),
    }


def _fit_prod_demand_ratio(
    data: DataBundle, window: int = 30
) -> dict | None:
    """Fit linear regression to prod/demand ratio over last 180 days."""
    fg_batches = data.fg_batches
    demand_ships = data.shipments[data.shipments["is_demand_endpoint"]]

    demand_daily = demand_ships.groupby("creation_day")["quantity"].sum()
    prod_daily = fg_batches.groupby("day_produced")["quantity"].sum()
    all_days = sorted(set(demand_daily.index) | set(prod_daily.index))
    demand_s = demand_daily.reindex(all_days, fill_value=0)
    prod_s = prod_daily.reindex(all_days, fill_value=0)

    demand_roll = demand_s.rolling(window, min_periods=1).mean()
    prod_roll = prod_s.rolling(window, min_periods=1).mean()
    ratio_roll = prod_roll / demand_roll.replace(0, np.nan)

    last_180_start = (
        max(all_days) - _STABILITY_WINDOW_DAYS
        if max(all_days) > _STABILITY_WINDOW_DAYS
        else 0
    )

    ratio_last180 = ratio_roll[ratio_roll.index >= last_180_start].dropna()
    if len(ratio_last180) <= _MIN_REGRESSION_POINTS:
        return None

    x = np.arange(len(ratio_last180), dtype=float)
    slope, intercept = np.polyfit(x, ratio_last180.values, 1)
    fitted = intercept + slope * x
    last_val = float(ratio_last180.iloc[-1])
    verdict = _classify_ratio_stability(slope, last_val, 1.0)

    return {
        "all_days": np.array(all_days, dtype=float),
        "ratio_all": ratio_roll.reindex(all_days).values,
        "last180_days": np.array(ratio_last180.index, dtype=float),
        "last180_vals": ratio_last180.values,
        "fitted": fitted,
        "slope": slope,
        "intercept": intercept,
        "last_val": last_val,
        "verdict": verdict,
        "last_180_start": last_180_start,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_inventory_timeseries(echelon_fits: dict, out_dir: Path) -> Path:
    """Plot 1: Per-echelon inventory with regression fit overlay."""
    echelons = [e for e in _ACTIVE_ECHELONS if e in echelon_fits]
    n = len(echelons)
    if n == 0:
        print("  No echelon data — skipping timeseries plot")
        return out_dir / "stability_timeseries.png"

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ech in zip(axes, echelons, strict=True):
        fit = echelon_fits[ech]
        all_days, all_vals = fit["all_days"], fit["all_vals"]
        day_vals, fitted = fit["day_vals"], fit["fitted"]
        last_180_start = day_vals[0]

        # Full timeseries (light)
        ax.plot(all_days, all_vals / 1e6, color="#bdc3c7", linewidth=0.5, label="All days")
        # Regression window (bold)
        ax.plot(day_vals, fit["y_vals"] / 1e6, color="#2c3e50", linewidth=0.8, label="Last 180d")
        # Fitted curve
        ax.plot(day_vals, fitted / 1e6, color="#e74c3c", linewidth=2, linestyle="--", label="Fit: a+bt+seasonal")
        # Linear component only
        ax.plot(day_vals, fit["linear_component"] / 1e6, color="#3498db", linewidth=1.5, linestyle=":", label="Linear trend (a+bt)")

        # Shade regression window
        ax.axvspan(last_180_start, day_vals[-1], alpha=0.06, color="#3498db")

        color = VERDICT_COLORS.get(fit["verdict"], "#95a5a6")
        ax.set_title(
            f"{ech}  —  slope={fit['slope']:.0f}/day  |  "
            f"pct_slope={fit['pct_slope']:.4f}%/day  |  "
            f"{fit['verdict']}",
            fontsize=11, fontweight="bold", color=color,
        )
        ax.set_ylabel("Inventory (M cases)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Simulation Day")
    fig.suptitle("K8 Stability: Per-Echelon Inventory + Regression Fit", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = out_dir / "stability_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_detrended(echelon_fits: dict, out_dir: Path) -> Path:
    """Plot 2: Detrended inventory (actual - linear trend, keeps seasonal)."""
    echelons = [e for e in _ACTIVE_ECHELONS if e in echelon_fits]
    n = len(echelons)
    if n == 0:
        print("  No echelon data — skipping detrended plot")
        return out_dir / "stability_detrended.png"

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ech in zip(axes, echelons, strict=True):
        fit = echelon_fits[ech]
        day_vals = fit["day_vals"]
        detrended = fit["detrended"] / 1e6
        seasonal_fit = fit["seasonal_component"] / 1e6
        residuals = fit["residuals"] / 1e6

        ax.plot(day_vals, detrended, color="#2c3e50", linewidth=0.7, label="Detrended (actual − linear)")
        ax.plot(day_vals, seasonal_fit, color="#e74c3c", linewidth=2, linestyle="--", label="Seasonal fit (c·sin + d·cos)")
        ax.fill_between(day_vals, 0, residuals, alpha=0.2, color="#f39c12", label="Residual (unexplained)")
        ax.axhline(0, color="#7f8c8d", linewidth=0.5)

        seasonal_amp = (max(seasonal_fit) - min(seasonal_fit)) / 2
        residual_std = float(np.std(residuals))
        ax.set_title(
            f"{ech}  —  seasonal amplitude={seasonal_amp:.2f}M  |  "
            f"residual σ={residual_std:.2f}M",
            fontsize=11,
        )
        ax.set_ylabel("Δ Inventory (M cases)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Simulation Day")
    fig.suptitle("Detrended Inventory (Linear Trend Removed)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = out_dir / "stability_detrended.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_prod_demand_ratio(ratio_fit: dict | None, out_dir: Path) -> Path:
    """Plot 3: Production/Demand ratio trend with regression."""
    path = out_dir / "stability_prod_demand.png"
    if ratio_fit is None:
        print("  No prod/demand data — skipping ratio plot")
        return path

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    all_days = ratio_fit["all_days"]
    ratio_all = ratio_fit["ratio_all"]
    last180_days = ratio_fit["last180_days"]
    fitted = ratio_fit["fitted"]
    last_180_start = ratio_fit["last_180_start"]

    # Full ratio (light)
    ax.plot(all_days, ratio_all, color="#bdc3c7", linewidth=0.5, label="All days (30d rolling)")
    # Last 180d
    ax.plot(last180_days, ratio_fit["last180_vals"], color="#2c3e50", linewidth=0.8, label="Last 180d")
    # Regression fit
    ax.plot(last180_days, fitted, color="#e74c3c", linewidth=2, linestyle="--",
            label=f"Linear fit (slope={ratio_fit['slope']:.6f}/day)")
    # Reference line at 1.0
    ax.axhline(1.0, color="#27ae60", linewidth=1.5, linestyle=":", label="Target = 1.0")
    # Shade regression window
    ax.axvspan(last_180_start, last180_days[-1], alpha=0.06, color="#3498db")

    color = VERDICT_COLORS.get(ratio_fit["verdict"], "#95a5a6")
    ax.set_title(
        f"Prod/Demand Ratio  —  slope={ratio_fit['slope']:.6f}/day  |  "
        f"last={ratio_fit['last_val']:.4f}  |  {ratio_fit['verdict']}",
        fontsize=12, fontweight="bold", color=color,
    )
    ax.set_xlabel("Simulation Day")
    ax.set_ylabel("Production / Demand (30d rolling)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_summary_dashboard(echelon_fits: dict, ratio_fit: dict | None, out_dir: Path) -> Path:
    """Plot 4: Bar chart of pct_slope per indicator with threshold lines."""
    indicators: list[tuple[str, float, str]] = []

    for ech in _ACTIVE_ECHELONS:
        if ech in echelon_fits:
            fit = echelon_fits[ech]
            indicators.append((f"{ech}\ninventory", fit["pct_slope"], fit["verdict"]))

    if ratio_fit:
        # Express ratio slope in comparable pct terms
        ratio_pct = ratio_fit["slope"] / max(ratio_fit["last_val"], 0.01) * 100
        indicators.append(("Prod/Demand\nratio", ratio_pct, ratio_fit["verdict"]))

    if not indicators:
        print("  No indicators — skipping dashboard")
        return out_dir / "stability_dashboard.png"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = [i[0] for i in indicators]
    values = [i[1] for i in indicators]
    verdicts = [i[2] for i in indicators]
    colors = [VERDICT_COLORS.get(v, "#95a5a6") for v in verdicts]

    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Threshold lines
    ax.axhline(_INV_SLOPE_STABLE, color="#27ae60", linewidth=1.5, linestyle="--",
               label=f"STABLE threshold (±{_INV_SLOPE_STABLE}%)")
    ax.axhline(-_INV_SLOPE_STABLE, color="#27ae60", linewidth=1.5, linestyle="--")
    ax.axhline(_INV_SLOPE_SLOW, color="#e74c3c", linewidth=1.5, linestyle=":",
               label=f"DIVERGING threshold (±{_INV_SLOPE_SLOW}%)")
    ax.axhline(-_INV_SLOPE_SLOW, color="#e74c3c", linewidth=1.5, linestyle=":")
    ax.axhline(0, color="#7f8c8d", linewidth=0.5)

    # Annotate bars with values and verdicts
    for bar, val, verdict in zip(bars, values, verdicts, strict=True):
        y = bar.get_height()
        offset = 0.003 if y >= 0 else -0.008
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                f"{val:.4f}%\n{verdict}", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Pct slope (%/day of mean)", fontsize=11)
    ax.set_title("K8 Stability Summary — Slope by Indicator", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Pad y-axis to fit annotations
    ymin, ymax = ax.get_ylim()
    margin = max(abs(ymin), abs(ymax)) * 0.3
    ax.set_ylim(ymin - margin, ymax + margin)

    fig.tight_layout()
    path = out_dir / "stability_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Text interpretation
# ---------------------------------------------------------------------------

def print_interpretation(echelon_fits: dict, ratio_fit: dict | None) -> None:
    """Print human-readable interpretation of the stability analysis."""
    width = 78
    print(f"\n{'  STABILITY DEEP-DIVE INTERPRETATION  ':=^{width}}")

    verdicts = []

    # Inventory echelons
    print(f"\n{'─ Inventory Stability (per-echelon) ─':─^{width}}")
    for ech in _ACTIVE_ECHELONS:
        if ech not in echelon_fits:
            continue
        fit = echelon_fits[ech]
        v = fit["verdict"]
        verdicts.append(v)
        marker = {"STABLE": "✓", "CONVERGING": "~", "DIVERGING": "✗", "DRAINING": "↓"}.get(v, "?")
        print(f"\n  [{marker}] {ech}: {v}")
        print(f"      Linear slope:  {fit['slope']:+.1f} cases/day")
        print(f"      Pct slope:     {fit['pct_slope']:+.4f} %/day of mean")
        print(f"      Mean inventory: {fit['mean_inv']:,.0f} cases")

        # Context: how much drift over 180 days
        drift_180 = fit["slope"] * 180
        drift_pct = drift_180 / fit["mean_inv"] * 100 if fit["mean_inv"] > 0 else 0
        print(f"      180-day drift: {drift_180:+,.0f} cases ({drift_pct:+.1f}% of mean)")

        # Regression quality
        ss_res = float(np.sum(fit["residuals"] ** 2))
        ss_tot = float(np.sum((fit["y_vals"] - fit["y_vals"].mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"      R² of full fit: {r_squared:.4f}")

        # Seasonal amplitude vs trend
        _a, b, c, d = fit["coeffs"]
        seasonal_amp = np.sqrt(c**2 + d**2)
        trend_180 = abs(b * 180)
        print(f"      Seasonal amplitude: {seasonal_amp:,.0f} cases")
        print(f"      Trend 180d vs seasonal: {trend_180:,.0f} vs {seasonal_amp:,.0f}"
              f" (ratio={trend_180 / seasonal_amp:.2f})" if seasonal_amp > 0 else "")

    # Prod/demand ratio
    print(f"\n{'─ Production/Demand Ratio ─':─^{width}}")
    if ratio_fit:
        v = ratio_fit["verdict"]
        verdicts.append(v)
        marker = {"STABLE": "✓", "CONVERGING": "~", "DIVERGING": "✗"}.get(v, "?")
        print(f"\n  [{marker}] Prod/Demand: {v}")
        print(f"      Last value:    {ratio_fit['last_val']:.4f}")
        print(f"      Slope:         {ratio_fit['slope']:+.8f}/day")
        print(f"      180-day drift: {ratio_fit['slope'] * 180:+.4f}")
    else:
        print("  Insufficient data for prod/demand ratio analysis")

    # Overall verdict
    print(f"\n{'─ Overall Assessment ─':─^{width}}")
    if all(v in ("STABLE", "CONVERGING") for v in verdicts):
        overall = "STABLE"
    elif any(v == "DIVERGING" for v in verdicts):
        overall = "DIVERGING"
    else:
        overall = "MIXED"

    diverging = [
        ech for ech in _ACTIVE_ECHELONS
        if ech in echelon_fits and echelon_fits[ech]["verdict"] == "DIVERGING"
    ]
    draining = [
        ech for ech in _ACTIVE_ECHELONS
        if ech in echelon_fits and echelon_fits[ech]["verdict"] == "DRAINING"
    ]

    print(f"\n  Overall: {overall}")
    if diverging:
        print(f"  DIVERGING echelons:  {', '.join(diverging)}")
    if draining:
        print(f"  DRAINING echelons:   {', '.join(draining)}")

    # Actionable commentary
    print(f"\n{'─ Commentary ─':─^{width}}")
    for ech in diverging:
        fit = echelon_fits[ech]
        if abs(fit["pct_slope"]) < 0.1:
            print(f"  {ech}: pct_slope={fit['pct_slope']:.4f}% — just over DIVERGING"
                  f" threshold ({_INV_SLOPE_SLOW}%). Could be measurement noise"
                  f" or very slow structural build. Check if 365 more days"
                  f" would push inventory to unrealistic levels.")
        else:
            print(f"  {ech}: pct_slope={fit['pct_slope']:.4f}% — significant structural"
                  f" inventory build. Likely a real replenishment/production imbalance.")

    for ech in draining:
        fit = echelon_fits[ech]
        print(f"  {ech}: pct_slope={fit['pct_slope']:.4f}% — inventory declining."
              f" Possible demand > production or deployment policy pull-down.")

    stable = [
        ech for ech in _ACTIVE_ECHELONS
        if ech in echelon_fits and echelon_fits[ech]["verdict"] == "STABLE"
    ]
    if stable:
        print(f"  Stable echelons ({', '.join(stable)}): pct_slope within ±{_INV_SLOPE_STABLE}%.")

    print(f"\n{'=' * width}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="K8 Stability Deep-Dive")
    parser.add_argument("--data-dir", type=str, default="data/output",
                        help="Simulation output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use Agg backend for headless environments
    plt.switch_backend("Agg")

    print("=" * 78)
    print("  K8 STABILITY DEEP-DIVE")
    print("=" * 78)

    # Load data via shared infrastructure
    data = load_all_data(data_dir)
    inv = data.inv_by_echelon
    seasonal = data.seasonality

    # Determine regression window
    if len(inv) == 0:
        print("ERROR: No inventory data loaded. Cannot perform stability analysis.")
        sys.exit(1)

    max_day = int(inv["day"].max())
    last_180_start = max_day - _STABILITY_WINDOW_DAYS if max_day > _STABILITY_WINDOW_DAYS else 0

    # Fit each echelon
    print("\nFitting regression models...")
    echelon_fits: dict[str, dict] = {}
    for ech in _ACTIVE_ECHELONS:
        ech_inv = inv[inv["echelon"] == ech]
        if len(ech_inv) == 0:
            continue
        daily_inv = ech_inv.groupby("day")["total"].sum()
        fit = _fit_inventory_echelon(daily_inv, last_180_start, seasonal)
        if fit is not None:
            echelon_fits[ech] = fit
            print(f"  {ech:>15}: slope={fit['slope']:+10.1f}/day  "
                  f"pct={fit['pct_slope']:+.4f}%  → {fit['verdict']}")

    # Fit prod/demand ratio
    print("\nFitting prod/demand ratio...")
    ratio_fit = _fit_prod_demand_ratio(data)
    if ratio_fit:
        print(f"  Prod/Demand: slope={ratio_fit['slope']:+.8f}/day  "
              f"last={ratio_fit['last_val']:.4f}  → {ratio_fit['verdict']}")

    # Generate plots
    print("\nGenerating plots...")
    plot_inventory_timeseries(echelon_fits, out_dir)
    plot_detrended(echelon_fits, out_dir)
    plot_prod_demand_ratio(ratio_fit, out_dir)
    plot_summary_dashboard(echelon_fits, ratio_fit, out_dir)

    # Print interpretation
    print_interpretation(echelon_fits, ratio_fit)


if __name__ == "__main__":
    main()
