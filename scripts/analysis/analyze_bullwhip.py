#!/usr/bin/env python3
"""Bullwhip effect analysis script for Prism Sim results."""

import argparse
from pathlib import Path

import pandas as pd


def load_data(results_dir: Path) -> dict:
    """Load all result CSVs from a results directory."""
    return {
        "orders": pd.read_csv(results_dir / "orders.csv"),
        "shipments": pd.read_csv(results_dir / "shipments.csv"),
        "batches": pd.read_csv(results_dir / "batches.csv"),
        "inventory": pd.read_csv(results_dir / "inventory.csv"),
    }


def analyze_echelon_variance(data: dict) -> None:
    """Analyze variance amplification through supply chain echelons."""
    orders = data["orders"]
    shipments = data["shipments"]

    print("=" * 70)
    print("ECHELON VARIANCE ANALYSIS (Bullwhip Detection)")
    print("=" * 70)

    # Store orders (downstream demand signal)
    store_orders = orders[orders["target_id"].str.startswith("STORE")]
    store_daily = store_orders.groupby("day")["quantity"].sum()

    # RDC to Store shipments
    rdc_to_store = shipments[
        shipments["source_id"].str.startswith("RDC")
        & shipments["target_id"].str.startswith("STORE")
    ]
    rdc_ship_daily = rdc_to_store.groupby("creation_day")["quantity"].sum()

    # Plant to RDC shipments (upstream flow)
    plant_to_rdc = shipments[
        shipments["source_id"].str.startswith("PLANT")
        & shipments["target_id"].str.startswith("RDC")
    ]
    plant_ship_daily = plant_to_rdc.groupby("creation_day")["quantity"].sum()

    print("\n--- Daily Variance by Echelon ---")
    for name, series in [
        ("Store Orders", store_daily),
        ("RDC→Store Shipments", rdc_ship_daily),
        ("Plant→RDC Shipments", plant_ship_daily),
    ]:
        cv = series.std() / series.mean() if series.mean() > 0 else 0
        print(f"\n{name}:")
        print(f"  Mean: {series.mean():>12,.0f}")
        print(f"  Std:  {series.std():>12,.0f}")
        print(f"  CV:   {cv:>12.3f}")

    # Bullwhip ratios
    store_cv = store_daily.std() / store_daily.mean()
    rdc_cv = rdc_ship_daily.std() / rdc_ship_daily.mean()
    plant_cv = plant_ship_daily.std() / plant_ship_daily.mean()

    print("\n" + "=" * 70)
    print("BULLWHIP RATIOS")
    print("=" * 70)
    print("\nVariance amplification (CV ratios):")
    print(f"  RDC/Store:   {rdc_cv/store_cv:.2f}x")
    print(f"  Plant/RDC:   {plant_cv/rdc_cv:.2f}x")
    print(f"  Plant/Store: {plant_cv/store_cv:.2f}x (total)")

    if plant_cv / store_cv < 1:
        print("\n⚠️  INVERSE BULLWHIP: Variance DECREASES upstream")
    else:
        print("\n✓ Classic bullwhip: Variance amplifies upstream")


def analyze_production_pattern(data: dict) -> None:
    """Analyze production oscillation patterns."""
    batches = data["batches"]

    print("\n" + "=" * 70)
    print("PRODUCTION OSCILLATION ANALYSIS")
    print("=" * 70)

    prod_daily = batches.groupby("day_produced")["quantity"].sum()

    print("\nDaily Production Statistics:")
    print(f"  Mean: {prod_daily.mean():>12,.0f}")
    print(f"  Std:  {prod_daily.std():>12,.0f}")
    print(f"  CV:   {prod_daily.std()/prod_daily.mean():>12.3f}")

    # Count zero production days
    max_day = batches["day_produced"].max()
    all_days = set(range(1, max_day + 1))
    prod_days = set(prod_daily.index)
    zero_prod_days = all_days - prod_days

    print(f"\nDays with ZERO production: {len(zero_prod_days)}")
    print(f"Days with production: {len(prod_days)}")


def analyze_order_batching(data: dict) -> None:
    """Analyze order size distribution for batching effects."""
    orders = data["orders"]

    print("\n" + "=" * 70)
    print("ORDER BATCHING ANALYSIS")
    print("=" * 70)

    order_sizes = orders.groupby(["day", "target_id"])["quantity"].sum()
    print("\nOrder Size Distribution:")
    print(f"  Mean:   {order_sizes.mean():>12,.0f}")
    print(f"  Std:    {order_sizes.std():>12,.0f}")
    print(f"  Min:    {order_sizes.min():>12,.0f}")
    print(f"  Max:    {order_sizes.max():>12,.0f}")
    print(f"  CV:     {order_sizes.std()/order_sizes.mean():>12.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze bullwhip effects in simulation results")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist")
        return 1

    print(f"Loading data from {args.results_dir}...")
    data = load_data(args.results_dir)

    analyze_echelon_variance(data)
    analyze_production_pattern(data)
    analyze_order_batching(data)

    return 0


if __name__ == "__main__":
    exit(main())
