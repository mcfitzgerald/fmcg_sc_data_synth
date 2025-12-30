#!/usr/bin/env python3
"""Compare two simulation scenarios (e.g., baseline vs risk events)."""

import argparse
import pandas as pd
from pathlib import Path


# Column name reference for result CSVs:
# orders.csv: order_id, day, source_id, target_id, product_id, quantity, status
# shipments.csv: shipment_id, creation_day, arrival_day, source_id, target_id, product_id, quantity, total_weight_kg, total_volume_m3, status
# batches.csv: batch_id, plant_id, product_id, day_produced, quantity, yield_pct, status, notes
# inventory.csv: day, node_id, product_id, perceived_inventory, actual_inventory


def load_scenario(results_dir: Path) -> dict:
    """Load all result CSVs from a results directory."""
    return {
        "orders": pd.read_csv(results_dir / "orders.csv"),
        "shipments": pd.read_csv(results_dir / "shipments.csv"),
        "batches": pd.read_csv(results_dir / "batches.csv"),
    }


def compare_shipments_around_days(
    scenario_a: dict, scenario_b: dict, day_ranges: list, label_a: str, label_b: str
) -> None:
    """Compare shipment volumes around specific day ranges."""
    print("=" * 60)
    print("SHIPMENT VOLUME COMPARISON")
    print("=" * 60)

    for name, (start, end) in day_ranges:
        print(f"\n{name} - Days {start}-{end}:")

        a_qty = scenario_a["shipments"][
            (scenario_a["shipments"]["creation_day"] >= start)
            & (scenario_a["shipments"]["creation_day"] <= end)
        ]["quantity"].sum()

        b_qty = scenario_b["shipments"][
            (scenario_b["shipments"]["creation_day"] >= start)
            & (scenario_b["shipments"]["creation_day"] <= end)
        ]["quantity"].sum()

        print(f"  {label_a}: {a_qty:>15,.0f} cases")
        print(f"  {label_b}: {b_qty:>15,.0f} cases")

        diff = b_qty - a_qty
        pct = (diff / a_qty * 100) if a_qty > 0 else 0
        print(f"  Difference: {diff:>+15,.0f} cases ({pct:+.1f}%)")


def compare_day_by_day(
    scenario_a: dict, scenario_b: dict, start_day: int, end_day: int, label_a: str, label_b: str
) -> None:
    """Show day-by-day shipment comparison."""
    print("\n" + "=" * 60)
    print(f"DAY-BY-DAY SHIPMENTS (Days {start_day}-{end_day})")
    print("=" * 60)

    a_daily = scenario_a["shipments"].groupby("creation_day")["quantity"].sum()
    b_daily = scenario_b["shipments"].groupby("creation_day")["quantity"].sum()

    print(f"{'Day':<6} {label_a:>15} {label_b:>15} {'Diff':>15}")
    print("-" * 54)

    for day in range(start_day, end_day + 1):
        a = a_daily.get(day, 0)
        b = b_daily.get(day, 0)
        diff = b - a
        print(f"{day:<6} {a:>15,.0f} {b:>15,.0f} {diff:>+15,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Compare two simulation scenarios")
    parser.add_argument("scenario_a", type=Path, help="Path to first scenario results")
    parser.add_argument("scenario_b", type=Path, help="Path to second scenario results")
    parser.add_argument("--label-a", default="Scenario A", help="Label for first scenario")
    parser.add_argument("--label-b", default="Scenario B", help="Label for second scenario")
    parser.add_argument("--day-range", type=str, help="Day range for comparison, e.g., '120-145'")
    args = parser.parse_args()

    print(f"Loading {args.label_a} from {args.scenario_a}...")
    a = load_scenario(args.scenario_a)

    print(f"Loading {args.label_b} from {args.scenario_b}...")
    b = load_scenario(args.scenario_b)

    # Default day ranges for risk events
    day_ranges = [
        ("Port Strike (Day 120, 14 days)", (115, 145)),
        ("Cyber Outage (Day 200, 3 days)", (195, 215)),
    ]

    compare_shipments_around_days(a, b, day_ranges, args.label_a, args.label_b)

    if args.day_range:
        start, end = map(int, args.day_range.split("-"))
        compare_day_by_day(a, b, start, end, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
