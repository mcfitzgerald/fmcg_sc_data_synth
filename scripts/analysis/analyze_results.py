#!/usr/bin/env python
"""Analyze simulation output and generate summary report."""

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_results(results_dir: str) -> dict:
    """Analyze simulation results and return summary statistics."""
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    stats = {}

    # Load metrics if available
    metrics_file = results_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            stats["metrics"] = json.load(f)

    # Load triangle report
    triangle_file = results_path / "triangle_report.txt"
    if triangle_file.exists():
        stats["triangle_report"] = triangle_file.read_text()

    # Analyze orders
    orders_file = results_path / "orders.csv"
    if orders_file.exists():
        orders = pd.read_csv(orders_file)
        stats["orders"] = {
            "total_lines": len(orders),
            "total_quantity": float(orders["quantity"].sum()),
            "unique_products": int(orders["product_id"].nunique()),
            "unique_destinations": int(orders["target_id"].nunique()),
            "status_breakdown": orders["status"].value_counts().to_dict(),
            "daily_volume": orders.groupby("day")["quantity"].sum().describe().to_dict(),
        }

    # Analyze shipments
    shipments_file = results_path / "shipments.csv"
    if shipments_file.exists():
        shipments = pd.read_csv(shipments_file)
        stats["shipments"] = {
            "total_lines": len(shipments),
            "total_quantity": float(shipments["quantity"].sum()),
            "unique_sources": int(shipments["source_id"].nunique()),
            "unique_destinations": int(shipments["target_id"].nunique()),
            "status_breakdown": shipments["status"].value_counts().to_dict(),
            "daily_volume": shipments.groupby("creation_day")["quantity"].sum().describe().to_dict(),
        }

    # Analyze production batches
    batches_file = results_path / "batches.csv"
    if batches_file.exists():
        batches = pd.read_csv(batches_file)
        stats["production"] = {
            "total_batches": len(batches),
            "total_quantity": float(batches["quantity"].sum()),
            "unique_products": int(batches["product_id"].nunique()),
            "by_plant": batches.groupby("plant_id")["quantity"].sum().to_dict(),
            "production_days": int(batches["day_produced"].nunique()),
            "daily_production": batches.groupby("day_produced")["quantity"].sum().describe().to_dict(),
        }

    # Analyze inventory (sampled due to file size)
    inventory_file = results_path / "inventory.csv"
    if inventory_file.exists():
        # Read in chunks to handle large files
        chunks = pd.read_csv(inventory_file, chunksize=500000)
        day_totals = {}
        for chunk in chunks:
            for day, group in chunk.groupby("day"):
                if day not in day_totals:
                    day_totals[day] = {"actual": 0, "perceived": 0}
                day_totals[day]["actual"] += group["actual_inventory"].sum()
                day_totals[day]["perceived"] += group["perceived_inventory"].sum()

        if day_totals:
            days = sorted(day_totals.keys())
            first_inv = day_totals[days[0]]["actual"]
            last_inv = day_totals[days[-1]]["actual"]

            stats["inventory"] = {
                "snapshot_days": len(days),
                "first_day": int(days[0]),
                "last_day": int(days[-1]),
                "starting_inventory": float(first_inv),
                "ending_inventory": float(last_inv),
                "depletion_pct": float((first_inv - last_inv) / first_inv * 100),
                "trend": {int(d): float(day_totals[d]["actual"]) for d in days},
            }

    return stats


def print_report(stats: dict) -> None:
    """Print formatted analysis report."""
    print("=" * 60)
    print("         SIMULATION RESULTS ANALYSIS REPORT")
    print("=" * 60)

    # Triangle report
    if "triangle_report" in stats:
        print("\n" + stats["triangle_report"])

    # Metrics summary
    if "metrics" in stats:
        print("\n--- KEY METRICS ---")
        for metric, data in stats["metrics"].items():
            status = data.get("status", "N/A")
            mean = data.get("mean", "N/A")
            if isinstance(mean, float):
                mean = f"{mean:.2%}" if mean < 1 else f"{mean:.2f}"
            print(f"  {metric}: {mean} [{status}]")

    # Orders summary
    if "orders" in stats:
        o = stats["orders"]
        print("\n--- ORDERS ---")
        print(f"  Total order lines:     {o['total_lines']:,}")
        print(f"  Total quantity:        {o['total_quantity']:,.0f} cases")
        print(f"  Unique products:       {o['unique_products']}")
        print(f"  Unique destinations:   {o['unique_destinations']}")
        print(f"  Status breakdown:      {o['status_breakdown']}")

    # Shipments summary
    if "shipments" in stats:
        s = stats["shipments"]
        print("\n--- SHIPMENTS ---")
        print(f"  Total shipment lines:  {s['total_lines']:,}")
        print(f"  Total quantity:        {s['total_quantity']:,.0f} cases")
        print(f"  Unique sources:        {s['unique_sources']}")
        print(f"  Unique destinations:   {s['unique_destinations']}")
        print(f"  Status breakdown:      {s['status_breakdown']}")

    # Production summary
    if "production" in stats:
        p = stats["production"]
        print("\n--- PRODUCTION ---")
        print(f"  Total batches:         {p['total_batches']}")
        print(f"  Total produced:        {p['total_quantity']:,.0f} cases")
        print(f"  Days with production:  {p['production_days']}")
        print(f"  Unique products:       {p['unique_products']}")
        print("  By plant:")
        for plant, qty in sorted(p["by_plant"].items(), key=lambda x: -x[1]):
            pct = qty / p["total_quantity"] * 100
            print(f"    {plant}: {qty:,.0f} ({pct:.1f}%)")

    # Inventory summary
    if "inventory" in stats:
        i = stats["inventory"]
        print("\n--- INVENTORY ---")
        print(f"  Snapshots recorded:    {i['snapshot_days']} days")
        print(f"  Starting inventory:    {i['starting_inventory']:,.0f} cases")
        print(f"  Ending inventory:      {i['ending_inventory']:,.0f} cases")
        print(f"  Depletion:             {i['depletion_pct']:.1f}%")

    # Supply vs demand
    if "orders" in stats and "shipments" in stats:
        demand = stats["orders"]["total_quantity"]
        shipped = stats["shipments"]["total_quantity"]
        print("\n--- SUPPLY VS DEMAND ---")
        print(f"  Total demand:          {demand:,.0f} cases")
        print(f"  Total shipped:         {shipped:,.0f} cases")
        print(f"  Fulfillment rate:      {shipped/demand*100:.1f}%")

    if "orders" in stats and "production" in stats:
        demand = stats["orders"]["total_quantity"]
        produced = stats["production"]["total_quantity"]
        print(f"  Production vs demand:  {produced/demand*100:.1f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="data/results/365day_analysis",
        help="Path to results directory (default: data/results/365day_analysis)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted report",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save report to file",
    )

    args = parser.parse_args()

    stats = analyze_results(args.results_dir)

    if args.json:
        output = json.dumps(stats, indent=2, default=str)
        if args.output:
            Path(args.output).write_text(output)
            print(f"JSON report saved to {args.output}")
        else:
            print(output)
    elif args.output:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        print_report(stats)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        Path(args.output).write_text(output)
        print(f"Report saved to {args.output}")
    else:
        print_report(stats)


if __name__ == "__main__":
    main()
