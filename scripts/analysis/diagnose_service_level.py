#!/usr/bin/env python3
"""
Diagnose service level degradation in Prism Sim results.

Analyzes:
1. Service level trend over time (weekly rolling average)
2. Service level by echelon (Stores vs DCs)
3. Service level by product category
4. Inventory availability vs demand patterns
5. Bottleneck identification (which nodes/products fail most)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(results_dir: Path) -> dict:
    """Load simulation results with chunked reading for large files."""
    data = {}

    # Orders - manageable size
    orders_file = results_dir / "orders.csv"
    if orders_file.exists():
        data["orders"] = pd.read_csv(orders_file)
        print(f"  Loaded {len(data['orders']):,} order lines")

    # Shipments - can be large
    shipments_file = results_dir / "shipments.csv"
    if shipments_file.exists():
        data["shipments"] = pd.read_csv(shipments_file)
        print(f"  Loaded {len(data['shipments']):,} shipment lines")

    # Inventory - sample for large files
    inventory_file = results_dir / "inventory.csv"
    if inventory_file.exists():
        # Read in chunks and aggregate by day/node type
        chunks = []
        for chunk in pd.read_csv(inventory_file, chunksize=500000):
            chunks.append(chunk)
        data["inventory"] = pd.concat(chunks, ignore_index=True)
        print(f"  Loaded {len(data['inventory']):,} inventory records")

    return data


def classify_node(node_id: str) -> str:
    """Classify node into echelon type."""
    if node_id.startswith("STORE-"):
        return "STORE"
    elif node_id.startswith("RET-DC-"):
        return "RETAILER_DC"
    elif node_id.startswith("DIST-DC-"):
        return "DISTRIBUTOR_DC"
    elif node_id.startswith("ECOM-FC-"):
        return "ECOM_FC"
    elif node_id.startswith("DTC-FC-"):
        return "DTC_FC"
    elif node_id.startswith("RDC-"):
        return "MFG_RDC"
    elif node_id.startswith("PLANT-"):
        return "PLANT"
    elif node_id.startswith("SUP-"):
        return "SUPPLIER"
    elif node_id.startswith("CLUB-"):
        return "CLUB"
    else:
        return "OTHER"


def get_product_category(product_id: str) -> str:
    """Extract product category from product ID."""
    if "ORAL" in product_id:
        return "ORAL_CARE"
    elif "PERSONAL" in product_id:
        return "PERSONAL_WASH"
    elif "HOME" in product_id:
        return "HOME_CARE"
    elif "ING-" in product_id or "PKG-" in product_id:
        return "RAW_MATERIAL"
    else:
        return "OTHER"


def analyze_service_level_trend(orders: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily and weekly service level trends."""

    # Aggregate orders by day and target
    daily_orders = orders.groupby(["day", "target_id", "product_id"])["quantity"].sum().reset_index()
    daily_orders.columns = ["day", "target_id", "product_id", "ordered"]

    # Aggregate shipments by arrival day and target
    daily_shipments = shipments.groupby(["arrival_day", "target_id", "product_id"])["quantity"].sum().reset_index()
    daily_shipments.columns = ["day", "target_id", "product_id", "shipped"]

    # Merge orders and shipments
    merged = daily_orders.merge(
        daily_shipments,
        on=["day", "target_id", "product_id"],
        how="left"
    ).fillna(0)

    # Calculate daily service level
    daily_sl = merged.groupby("day").agg({
        "ordered": "sum",
        "shipped": "sum"
    }).reset_index()
    daily_sl["service_level"] = (daily_sl["shipped"] / daily_sl["ordered"]).clip(0, 1)

    # Add 7-day rolling average
    daily_sl["sl_7d_avg"] = daily_sl["service_level"].rolling(7, min_periods=1).mean()

    # Add 30-day rolling average
    daily_sl["sl_30d_avg"] = daily_sl["service_level"].rolling(30, min_periods=1).mean()

    return daily_sl


def analyze_service_by_echelon(orders: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    """Analyze service level by supply chain echelon."""

    # Add echelon classification
    orders["echelon"] = orders["target_id"].apply(classify_node)
    shipments["echelon"] = shipments["target_id"].apply(classify_node)

    # Aggregate by echelon
    order_by_echelon = orders.groupby("echelon")["quantity"].sum()
    ship_by_echelon = shipments.groupby("echelon")["quantity"].sum()

    result = pd.DataFrame({
        "ordered": order_by_echelon,
        "shipped": ship_by_echelon
    }).fillna(0)
    result["service_level"] = (result["shipped"] / result["ordered"]).clip(0, 1)
    result["gap"] = result["ordered"] - result["shipped"]

    return result.sort_values("gap", ascending=False)


def analyze_service_by_product(orders: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    """Analyze service level by product category."""

    orders["category"] = orders["product_id"].apply(get_product_category)
    shipments["category"] = shipments["product_id"].apply(get_product_category)

    order_by_cat = orders.groupby("category")["quantity"].sum()
    ship_by_cat = shipments.groupby("category")["quantity"].sum()

    result = pd.DataFrame({
        "ordered": order_by_cat,
        "shipped": ship_by_cat
    }).fillna(0)
    result["service_level"] = (result["shipped"] / result["ordered"]).clip(0, 1)
    result["gap"] = result["ordered"] - result["shipped"]

    return result.sort_values("service_level")


def analyze_worst_performers(orders: pd.DataFrame, shipments: pd.DataFrame, top_n: int = 20) -> dict:
    """Identify worst performing nodes and products."""

    # By target node
    order_by_node = orders.groupby("target_id")["quantity"].sum()
    ship_by_node = shipments.groupby("target_id")["quantity"].sum()

    node_sl = pd.DataFrame({
        "ordered": order_by_node,
        "shipped": ship_by_node
    }).fillna(0)
    node_sl["service_level"] = (node_sl["shipped"] / node_sl["ordered"]).clip(0, 1)
    node_sl["gap"] = node_sl["ordered"] - node_sl["shipped"]

    # By product
    order_by_prod = orders.groupby("product_id")["quantity"].sum()
    ship_by_prod = shipments.groupby("product_id")["quantity"].sum()

    prod_sl = pd.DataFrame({
        "ordered": order_by_prod,
        "shipped": ship_by_prod
    }).fillna(0)
    prod_sl["service_level"] = (prod_sl["shipped"] / prod_sl["ordered"]).clip(0, 1)
    prod_sl["gap"] = prod_sl["ordered"] - prod_sl["shipped"]

    return {
        "worst_nodes": node_sl.nsmallest(top_n, "service_level"),
        "worst_products": prod_sl.nsmallest(top_n, "service_level"),
        "biggest_gaps_nodes": node_sl.nlargest(top_n, "gap"),
        "biggest_gaps_products": prod_sl.nlargest(top_n, "gap"),
    }


def analyze_inventory_availability(inventory: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Compare inventory levels against demand over time."""

    # Filter to stores only
    store_inv = inventory[inventory["node_id"].str.startswith("STORE-")].copy()
    store_orders = orders[orders["target_id"].str.startswith("STORE-")].copy()

    # Aggregate by day
    inv_by_day = store_inv.groupby("day")["actual_inventory"].sum()
    orders_by_day = store_orders.groupby("day")["quantity"].sum()

    result = pd.DataFrame({
        "inventory": inv_by_day,
        "demand": orders_by_day
    }).fillna(0)

    result["days_of_supply"] = result["inventory"] / result["demand"].replace(0, np.nan)
    result["inventory_ratio"] = result["inventory"] / result["inventory"].iloc[0] if len(result) > 0 else 0

    return result


def analyze_service_degradation_phases(daily_sl: pd.DataFrame) -> dict:
    """Identify phases of service level degradation."""

    # Split into quarters
    max_day = daily_sl["day"].max()
    q1 = daily_sl[daily_sl["day"] <= max_day * 0.25]["service_level"].mean()
    q2 = daily_sl[(daily_sl["day"] > max_day * 0.25) & (daily_sl["day"] <= max_day * 0.5)]["service_level"].mean()
    q3 = daily_sl[(daily_sl["day"] > max_day * 0.5) & (daily_sl["day"] <= max_day * 0.75)]["service_level"].mean()
    q4 = daily_sl[daily_sl["day"] > max_day * 0.75]["service_level"].mean()

    # Find inflection points (where degradation accelerates)
    sl_diff = daily_sl["sl_7d_avg"].diff()
    worst_decline_day = daily_sl.loc[sl_diff.idxmin(), "day"] if len(sl_diff) > 0 else 0

    # Find stabilization point (if any)
    late_variance = daily_sl[daily_sl["day"] > max_day * 0.8]["service_level"].std()

    return {
        "q1_avg": q1,
        "q2_avg": q2,
        "q3_avg": q3,
        "q4_avg": q4,
        "degradation_q1_to_q4": q1 - q4,
        "worst_decline_day": worst_decline_day,
        "late_variance": late_variance,
        "stabilized": late_variance < 0.05,
    }


def print_diagnostic_report(
    daily_sl: pd.DataFrame,
    echelon_sl: pd.DataFrame,
    product_sl: pd.DataFrame,
    worst: dict,
    inventory_avail: pd.DataFrame,
    phases: dict,
) -> None:
    """Print comprehensive diagnostic report."""

    print("=" * 70)
    print("         SERVICE LEVEL DIAGNOSTIC REPORT")
    print("=" * 70)

    # Overall stats
    print("\n--- OVERALL SERVICE LEVEL ---")
    print(f"  Mean:     {daily_sl['service_level'].mean():.1%}")
    print(f"  Std Dev:  {daily_sl['service_level'].std():.1%}")
    print(f"  Min:      {daily_sl['service_level'].min():.1%}")
    print(f"  Max:      {daily_sl['service_level'].max():.1%}")
    print(f"  Day 1:    {daily_sl.iloc[0]['service_level']:.1%}")
    print(f"  Day {int(daily_sl.iloc[-1]['day'])}:  {daily_sl.iloc[-1]['service_level']:.1%}")

    # Degradation analysis
    print("\n--- DEGRADATION ANALYSIS ---")
    print(f"  Q1 Average:  {phases['q1_avg']:.1%}")
    print(f"  Q2 Average:  {phases['q2_avg']:.1%}")
    print(f"  Q3 Average:  {phases['q3_avg']:.1%}")
    print(f"  Q4 Average:  {phases['q4_avg']:.1%}")
    print(f"  Total Drop:  {phases['degradation_q1_to_q4']:.1%}")
    print(f"  Worst Decline Day: {phases['worst_decline_day']}")
    print(f"  Stabilized in Q4:  {'Yes' if phases['stabilized'] else 'No'}")

    # By echelon
    print("\n--- SERVICE LEVEL BY ECHELON ---")
    for echelon, row in echelon_sl.iterrows():
        print(f"  {echelon:20s}: {row['service_level']:.1%} (gap: {row['gap']:,.0f})")

    # By product category
    print("\n--- SERVICE LEVEL BY PRODUCT CATEGORY ---")
    for cat, row in product_sl.iterrows():
        print(f"  {cat:20s}: {row['service_level']:.1%} (gap: {row['gap']:,.0f})")

    # Worst performers
    print("\n--- WORST PERFORMING PRODUCTS (by fill rate) ---")
    for prod_id, row in worst["worst_products"].head(10).iterrows():
        print(f"  {prod_id:20s}: {row['service_level']:.1%} (ordered: {row['ordered']:,.0f})")

    print("\n--- BIGGEST SUPPLY GAPS (by volume) ---")
    for prod_id, row in worst["biggest_gaps_products"].head(10).iterrows():
        print(f"  {prod_id:20s}: gap={row['gap']:,.0f} ({row['service_level']:.1%})")

    # Inventory analysis
    if len(inventory_avail) > 0:
        print("\n--- STORE INVENTORY ANALYSIS ---")
        print(f"  Starting DOS:  {inventory_avail['days_of_supply'].iloc[0]:.1f} days")
        print(f"  Ending DOS:    {inventory_avail['days_of_supply'].iloc[-1]:.1f} days")
        print(f"  Inventory retained: {inventory_avail['inventory_ratio'].iloc[-1]:.1%}")

    # Root cause hypothesis
    print("\n" + "=" * 70)
    print("         ROOT CAUSE HYPOTHESIS")
    print("=" * 70)

    if phases["degradation_q1_to_q4"] > 0.15:
        print("\n  SIGNIFICANT DEGRADATION DETECTED")

        # Check if it's an inventory drain
        if len(inventory_avail) > 0 and inventory_avail['inventory_ratio'].iloc[-1] < 0.5:
            print("  -> Inventory is depleting faster than replenishment")
            print("     Likely cause: Upstream supply cannot keep pace with demand")

        # Check echelon patterns
        store_sl = echelon_sl.loc["STORE", "service_level"] if "STORE" in echelon_sl.index else 1.0
        dc_sl = echelon_sl.loc["RETAILER_DC", "service_level"] if "RETAILER_DC" in echelon_sl.index else 1.0

        if store_sl < dc_sl - 0.1:
            print("  -> Store service worse than DC service")
            print("     Likely cause: DC-to-Store replenishment bottleneck")
        elif dc_sl < store_sl - 0.1:
            print("  -> DC service worse than Store service")
            print("     Likely cause: RDC-to-DC supply constraint")

        # Check product patterns
        worst_cat = product_sl.idxmin()["service_level"]
        print(f"  -> Worst category: {worst_cat}")

        if not phases["stabilized"]:
            print("  -> System NOT stabilizing - continuous degradation")
    else:
        print("\n  Service level relatively stable")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose service level issues")
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("data/results/v0.19_365day"),
        help="Path to results directory",
    )
    parser.add_argument("--csv", action="store_true", help="Export detailed CSV files")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist")
        return 1

    print(f"Loading data from {args.results_dir}...")
    data = load_data(args.results_dir)

    if "orders" not in data or "shipments" not in data:
        print("Error: Missing orders.csv or shipments.csv")
        return 1

    print("\nAnalyzing service level trends...")
    daily_sl = analyze_service_level_trend(data["orders"], data["shipments"])

    print("Analyzing by echelon...")
    echelon_sl = analyze_service_by_echelon(data["orders"], data["shipments"])

    print("Analyzing by product...")
    product_sl = analyze_service_by_product(data["orders"], data["shipments"])

    print("Finding worst performers...")
    worst = analyze_worst_performers(data["orders"], data["shipments"])

    inventory_avail = pd.DataFrame()
    if "inventory" in data:
        print("Analyzing inventory availability...")
        inventory_avail = analyze_inventory_availability(data["inventory"], data["orders"])

    print("Analyzing degradation phases...")
    phases = analyze_service_degradation_phases(daily_sl)

    print_diagnostic_report(daily_sl, echelon_sl, product_sl, worst, inventory_avail, phases)

    if args.csv:
        output_dir = args.results_dir / "diagnostics"
        output_dir.mkdir(exist_ok=True)
        daily_sl.to_csv(output_dir / "service_level_daily.csv", index=False)
        echelon_sl.to_csv(output_dir / "service_level_by_echelon.csv")
        product_sl.to_csv(output_dir / "service_level_by_product.csv")
        worst["worst_products"].to_csv(output_dir / "worst_products.csv")
        worst["biggest_gaps_products"].to_csv(output_dir / "biggest_gaps.csv")
        print(f"\nCSV files exported to {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
