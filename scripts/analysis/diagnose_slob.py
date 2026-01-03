#!/usr/bin/env python3
"""
Diagnose SLOB (Slow/Obsolete) inventory in Prism Sim results.

Analyzes:
1. Inventory distribution by echelon (where is stock sitting?)
2. Days of Supply by product and location
3. Inventory velocity (turnover) by SKU
4. Stagnant inventory identification
5. Imbalance between upstream and downstream
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(results_dir: Path) -> dict:
    """Load simulation results."""
    data = {}

    # Inventory
    inventory_file = results_dir / "inventory.csv"
    if inventory_file.exists():
        chunks = []
        for chunk in pd.read_csv(inventory_file, chunksize=500000):
            chunks.append(chunk)
        data["inventory"] = pd.concat(chunks, ignore_index=True)
        print(f"  Loaded {len(data['inventory']):,} inventory records")

    # Shipments for velocity calculation
    shipments_file = results_dir / "shipments.csv"
    if shipments_file.exists():
        data["shipments"] = pd.read_csv(shipments_file)
        print(f"  Loaded {len(data['shipments']):,} shipment lines")

    # Batches for production reference
    batches_file = results_dir / "batches.csv"
    if batches_file.exists():
        data["batches"] = pd.read_csv(batches_file)
        print(f"  Loaded {len(data['batches']):,} batch records")

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


def is_finished_good(product_id: str) -> bool:
    """Check if product is a finished good (not raw material)."""
    return product_id.startswith("SKU-")


def analyze_inventory_by_echelon(inventory: pd.DataFrame) -> pd.DataFrame:
    """Analyze inventory distribution across echelons."""

    # Get latest day snapshot
    latest_day = inventory["day"].max()
    latest_inv = inventory[inventory["day"] == latest_day].copy()

    # Filter to finished goods only
    latest_inv = latest_inv[latest_inv["product_id"].apply(is_finished_good)]

    # Classify nodes
    latest_inv["echelon"] = latest_inv["node_id"].apply(classify_node)

    # Aggregate by echelon
    echelon_inv = latest_inv.groupby("echelon").agg({
        "actual_inventory": "sum",
        "node_id": "nunique",
        "product_id": "nunique"
    }).rename(columns={
        "actual_inventory": "inventory",
        "node_id": "node_count",
        "product_id": "sku_count"
    })

    # Calculate percentages
    total_inv = echelon_inv["inventory"].sum()
    echelon_inv["pct_of_total"] = echelon_inv["inventory"] / total_inv * 100
    echelon_inv["avg_per_node"] = echelon_inv["inventory"] / echelon_inv["node_count"]

    return echelon_inv.sort_values("inventory", ascending=False)


def analyze_inventory_trend_by_echelon(inventory: pd.DataFrame) -> pd.DataFrame:
    """Track inventory levels over time by echelon."""

    # Filter to finished goods
    fg_inv = inventory[inventory["product_id"].apply(is_finished_good)].copy()
    fg_inv["echelon"] = fg_inv["node_id"].apply(classify_node)

    # Aggregate by day and echelon
    trend = fg_inv.groupby(["day", "echelon"])["actual_inventory"].sum().unstack(fill_value=0)

    return trend


def analyze_days_of_supply(inventory: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    """Calculate days of supply by product at each echelon."""

    # Get latest inventory
    latest_day = inventory["day"].max()
    latest_inv = inventory[inventory["day"] == latest_day].copy()
    latest_inv = latest_inv[latest_inv["product_id"].apply(is_finished_good)]
    latest_inv["echelon"] = latest_inv["node_id"].apply(classify_node)

    # Calculate average daily demand from shipments (outbound flow)
    # Use last 30 days of shipments
    recent_ships = shipments[shipments["creation_day"] > latest_day - 30]
    daily_demand = recent_ships.groupby("product_id")["quantity"].sum() / 30

    # Aggregate inventory by product and echelon
    inv_by_prod_echelon = latest_inv.groupby(["product_id", "echelon"])["actual_inventory"].sum().reset_index()

    # Merge with demand
    inv_by_prod_echelon = inv_by_prod_echelon.merge(
        daily_demand.reset_index().rename(columns={"quantity": "daily_demand"}),
        on="product_id",
        how="left"
    )

    # Calculate DOS
    inv_by_prod_echelon["days_of_supply"] = (
        inv_by_prod_echelon["actual_inventory"] /
        inv_by_prod_echelon["daily_demand"].replace(0, np.nan)
    )

    return inv_by_prod_echelon


def analyze_inventory_velocity(inventory: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    """Calculate inventory turnover velocity by product."""

    # Get first and last day inventory
    first_day = inventory["day"].min()
    last_day = inventory["day"].max()
    sim_days = last_day - first_day + 1

    first_inv = inventory[inventory["day"] == first_day].copy()
    last_inv = inventory[inventory["day"] == last_day].copy()

    # Filter to finished goods
    first_inv = first_inv[first_inv["product_id"].apply(is_finished_good)]
    last_inv = last_inv[last_inv["product_id"].apply(is_finished_good)]

    # Calculate average inventory by product
    first_by_prod = first_inv.groupby("product_id")["actual_inventory"].sum()
    last_by_prod = last_inv.groupby("product_id")["actual_inventory"].sum()
    avg_inv = (first_by_prod + last_by_prod) / 2

    # Calculate total throughput (shipments)
    ship_by_prod = shipments[shipments["product_id"].apply(is_finished_good)].groupby("product_id")["quantity"].sum()

    # Calculate annualized turns
    result = pd.DataFrame({
        "avg_inventory": avg_inv,
        "total_shipped": ship_by_prod,
        "first_inv": first_by_prod,
        "last_inv": last_by_prod,
    }).fillna(0)

    result["annualized_throughput"] = result["total_shipped"] * (365 / sim_days)
    result["turns"] = result["annualized_throughput"] / result["avg_inventory"].replace(0, np.nan)
    result["inv_change"] = result["last_inv"] - result["first_inv"]
    result["inv_change_pct"] = result["inv_change"] / result["first_inv"].replace(0, np.nan) * 100

    return result.sort_values("turns")


def identify_slob_products(dos_by_prod: pd.DataFrame, threshold_days: int = 60) -> pd.DataFrame:
    """Identify products with SLOB (>threshold DOS)."""

    # Focus on MFG_RDC since that's where SLOB typically sits
    rdc_dos = dos_by_prod[dos_by_prod["echelon"] == "MFG_RDC"].copy()

    slob = rdc_dos[rdc_dos["days_of_supply"] > threshold_days].copy()
    slob = slob.sort_values("days_of_supply", ascending=False)

    return slob


def analyze_inventory_imbalance(echelon_inv: pd.DataFrame) -> dict:
    """Analyze inventory imbalance between echelons."""

    total = echelon_inv["inventory"].sum()

    # Calculate ideal distribution (rough heuristic)
    # Stores: ~20%, Customer DCs: ~30%, MFG RDCs: ~40%, Plants: ~10%

    actual = {}
    for echelon in echelon_inv.index:
        actual[echelon] = echelon_inv.loc[echelon, "pct_of_total"]

    # Check for common imbalances
    issues = []

    store_pct = actual.get("STORE", 0)
    rdc_pct = actual.get("MFG_RDC", 0)
    dc_pct = actual.get("RETAILER_DC", 0) + actual.get("DISTRIBUTOR_DC", 0)

    if store_pct < 5:
        issues.append(f"CRITICAL: Only {store_pct:.1f}% inventory at stores (should be ~20%)")
    if rdc_pct > 60:
        issues.append(f"WARNING: {rdc_pct:.1f}% inventory stuck at MFG RDCs (should be ~40%)")
    if dc_pct < 10:
        issues.append(f"WARNING: Only {dc_pct:.1f}% at customer DCs (should be ~30%)")

    return {
        "actual_distribution": actual,
        "store_pct": store_pct,
        "rdc_pct": rdc_pct,
        "dc_pct": dc_pct,
        "issues": issues,
    }


def analyze_production_vs_demand(batches: pd.DataFrame, shipments: pd.DataFrame) -> dict:
    """Compare production output to demand signal."""

    # Total production
    total_produced = batches["quantity"].sum()
    prod_days = batches["day_produced"].max() - batches["day_produced"].min() + 1

    # Total shipped (demand proxy)
    total_shipped = shipments["quantity"].sum()
    ship_days = shipments["creation_day"].max() - shipments["creation_day"].min() + 1

    daily_prod = total_produced / prod_days
    daily_ship = total_shipped / ship_days

    return {
        "total_produced": total_produced,
        "total_shipped": total_shipped,
        "daily_production": daily_prod,
        "daily_shipments": daily_ship,
        "production_to_ship_ratio": total_produced / total_shipped if total_shipped > 0 else 0,
        "excess_production": total_produced - total_shipped,
    }


def print_diagnostic_report(
    echelon_inv: pd.DataFrame,
    dos_analysis: pd.DataFrame,
    velocity: pd.DataFrame,
    slob_products: pd.DataFrame,
    imbalance: dict,
    prod_vs_demand: dict,
    inv_trend: pd.DataFrame,
) -> None:
    """Print comprehensive SLOB diagnostic report."""

    print("=" * 70)
    print("         SLOB (SLOW/OBSOLETE) INVENTORY DIAGNOSTIC REPORT")
    print("=" * 70)

    # Inventory distribution
    print("\n--- INVENTORY DISTRIBUTION BY ECHELON ---")
    for echelon, row in echelon_inv.iterrows():
        print(f"  {echelon:20s}: {row['inventory']:>15,.0f} ({row['pct_of_total']:>5.1f}%)  [{int(row['node_count'])} nodes]")

    total_inv = echelon_inv["inventory"].sum()
    print(f"  {'TOTAL':20s}: {total_inv:>15,.0f}")

    # Imbalance analysis
    print("\n--- INVENTORY IMBALANCE ANALYSIS ---")
    print(f"  Store inventory:     {imbalance['store_pct']:.1f}% (target: ~20%)")
    print(f"  Customer DC inv:     {imbalance['dc_pct']:.1f}% (target: ~30%)")
    print(f"  MFG RDC inventory:   {imbalance['rdc_pct']:.1f}% (target: ~40%)")

    if imbalance["issues"]:
        print("\n  ISSUES DETECTED:")
        for issue in imbalance["issues"]:
            print(f"    - {issue}")

    # Production vs demand
    print("\n--- PRODUCTION VS DEMAND ---")
    print(f"  Total produced:      {prod_vs_demand['total_produced']:>15,.0f}")
    print(f"  Total shipped:       {prod_vs_demand['total_shipped']:>15,.0f}")
    print(f"  Excess production:   {prod_vs_demand['excess_production']:>15,.0f}")
    print(f"  Production ratio:    {prod_vs_demand['production_to_ship_ratio']:.2f}x")

    # Velocity analysis
    print("\n--- INVENTORY VELOCITY (TURNS) ---")
    print("  Slowest moving products:")
    for prod_id, row in velocity.head(10).iterrows():
        turns = row['turns'] if not np.isnan(row['turns']) else 0
        print(f"    {prod_id:20s}: {turns:>6.1f}x turns  (inv: {row['avg_inventory']:>12,.0f})")

    print("\n  Fastest moving products:")
    for prod_id, row in velocity.tail(5).iloc[::-1].iterrows():
        turns = row['turns'] if not np.isnan(row['turns']) else 0
        print(f"    {prod_id:20s}: {turns:>6.1f}x turns  (inv: {row['avg_inventory']:>12,.0f})")

    # SLOB products
    print("\n--- SLOB PRODUCTS (>60 Days of Supply at RDCs) ---")
    if len(slob_products) > 0:
        for _, row in slob_products.head(15).iterrows():
            print(f"    {row['product_id']:20s}: {row['days_of_supply']:>6.0f} DOS  (inv: {row['actual_inventory']:>12,.0f})")
        print(f"\n  Total SLOB products: {len(slob_products)}")
        slob_inv = slob_products["actual_inventory"].sum()
        print(f"  Total SLOB inventory: {slob_inv:,.0f} ({slob_inv/total_inv*100:.1f}% of total)")
    else:
        print("    No SLOB products identified")

    # Inventory trend over time
    print("\n--- INVENTORY TREND BY ECHELON ---")
    if len(inv_trend) > 0:
        first_day = inv_trend.index.min()
        last_day = inv_trend.index.max()

        print(f"  {'Echelon':20s} {'Day '+str(first_day):>15s} {'Day '+str(last_day):>15s} {'Change':>10s}")
        print("  " + "-" * 62)
        for col in inv_trend.columns:
            start = inv_trend.iloc[0][col]
            end = inv_trend.iloc[-1][col]
            change = (end - start) / start * 100 if start > 0 else 0
            print(f"  {col:20s} {start:>15,.0f} {end:>15,.0f} {change:>+9.1f}%")

    # Root cause hypothesis
    print("\n" + "=" * 70)
    print("         ROOT CAUSE HYPOTHESIS")
    print("=" * 70)

    if imbalance["rdc_pct"] > 50:
        print("\n  INVENTORY STUCK UPSTREAM")
        print("  -> Majority of inventory sitting at MFG RDCs")
        print("  -> Flow bottleneck between RDCs and Customer DCs")
        print("  -> Possible causes:")
        print("     1. Customer DC replenishment signals too weak")
        print("     2. RDC-to-DC shipping constraints")
        print("     3. Customer DC order frequency too low")

    if imbalance["store_pct"] < 10:
        print("\n  STORE INVENTORY STARVATION")
        print("  -> Stores holding insufficient inventory")
        print("  -> Possible causes:")
        print("     1. DC-to-Store replenishment bottleneck")
        print("     2. Store reorder points too low")
        print("     3. LTL shipping delays")

    if prod_vs_demand["production_to_ship_ratio"] > 1.1:
        print("\n  OVERPRODUCTION")
        print(f"  -> Producing {(prod_vs_demand['production_to_ship_ratio']-1)*100:.0f}% more than shipping")
        print("  -> MRP demand signal may be inflated")
        print("  -> Or downstream flow constraints blocking shipments")

    avg_turns = velocity["turns"].median()
    if avg_turns < 6:
        print(f"\n  LOW INVENTORY TURNS ({avg_turns:.1f}x)")
        print("  -> Inventory not flowing fast enough")
        print("  -> Either too much stock or too little demand")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose SLOB inventory issues")
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("data/results/v0.19_365day"),
        help="Path to results directory",
    )
    parser.add_argument("--csv", action="store_true", help="Export detailed CSV files")
    parser.add_argument("--dos-threshold", type=int, default=60, help="DOS threshold for SLOB (default: 60)")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist")
        return 1

    print(f"Loading data from {args.results_dir}...")
    data = load_data(args.results_dir)

    if "inventory" not in data:
        print("Error: Missing inventory.csv")
        return 1

    print("\nAnalyzing inventory by echelon...")
    echelon_inv = analyze_inventory_by_echelon(data["inventory"])

    print("Analyzing inventory trend...")
    inv_trend = analyze_inventory_trend_by_echelon(data["inventory"])

    dos_analysis = pd.DataFrame()
    slob_products = pd.DataFrame()
    if "shipments" in data:
        print("Analyzing days of supply...")
        dos_analysis = analyze_days_of_supply(data["inventory"], data["shipments"])

        print("Identifying SLOB products...")
        slob_products = identify_slob_products(dos_analysis, args.dos_threshold)

    velocity = pd.DataFrame()
    if "shipments" in data:
        print("Analyzing inventory velocity...")
        velocity = analyze_inventory_velocity(data["inventory"], data["shipments"])

    print("Analyzing imbalances...")
    imbalance = analyze_inventory_imbalance(echelon_inv)

    prod_vs_demand = {}
    if "batches" in data and "shipments" in data:
        print("Analyzing production vs demand...")
        prod_vs_demand = analyze_production_vs_demand(data["batches"], data["shipments"])

    print_diagnostic_report(
        echelon_inv,
        dos_analysis,
        velocity,
        slob_products,
        imbalance,
        prod_vs_demand,
        inv_trend,
    )

    if args.csv:
        output_dir = args.results_dir / "diagnostics"
        output_dir.mkdir(exist_ok=True)
        echelon_inv.to_csv(output_dir / "inventory_by_echelon.csv")
        velocity.to_csv(output_dir / "inventory_velocity.csv")
        if len(slob_products) > 0:
            slob_products.to_csv(output_dir / "slob_products.csv", index=False)
        if len(dos_analysis) > 0:
            dos_analysis.to_csv(output_dir / "days_of_supply.csv", index=False)
        print(f"\nCSV files exported to {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
