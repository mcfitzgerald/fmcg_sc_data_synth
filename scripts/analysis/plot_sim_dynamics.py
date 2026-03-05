#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_sim_dynamics(data_dir="data/output"):
    print(f"Extracting sim dynamics from {data_dir}...")
    data_path = Path(data_dir)
    
    # 1. Orders & Demand Proxy
    orders_path = data_path / "orders.parquet"
    if orders_path.exists():
        orders_df = pd.read_parquet(orders_path, columns=["day", "quantity", "requested_date"])
        daily_ord = orders_df.groupby("day")["quantity"].sum()
        daily_dmd = orders_df.groupby("requested_date")["quantity"].sum()
    else:
        daily_ord = pd.Series()
        daily_dmd = pd.Series()

    # 2. Shipments (Ship)
    ships_path = data_path / "shipments.parquet"
    if ships_path.exists():
        ships_df = pd.read_parquet(ships_path, columns=["creation_day", "quantity", "arrival_day"])
        daily_ship = ships_df.groupby("creation_day")["quantity"].sum()
        daily_arr = ships_df.groupby("arrival_day")["quantity"].sum()
    else:
        daily_ship = pd.Series()
        daily_arr = pd.Series()

    # 3. Production (Prod)
    batches_path = data_path / "batches.parquet"
    if batches_path.exists():
        batches_df = pd.read_parquet(batches_path, columns=["day_produced", "quantity"])
        daily_prod = batches_df.groupby("day_produced")["quantity"].sum()
    else:
        daily_prod = pd.Series()

    # 4. Inventory Mean (InvMean) - v0.91.0: Robust SKU-level mean
    inv_path = data_path / "inventory.parquet"
    if inv_path.exists():
        inv_df = pd.read_parquet(inv_path, columns=["day", "actual_inventory"])
        print("  Filtering out zero-inventory rows for accurate mean...")
        active_inv = inv_df[inv_df["actual_inventory"] > 0]
        daily_inv = active_inv.groupby("day")["actual_inventory"].mean()
    else:
        daily_inv = pd.Series()

    # Combine into single DataFrame
    # Note: Use fillna(0) for flows, but ffill() for state (inventory)
    df = pd.DataFrame({
        "Dmd": daily_dmd,
        "Ord": daily_ord,
        "Ship": daily_ship,
        "Arr": daily_arr,
        "Prod": daily_prod,
    }).fillna(0)
    
    # Merge inventory and forward fill snapshots
    df = df.join(pd.DataFrame({"InvMean": daily_inv}), how='left')
    df["InvMean"] = df["InvMean"].ffill().bfill()
    
    # Filter to actual data range (exclude day 0 if needed)
    df = df[df.index >= 1]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(16, 9))

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Flow Quantity (Cases)")
    ax1.plot(df.index, df["Dmd"], label="Consumer Demand (Proxy)", color="black", linestyle=":", alpha=0.5)
    ax1.plot(df.index, df["Ord"], label="Replenishment Orders", color="blue", alpha=0.3)
    ax1.plot(df.index, df["Ship"], label="Shipments Created", color="green", alpha=0.3)
    ax1.plot(df.index, df["Arr"], label="Arrivals (Receipts)", color="orange", alpha=0.3)
    ax1.plot(df.index, df["Prod"], label="Production Completed", color="red", alpha=0.8, linewidth=2)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Inventory (per active node SKU)")
    ax2.plot(df.index, df["InvMean"], label="Mean Inventory", color="purple", linestyle="--", linewidth=2)
    ax2.tick_params(axis='y', labelcolor="purple")
    # Axis padding
    if not df["InvMean"].empty:
        ax2.set_ylim(0, df["InvMean"].max() * 1.5)
    ax2.legend(loc='upper right')

    plt.title(f"Simulation Dynamics ({len(df)} Days) - Demand-Centric model v0.91.0\nIntegral Priming Convergence Check")
    plt.grid(True, alpha=0.3)
    
    output_file = data_path / "sim_dynamics.png"
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    
    # Summary stats
    print(f"\nSimulation Metrics Summary ({len(df)} days):")
    stats = df.describe(percentiles=[0.1, 0.5, 0.9]).T
    print(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/output")
    args = parser.parse_args()
    plot_sim_dynamics(args.data_dir)
