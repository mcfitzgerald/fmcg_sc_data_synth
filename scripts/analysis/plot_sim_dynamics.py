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

    # 4. Inventory Mean (InvMean) - v0.90.0: Robust SKU-level mean
    inv_path = data_path / "inventory.parquet"
    if inv_path.exists():
        # Read day and inventory
        inv_df = pd.read_parquet(inv_path, columns=["day", "actual_inventory"])
        
        # v0.90.0 Fix: The log uses mean(inventory[demand > 0]).
        # To match the log's 15-20 scale on the chart, we filter out zeros.
        # This removes the sparsity bias (90% of node-SKU combos are zero).
        print("  Filtering out zero-inventory rows for accurate mean...")
        active_inv = inv_df[inv_df["actual_inventory"] > 0]
        daily_inv = active_inv.groupby("day")["actual_inventory"].mean()
    else:
        daily_inv = pd.Series()

    # Combine into single DataFrame
    df = pd.DataFrame({
        "Dmd": daily_dmd,
        "Ord": daily_ord,
        "Ship": daily_ship,
        "Arr": daily_arr,
        "Prod": daily_prod,
        "InvMean": daily_inv
    }).fillna(0)
    
    # Filter to 365 days
    df = df[(df.index >= 1) & (df.index <= 365)]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Quantity (Cases)")
    # Plot Dmd, Ord, Ship, Arr, Prod
    ax1.plot(df.index, df["Dmd"], label="Consumer Demand (Proxy)", color="black", linestyle=":", alpha=0.5)
    ax1.plot(df.index, df["Ord"], label="Replenishment Orders", color="blue", alpha=0.4)
    ax1.plot(df.index, df["Ship"], label="Shipments Created", color="green", alpha=0.4)
    ax1.plot(df.index, df["Arr"], label="Arrivals (Receipts)", color="orange", alpha=0.4)
    ax1.plot(df.index, df["Prod"], label="Production Completed", color="red", alpha=0.8, linewidth=2)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Inventory (per active node SKU)")
    ax2.plot(df.index, df["InvMean"], label="Mean Inventory", color="purple", linestyle="--", linewidth=2)
    ax2.tick_params(axis='y', labelcolor="purple")
    # Force axis to show lean range if possible, or let it auto-scale
    ax2.set_ylim(0, df["InvMean"].max() * 1.2)
    ax2.legend(loc='upper right')

    plt.title("Simulation Dynamics (365 Days) - Demand-Centric Model v0.90.0\nLean Priming + Pipeline Synchronization")
    plt.grid(True, alpha=0.3)
    
    output_file = data_path / "sim_dynamics.png"
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    
    # Also print a summary table for the terminal
    print("\nSimulation Metrics Summary:")
    print(df.describe().T)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/output")
    args = parser.parse_args()
    plot_sim_dynamics(args.data_dir)
