#!/usr/bin/env python3
import pandas as pd
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
        # day = creation_day, requested_date = target arrival day (proxy for demand timing)
        orders_df = pd.read_parquet(orders_path, columns=["day", "quantity", "requested_date"])
        daily_ord = orders_df.groupby("day")["quantity"].sum()
        daily_dmd = orders_df.groupby("requested_date")["quantity"].sum()
    else:
        daily_ord = pd.Series()
        daily_dmd = pd.Series()

    # 2. Shipments (Ship)
    ships_path = data_path / "shipments.parquet"
    if ships_path.exists():
        # creation_day, arrival_day, quantity
        ships_df = pd.read_parquet(ships_path, columns=["creation_day", "quantity", "arrival_day"])
        daily_ship = ships_df.groupby("creation_day")["quantity"].sum()
        daily_arr = ships_df.groupby("arrival_day")["quantity"].sum()
    else:
        daily_ship = pd.Series()
        daily_arr = pd.Series()

    # 3. Production (Prod)
    batches_path = data_path / "batches.parquet"
    if batches_path.exists():
        # day_produced
        batches_df = pd.read_parquet(batches_path, columns=["day_produced", "quantity"])
        daily_prod = batches_df.groupby("day_produced")["quantity"].sum()
    else:
        daily_prod = pd.Series()

    # 4. Inventory Mean (InvMean)
    inv_path = data_path / "inventory.parquet"
    if inv_path.exists():
        # day, actual_inventory
        inv_df = pd.read_parquet(inv_path, columns=["day", "actual_inventory"])
        daily_inv = inv_df.groupby("day")["actual_inventory"].mean()
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
    df = df[df.index <= 365]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Quantity (Cases)")
    ax1.plot(df.index, df["Dmd"], label="Consumer Demand (Proxy)", color="black", linestyle=":", alpha=0.5)
    ax1.plot(df.index, df["Ord"], label="Replenishment Orders", color="blue", alpha=0.6)
    ax1.plot(df.index, df["Ship"], label="Shipments Created", color="green", alpha=0.6)
    ax1.plot(df.index, df["Arr"], label="Arrivals (Receipts)", color="orange", alpha=0.6)
    ax1.plot(df.index, df["Prod"], label="Production Completed", color="red", alpha=0.8, linewidth=2)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Inventory (per node SKU)")
    ax2.plot(df.index, df["InvMean"], label="Mean Inventory", color="purple", linestyle="--", linewidth=2)
    ax2.tick_params(axis='y', labelcolor="purple")
    ax2.legend(loc='upper right')

    plt.title("Simulation Dynamics (365 Days) - Demand-Centric Model v0.90.0\nSupply follows Demand nervous system")
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
