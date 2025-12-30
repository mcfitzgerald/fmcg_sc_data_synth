#!/usr/bin/env python3
"""
Phase 1 Diagnostic V2: Run short simulation to trace ingredient flow.

This script runs a 30-day simulation with detailed logging to identify
where and why ingredient replenishment fails.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.orchestrator import Orchestrator


def run_diagnostic_simulation():
    """Run a 30-day simulation with ingredient tracking."""
    print("="*80)
    print("PHASE 1 DIAGNOSTIC V2: INGREDIENT FLOW TRACING")
    print("="*80)

    # Create orchestrator (initializes world, state, engines)
    orch = Orchestrator(enable_logging=False)

    # Get references
    world = orch.world
    state = orch.state
    mrp = orch.mrp_engine

    # Get plant IDs and ingredient info
    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    suppliers = [n for n in world.nodes.values() if n.type == NodeType.SUPPLIER]
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    print(f"\nNetwork: {len(plants)} plants, {len(suppliers)} suppliers, {len(ingredients)} ingredients")

    # Track metrics over time
    daily_metrics = []

    # Get initial plant inventory
    print(f"\n--- Initial Plant Inventory (After Priming) ---")
    for plant in plants[:2]:
        plant_idx = state.node_id_to_idx[plant.id]
        total_ing_inv = 0
        for ing in ingredients:
            ing_idx = state.product_id_to_idx.get(ing.id)
            if ing_idx is not None:
                total_ing_inv += state.inventory[plant_idx, ing_idx]
        print(f"  {plant.id}: Total ingredient inventory = {total_ing_inv:,.0f} units")

        # Sample specific ingredients
        for ing in ingredients[:3]:
            ing_idx = state.product_id_to_idx.get(ing.id)
            if ing_idx is not None:
                inv = state.inventory[plant_idx, ing_idx]
                print(f"    {ing.id}: {inv:,.0f} units")

    # Run simulation for N days with detailed tracking
    DAYS = 30
    print(f"\n--- Running {DAYS}-Day Simulation ---")

    for day in range(1, DAYS + 1):
        # Capture state before step
        plant_inv_before = {}
        for plant in plants:
            plant_idx = state.node_id_to_idx[plant.id]
            total = 0
            for ing in ingredients:
                ing_idx = state.product_id_to_idx.get(ing.id)
                if ing_idx is not None:
                    total += state.inventory[plant_idx, ing_idx]
            plant_inv_before[plant.id] = total

        # Run daily step
        orch._step(day)

        # Capture state after step
        plant_inv_after = {}
        for plant in plants:
            plant_idx = state.node_id_to_idx[plant.id]
            total = 0
            for ing in ingredients:
                ing_idx = state.product_id_to_idx.get(ing.id)
                if ing_idx is not None:
                    total += state.inventory[plant_idx, ing_idx]
            plant_inv_after[plant.id] = total

        # Count ingredient purchase orders created today
        po_count = 0
        po_qty = 0
        for order in orch.replenisher._last_orders if hasattr(orch.replenisher, '_last_orders') else []:
            pass

        # Track active shipments to plants
        shipments_to_plants = 0
        shipment_qty = 0
        for shipment in state.active_shipments:
            if shipment.target_id in [p.id for p in plants]:
                shipments_to_plants += 1
                for line in shipment.lines:
                    shipment_qty += line.quantity

        # Record daily metrics
        metrics = {
            "day": day,
            "plant_inv": sum(plant_inv_after.values()),
            "inv_change": sum(plant_inv_after.values()) - sum(plant_inv_before.values()),
            "active_shipments_to_plants": shipments_to_plants,
            "shipment_qty_to_plants": shipment_qty,
            "production_orders": len(orch.active_production_orders),
        }
        daily_metrics.append(metrics)

        if day <= 10 or day % 10 == 0:
            print(f"  Day {day:3d}: Plant Inv = {metrics['plant_inv']:>15,.0f} | "
                  f"Change = {metrics['inv_change']:>+12,.0f} | "
                  f"Shipments to Plants = {shipments_to_plants}")

    # Final analysis
    print("\n" + "="*80)
    print("INGREDIENT FLOW ANALYSIS")
    print("="*80)

    # Check final plant inventory
    print(f"\n--- Final Plant Inventory (Day {DAYS}) ---")
    critical_low = []
    for plant in plants:
        plant_idx = state.node_id_to_idx[plant.id]
        print(f"\n  {plant.id}:")
        total = 0
        for ing in ingredients:
            ing_idx = state.product_id_to_idx.get(ing.id)
            if ing_idx is not None:
                inv = state.inventory[plant_idx, ing_idx]
                total += inv
                if inv < 100000:  # Less than 100k units is critical
                    critical_low.append((plant.id, ing.id, inv))
        print(f"    Total: {total:,.0f} units")

    if critical_low:
        print(f"\n--- CRITICAL LOW INVENTORY ALERTS ---")
        for plant_id, ing_id, qty in critical_low[:20]:
            print(f"  {plant_id} / {ing_id}: {qty:,.0f} units")

    # Analyze supplier shipment arrivals
    print(f"\n--- Active Shipments to Plants ---")
    supplier_shipments = [s for s in state.active_shipments
                         if s.target_id in [p.id for p in plants]]
    print(f"  Active shipments: {len(supplier_shipments)}")

    if supplier_shipments:
        # Group by supplier
        by_supplier = defaultdict(list)
        for s in supplier_shipments:
            by_supplier[s.source_id].append(s)

        print(f"\n  By Source:")
        for source_id, shipments in sorted(by_supplier.items()):
            total_qty = sum(line.quantity for s in shipments for line in s.lines)
            print(f"    {source_id}: {len(shipments)} shipments, {total_qty:,.0f} units total")

    # Check supplier capacity constraint
    print(f"\n--- Supplier Capacity Check ---")
    sup_001 = world.nodes.get("SUP-001")
    if sup_001:
        print(f"  SUP-001 capacity: {sup_001.throughput_capacity:,.0f} units/day")

    # Get SPOF config
    config = orch.config
    spof_config = config.get("simulation_parameters", {}).get("manufacturing", {}).get("spof", {})
    spof_ing = spof_config.get("ingredient_id")
    print(f"  SPOF ingredient: {spof_ing}")

    # Check if SPOF ingredient inventory is depleting faster
    if spof_ing:
        print(f"\n--- SPOF Ingredient ({spof_ing}) Status ---")
        for plant in plants:
            plant_idx = state.node_id_to_idx[plant.id]
            ing_idx = state.product_id_to_idx.get(spof_ing)
            if ing_idx is not None:
                inv = state.inventory[plant_idx, ing_idx]
                print(f"    {plant.id}: {inv:,.0f} units")

    # Trend analysis
    print(f"\n--- Inventory Trend Summary ---")
    first_5_avg = np.mean([m["plant_inv"] for m in daily_metrics[:5]])
    last_5_avg = np.mean([m["plant_inv"] for m in daily_metrics[-5:]])
    print(f"  First 5 days avg: {first_5_avg:,.0f} units")
    print(f"  Last 5 days avg:  {last_5_avg:,.0f} units")
    print(f"  Change: {last_5_avg - first_5_avg:+,.0f} units ({100*(last_5_avg/first_5_avg - 1):+.1f}%)")

    if last_5_avg < first_5_avg * 0.9:
        print("\n  [WARNING] Plant inventory is DECLINING - replenishment may be inadequate!")
    else:
        print("\n  [OK] Plant inventory appears stable")


def analyze_po_generation_detail():
    """Detailed analysis of purchase order generation."""
    print("\n" + "="*80)
    print("PURCHASE ORDER GENERATION DETAIL")
    print("="*80)

    from prism_sim.config.loader import load_manifest, load_simulation_config
    from prism_sim.simulation.builder import WorldBuilder
    from prism_sim.simulation.state import StateManager
    from prism_sim.simulation.mrp import MRPEngine

    # Build world
    manifest = load_manifest()
    config = load_simulation_config()
    builder = WorldBuilder(manifest)
    world = builder.build()
    state = StateManager(world)

    # Initialize with typical starting inventory
    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    # Prime plants with 5M units per ingredient (simulating orchestrator init)
    mfg_config = config.get("simulation_parameters", {}).get("manufacturing", {})
    initial_plant_inv = mfg_config.get("initial_plant_inventory", {})

    for plant in plants:
        for ing in ingredients:
            qty = initial_plant_inv.get(ing.id, 5000000.0)
            state.update_inventory(plant.id, ing.id, qty)

    # Create MRP engine
    mrp = MRPEngine(world, state, config)

    # Generate fake demand to trigger PO generation
    daily_demand = np.ones((state.n_nodes, state.n_products), dtype=np.float64) * 100.0

    print(f"\nSimulating high consumption scenario...")

    # Simulate 30 days of consumption without replenishment arrivals
    for day in range(1, 31):
        # Generate purchase orders
        pos = mrp.generate_purchase_orders(day, daily_demand)

        # Track orders
        if day <= 5 or day % 10 == 0:
            total_qty = sum(line.quantity for po in pos for line in po.lines)
            print(f"  Day {day}: Generated {len(pos)} POs totaling {total_qty:,.0f} units")

        # Simulate consumption (deduct from inventory to trigger ROP)
        for plant in plants:
            plant_idx = state.node_id_to_idx[plant.id]
            for ing in ingredients:
                ing_idx = state.product_id_to_idx.get(ing.id)
                if ing_idx is not None:
                    # Deduct 50k units per ingredient per day (simulating production)
                    state.inventory[plant_idx, ing_idx] -= 50000.0

    # Check final inventory after 30 days of no replenishment
    print(f"\n--- After 30 Days of Consumption (No Arrivals) ---")
    for plant in plants[:2]:
        plant_idx = state.node_id_to_idx[plant.id]
        print(f"\n  {plant.id}:")
        for ing in ingredients[:5]:
            ing_idx = state.product_id_to_idx.get(ing.id)
            if ing_idx is not None:
                inv = state.inventory[plant_idx, ing_idx]
                print(f"    {ing.id}: {inv:,.0f} units {'(DEPLETED!)' if inv <= 0 else ''}")


if __name__ == "__main__":
    run_diagnostic_simulation()
    print("\n")
    analyze_po_generation_detail()
