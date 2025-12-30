#!/usr/bin/env python3
"""
Phase 1 Diagnostic V3: Trace purchase order pipeline in detail.

Traces: MRP generates PO → Allocation → Logistics → Shipment
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from prism_sim.config.loader import load_manifest, load_simulation_config
from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.mrp import MRPEngine
from prism_sim.agents.allocation import AllocationAgent
from prism_sim.simulation.logistics import LogisticsEngine


def trace_po_pipeline():
    """Trace a single day's purchase order through the entire pipeline."""
    print("="*80)
    print("PHASE 1 DIAGNOSTIC V3: PURCHASE ORDER PIPELINE TRACE")
    print("="*80)

    # Build world
    manifest = load_manifest()
    config = load_simulation_config()
    builder = WorldBuilder(manifest)
    world = builder.build()
    state = StateManager(world)

    # Get nodes
    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    suppliers = [n for n in world.nodes.values() if n.type == NodeType.SUPPLIER]
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    print(f"\nNetwork: {len(plants)} plants, {len(suppliers)} suppliers, {len(ingredients)} ingredients")

    # Initialize plant inventory (simulating orchestrator priming)
    mfg_config = config.get("simulation_parameters", {}).get("manufacturing", {})
    initial_plant_inv = mfg_config.get("initial_plant_inventory", {})

    for plant in plants:
        for ing in ingredients:
            qty = initial_plant_inv.get(ing.id, 5000000.0)
            state.update_inventory(plant.id, ing.id, qty)

    # Create engines
    mrp = MRPEngine(world, state, config)
    allocator = AllocationAgent(state, config)
    logistics = LogisticsEngine(world, state, config)

    # Check route map
    print(f"\n--- Route Map Analysis ---")
    sup_to_plant_routes = []
    for (src, tgt), link in logistics.route_map.items():
        src_node = world.nodes.get(src)
        tgt_node = world.nodes.get(tgt)
        if src_node and tgt_node:
            if src_node.type == NodeType.SUPPLIER and tgt_node.type == NodeType.PLANT:
                sup_to_plant_routes.append((src, tgt, link.lead_time_days))

    print(f"  Supplier→Plant routes in route_map: {len(sup_to_plant_routes)}")
    for src, tgt, lt in sup_to_plant_routes[:5]:
        print(f"    {src} → {tgt} (lead time: {lt:.1f} days)")

    # Generate purchase orders
    print(f"\n--- Step 1: MRP.generate_purchase_orders() ---")
    daily_demand = np.ones((state.n_nodes, state.n_products), dtype=np.float64) * 100.0
    purchase_orders = mrp.generate_purchase_orders(current_day=1, daily_demand=daily_demand)

    print(f"  Purchase orders created: {len(purchase_orders)}")

    # Analyze orders by source
    by_source = {}
    for po in purchase_orders:
        src = po.source_id
        if src not in by_source:
            by_source[src] = {"count": 0, "total_qty": 0}
        by_source[src]["count"] += 1
        by_source[src]["total_qty"] += sum(l.quantity for l in po.lines)

    print(f"\n  Orders by source supplier:")
    for src, stats in sorted(by_source.items()):
        sup = world.nodes.get(src)
        cap = sup.throughput_capacity if sup else 0
        cap_str = f"{cap:,.0f}" if cap != float('inf') else "INFINITE"
        print(f"    {src}: {stats['count']} orders, {stats['total_qty']:,.0f} units (capacity: {cap_str})")

    # Sample a few orders
    print(f"\n  Sample orders:")
    for po in purchase_orders[:3]:
        total_qty = sum(l.quantity for l in po.lines)
        print(f"    {po.id}: {po.source_id} → {po.target_id}, "
              f"{len(po.lines)} lines, {total_qty:,.0f} units, status={po.status}")

    # Run allocation
    print(f"\n--- Step 2: AllocationAgent.allocate_orders() ---")
    allocated_orders = allocator.allocate_orders(purchase_orders)

    print(f"  Allocated orders: {len(allocated_orders)}")

    # Analyze allocated orders
    by_source_alloc = {}
    for ao in allocated_orders:
        src = ao.source_id
        if src not in by_source_alloc:
            by_source_alloc[src] = {"count": 0, "total_qty": 0}
        by_source_alloc[src]["count"] += 1
        by_source_alloc[src]["total_qty"] += sum(l.quantity for l in ao.lines)

    print(f"\n  Allocated orders by source:")
    for src, stats in sorted(by_source_alloc.items()):
        orig_qty = by_source.get(src, {}).get("total_qty", 0)
        ratio = stats["total_qty"] / orig_qty * 100 if orig_qty > 0 else 0
        print(f"    {src}: {stats['count']} orders, {stats['total_qty']:,.0f} units ({ratio:.1f}% of requested)")

    # Check if any orders were dropped (have empty lines)
    orders_with_lines = [ao for ao in allocated_orders if ao.lines]
    orders_without_lines = [ao for ao in allocated_orders if not ao.lines]
    print(f"\n  Orders with lines: {len(orders_with_lines)}")
    print(f"  Orders without lines (dropped): {len(orders_without_lines)}")

    # Sample allocated orders
    print(f"\n  Sample allocated orders:")
    for ao in allocated_orders[:3]:
        total_qty = sum(l.quantity for l in ao.lines)
        print(f"    {ao.id}: {ao.source_id} → {ao.target_id}, "
              f"{len(ao.lines)} lines, {total_qty:,.0f} units, status={ao.status}")

    # Run logistics
    print(f"\n--- Step 3: LogisticsEngine.create_shipments() ---")
    shipments = logistics.create_shipments(allocated_orders, current_day=1)

    print(f"  Shipments created: {len(shipments)}")
    print(f"  Held orders (waiting for FTL): {len(logistics.held_orders)}")

    # Analyze shipments
    sup_to_plant_shipments = []
    for s in shipments:
        src_node = world.nodes.get(s.source_id)
        if src_node and src_node.type == NodeType.SUPPLIER:
            sup_to_plant_shipments.append(s)

    print(f"\n  Supplier→Plant shipments: {len(sup_to_plant_shipments)}")

    if sup_to_plant_shipments:
        print(f"  Sample shipments:")
        for s in sup_to_plant_shipments[:3]:
            total_qty = sum(l.quantity for l in s.lines)
            print(f"    {s.id}: {s.source_id} → {s.target_id}, "
                  f"{len(s.lines)} lines, {total_qty:,.0f} units")
    else:
        print(f"\n  !!! NO SUPPLIER→PLANT SHIPMENTS CREATED !!!")

        # Analyze held orders
        held_sup_orders = [o for o in logistics.held_orders
                          if world.nodes.get(o.source_id) and
                          world.nodes[o.source_id].type == NodeType.SUPPLIER]
        print(f"\n  Held supplier orders: {len(held_sup_orders)}")

        if held_sup_orders:
            for ho in held_sup_orders[:3]:
                target_node = world.nodes.get(ho.target_id)
                channel = target_node.channel if target_node else None
                print(f"    {ho.id}: {ho.source_id} → {ho.target_id}, channel={channel}")

    # Why might orders be held?
    print(f"\n--- FTL Constraint Analysis ---")
    print(f"  Channel rules config:")
    for channel, rules in logistics.channel_rules.items():
        print(f"    {channel}: min_pallets={rules.get('min_order_pallets', 0)}")

    # Check a supplier order's target channel
    if allocated_orders:
        sample = allocated_orders[0]
        target_node = world.nodes.get(sample.target_id)
        if target_node:
            print(f"\n  Sample target: {sample.target_id}")
            print(f"    Type: {target_node.type}")
            print(f"    Channel: {getattr(target_node, 'channel', 'N/A')}")
            rules = logistics._get_channel_rules(getattr(target_node, 'channel', None))
            print(f"    Rules: {rules}")
            pallets = logistics._calculate_pallets(sample)
            print(f"    Order pallets: {pallets:.2f}")
            min_pallets = rules.get("min_order_pallets", 0)
            print(f"    Min pallets required: {min_pallets}")
            if pallets < min_pallets:
                print(f"    !!! BELOW FTL THRESHOLD - ORDER WOULD BE HELD !!!")


def check_ingredient_products():
    """Check if ingredient products have required attributes for logistics."""
    print("\n" + "="*80)
    print("INGREDIENT PRODUCT ATTRIBUTES CHECK")
    print("="*80)

    manifest = load_manifest()
    config = load_simulation_config()
    builder = WorldBuilder(manifest)
    world = builder.build()

    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    print(f"\nChecking {len(ingredients)} ingredients for logistics attributes...")

    missing_attrs = []
    for ing in ingredients[:10]:
        issues = []
        if not hasattr(ing, 'weight_kg') or ing.weight_kg <= 0:
            issues.append("weight_kg")
        if not hasattr(ing, 'volume_m3') or ing.volume_m3 <= 0:
            issues.append("volume_m3")
        if not hasattr(ing, 'cases_per_pallet') or ing.cases_per_pallet <= 0:
            issues.append("cases_per_pallet")

        if issues:
            missing_attrs.append((ing.id, issues))
        else:
            print(f"  {ing.id}: weight={ing.weight_kg}kg, vol={ing.volume_m3}m³, cpp={ing.cases_per_pallet}")

    if missing_attrs:
        print(f"\n  !!! {len(missing_attrs)} ingredients missing attributes !!!")
        for ing_id, issues in missing_attrs[:5]:
            print(f"    {ing_id}: missing {issues}")


if __name__ == "__main__":
    trace_po_pipeline()
    check_ingredient_products()
