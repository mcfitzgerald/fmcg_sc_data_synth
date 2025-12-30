#!/usr/bin/env python3
"""
Phase 1 Diagnostic V4: Trace actual orchestrator PO flow.

This directly instruments the orchestrator to see what's happening
with ingredient purchase orders during a real simulation step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.orchestrator import Orchestrator


def trace_real_orchestrator_flow():
    """Run actual orchestrator with tracing."""
    print("="*80)
    print("PHASE 1 DIAGNOSTIC V4: REAL ORCHESTRATOR FLOW TRACE")
    print("="*80)

    # Create orchestrator
    orch = Orchestrator(enable_logging=False)
    world = orch.world
    state = orch.state

    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    print(f"\nInitial plant ingredient inventory:")
    for plant in plants[:2]:
        plant_idx = state.node_id_to_idx[plant.id]
        total = sum(state.inventory[plant_idx, state.product_id_to_idx[ing.id]]
                    for ing in ingredients if ing.id in state.product_id_to_idx)
        print(f"  {plant.id}: {total:,.0f} units")

    # Manually step through Day 1 with tracing
    print(f"\n--- Day 1 Detailed Trace ---")
    day = 1

    # Step 1: Generate demand
    daily_demand = orch.pos_engine.generate_demand(day)
    print(f"\n1. Demand generated: {np.sum(daily_demand):,.0f} units")

    # Step 2: Consume inventory (sales)
    available = np.maximum(0, state.actual_inventory)
    actual_sales = np.minimum(daily_demand, available)
    state.update_inventory_batch(-actual_sales)
    print(f"2. Actual sales: {np.sum(actual_sales):,.0f} units")

    # Step 3: Replenishment orders (store->RDC, RDC->plant)
    raw_orders = orch.replenisher.generate_orders(day, daily_demand)
    replen_qty = sum(l.quantity for o in raw_orders for l in o.lines)
    print(f"3. Replenishment orders: {len(raw_orders)} orders, {replen_qty:,.0f} units")

    # Count replenishment orders by type
    store_to_rdc = [o for o in raw_orders
                   if world.nodes.get(o.target_id) and
                   world.nodes[o.target_id].type == NodeType.DC]
    rdc_to_plant = [o for o in raw_orders
                   if world.nodes.get(o.target_id) and
                   world.nodes[o.target_id].type == NodeType.PLANT]
    print(f"   - Store→RDC orders: {len(store_to_rdc)}")
    print(f"   - RDC→Plant orders: {len(rdc_to_plant)}")

    # Step 4: Ingredient purchase orders
    ing_orders = orch.mrp_engine.generate_purchase_orders(day, daily_demand)
    ing_qty = sum(l.quantity for o in ing_orders for l in o.lines)
    print(f"4. Ingredient purchase orders: {len(ing_orders)} orders, {ing_qty:,.0f} units")

    # Analyze ingredient orders by supplier
    if ing_orders:
        by_supplier = {}
        for o in ing_orders:
            src = o.source_id
            if src not in by_supplier:
                by_supplier[src] = {"count": 0, "qty": 0}
            by_supplier[src]["count"] += 1
            by_supplier[src]["qty"] += sum(l.quantity for l in o.lines)

        print(f"   By supplier:")
        for src, stats in sorted(by_supplier.items()):
            sup = world.nodes.get(src)
            cap = sup.throughput_capacity if sup else 0
            print(f"     {src}: {stats['count']} orders, {stats['qty']:,.0f} units (cap: {cap:,.0f})")

    # Combine orders
    raw_orders.extend(ing_orders)
    print(f"\n   Combined orders: {len(raw_orders)}")

    # Step 5: Allocation
    allocated_orders = orch.allocator.allocate_orders(raw_orders)
    alloc_qty = sum(l.quantity for o in allocated_orders for l in o.lines)
    print(f"5. Allocated orders: {len(allocated_orders)} orders, {alloc_qty:,.0f} units")

    # Count supplier orders after allocation
    sup_orders_alloc = [o for o in allocated_orders
                       if world.nodes.get(o.source_id) and
                       world.nodes[o.source_id].type == NodeType.SUPPLIER]
    sup_alloc_qty = sum(l.quantity for o in sup_orders_alloc for l in o.lines)
    print(f"   - Supplier→Plant orders: {len(sup_orders_alloc)}, {sup_alloc_qty:,.0f} units")

    # Step 6: Logistics - create shipments
    new_shipments = orch.logistics.create_shipments(allocated_orders, day)
    ship_qty = sum(l.quantity for s in new_shipments for l in s.lines)
    print(f"6. Shipments created: {len(new_shipments)} shipments, {ship_qty:,.0f} units")

    # Count supplier shipments
    sup_shipments = [s for s in new_shipments
                    if world.nodes.get(s.source_id) and
                    world.nodes[s.source_id].type == NodeType.SUPPLIER]
    sup_ship_qty = sum(l.quantity for s in sup_shipments for l in s.lines)
    print(f"   - Supplier→Plant shipments: {len(sup_shipments)}, {sup_ship_qty:,.0f} units")

    # Check held orders
    held = orch.logistics.held_orders
    print(f"   - Held orders (waiting FTL): {len(held)}")

    # Add shipments to active
    orch.state.active_shipments.extend(new_shipments)

    # Now run a few more days and see what arrives
    print(f"\n--- Running Days 2-5 to see arrivals ---")

    for d in range(2, 6):
        # Simplified step - just do arrival processing
        active, arrived = orch.logistics.update_shipments(state.active_shipments, d)
        state.active_shipments = active

        # Count supplier arrivals
        sup_arrivals = [s for s in arrived
                       if world.nodes.get(s.source_id) and
                       world.nodes[s.source_id].type == NodeType.SUPPLIER]
        sup_arr_qty = sum(l.quantity for s in sup_arrivals for l in s.lines)

        # Process arrivals
        for shipment in arrived:
            target_node = world.nodes.get(shipment.target_id)
            if target_node:
                for line in shipment.lines:
                    state.update_inventory(shipment.target_id, line.product_id, line.quantity)

        print(f"  Day {d}: {len(arrived)} arrivals, {len(sup_arrivals)} from suppliers ({sup_arr_qty:,.0f} units)")

    # Final plant inventory
    print(f"\n--- Final Plant Inventory (Day 5) ---")
    for plant in plants[:2]:
        plant_idx = state.node_id_to_idx[plant.id]
        total = sum(state.inventory[plant_idx, state.product_id_to_idx[ing.id]]
                    for ing in ingredients if ing.id in state.product_id_to_idx)
        print(f"  {plant.id}: {total:,.0f} units")


if __name__ == "__main__":
    trace_real_orchestrator_flow()
