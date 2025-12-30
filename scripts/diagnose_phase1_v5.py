#!/usr/bin/env python3
"""
Phase 1 Diagnostic V5: Why are zero ingredient POs being generated?

Trace the MRP.generate_purchase_orders() logic step by step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.orchestrator import Orchestrator


def trace_mrp_po_generation():
    """Trace MRP purchase order generation logic."""
    print("="*80)
    print("PHASE 1 DIAGNOSTIC V5: MRP PURCHASE ORDER GENERATION TRACE")
    print("="*80)

    # Create orchestrator
    orch = Orchestrator(enable_logging=False)
    world = orch.world
    state = orch.state
    mrp = orch.mrp_engine

    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    print(f"\nNetwork: {len(plants)} plants, {len(ingredients)} ingredients")

    # Generate day 1 demand
    day = 1
    daily_demand = orch.pos_engine.generate_demand(day)
    print(f"\n--- Step 1: Analyze Daily Demand ---")
    print(f"  daily_demand shape: {daily_demand.shape}")
    print(f"  Total demand: {np.sum(daily_demand):,.0f} units")

    # Step through MRP logic
    print(f"\n--- Step 2: MRP.generate_purchase_orders() Internals ---")

    # 2.1 Estimate Aggregate Demand
    total_demand = np.sum(daily_demand, axis=0)
    total_demand = np.maximum(total_demand, 1.0)
    print(f"  total_demand (per product): shape={total_demand.shape}, sum={np.sum(total_demand):,.0f}")
    print(f"  Non-zero products: {np.count_nonzero(total_demand > 1.0)}")

    # 2.2 Distribute to plants
    n_plants = len(mrp._plant_ids)
    plant_demand_share = total_demand / n_plants
    print(f"  plant_demand_share: {np.sum(plant_demand_share):,.0f} per plant")

    # 2.3 Calculate Ingredient Requirements via Recipe Matrix
    print(f"\n--- Step 3: Recipe Matrix Analysis ---")
    recipe_matrix = state.recipe_matrix
    print(f"  recipe_matrix shape: {recipe_matrix.shape}")
    print(f"  Recipe matrix non-zeros: {np.count_nonzero(recipe_matrix)}")
    print(f"  Recipe matrix sum: {np.sum(recipe_matrix):,.2f}")

    # Check a row (finished good → ingredients)
    # Find a finished good that has non-zero demand
    finished_products = [p for p in world.products.values() if p.category != ProductCategory.INGREDIENT]
    print(f"  Finished products: {len(finished_products)}")

    # Sample recipe matrix row
    sample_product = finished_products[0] if finished_products else None
    if sample_product:
        p_idx = state.product_id_to_idx.get(sample_product.id)
        if p_idx is not None:
            row = recipe_matrix[p_idx, :]
            non_zeros = np.where(row > 0)[0]
            print(f"  Sample product {sample_product.id} (idx={p_idx}):")
            print(f"    Recipe row non-zeros: {len(non_zeros)} ingredients")
            if len(non_zeros) > 0:
                for ing_idx in non_zeros[:3]:
                    ing_id = state.product_idx_to_id[ing_idx]
                    print(f"      {ing_id}: {row[ing_idx]:.2f} units per case")

    # 2.4 Calculate ingredient requirements
    ingredient_reqs = plant_demand_share @ recipe_matrix
    print(f"\n--- Step 4: Ingredient Requirements ---")
    print(f"  ingredient_reqs shape: {ingredient_reqs.shape}")
    print(f"  ingredient_reqs non-zeros: {np.count_nonzero(ingredient_reqs > 0)}")
    print(f"  ingredient_reqs sum: {np.sum(ingredient_reqs):,.2f}")

    # 2.5 Calculate ROP and Target levels
    target_levels = ingredient_reqs * mrp.target_vector
    rop_levels = ingredient_reqs * mrp.rop_vector
    print(f"  target_levels sum: {np.sum(target_levels):,.2f}")
    print(f"  rop_levels sum: {np.sum(rop_levels):,.2f}")

    # 2.6 Check inventory position vs ROP
    print(f"\n--- Step 5: Inventory Position Check ---")

    for plant_id in mrp._plant_ids[:2]:  # First 2 plants
        plant_idx = state.node_id_to_idx.get(plant_id)
        if plant_idx is None:
            continue

        on_hand = state.inventory[plant_idx]
        inv_position = on_hand  # No in-transit for now

        # Check condition
        needs_ordering = (inv_position < rop_levels) & (ingredient_reqs > 0)
        count_below_rop = np.sum(needs_ordering)

        print(f"\n  {plant_id}:")
        print(f"    On-hand inventory sum: {np.sum(on_hand):,.0f}")

        # Sample a few ingredients
        for ing in ingredients[:5]:
            ing_idx = state.product_id_to_idx.get(ing.id)
            if ing_idx is not None:
                oh = on_hand[ing_idx]
                req = ingredient_reqs[ing_idx]
                rop = rop_levels[ing_idx]
                target = target_levels[ing_idx]
                below = needs_ordering[ing_idx]
                print(f"    {ing.id}: OH={oh:,.0f}, Req={req:.1f}, ROP={rop:.1f}, Target={target:.1f}, NeedsOrder={below}")

        print(f"    Items below ROP: {count_below_rop}")

    # The key insight: WHY is needs_ordering always False?
    print(f"\n--- Step 6: Root Cause Analysis ---")
    print(f"\n  Initial inventory per ingredient at plant: 5,000,000 units")
    print(f"  Daily ingredient requirement (calculated): ~{np.mean(ingredient_reqs[ingredient_reqs > 0]):.1f} units")
    print(f"  ROP days: {mrp.reorder_point_days}")
    print(f"  ROP = daily_req * ROP_days = {np.mean(ingredient_reqs[ingredient_reqs > 0]) * mrp.reorder_point_days:.1f} units")
    print(f"\n  Inventory (5M) >> ROP (~{np.mean(rop_levels[rop_levels > 0]):.1f})")
    print(f"  Therefore: inv_position < rop_levels is FALSE for all ingredients!")

    # What would it take to trigger reorder?
    print(f"\n--- Step 7: When Would Reorder Trigger? ---")
    avg_daily_req = np.mean(ingredient_reqs[ingredient_reqs > 0])
    avg_rop = np.mean(rop_levels[rop_levels > 0])
    days_until_trigger = (5000000 - avg_rop) / avg_daily_req if avg_daily_req > 0 else float('inf')
    print(f"  Average daily ingredient consumption: {avg_daily_req:.1f} units")
    print(f"  Average ROP level: {avg_rop:.1f} units")
    print(f"  Days until ROP reached: {days_until_trigger:,.0f} days")


if __name__ == "__main__":
    trace_mrp_po_generation()
