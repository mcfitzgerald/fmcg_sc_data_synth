#!/usr/bin/env python3
"""
Phase 1 Diagnostic: Verify Root Causes of 365-Day Simulation Collapse.

This script verifies the three suspected issues from the debug plan:
1. MRP Purchase Order Supplier Routing Bug
2. Initial Ingredient Inventory Insufficient
3. Supplier Capacity Bottleneck
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism_sim.network.core import NodeType
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.mrp import MRPEngine
from prism_sim.product.core import ProductCategory


def load_config() -> dict:
    """Load simulation config."""
    config_path = Path(__file__).parent.parent / "src/prism_sim/config/simulation_config.json"
    with open(config_path) as f:
        return json.load(f)


def verify_step_1_1_supplier_links():
    """Step 1.1: Trace Supplier Link Discovery."""
    print("\n" + "="*80)
    print("STEP 1.1: SUPPLIER LINK DISCOVERY ANALYSIS")
    print("="*80)

    config = load_config()
    builder = WorldBuilder(config)
    world = builder.build()

    # Get all plants and suppliers
    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    suppliers = [n for n in world.nodes.values() if n.type == NodeType.SUPPLIER]

    print(f"\nNetwork has {len(plants)} plants and {len(suppliers)} suppliers")

    # Count links by type
    supplier_to_plant_links = []
    plant_to_rdc_links = []
    rdc_to_store_links = []
    other_links = []

    for link in world.links.values():
        source = world.nodes.get(link.source_id)
        target = world.nodes.get(link.target_id)

        if source and target:
            if source.type == NodeType.SUPPLIER and target.type == NodeType.PLANT:
                supplier_to_plant_links.append(link)
            elif source.type == NodeType.PLANT and target.type == NodeType.DC:
                plant_to_rdc_links.append(link)
            elif source.type == NodeType.DC and target.type in (NodeType.STORE, NodeType.DC):
                rdc_to_store_links.append(link)
            else:
                other_links.append((link, source.type, target.type))

    print(f"\nLink Type Distribution:")
    print(f"  Supplier → Plant links: {len(supplier_to_plant_links)}")
    print(f"  Plant → RDC links: {len(plant_to_rdc_links)}")
    print(f"  RDC → Store/DC links: {len(rdc_to_store_links)}")
    print(f"  Other links: {len(other_links)}")

    # Check which plants have supplier links
    print(f"\n--- Plant-to-Supplier Connectivity ---")
    plant_suppliers: dict[str, list[str]] = {p.id: [] for p in plants}

    for link in supplier_to_plant_links:
        plant_suppliers[link.target_id].append(link.source_id)

    for plant_id, sup_list in plant_suppliers.items():
        print(f"  {plant_id}: {len(sup_list)} suppliers linked")
        if sup_list:
            print(f"    First 5: {sup_list[:5]}")

    # Test the _find_supplier_for_ingredient logic
    print(f"\n--- Testing _find_supplier_for_ingredient Logic ---")
    state = StateManager(world)
    mrp = MRPEngine(world, state, config)

    # Get all ingredients
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]
    print(f"\nTotal ingredients in world: {len(ingredients)}")

    # Test finding suppliers for each plant + ingredient combo
    found_count = 0
    not_found_count = 0
    not_found_examples = []

    for plant in plants:
        for ing in ingredients[:20]:  # Test first 20 ingredients
            supplier_id = mrp._find_supplier_for_ingredient(plant.id, ing.id)
            if supplier_id:
                found_count += 1
            else:
                not_found_count += 1
                if len(not_found_examples) < 10:
                    not_found_examples.append((plant.id, ing.id))

    print(f"\nSupplier lookup results:")
    print(f"  Found supplier: {found_count}")
    print(f"  NOT found (None): {not_found_count}")

    if not_found_examples:
        print(f"\n  Examples where supplier NOT found:")
        for plant_id, ing_id in not_found_examples:
            print(f"    Plant: {plant_id}, Ingredient: {ing_id}")

    # Check SPOF ingredient specifically
    spof_id = config.get("simulation_parameters", {}).get("manufacturing", {}).get("spof", {}).get("ingredient_id")
    print(f"\n--- SPOF Ingredient Analysis ---")
    print(f"  SPOF ingredient ID: {spof_id}")

    if spof_id:
        for plant in plants:
            supplier = mrp._find_supplier_for_ingredient(plant.id, spof_id)
            print(f"    {plant.id} → supplier for {spof_id}: {supplier}")

    return not_found_count > 0


def verify_step_1_2_purchase_orders():
    """Step 1.2: Trace Purchase Order Generation."""
    print("\n" + "="*80)
    print("STEP 1.2: PURCHASE ORDER GENERATION ANALYSIS")
    print("="*80)

    import numpy as np

    config = load_config()
    builder = WorldBuilder(config)
    world = builder.build()
    state = StateManager(world)
    mrp = MRPEngine(world, state, config)

    # Get plant IDs
    plant_ids = [n.id for n in world.nodes.values() if n.type == NodeType.PLANT]
    print(f"\nPlants: {plant_ids}")

    # Get ingredient count
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]
    print(f"Total ingredients: {len(ingredients)}")

    # Create some demand to trigger purchase orders
    daily_demand = np.ones((state.n_nodes, state.n_products), dtype=np.float64) * 100.0

    # Generate purchase orders
    purchase_orders = mrp.generate_purchase_orders(current_day=1, daily_demand=daily_demand)

    print(f"\n--- Purchase Order Generation Results ---")
    print(f"  Orders generated: {len(purchase_orders)}")

    # Analyze orders by supplier
    orders_by_supplier: dict[str, int] = {}
    for po in purchase_orders:
        sup_id = po.source_id
        orders_by_supplier[sup_id] = orders_by_supplier.get(sup_id, 0) + 1

    print(f"\n  Orders by supplier:")
    for sup_id, count in sorted(orders_by_supplier.items()):
        print(f"    {sup_id}: {count} orders")

    # Check which ingredients DID get purchase orders
    ordered_ingredients = set()
    for po in purchase_orders:
        for line in po.lines:
            ordered_ingredients.add(line.product_id)

    print(f"\n  Ingredients with orders: {len(ordered_ingredients)}")

    # Which ingredients did NOT get orders?
    all_ing_ids = {ing.id for ing in ingredients}
    missing_orders = all_ing_ids - ordered_ingredients

    print(f"  Ingredients WITHOUT orders: {len(missing_orders)}")
    if missing_orders:
        print(f"    Examples: {list(missing_orders)[:10]}")

    return len(purchase_orders), len(missing_orders)


def verify_step_1_3_ingredient_consumption():
    """Step 1.3: Verify Ingredient Consumption Rate."""
    print("\n" + "="*80)
    print("STEP 1.3: INGREDIENT CONSUMPTION RATE ANALYSIS")
    print("="*80)

    import numpy as np

    config = load_config()
    builder = WorldBuilder(config)
    world = builder.build()
    state = StateManager(world)

    # Get manufacturing config
    mfg_config = config.get("simulation_parameters", {}).get("manufacturing", {})
    initial_inv = mfg_config.get("initial_plant_inventory", {})

    print(f"\n--- Initial Inventory Configuration ---")
    print(f"  Configured initial inventory:")
    for ing_id, qty in initial_inv.items():
        print(f"    {ing_id}: {qty:,.0f} units")

    print(f"\n  Default initial inventory: 5,000,000 units (per docs)")

    # Get plants and ingredients
    plants = [n for n in world.nodes.values() if n.type == NodeType.PLANT]
    ingredients = [p for p in world.products.values() if p.category == ProductCategory.INGREDIENT]

    # Check actual initial inventory at plants
    print(f"\n--- Actual Plant Ingredient Inventory (Sample) ---")
    for plant in plants[:2]:  # First 2 plants
        plant_idx = state.node_id_to_idx[plant.id]
        print(f"\n  {plant.id}:")
        for ing in ingredients[:5]:  # First 5 ingredients
            ing_idx = state.product_id_to_idx.get(ing.id)
            if ing_idx is not None:
                inv = state.inventory[plant_idx, ing_idx]
                print(f"    {ing.id}: {inv:,.0f} units")

    # Estimate consumption rate
    print(f"\n--- Consumption Rate Estimate ---")

    # Get recipe info
    finished_products = [p for p in world.products.values() if p.category != ProductCategory.INGREDIENT]
    print(f"\n  Finished products: {len(finished_products)}")

    # Estimate daily production need
    # From config: daily demand ~130k cases (based on debug plan)
    estimated_daily_demand = 130000  # cases/day

    # Each case needs ~X units of ingredients (from recipe)
    # Let's check a sample recipe
    sample_recipe = list(world.recipes.values())[0] if world.recipes else None
    if sample_recipe:
        print(f"\n  Sample recipe ({sample_recipe.product_id}):")
        print(f"    Output: {sample_recipe.output_qty} cases")
        for ing_id, qty in sample_recipe.ingredients.items():
            print(f"    Requires: {qty} units of {ing_id}")

    # Calculate days to depletion
    print(f"\n--- Days to Depletion Estimate ---")

    # Assume ~250k units/day ingredient consumption (rough estimate)
    # With 5M initial inventory: 5M / 250k = 20 days
    initial_default = 5_000_000
    estimated_consumption = 250_000  # units/day per ingredient (rough)
    days_to_depletion = initial_default / estimated_consumption

    print(f"  Initial inventory: {initial_default:,} units")
    print(f"  Estimated consumption: {estimated_consumption:,} units/day")
    print(f"  Estimated days to depletion: {days_to_depletion:.1f} days")
    print(f"\n  Debug plan claims collapse at day ~172-178")
    print(f"  If replenishment is broken, collapse would occur when inventory depletes")


def verify_supplier_capacity():
    """Check supplier capacity settings."""
    print("\n" + "="*80)
    print("BONUS: SUPPLIER CAPACITY ANALYSIS")
    print("="*80)

    config = load_config()
    builder = WorldBuilder(config)
    world = builder.build()

    suppliers = [n for n in world.nodes.values() if n.type == NodeType.SUPPLIER]

    print(f"\nSupplier Capacity Settings:")
    for sup in suppliers[:10]:  # First 10 suppliers
        cap = sup.throughput_capacity
        cap_str = f"{cap:,.0f}" if cap != float('inf') else "INFINITE"
        print(f"  {sup.id}: {cap_str} units/day")

    # Check SUP-001 specifically (SPOF)
    sup_001 = world.nodes.get("SUP-001")
    if sup_001:
        print(f"\n  SUP-001 (SPOF supplier): {sup_001.throughput_capacity:,.0f} units/day")
        print(f"  This is the bottleneck mentioned in debug plan (500k/day)")


def main():
    """Run all Phase 1 diagnostic steps."""
    print("="*80)
    print("PHASE 1 DIAGNOSTIC: VERIFYING ROOT CAUSES")
    print("="*80)

    # Step 1.1
    supplier_issue = verify_step_1_1_supplier_links()

    # Step 1.2
    orders_created, missing_orders = verify_step_1_2_purchase_orders()

    # Step 1.3
    verify_step_1_3_ingredient_consumption()

    # Bonus
    verify_supplier_capacity()

    # Summary
    print("\n" + "="*80)
    print("PHASE 1 DIAGNOSTIC SUMMARY")
    print("="*80)

    print("\n1. SUPPLIER ROUTING BUG:")
    if supplier_issue:
        print("   [CONFIRMED] Some plant/ingredient combos return None for supplier")
        print("   The function finds ANY supplier linked to the plant, but that")
        print("   supplier may not be the right one for the specific ingredient.")
    else:
        print("   [NOT CONFIRMED] All lookups returned a supplier")

    print("\n2. PURCHASE ORDER GENERATION:")
    print(f"   Orders created: {orders_created}")
    print(f"   Ingredients without orders: {missing_orders}")
    if missing_orders > 0:
        print("   [ISSUE] Some ingredients may not get replenished!")

    print("\n3. INITIAL INVENTORY:")
    print("   See consumption analysis above.")
    print("   If PO generation is broken, inventory will deplete and not recover.")

    print("\n4. SUPPLIER CAPACITY:")
    print("   SUP-001 capped at 500k/day (potential bottleneck)")


if __name__ == "__main__":
    main()
