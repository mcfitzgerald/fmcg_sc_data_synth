
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from prism_sim.simulation.builder import WorldBuilder
from prism_sim.config.loader import load_simulation_config

def check_world_integrity():
    print("Loading World...")
    config = load_simulation_config("src/prism_sim/config/simulation_config.json")
    
    # WorldBuilder is a class, we need to instantiate it or use it correctly.
    # Looking at builder.py signature from memory/context:
    # It likely takes config in __init__
    builder = WorldBuilder(config)
    world = builder.build()
    
    print(f"Total Nodes: {len(world.nodes)}")
    print(f"Total Products: {len(world.products)}")
    
    finished_goods = [p for p in world.products.values() if p.category.name != "INGREDIENT"]
    print(f"Finished Goods: {len(finished_goods)}")
    
    # Check if these match the missing ones
    # We suspect ~500 finished goods.
    
    if len(finished_goods) < 490:
        print("FAIL: World generation did not create enough SKUs.")
    else:
        print("PASS: World has correct SKU count.")
        
    # Check Recipes
    recipes = world.recipes
    print(f"Total Recipes: {len(recipes)}")
    
    # Check if every FG has a recipe
    fg_ids = set(p.id for p in finished_goods)
    recipe_product_ids = set(r.product_id for r in recipes.values())
    
    missing_recipes = fg_ids - recipe_product_ids
    print(f"Products without Recipes: {len(missing_recipes)}")
    if missing_recipes:
        print(f"Example Orphan Products: {list(missing_recipes)[:5]}")

if __name__ == "__main__":
    check_world_integrity()
