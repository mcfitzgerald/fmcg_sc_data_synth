"""Script to generate the static world (Level 0-4) for Deep NAM expansion."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prism_sim.generators.hierarchy import ProductGenerator
from prism_sim.generators.network import NetworkGenerator
from prism_sim.writers.static_writer import StaticWriter


def main() -> None:
    # 0. Load Configs
    world_config_path = Path("src/prism_sim/config/world_definition.json")
    sim_config_path = Path("src/prism_sim/config/simulation_config.json")
    
    with open(world_config_path) as f:
        world_config = json.load(f)
        
    with open(sim_config_path) as f:
        sim_config = json.load(f)
        
    # Merge simulation parameters into world config for generators
    # This allows geospatial paths and jitter to be available
    world_config.update(sim_config)

    output_dir = Path("data/output/static_world")
    print(f"Generating static world to {output_dir}...")

    # 1. Generators
    prod_gen = ProductGenerator(config=world_config, seed=42)
    net_gen = NetworkGenerator(seed=42, config=world_config)
    writer = StaticWriter(str(output_dir))

    # 2. Products
    print("Generating Products...")
    counts = world_config.get("topology", {}).get("target_counts", {})
    n_skus = counts.get("skus", 50)

    finished_goods = prod_gen.generate_products(n_skus=n_skus)

    # 3. Ingredients (Procedural)
    print("Generating Ingredients...")
    ingredients = prod_gen.generate_ingredients(n_per_type=5)

    all_products = ingredients + finished_goods

    # 3. Recipes
    print("Generating Recipes...")
    recipes = prod_gen.generate_recipes(finished_goods, ingredients)

    # 4. Network
    print("Generating Network (4500 stores)... this may take a moment.")
    nodes, links = net_gen.generate_network(
        n_stores=counts.get("stores", 4500),
        n_suppliers=counts.get("suppliers", 50),
        n_plants=counts.get("plants", 4),
        n_rdcs=counts.get("rdcs", 4)
    )

    # 5. Write
    print("Writing files...")
    writer.write_products(all_products)
    writer.write_recipes(recipes)
    writer.write_locations(nodes)
    writer.write_partners(nodes)
    writer.write_links(links)

    print("Done!")
    print("Stats:")
    print(f"  Products: {len(all_products)}")
    print(f"  Recipes:  {len(recipes)}")
    print(f"  Nodes:    {len(nodes)}")
    print(f"  Links:    {len(links)}")

if __name__ == "__main__":
    main()
