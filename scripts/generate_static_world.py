"""Script to generate the static world (Level 0-4) for Deep NAM expansion."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prism_sim.generators.hierarchy import ProductGenerator
from prism_sim.generators.network import NetworkGenerator
from prism_sim.writers.static_writer import StaticWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static world data.")
    parser.add_argument(
        "--skus", type=int, default=0,
        help="Number of SKUs (overrides config)",
    )
    parser.add_argument(
        "--stores", type=int, default=0,
        help="Number of stores (overrides config)",
    )
    args = parser.parse_args()

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

    # 2. Products (Level 0 — finished SKUs)
    print("Generating Products...")
    counts = world_config.get("topology", {}).get("target_counts", {})
    n_skus = args.skus if args.skus > 0 else counts.get("skus", 50)
    finished_goods = prod_gen.generate_products(n_skus=n_skus)

    # 3. Raw Materials (Level 2)
    print("Generating Raw Materials...")
    ingredients = prod_gen.generate_ingredients()

    # 4. Bulk Intermediates (Level 1)
    print("Generating Bulk Intermediates...")
    bulk_intermediates = prod_gen.generate_bulk_intermediates(finished_goods)

    all_products = ingredients + bulk_intermediates + finished_goods

    # 5. Recipes (3-level BOM)
    print("Generating Recipes...")
    recipes = prod_gen.generate_recipes(
        finished_goods, bulk_intermediates, ingredients
    )

    # 6. Network
    n_stores = args.stores if args.stores > 0 else counts.get("small_retailers", 4000)
    print(f"Generating Network ({n_stores} stores)... this may take a moment.")
    nodes, links = net_gen.generate_network(
        n_stores=n_stores,
        n_suppliers=counts.get("suppliers", 50),
        n_plants=counts.get("plants", 4),
        n_rdcs=counts.get("rdcs", 4)
    )

    # 7. Write
    print("Writing files...")
    writer.write_products(all_products)
    writer.write_recipes(recipes)
    writer.write_locations(nodes)
    writer.write_partners(nodes)
    writer.write_links(links)

    print("Done!")
    print("Stats:")
    print(f"  Products: {len(all_products)}")
    print(f"    - Raw Materials: {len(ingredients)}")
    print(f"    - Bulk Intermediates: {len(bulk_intermediates)}")
    print(f"    - Finished SKUs: {len(finished_goods)}")
    print(f"  Recipes:  {len(recipes)}")
    print(f"  Nodes:    {len(nodes)}")
    print(f"  Links:    {len(links)}")

if __name__ == "__main__":
    main()
