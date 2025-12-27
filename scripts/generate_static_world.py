"""Script to generate the static world (Level 0-4) for Deep NAM expansion."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prism_sim.generators.hierarchy import ProductGenerator
from prism_sim.generators.network import NetworkGenerator
from prism_sim.product.core import Product, ProductCategory
from prism_sim.writers.static_writer import StaticWriter


def main() -> None:
    output_dir = Path("data/output/static_world")
    print(f"Generating static world to {output_dir}...")

    # 1. Generators
    prod_gen = ProductGenerator(seed=42)
    net_gen = NetworkGenerator(seed=42)
    writer = StaticWriter(str(output_dir))

    # 2. Products
    print("Generating Products...")
    finished_goods = prod_gen.generate_products(n_skus=50)

    # Create base ingredients (hardcoded for now as per scor_reference)
    ingredients = [
        Product(
            id="ING-SURF-SPEC",
            name="Specialty Surfactant",
            category=ProductCategory.INGREDIENT,
            weight_kg=200,
            length_cm=60,
            width_cm=60,
            height_cm=90,
            cases_per_pallet=4,
            cost_per_case=500.0,
        ),
        Product(
            id="ING-BASE-LIQ",
            name="Purified Water Base",
            category=ProductCategory.INGREDIENT,
            weight_kg=1000,
            length_cm=100,
            width_cm=100,
            height_cm=100,
            cases_per_pallet=1,
            cost_per_case=50.0,
        ),
    ]

    all_products = ingredients + finished_goods

    # 3. Recipes
    print("Generating Recipes...")
    recipes = prod_gen.generate_recipes(finished_goods, ingredients)

    # 4. Network
    print("Generating Network (4500 stores)... this may take a moment.")
    nodes, links = net_gen.generate_network(
        n_stores=4500,
        n_suppliers=50,
        n_plants=4,
        n_rdcs=4
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
