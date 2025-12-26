from typing import Any

from prism_sim.network.core import Link, Node, NodeType
from prism_sim.product.core import Product, ProductCategory, Recipe
from prism_sim.simulation.world import World


class WorldBuilder:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.world = World()

    def build(self) -> World:
        self._build_products()
        self._build_recipes()
        self._build_network()
        return self.world

    def _build_products(self) -> None:
        # --- Ingredients ---
        # 1. SPOF Ingredient
        self.world.add_product(
            Product(
                id="ING-SURF-SPEC",
                name="Specialty Surfactant",
                category=ProductCategory.INGREDIENT,
                weight_kg=200,  # Drum
                length_cm=60,
                width_cm=60,
                height_cm=90,
                cases_per_pallet=4,
                cost_per_case=500.0,
            )
        )

        # 2. Generic Base (Water/Filler)
        self.world.add_product(
            Product(
                id="ING-BASE-LIQ",
                name="Purified Water Base",
                category=ProductCategory.INGREDIENT,
                weight_kg=1000,  # IBC Tote
                length_cm=100,
                width_cm=100,
                height_cm=100,
                cases_per_pallet=1,
                cost_per_case=50.0,
            )
        )

        # --- Finished Goods ---
        # 1. Oral Care (Toothpaste) - Cubes out
        self.world.add_product(
            Product(
                id="SKU-PASTE-001",
                name="Prism Whitening Paste 12pk",
                category=ProductCategory.ORAL_CARE,
                weight_kg=1.2,  # Light
                length_cm=20,
                width_cm=15,
                height_cm=10,  # Small box
                cases_per_pallet=120,  # High density
                cost_per_case=25.0,
            )
        )

        # 2. Personal Wash (Bar Soap) - Weighs out
        self.world.add_product(
            Product(
                id="SKU-SOAP-001",
                name="Prism Pure Soap 48pk",
                category=ProductCategory.PERSONAL_WASH,
                weight_kg=8.5,  # Heavy brick
                length_cm=30,
                width_cm=20,
                height_cm=15,
                cases_per_pallet=80,
                cost_per_case=18.0,
            )
        )

        # 3. Home Care (Detergent) - Fluid Heavyweight
        self.world.add_product(
            Product(
                id="SKU-DET-001",
                name="Prism Dish Liquid 6pk",
                category=ProductCategory.HOME_CARE,
                weight_kg=6.0,
                length_cm=35,
                width_cm=25,
                height_cm=30,  # Tall bottle, air gaps
                cases_per_pallet=40,
                cost_per_case=12.0,
            )
        )

    def _build_recipes(self) -> None:
        # Recipe for Detergent (Requires SPOF)
        self.world.add_recipe(
            Recipe(
                product_id="SKU-DET-001",
                ingredients={
                    "ING-SURF-SPEC": 0.05,  # 5% per case equivalent
                    "ING-BASE-LIQ": 0.95,
                },
                run_rate_cases_per_hour=1200,
                changeover_time_hours=4.0,
            )
        )

    def _build_network(self) -> None:
        # The Big 4 RDCs
        rdcs = [
            ("RDC-NAM-NE", "Northeast RDC", "Pennsylvania"),
            ("RDC-NAM-ATL", "Atlanta RDC", "Atlanta, GA"),
            ("RDC-NAM-CHI", "Chicago RDC", "Chicago, IL"),
            ("RDC-NAM-CAL", "West RDC", "Inland Empire, CA"),
        ]

        for rdc_id, name, location in rdcs:
            self.world.add_node(
                Node(
                    id=rdc_id,
                    name=name,
                    type=NodeType.DC,
                    location=location,
                    storage_capacity=100_000,  # Pallets placeholder
                )
            )

        # Add the Specialty Ingredient Supplier (SPOF)
        self.world.add_node(
            Node(
                id="SUP-SURF-SPEC",
                name="German Surfactants GmbH",
                type=NodeType.SUPPLIER,
                location="Hamburg, DE",
                throughput_capacity=float("inf"),
            )
        )

        # --- Retail Stores ---
        # 1. MegaMart (National Chain) - Assigned to RDCs
        # For simulation scale, we'll create one "Aggregate Store" per RDC
        # in a real run, this would be thousands of nodes.

        rdc_map = {
            "RDC-NAM-NE": ["STORE-MEGA-NE-001", "MegaMart Northeast"],
            "RDC-NAM-ATL": ["STORE-MEGA-ATL-001", "MegaMart South"],
            "RDC-NAM-CHI": ["STORE-MEGA-CHI-001", "MegaMart Midwest"],
            "RDC-NAM-CAL": ["STORE-MEGA-CAL-001", "MegaMart West"],
        }

        for rdc_id, (store_id, store_name) in rdc_map.items():
            self.world.add_node(
                Node(
                    id=store_id,
                    name=store_name,
                    type=NodeType.STORE,
                    location="Various",
                    storage_capacity=5000,  # Backroom capacity
                )
            )

            # Establish Link (RDC -> Store)
            # In a full model, this is explicit. For now, we imply it via logic,
            # but let's create the link object for completeness.
            self.world.add_link(
                Link(
                    id=f"LINK-{rdc_id}-{store_id}",
                    source_id=rdc_id,
                    target_id=store_id,
                    mode="truck",
                    lead_time_days=2.0,  # Standard retail replenishment
                    variability_sigma=0.5,
                )
            )
