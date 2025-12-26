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
        # Recipe for Detergent (Requires SPOF - Specialty Surfactant)
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

        # Recipe for Toothpaste (Generic ingredients)
        self.world.add_recipe(
            Recipe(
                product_id="SKU-PASTE-001",
                ingredients={
                    "ING-BASE-LIQ": 1.0,  # Simplified: 1 unit base per case
                },
                run_rate_cases_per_hour=1500,
                changeover_time_hours=2.0,
            )
        )

        # Recipe for Soap (Generic ingredients)
        self.world.add_recipe(
            Recipe(
                product_id="SKU-SOAP-001",
                ingredients={
                    "ING-BASE-LIQ": 1.0,  # Simplified: 1 unit base per case
                },
                run_rate_cases_per_hour=1800,
                changeover_time_hours=3.0,
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

        # Add the Specialty Ingredient Supplier (SPOF - Primary)
        self.world.add_node(
            Node(
                id="SUP-SURF-SPEC",
                name="German Surfactants GmbH",
                type=NodeType.SUPPLIER,
                location="Hamburg, DE",
                throughput_capacity=float("inf"),
            )
        )

        # Add Backup Supplier (Mexico - 25% cost premium, 2x variability)
        self.world.add_node(
            Node(
                id="SUP-SURF-BACKUP",
                name="Mexican Chemicals SA",
                type=NodeType.SUPPLIER,
                location="Monterrey, MX",
                throughput_capacity=float("inf"),
            )
        )

        # Add Generic Base Liquid Supplier
        self.world.add_node(
            Node(
                id="SUP-BASE-LIQ",
                name="Industrial Water Works",
                type=NodeType.SUPPLIER,
                location="Houston, TX",
                throughput_capacity=float("inf"),
            )
        )

        # --- Manufacturing Plants ---
        plants = [
            ("PLANT-OH", "Ohio Plant", "Columbus, OH"),
            ("PLANT-TX", "Texas Plant", "Houston, TX"),
        ]

        for plant_id, name, location in plants:
            self.world.add_node(
                Node(
                    id=plant_id,
                    name=name,
                    type=NodeType.PLANT,
                    location=location,
                    throughput_capacity=50_000,  # Cases per day
                    storage_capacity=10_000,  # Pallets
                )
            )

        # --- Links: Suppliers -> Plants ---
        # Primary Surfactant Supplier -> Plants (Long lead time - transatlantic)
        for plant_id in ["PLANT-OH", "PLANT-TX"]:
            self.world.add_link(
                Link(
                    id=f"LINK-SUP-SURF-{plant_id}",
                    source_id="SUP-SURF-SPEC",
                    target_id=plant_id,
                    mode="ocean",
                    lead_time_days=21.0,  # Transatlantic shipping
                    variability_sigma=3.0,
                )
            )

        # Backup Surfactant Supplier -> Plants (Shorter lead time - domestic)
        for plant_id in ["PLANT-OH", "PLANT-TX"]:
            self.world.add_link(
                Link(
                    id=f"LINK-SUP-SURF-BACKUP-{plant_id}",
                    source_id="SUP-SURF-BACKUP",
                    target_id=plant_id,
                    mode="truck",
                    lead_time_days=5.0,
                    variability_sigma=2.0,  # 2x variability vs normal
                )
            )

        # Base Liquid Supplier -> Plants
        for plant_id in ["PLANT-OH", "PLANT-TX"]:
            self.world.add_link(
                Link(
                    id=f"LINK-SUP-BASE-{plant_id}",
                    source_id="SUP-BASE-LIQ",
                    target_id=plant_id,
                    mode="truck",
                    lead_time_days=2.0,
                    variability_sigma=0.5,
                )
            )

        # --- Links: Plants -> RDCs ---
        plant_rdc_distances = [
            ("PLANT-OH", "RDC-NAM-NE", 3.0),
            ("PLANT-OH", "RDC-NAM-ATL", 4.0),
            ("PLANT-OH", "RDC-NAM-CHI", 2.0),
            ("PLANT-OH", "RDC-NAM-CAL", 5.0),
            ("PLANT-TX", "RDC-NAM-NE", 5.0),
            ("PLANT-TX", "RDC-NAM-ATL", 3.0),
            ("PLANT-TX", "RDC-NAM-CHI", 3.0),
            ("PLANT-TX", "RDC-NAM-CAL", 4.0),
        ]

        for plant_id, rdc_id, lead_time in plant_rdc_distances:
            self.world.add_link(
                Link(
                    id=f"LINK-{plant_id}-{rdc_id}",
                    source_id=plant_id,
                    target_id=rdc_id,
                    mode="truck",
                    lead_time_days=lead_time,
                    variability_sigma=0.5,
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
