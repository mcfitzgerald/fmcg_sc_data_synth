from typing import Any

from prism_sim.config.loader import load_world_definition
from prism_sim.network.core import Link, Node, NodeType
from prism_sim.product.core import Product, ProductCategory, Recipe
from prism_sim.simulation.world import World


class WorldBuilder:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.world_config = load_world_definition()
        self.world = World()

    def build(self) -> World:
        self._build_products()
        self._build_recipes()
        self._build_network()
        return self.world

    def _build_products(self) -> None:
        for p_data in self.world_config.get("products", []):
            self.world.add_product(
                Product(
                    id=p_data["id"],
                    name=p_data["name"],
                    category=ProductCategory[p_data["category"]],
                    weight_kg=p_data["weight_kg"],
                    length_cm=p_data["length_cm"],
                    width_cm=p_data["width_cm"],
                    height_cm=p_data["height_cm"],
                    cases_per_pallet=p_data["cases_per_pallet"],
                    cost_per_case=p_data["cost_per_case"],
                )
            )

    def _build_recipes(self) -> None:
        for r_data in self.world_config.get("recipes", []):
            self.world.add_recipe(
                Recipe(
                    product_id=r_data["product_id"],
                    ingredients=r_data["ingredients"],
                    run_rate_cases_per_hour=r_data["run_rate_cases_per_hour"],
                    changeover_time_hours=r_data["changeover_time_hours"],
                )
            )

    def _build_network(self) -> None:
        net_config = self.world_config.get("network", {})
        
        # 1. RDCs
        for rdc in net_config.get("rdcs", []):
            self.world.add_node(
                Node(
                    id=rdc["id"],
                    name=rdc["name"],
                    type=NodeType.DC,
                    location=rdc["location"],
                    storage_capacity=rdc["storage_capacity"],
                )
            )

        # 2. Suppliers
        for sup in net_config.get("suppliers", []):
            self.world.add_node(
                Node(
                    id=sup["id"],
                    name=sup["name"],
                    type=NodeType.SUPPLIER,
                    location=sup["location"],
                    throughput_capacity=float(sup["throughput_capacity"]) if sup["throughput_capacity"] else float("inf"),
                )
            )

        # 3. Plants
        for plant in net_config.get("plants", []):
            self.world.add_node(
                Node(
                    id=plant["id"],
                    name=plant["name"],
                    type=NodeType.PLANT,
                    location=plant["location"],
                    throughput_capacity=plant["throughput_capacity"],
                    storage_capacity=plant["storage_capacity"],
                )
            )

        # 4. Links
        links_config = net_config.get("links", {})
        
        # Suppliers -> Plants
        for link_def in links_config.get("suppliers_to_plants", []):
            source_id = link_def["source_id"]
            for target_id in link_def["target_ids"]:
                # Construct a unique link ID
                link_id = f"LINK-{source_id}-{target_id}"
                # Handle simplified naming for known patterns if needed, 
                # but unique ID is safer.
                # Current hardcoded pattern was: f"LINK-SUP-SURF-{plant_id}" etc.
                # We will standardize on LINK-{source}-{target} for config-driven generation
                
                self.world.add_link(
                    Link(
                        id=link_id,
                        source_id=source_id,
                        target_id=target_id,
                        mode=link_def["mode"],
                        lead_time_days=link_def["lead_time_days"],
                        variability_sigma=link_def["variability_sigma"],
                    )
                )

        # Plants -> RDCs
        for link_def in links_config.get("plants_to_rdcs", []):
            source_id = link_def["source"]
            target_id = link_def["target"]
            self.world.add_link(
                Link(
                    id=f"LINK-{source_id}-{target_id}",
                    source_id=source_id,
                    target_id=target_id,
                    mode="truck",
                    lead_time_days=link_def["lead_time"],
                    variability_sigma=0.5, # Default, or could be in config
                )
            )

        # 5. Stores and RDC->Store Links
        stores_config = net_config.get("stores", {})
        rdc_store_links = links_config.get("rdcs_to_stores", {})
        
        rdc_map = rdc_store_links.get("map", {})
        link_params = rdc_store_links.get("params", {})

        for rdc_id, store_info in rdc_map.items():
            store_id = store_info["id"]
            store_name = store_info["name"]
            
            self.world.add_node(
                Node(
                    id=store_id,
                    name=store_name,
                    type=NodeType.STORE,
                    location=stores_config.get("location", "Various"),
                    storage_capacity=stores_config.get("storage_capacity", 5000),
                )
            )

            self.world.add_link(
                Link(
                    id=f"LINK-{rdc_id}-{store_id}",
                    source_id=rdc_id,
                    target_id=store_id,
                    mode=link_params.get("mode", "truck"),
                    lead_time_days=link_params.get("lead_time_days", 2.0),
                    variability_sigma=link_params.get("variability_sigma", 0.5),
                )
            )