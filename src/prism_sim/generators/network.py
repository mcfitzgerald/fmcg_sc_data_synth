"""Network topology generator for the Deep NAM expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from prism_sim.generators.static_pool import StaticDataPool
from prism_sim.network.core import CustomerChannel, Link, Node, NodeType, StoreFormat

if TYPE_CHECKING:
    from numpy.random import Generator


EARTH_RADIUS_KM = 6371.0


class NetworkGenerator:
    """Generates the physical supply chain network (Nodes and Links)."""

    def __init__(self, seed: int = 42, config: dict[str, Any] | None = None) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.config = config or {}
        # Pass config to pool so it knows city data path
        self.pool = StaticDataPool(seed=seed, config=self.config)

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in KM between two points."""
        R = EARTH_RADIUS_KM
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return float(R * c)

    def _get_jittered_coords(
        self, lat: float, lon: float, scale: float = 0.05
    ) -> tuple[float, float]:
        """Apply random jitter to coordinates to simulate metropolitan spread."""
        j_lat = lat + self.rng.uniform(-scale, scale)
        j_lon = lon + self.rng.uniform(-scale, scale)
        return float(j_lat), float(j_lon)

    def generate_network(
        self,
        n_stores: int = 4500,  # Legacy param, now driven by config
        n_suppliers: int = 50,
        n_plants: int = 4,
        n_rdcs: int = 4,
    ) -> tuple[list[Node], list[Link]]:
        """
        Generate the full network topology with geospatial logic.

        Structure:
            Suppliers -> Plants
            Plants -> RDCs (Fixed locations)
            RDCs -> Customer DCs (Linked by proximity)
            Customer DCs -> Stores (Linked by hierarchy)
        """
        nodes: list[Node] = []
        links: list[Link] = []

        # Load topology config
        topology_conf = self.config.get("topology", {})
        fixed_nodes_conf = topology_conf.get("fixed_nodes", {})
        target_counts = topology_conf.get("target_counts", {})

        # Geospatial params
        sim_params = self.config.get("simulation_parameters", {})
        geo_params = sim_params.get("geospatial", {})
        jitter = geo_params.get("metropolitan_jitter", 0.05)
        speed = geo_params.get("avg_truck_speed_kmh", 80.0)
        handling = geo_params.get("base_handling_days", 1.0)

        # --- 1. Generate Upstream Nodes ---

        # Suppliers (Sample from cities)
        suppliers: list[Node] = []
        supplier_cities = self.pool.sample_cities(n_suppliers)
        for i, city_data in enumerate(supplier_cities):
            node_id = f"SUP-{i+1:03d}"
            # Plant-TX is often a SPOF for surfactant in this twin
            capacity = 500000.0 if i == 0 else float("inf")
            suppliers.append(Node(
                id=node_id,
                name=f"Supplier {i+1}",
                type=NodeType.SUPPLIER,
                location=f"{city_data['city']}, {city_data['state']}",
                lat=city_data["lat"],
                lon=city_data["lon"],
                throughput_capacity=capacity
            ))
        nodes.extend(suppliers)

        # Plants (From fixed config)
        plants: list[Node] = []
        for p_data in fixed_nodes_conf.get("plants", []):
            plants.append(Node(
                id=p_data["id"],
                name=p_data["name"],
                type=NodeType.PLANT,
                location=p_data["city"],
                lat=p_data["lat"],
                lon=p_data["lon"]
            ))
        nodes.extend(plants)

        # RDCs (From fixed config)
        rdcs: list[Node] = []
        for r_data in fixed_nodes_conf.get("rdcs", []):
            rdcs.append(Node(
                id=r_data["id"],
                name=r_data["name"],
                type=NodeType.DC,
                location=r_data["city"],
                lat=r_data["lat"],
                lon=r_data["lon"]
            ))
        nodes.extend(rdcs)

        # --- 2. Generate Downstream Nodes ---

        customer_dcs: list[Node] = []
        stores: list[Node] = []
        dc_to_stores: dict[str, list[Node]] = {}

        # A. B2M_LARGE (Retailer DCs + their stores)
        n_retailer_dcs = target_counts.get("retailer_dcs", 20)
        stores_per_retailer_dc = target_counts.get("stores_per_retailer_dc", 100)

        retailer_companies = self.pool.sample_companies(n_retailer_dcs)
        dc_cities = self.pool.sample_cities(n_retailer_dcs)

        for i in range(n_retailer_dcs):
            account_name = retailer_companies[i]
            city_data = dc_cities[i]
            dc_id = f"RET-DC-{i+1:03d}"

            dc_node = Node(
                id=dc_id,
                name=f"{account_name} DC",
                type=NodeType.DC,
                location=f"{city_data['city']}, {city_data['state']}",
                lat=city_data["lat"],
                lon=city_data["lon"],
                channel=CustomerChannel.B2M_LARGE,
                store_format=StoreFormat.RETAILER_DC,
                parent_account_id=f"ACCT-RET-{i+1:03d}"
            )
            customer_dcs.append(dc_node)
            dc_to_stores[dc_id] = []

            # Stores under this DC
            for j in range(stores_per_retailer_dc):
                lat, lon = self._get_jittered_coords(city_data["lat"], city_data["lon"], jitter)
                store_node = Node(
                    id=f"STORE-RET-{i+1:03d}-{j+1:04d}",
                    name=f"{account_name} Store #{j+1}",
                    type=NodeType.STORE,
                    location=dc_node.location,
                    lat=lat,
                    lon=lon,
                    channel=CustomerChannel.B2M_LARGE,
                    store_format=StoreFormat.SUPERMARKET,
                    parent_account_id=dc_id
                )
                stores.append(store_node)
                dc_to_stores[dc_id].append(store_node)

        # B. B2M_CLUB (Direct to RDC)
        n_club_stores = target_counts.get("club_stores", 30)
        club_cities = self.pool.sample_cities(n_club_stores)

        for i, city in enumerate(club_cities):
            stores.append(Node(
                id=f"STORE-CLUB-{i+1:04d}",
                name=f"Club Store {i+1}",
                type=NodeType.STORE,
                location=f"{city['city']}, {city['state']}",
                lat=city["lat"],
                lon=city["lon"],
                channel=CustomerChannel.B2M_CLUB,
                store_format=StoreFormat.CLUB
            ))

        # C. B2M_DISTRIBUTOR (Distributor DCs + small retailers)
        n_distributor_dcs = target_counts.get("distributor_dcs", 8)
        n_small_retailers = target_counts.get("small_retailers", 4000)
        stores_per_distributor = n_small_retailers // n_distributor_dcs

        dist_cities = self.pool.sample_cities(n_distributor_dcs)
        for i, city in enumerate(dist_cities):
            dc_id = f"DIST-DC-{i+1:03d}"
            dc_node = Node(
                id=dc_id,
                name=f"Distributor DC {i+1}",
                type=NodeType.DC,
                location=f"{city['city']}, {city['state']}",
                lat=city["lat"],
                lon=city["lon"],
                channel=CustomerChannel.B2M_DISTRIBUTOR,
                store_format=StoreFormat.DISTRIBUTOR_DC
            )
            customer_dcs.append(dc_node)
            dc_to_stores[dc_id] = []

            for j in range(stores_per_distributor):
                lat, lon = self._get_jittered_coords(city["lat"], city["lon"], jitter)
                store_node = Node(
                    id=f"STORE-SMB-{i+1:03d}-{j+1:04d}",
                    name=f"SMB Store {i+1}-{j+1}",
                    type=NodeType.STORE,
                    location=dc_node.location,
                    lat=lat,
                    lon=lon,
                    channel=CustomerChannel.B2M_DISTRIBUTOR,
                    store_format=StoreFormat.CONVENIENCE,
                    parent_account_id=dc_id
                )
                stores.append(store_node)
                dc_to_stores[dc_id].append(store_node)

        # D. ECOMMERCE (Leaf nodes)
        n_ecom_fcs = target_counts.get("ecom_fcs", 10)
        ecom_cities = self.pool.sample_cities(n_ecom_fcs)
        for i, city in enumerate(ecom_cities):
            customer_dcs.append(Node(
                id=f"ECOM-FC-{i+1:03d}",
                name=f"ECOM FC {i+1}",
                type=NodeType.DC,
                location=f"{city['city']}, {city['state']}",
                lat=city["lat"],
                lon=city["lon"],
                channel=CustomerChannel.ECOMMERCE,
                store_format=StoreFormat.ECOM_FC
            ))

        nodes.extend(customer_dcs)
        nodes.extend(stores)

        # --- 3. Generate Links (Physics-Based) ---

        # Helper for physics-based link
        def add_geo_link(s_node: Node, t_node: Node) -> None:
            dist = self._haversine(s_node.lat, s_node.lon, t_node.lat, t_node.lon)
            # Convert driving hours to days: (km / km/h) / 24 hours/day + handling
            lt = (dist / speed / 24) + handling
            links.append(Link(
                id=f"L-{s_node.id}-{t_node.id}",
                source_id=s_node.id,
                target_id=t_node.id,
                distance_km=float(dist),
                lead_time_days=float(lt)
            ))

        # Suppliers -> Plants (Nearest 2)
        for supplier in suppliers:
            dists = [(self._haversine(supplier.lat, supplier.lon, p.lat, p.lon), p) for p in plants]
            dists.sort(key=lambda x: x[0])
            for _, plant in dists[:2]:
                add_geo_link(supplier, plant)

        # Plants -> RDCs (Full Mesh)
        for plant in plants:
            for rdc in rdcs:
                add_geo_link(plant, rdc)

        # RDCs -> Customer DCs (Nearest RDC)
        # This replaces preferential attachment with geography
        for dc in customer_dcs:
            dists = [(self._haversine(dc.lat, dc.lon, r.lat, r.lon), r) for r in rdcs]
            dists.sort(key=lambda x: x[0])
            closest_rdc = dists[0][1]
            add_geo_link(closest_rdc, dc)

        # Customer DCs -> Stores (Hierarchical)
        for dc_id, dc_stores in dc_to_stores.items():
            dc_node = next(n for n in customer_dcs if n.id == dc_id)
            for store in dc_stores:
                add_geo_link(dc_node, store)

        # RDCs -> Club Stores (Nearest RDC)
        club_stores = [s for s in stores if s.store_format == StoreFormat.CLUB]
        for store in club_stores:
            dists = [(self._haversine(store.lat, store.lon, r.lat, r.lon), r) for r in rdcs]
            dists.sort(key=lambda x: x[0])
            closest_rdc = dists[0][1]
            add_geo_link(closest_rdc, store)

        return nodes, links

    def _generate_suppliers(self, n: int) -> list[Node]:
        nodes = []
        companies = self.pool.sample_companies(n)
        cities = self.pool.sample_cities(n)
        for i in range(n):
            node_id = f"SUP-{i + 1:03d}"
            capacity = 500000.0 if i == 0 else float("inf")
            city_data = cities[i]
            nodes.append(
                Node(
                    id=node_id,
                    name=companies[i],
                    type=NodeType.SUPPLIER,
                    location=f"{city_data['city']}, {city_data['state']}",
                    lat=city_data.get("lat", 0.0),
                    lon=city_data.get("lon", 0.0),
                    throughput_capacity=capacity,
                )
            )
        return nodes

    def _generate_plants(self, n: int) -> list[Node]:
        plants = [
            Node("PLANT-OH", "Ohio Plant", NodeType.PLANT, "Columbus, OH"),
            Node("PLANT-TX", "Texas Plant", NodeType.PLANT, "Dallas, TX"),
            Node("PLANT-GA", "Georgia Plant", NodeType.PLANT, "Atlanta, GA"),
            Node("PLANT-CA", "California Plant", NodeType.PLANT, "Sacramento, CA"),
        ]
        return plants[:n]

    def _generate_rdcs(self, n: int) -> list[Node]:
        rdcs = [
            Node("RDC-NE", "Northeast DC", NodeType.DC, "Allentown, PA"),
            Node("RDC-MW", "Midwest DC", NodeType.DC, "Chicago, IL"),
            Node("RDC-SO", "South DC", NodeType.DC, "Memphis, TN"),
            Node("RDC-WE", "West DC", NodeType.DC, "Reno, NV"),
        ]
        return rdcs[:n]

    def _generate_stores(self, n: int) -> list[Node]:
        """Legacy fallback."""
        nodes = []
        cities = self.pool.sample_cities(n, replace=True)
        for i in range(n):
            city_data = cities[i]
            location = f"{city_data['city']}, {city_data['state']}"
            nodes.append(
                Node(
                    id=f"STORE-{i + 1:05d}",
                    name=f"Store {location} #{i + 1}",
                    type=NodeType.STORE,
                    location=location,
                    lat=city_data.get("lat", 0.0),
                    lon=city_data.get("lon", 0.0),
                )
            )
        return nodes
