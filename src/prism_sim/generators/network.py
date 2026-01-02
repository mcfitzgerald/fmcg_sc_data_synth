"""Network topology generator for the Deep NAM expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from prism_sim.generators.distributions import preferential_attachment
from prism_sim.generators.static_pool import StaticDataPool
from prism_sim.network.core import CustomerChannel, Link, Node, NodeType, StoreFormat

if TYPE_CHECKING:
    from numpy.random import Generator


class NetworkGenerator:
    """Generates the physical supply chain network (Nodes and Links)."""

    def __init__(self, seed: int = 42, config: dict[str, Any] | None = None) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.pool = StaticDataPool(seed=seed)
        self.config = config or {}

    def generate_network(
        self,
        n_stores: int = 4500,  # Legacy param, now driven by config
        n_suppliers: int = 50,
        n_plants: int = 4,
        n_rdcs: int = 4,
    ) -> tuple[list[Node], list[Link]]:
        """
        Generate the full network topology with channel structure.

        Structure (Option C - Hierarchical):
            Suppliers -> Plants
            Plants -> RDCs
            RDCs -> Customer DCs (logistics layer)
            Customer DCs -> Stores (POS layer)
        """
        nodes: list[Node] = []
        links: list[Link] = []

        # Load topology config if available, else defaults
        topology_conf = self.config.get("topology", {})
        target_counts = topology_conf.get("target_counts", {})
        channel_conf = topology_conf.get("channels", {})

        # --- 1. Generate Upstream Nodes (Suppliers, Plants, RDCs) ---

        # Suppliers
        suppliers = self._generate_suppliers(n_suppliers)
        nodes.extend(suppliers)

        # Plants
        plants = self._generate_plants(n_plants)
        nodes.extend(plants)

        # RDCs
        rdcs = self._generate_rdcs(n_rdcs)
        nodes.extend(rdcs)

        # --- 2. Generate Downstream Nodes (DCs + Stores) ---

        customer_dcs: list[Node] = []  # Logistics layer
        stores: list[Node] = []         # POS layer
        dc_to_stores: dict[str, list[Node]] = {}  # For linking

        # Helper to get location
        def get_loc() -> str:
            return self.pool.sample_cities(1)[0]

        # A. B2M_LARGE (Retailer DCs + their stores)
        n_retailer_dcs = target_counts.get("retailer_dcs", 20)
        # Distribute stores across retailer DCs (assume ~100 stores per DC)
        stores_per_retailer_dc = target_counts.get("stores_per_retailer_dc", 100)

        retailer_companies = self.pool.sample_companies(n_retailer_dcs)
        for i in range(n_retailer_dcs):
            account_name = retailer_companies[i]
            dc_id = f"RET-DC-{i+1:03d}"
            dc_loc = get_loc()

            dc_node = Node(
                id=dc_id,
                name=f"{account_name} DC",
                type=NodeType.DC,
                location=dc_loc,
                channel=CustomerChannel.B2M_LARGE,
                store_format=StoreFormat.RETAILER_DC,
                parent_account_id=f"ACCT-RET-{i+1:03d}"
            )
            customer_dcs.append(dc_node)
            dc_to_stores[dc_id] = []

            # Generate stores under this DC
            for j in range(stores_per_retailer_dc):
                store_id = f"STORE-RET-{i+1:03d}-{j+1:04d}"
                store_node = Node(
                    id=store_id,
                    name=f"{account_name} Store #{j+1}",
                    type=NodeType.STORE,
                    location=get_loc(),
                    channel=CustomerChannel.B2M_LARGE,
                    store_format=StoreFormat.SUPERMARKET,
                    parent_account_id=dc_id  # Link to parent DC
                )
                stores.append(store_node)
                dc_to_stores[dc_id].append(store_node)

        # B. B2M_CLUB (Club stores - no DC layer, direct to RDC)
        n_club_stores = target_counts.get("club_stores", 30)
        club_companies = self.pool.sample_companies(min(3, n_club_stores))

        for i in range(n_club_stores):
            company_idx = i % len(club_companies)
            account_name = club_companies[company_idx] + " Club"
            store_id = f"STORE-CLUB-{i+1:04d}"

            stores.append(Node(
                id=store_id,
                name=f"{account_name} #{i+1}",
                type=NodeType.STORE,
                location=get_loc(),
                channel=CustomerChannel.B2M_CLUB,
                store_format=StoreFormat.CLUB,
                parent_account_id=f"ACCT-CLUB-{company_idx+1:03d}"
            ))

        # C. B2M_DISTRIBUTOR (Distributor DCs + small retailers)
        n_distributor_dcs = target_counts.get("distributor_dcs", 8)
        n_small_retailers = target_counts.get("small_retailers", 4000)
        stores_per_distributor = n_small_retailers // n_distributor_dcs

        dist_companies = self.pool.sample_companies(n_distributor_dcs)
        for i in range(n_distributor_dcs):
            account_name = dist_companies[i] + " Distribution"
            dc_id = f"DIST-DC-{i+1:03d}"

            dc_node = Node(
                id=dc_id,
                name=f"{account_name} DC",
                type=NodeType.DC,
                location=get_loc(),
                channel=CustomerChannel.B2M_DISTRIBUTOR,
                store_format=StoreFormat.DISTRIBUTOR_DC,
                parent_account_id=f"ACCT-DIST-{i+1:03d}"
            )
            customer_dcs.append(dc_node)
            dc_to_stores[dc_id] = []

            # Generate small retailers under this distributor
            for j in range(stores_per_distributor):
                store_id = f"STORE-SMB-{i+1:03d}-{j+1:04d}"
                store_node = Node(
                    id=store_id,
                    name=f"Independent Store {i+1}-{j+1}",
                    type=NodeType.STORE,
                    location=get_loc(),
                    channel=CustomerChannel.B2M_DISTRIBUTOR,
                    store_format=StoreFormat.CONVENIENCE,
                    parent_account_id=dc_id
                )
                stores.append(store_node)
                dc_to_stores[dc_id].append(store_node)

        # D. ECOMMERCE (Ecom FCs - leaf nodes, no stores)
        n_ecom_fcs = target_counts.get("ecom_fcs", 10)
        ecom_companies = self.pool.sample_companies(min(5, n_ecom_fcs))

        for i in range(n_ecom_fcs):
            company_idx = i % len(ecom_companies)
            account_name = "E-Shop " + ecom_companies[company_idx]
            fc_id = f"ECOM-FC-{i+1:03d}"

            customer_dcs.append(Node(
                id=fc_id,
                name=f"{account_name} FC #{i+1}",
                type=NodeType.DC,
                location=get_loc(),
                channel=CustomerChannel.ECOMMERCE,
                store_format=StoreFormat.ECOM_FC,
                parent_account_id=f"ACCT-ECOM-{company_idx+1:03d}"
            ))

        # E. DTC (Direct Fulfillment - leaf nodes)
        n_dtc = target_counts.get("dtc_fc", 2)
        for i in range(n_dtc):
            customer_dcs.append(Node(
                id=f"DTC-FC-{i+1:02d}",
                name=f"Prism DTC Center #{i+1}",
                type=NodeType.DC,
                location=get_loc(),
                channel=CustomerChannel.DTC,
                store_format=StoreFormat.ECOM_FC,
                parent_account_id="PRISM-DTC"
            ))

        nodes.extend(customer_dcs)
        nodes.extend(stores)

        # --- 3. Generate Links ---

        # Suppliers -> Plants
        for supplier in suppliers:
            # Pick 1-2 random plants
            indices = self.rng.choice(
                len(plants),
                size=self.rng.integers(1, 3),
                replace=False,
            )
            for idx in indices:
                plant = plants[int(idx)]
                links.append(
                    Link(
                        id=f"L-{supplier.id}-{plant.id}",
                        source_id=supplier.id,
                        target_id=plant.id,
                        mode="truck",
                        distance_km=self.rng.uniform(100, 1000),
                        lead_time_days=self.rng.normal(2.0, 0.5),
                    )
                )

        # Plants -> RDCs (Full Mesh)
        for plant in plants:
            for rdc in rdcs:
                dist = self.rng.uniform(500, 2500)
                links.append(
                    Link(
                        id=f"L-{plant.id}-{rdc.id}",
                        source_id=plant.id,
                        target_id=rdc.id,
                        mode="truck",
                        distance_km=dist,
                        lead_time_days=dist / 800.0 * 1.5 + 1.0,
                    )
                )

        # RDCs -> Customer DCs (logistics layer)
        rdc_degrees = [0] * len(rdcs)

        for dc in customer_dcs:
            # Select RDC using preferential attachment
            selected_indices = preferential_attachment(rdc_degrees, m=1, rng=self.rng)
            rdc_idx = selected_indices[0]
            rdc = rdcs[rdc_idx]
            rdc_degrees[rdc_idx] += 1

            dist = self.rng.exponential(200) + 50
            links.append(
                Link(
                    id=f"L-{rdc.id}-{dc.id}",
                    source_id=rdc.id,
                    target_id=dc.id,
                    mode="truck",
                    distance_km=dist,
                    lead_time_days=self.rng.normal(1.0, 0.2),
                )
            )

        # Customer DCs -> Stores (replenishment to POS layer)
        for dc_id, dc_stores in dc_to_stores.items():
            for store in dc_stores:
                dist = self.rng.exponential(50) + 10  # Shorter distance for last-mile
                links.append(
                    Link(
                        id=f"L-{dc_id}-{store.id}",
                        source_id=dc_id,
                        target_id=store.id,
                        mode="truck",
                        distance_km=dist,
                        lead_time_days=self.rng.normal(0.5, 0.1),  # Same-day/next-day
                    )
                )

        # RDCs -> Club Stores (direct, no DC layer)
        club_stores = [s for s in stores if s.store_format == StoreFormat.CLUB]
        for store in club_stores:
            selected_indices = preferential_attachment(rdc_degrees, m=1, rng=self.rng)
            rdc_idx = selected_indices[0]
            rdc = rdcs[rdc_idx]
            rdc_degrees[rdc_idx] += 1

            dist = self.rng.exponential(150) + 30
            links.append(
                Link(
                    id=f"L-{rdc.id}-{store.id}",
                    source_id=rdc.id,
                    target_id=store.id,
                    mode="truck",
                    distance_km=dist,
                    lead_time_days=self.rng.normal(1.0, 0.2),
                )
            )

        return nodes, links

    def _generate_suppliers(self, n: int) -> list[Node]:
        nodes = []
        companies = self.pool.sample_companies(n)
        cities = self.pool.sample_cities(n)
        for i in range(n):
            node_id = f"SUP-{i + 1:03d}"
            capacity = 500000.0 if i == 0 else float("inf")
            nodes.append(
                Node(
                    id=node_id,
                    name=companies[i],
                    type=NodeType.SUPPLIER,
                    location=cities[i],
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
            nodes.append(
                Node(
                    id=f"STORE-{i + 1:05d}",
                    name=f"Store {cities[i]} #{i + 1}",
                    type=NodeType.STORE,
                    location=cities[i],
                )
            )
        return nodes
