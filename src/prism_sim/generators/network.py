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

        Structure:
            Suppliers -> Plants
            Plants -> RDCs
            RDCs -> [Retailer DCs, Club Stores, Distributors, Ecom FCs]
            Distributors -> Small Retailers (Out of Scope for direct sim)
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

        # --- 2. Generate Downstream Nodes (Customers) ---
        
        customers: list[Node] = []
        
        # Helper to get location
        def get_loc() -> str:
            return self.pool.sample_city()

        # A. B2M_LARGE (Retailer DCs)
        conf = channel_conf.get("B2M_LARGE", {})
        n_accounts = conf.get("accounts", 15)
        loc_range = conf.get("locations_per_account", [1, 5])
        
        for i in range(n_accounts):
            account_name = self.pool.sample_company()
            account_id = f"ACCT-LKG-{i+1:03d}"
            n_locs = self.rng.integers(loc_range[0], loc_range[1] + 1)
            
            for j in range(n_locs):
                customers.append(Node(
                    id=f"{account_id}-DC-{j+1:02d}",
                    name=f"{account_name} DC #{j+1}",
                    type=NodeType.DC, # Customer DC
                    location=get_loc(),
                    channel=CustomerChannel.B2M_LARGE,
                    store_format=StoreFormat.RETAILER_DC,
                    parent_account_id=account_id
                ))

        # B. B2M_CLUB (Costco/Sam's)
        conf = channel_conf.get("B2M_CLUB", {})
        n_accounts = conf.get("accounts", 3)
        loc_range = conf.get("locations_per_account", [5, 15])
        
        for i in range(n_accounts):
            account_name = self.pool.sample_company() + " Club"
            account_id = f"ACCT-CLUB-{i+1:03d}"
            n_locs = self.rng.integers(loc_range[0], loc_range[1] + 1)
            
            for j in range(n_locs):
                customers.append(Node(
                    id=f"{account_id}-LOC-{j+1:02d}",
                    name=f"{account_name} Store #{j+1}",
                    type=NodeType.STORE,
                    location=get_loc(),
                    channel=CustomerChannel.B2M_CLUB,
                    store_format=StoreFormat.CLUB,
                    parent_account_id=account_id
                ))

        # C. B2M_DISTRIBUTOR (3P Distributors)
        conf = channel_conf.get("B2M_DISTRIBUTOR", {})
        n_accounts = conf.get("accounts", 8)
        
        for i in range(n_accounts):
            account_name = self.pool.sample_company() + " Dist"
            account_id = f"ACCT-DIST-{i+1:03d}"
            customers.append(Node(
                id=f"{account_id}-DC-01",
                name=f"{account_name} Main DC",
                type=NodeType.DC,
                location=get_loc(),
                channel=CustomerChannel.B2M_DISTRIBUTOR,
                store_format=StoreFormat.DISTRIBUTOR_DC,
                parent_account_id=account_id
            ))

        # D. ECOMMERCE (Amazon FCs)
        conf = channel_conf.get("ECOMMERCE", {})
        n_accounts = conf.get("accounts", 5)
        loc_range = conf.get("locations_per_account", [1, 3])
        
        for i in range(n_accounts):
            account_name = "E-Shop " + self.pool.sample_company()
            account_id = f"ACCT-ECOM-{i+1:03d}"
            n_locs = self.rng.integers(loc_range[0], loc_range[1] + 1)
            
            for j in range(n_locs):
                customers.append(Node(
                    id=f"{account_id}-FC-{j+1:02d}",
                    name=f"{account_name} FC #{j+1}",
                    type=NodeType.DC,
                    location=get_loc(),
                    channel=CustomerChannel.ECOMMERCE,
                    store_format=StoreFormat.ECOM_FC,
                    parent_account_id=account_id
                ))
                
        # E. DTC (Direct Fulfillment)
        # Using target_counts directly or hardcoding small number
        n_dtc = target_counts.get("dtc_fc", 2)
        for i in range(n_dtc):
            customers.append(Node(
                id=f"PRISM-DTC-FC-{i+1:02d}",
                name=f"Prism DTC Center #{i+1}",
                type=NodeType.DC,
                location=get_loc(),
                channel=CustomerChannel.DTC,
                store_format=StoreFormat.ECOM_FC,
                parent_account_id="PRISM-DTC"
            ))

        # If no customers generated (fallback to legacy stores if config missing)
        if not customers and n_stores > 0:
             # Legacy fallback
             legacy_stores = self._generate_stores(n_stores)
             # Assign them a channel so logistics doesn't break
             for s in legacy_stores:
                 s.channel = CustomerChannel.B2M_LARGE # Assume direct for legacy
                 s.store_format = StoreFormat.SUPERMARKET
             customers.extend(legacy_stores)

        nodes.extend(customers)

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

        # RDCs -> Customers
        # Preferential Attachment
        rdc_degrees = [0] * len(rdcs)

        for customer in customers:
            # DTC served by specific logic? For now assume RDC replenishes DTC FCs
            # or Plants replenish DTC FCs. Let's assume RDC -> All Customers
            
            # Select RDC
            selected_indices = preferential_attachment(rdc_degrees, m=1, rng=self.rng)
            rdc_idx = selected_indices[0]
            rdc = rdcs[rdc_idx]
            rdc_degrees[rdc_idx] += 1

            dist = self.rng.exponential(200) + 50
            links.append(
                Link(
                    id=f"L-{rdc.id}-{customer.id}",
                    source_id=rdc.id,
                    target_id=customer.id,
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