"""Network topology generator for the Deep NAM expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from prism_sim.generators.distributions import preferential_attachment
from prism_sim.generators.static_pool import StaticDataPool
from prism_sim.network.core import Link, Node, NodeType

if TYPE_CHECKING:
    from numpy.random import Generator


class NetworkGenerator:
    """Generates the physical supply chain network (Nodes and Links)."""

    def __init__(self, seed: int = 42) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.pool = StaticDataPool(seed=seed)

    def generate_network(
        self,
        n_stores: int = 4500,
        n_suppliers: int = 50,
        n_plants: int = 4,
        n_rdcs: int = 4,
    ) -> tuple[list[Node], list[Link]]:
        """
        Generate the full network topology.

        Structure:
            Suppliers (Raw Materials) -> Plants (Manufacturing)
            Plants -> RDCs (Distribution)
            RDCs -> Stores (Retail)

        Args:
            n_stores: Number of retail locations.
            n_suppliers: Number of raw material suppliers.
            n_plants: Number of manufacturing plants.
            n_rdcs: Number of Regional Distribution Centers.

        Returns:
            Tuple of (List[Node], List[Link])
        """
        nodes: list[Node] = []
        links: list[Link] = []

        # --- 1. Generate Nodes ---

        # Suppliers
        suppliers = self._generate_suppliers(n_suppliers)
        nodes.extend(suppliers)

        # Plants (Fixed locations for realism)
        plants = [
            Node(
                id="PLANT-OH",
                name="Ohio Plant",
                type=NodeType.PLANT,
                location="Columbus, OH",
            ),
            Node(
                id="PLANT-TX",
                name="Texas Plant",
                type=NodeType.PLANT,
                location="Dallas, TX",
            ),
            Node(
                id="PLANT-GA",
                name="Georgia Plant",
                type=NodeType.PLANT,
                location="Atlanta, GA",
            ),
            Node(
                id="PLANT-CA",
                name="California Plant",
                type=NodeType.PLANT,
                location="Sacramento, CA",
            ),
        ]
        # Adjust if n_plants != 4, but defaulting to fixed set is safer for now
        nodes.extend(plants[:n_plants])
        active_plants = plants[:n_plants]

        # RDCs
        rdcs = [
            Node(
                id="RDC-NE",
                name="Northeast DC",
                type=NodeType.DC,
                location="Allentown, PA",
            ),
            Node(
                id="RDC-MW",
                name="Midwest DC",
                type=NodeType.DC,
                location="Chicago, IL",
            ),
            Node(
                id="RDC-SO",
                name="South DC",
                type=NodeType.DC,
                location="Memphis, TN",
            ),
            Node(
                id="RDC-WE",
                name="West DC",
                type=NodeType.DC,
                location="Reno, NV",
            ),
        ]
        nodes.extend(rdcs[:n_rdcs])
        active_rdcs = rdcs[:n_rdcs]

        # Stores
        stores = self._generate_stores(n_stores)
        nodes.extend(stores)

        # --- 2. Generate Links ---

        # Suppliers -> Plants
        # Each supplier supplies at least one plant.
        # Using Pareto: Some suppliers supply many plants, some few.
        for supplier in suppliers:
            # Pick 1-2 random plants to supply
            # Use range of indices instead of objects for numpy choice compatibility
            indices = self.rng.choice(
                len(active_plants),
                size=self.rng.integers(1, 3),
                replace=False,
            )
            for idx in indices:
                plant = active_plants[int(idx)]
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

        # Plants -> RDCs
        # Fully connected mesh (every plant ships to every DC)
        for plant in active_plants:
            for rdc in active_rdcs:
                dist = self.rng.uniform(500, 2500)
                links.append(
                    Link(
                        id=f"L-{plant.id}-{rdc.id}",
                        source_id=plant.id,
                        target_id=rdc.id,
                        mode="truck",
                        distance_km=dist,
                        lead_time_days=dist / 800.0 * 1.5 + 1.0,  # Approx calc
                    )
                )

        # RDCs -> Stores
        # Preferential Attachment / Clustering
        # Each store is assigned to ONE primary RDC (usually closest)
        # We simulate "closest" by randomly assigning with weights or just
        # uniformly for now
        # Expansion plan mentions: "Barabási-Albert (Hubs)"

        # Initialize degrees for preferential attachment
        # (just count of stores per RDC)
        rdc_degrees = [0] * len(active_rdcs)

        for _i, store in enumerate(stores):
            # Preferential attachment logic:
            # New stores are more likely to attach to "busy" RDCs
            # (simulating population centers)
            # OR we just assign round robin / random for simpler
            # geo-clustering simulation.

            # Using the helper from distributions.py
            selected_indices = preferential_attachment(rdc_degrees, m=1, rng=self.rng)
            rdc_idx = selected_indices[0]
            rdc = active_rdcs[rdc_idx]

            rdc_degrees[rdc_idx] += 1

            dist = self.rng.exponential(200) + 50  # Most stores close, tail are far
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
        """Generate supplier nodes."""
        nodes = []
        companies = self.pool.sample_companies(n)
        cities = self.pool.sample_cities(n)

        for i in range(n):
            node_id = f"SUP-{i + 1:03d}"
            # Constrain SUP-001 (SPOF) to finite but sufficient capacity
            # Others get infinite capacity (default)
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

    def _generate_stores(self, n: int) -> list[Node]:
        """Generate retail store nodes."""
        nodes = []
        # Sample with replacement allowed for cities (many stores in one city)
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
