import numpy as np

from prism_sim.network.core import Shipment
from prism_sim.network.recipe_matrix import RecipeMatrixBuilder
from prism_sim.simulation.world import World


class StateManager:
    """
    Manages the vectorized state of the simulation using numpy arrays.
    Maps object IDs to integer indices for O(1) access.
    """

    def __init__(self, world: World) -> None:
        self.world = world

        # 1. Create Index Maps
        self.node_id_to_idx: dict[str, int] = {}
        self.node_idx_to_id: dict[int, str] = {}

        self.product_id_to_idx: dict[str, int] = {}
        self.product_idx_to_id: dict[int, str] = {}

        self._index_entities()

        # 2. Build Recipe Matrix (Vectorized BOM)
        # Ensure products are passed in the EXACT same order as _index_entities (sorted)
        sorted_pids = sorted(self.world.products.keys())
        sorted_products = [self.world.products[pid] for pid in sorted_pids]
        recipes = list(self.world.recipes.values())

        builder = RecipeMatrixBuilder(sorted_products, recipes)
        self.recipe_matrix = builder.build_matrix()

        # 3. Allocate State Tensors
        self.n_nodes = len(self.world.nodes)
        self.n_products = len(self.world.products)

        # Shape: [Nodes, Products]
        # Represents: Inventory level of Product P at Node N
        # Perceived Inventory: What the WMS/ERP thinks we have
        self.perceived_inventory = np.zeros(
            (self.n_nodes, self.n_products), dtype=np.float32
        )

        # Actual Inventory: What is physically in the bin (Ground Truth)
        # In a perfect world, Perceived == Actual.
        self.actual_inventory = np.zeros(
            (self.n_nodes, self.n_products), dtype=np.float32
        )

        # Shape: [Nodes, Products]
        # Represents: Backlog quantity
        self.backlog = np.zeros((self.n_nodes, self.n_products), dtype=np.float32)

        # Shape: [Nodes]
        # Represents: Cash balance at each node (if applicable, or just 1 global)
        # For now, let's track cash per node (e.g. budget)
        self.cash = np.zeros((self.n_nodes,), dtype=np.float32)

        # Discrete State (Levels 10-11)
        self.active_shipments: list[Shipment] = []

        # PERF: Cached in-transit tensor - updated incrementally instead of
        # recomputing from all shipments each time. Shape [n_nodes, n_products].
        self._in_transit_tensor = np.zeros(
            (self.n_nodes, self.n_products), dtype=np.float64
        )

    @property
    def inventory(self) -> np.ndarray:
        """Alias for perceived_inventory (System View)."""
        return self.perceived_inventory

    @inventory.setter
    def inventory(self, value: np.ndarray) -> None:
        """
        Safety setter to prevent accidental overwrites.
        Ideally use update_inventory_batch.
        """
        self.perceived_inventory = value
        # Note: We do NOT automatically sync actual here to allow divergence.
        # But for initialization they should be synced manually or via specific init.

    def _index_entities(self) -> None:
        # Sort keys for deterministic indexing
        sorted_nodes = sorted(self.world.nodes.keys())
        for i, node_id in enumerate(sorted_nodes):
            self.node_id_to_idx[node_id] = i
            self.node_idx_to_id[i] = node_id

        sorted_products = sorted(self.world.products.keys())
        for i, prod_id in enumerate(sorted_products):
            self.product_id_to_idx[prod_id] = i
            self.product_idx_to_id[i] = prod_id

    def get_node_idx(self, node_id: str) -> int:
        return self.node_id_to_idx[node_id]

    def get_product_idx(self, product_id: str) -> int:
        return self.product_id_to_idx[product_id]

    def get_inventory(self, node_id: str, product_id: str) -> float:
        n_idx = self.get_node_idx(node_id)
        p_idx = self.get_product_idx(product_id)
        return float(self.perceived_inventory[n_idx, p_idx])

    def update_inventory(self, node_id: str, product_id: str, delta: float) -> None:
        """Updates BOTH actual and perceived inventory (default behavior)."""
        n_idx = self.get_node_idx(node_id)
        p_idx = self.get_product_idx(product_id)
        self.perceived_inventory[n_idx, p_idx] += delta
        self.actual_inventory[n_idx, p_idx] += delta
        # Floor to zero - prevent floating point noise from creating tiny negatives
        if self.perceived_inventory[n_idx, p_idx] < 0:
            self.perceived_inventory[n_idx, p_idx] = 0.0
        if self.actual_inventory[n_idx, p_idx] < 0:
            self.actual_inventory[n_idx, p_idx] = 0.0

    def update_inventory_batch(self, delta_tensor: np.ndarray) -> None:
        """
        Updates inventory for all nodes and products using a tensor of deltas.
        Updates BOTH actual and perceived.
        delta_tensor shape must match (n_nodes, n_products).
        """
        if delta_tensor.shape != self.perceived_inventory.shape:
            raise ValueError(
                f"Shape mismatch: {delta_tensor.shape} "
                f"!= {self.perceived_inventory.shape}"
            )
        self.perceived_inventory += delta_tensor
        self.actual_inventory += delta_tensor
        # Floor to zero - prevent floating point noise from creating tiny negatives
        np.maximum(self.perceived_inventory, 0, out=self.perceived_inventory)
        np.maximum(self.actual_inventory, 0, out=self.actual_inventory)

    def add_shipment(self, shipment: Shipment) -> None:
        """
        PERF: Add a shipment and update the in-transit tensor incrementally.
        Use this instead of directly appending to active_shipments.
        """
        self.active_shipments.append(shipment)

        # Update in-transit tensor
        target_idx = self.node_id_to_idx.get(shipment.target_id)
        if target_idx is not None:
            for line in shipment.lines:
                p_idx = self.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    self._in_transit_tensor[target_idx, p_idx] += line.quantity

    def remove_shipment(self, shipment: Shipment) -> None:
        """
        PERF: Remove a shipment and update the in-transit tensor incrementally.
        Use this when shipments arrive instead of filtering active_shipments.
        """
        # Update in-transit tensor (subtract quantities)
        target_idx = self.node_id_to_idx.get(shipment.target_id)
        if target_idx is not None:
            for line in shipment.lines:
                p_idx = self.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    self._in_transit_tensor[target_idx, p_idx] -= line.quantity

        # Remove from active list (O(n) but unavoidable for list)
        if shipment in self.active_shipments:
            self.active_shipments.remove(shipment)

    def add_shipments_batch(self, shipments: list[Shipment]) -> None:
        """
        PERF: Add multiple shipments and update in-transit tensor in batch.
        More efficient than calling add_shipment() repeatedly.
        """
        if not shipments:
            return

        # Build delta tensor for all shipments
        delta = np.zeros((self.n_nodes, self.n_products), dtype=np.float64)

        for shipment in shipments:
            self.active_shipments.append(shipment)
            target_idx = self.node_id_to_idx.get(shipment.target_id)
            if target_idx is not None:
                for line in shipment.lines:
                    p_idx = self.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        delta[target_idx, p_idx] += line.quantity

        self._in_transit_tensor += delta

    def remove_arrived_shipments(self, arrived: list[Shipment]) -> None:
        """
        PERF: Remove multiple arrived shipments and update in-transit tensor.
        More efficient than calling remove_shipment() repeatedly.
        """
        if not arrived:
            return

        # Build delta tensor for all arrived shipments
        delta = np.zeros((self.n_nodes, self.n_products), dtype=np.float64)
        arrived_set = set(id(s) for s in arrived)

        for shipment in arrived:
            target_idx = self.node_id_to_idx.get(shipment.target_id)
            if target_idx is not None:
                for line in shipment.lines:
                    p_idx = self.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        delta[target_idx, p_idx] += line.quantity

        self._in_transit_tensor -= delta

        # Filter active_shipments list (O(n) but only done once per batch)
        self.active_shipments = [
            s for s in self.active_shipments if id(s) not in arrived_set
        ]

    def get_in_transit_by_target(self) -> np.ndarray:
        """
        PERF: Return cached in-transit tensor (O(1) instead of O(N*M)).

        Returns the quantity currently in transit TO each node, aggregated
        across all active shipments. Used for Inventory Position calculation
        in (s,S) replenishment decisions.

        Inventory Position = On-Hand + In-Transit

        This is a fundamental fix for (s,S) theory which requires decisions
        to be based on IP, not just on-hand. Using only on-hand causes
        double-ordering oscillation when shipments are in transit.

        Returns:
            np.ndarray: Shape [n_nodes, n_products] - in-transit qty per target
        """
        return self._in_transit_tensor.copy()
