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
