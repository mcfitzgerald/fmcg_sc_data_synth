
import numpy as np

from prism_sim.network.core import Shipment
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

        # 2. Allocate State Tensors
        self.n_nodes = len(self.world.nodes)
        self.n_products = len(self.world.products)

        # Shape: [Nodes, Products]
        # Represents: Inventory level of Product P at Node N
        self.inventory = np.zeros((self.n_nodes, self.n_products), dtype=np.float32)

        # Shape: [Nodes, Products]
        # Represents: Backlog quantity
        self.backlog = np.zeros((self.n_nodes, self.n_products), dtype=np.float32)

        # Shape: [Nodes]
        # Represents: Cash balance at each node (if applicable, or just 1 global)
        # For now, let's track cash per node (e.g. budget)
        self.cash = np.zeros((self.n_nodes,), dtype=np.float32)

        # Discrete State (Levels 10-11)
        self.active_shipments: list[Shipment] = []

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
        return float(self.inventory[n_idx, p_idx])

    def update_inventory(self, node_id: str, product_id: str, delta: float) -> None:
        n_idx = self.get_node_idx(node_id)
        p_idx = self.get_product_idx(product_id)
        self.inventory[n_idx, p_idx] += delta

    def update_inventory_batch(self, delta_tensor: np.ndarray) -> None:
        """
        Updates inventory for all nodes and products using a tensor of deltas.
        delta_tensor shape must match (n_nodes, n_products).
        """
        if delta_tensor.shape != self.inventory.shape:
            raise ValueError(
                f"Shape mismatch: {delta_tensor.shape} != {self.inventory.shape}"
            )
        self.inventory += delta_tensor
