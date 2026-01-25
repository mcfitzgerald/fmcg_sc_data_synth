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

        # v0.38.0: Unmet demand tracking for improved replenishment signal
        # Tracks demand that couldn't be fulfilled due to inventory shortage.
        # This prevents the "demand signal collapse" where unfilled orders
        # don't register as demand, causing under-forecasting.
        # Shape: [n_nodes, n_products] - accumulated unmet demand per node
        self._unmet_demand = np.zeros(
            (self.n_nodes, self.n_products), dtype=np.float64
        )

        # v0.39.2: Inventory age tracking for SLOB calculation (SLOB fix)
        # Tracks weighted average age of inventory per (node, product).
        # Industry SLOB definition uses inventory AGE (how long it's been sitting),
        # not Days of Supply (how long it COULD last).
        # Shape: [n_nodes, n_products] - weighted average age in days
        self.inventory_age = np.zeros(
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

        v0.39.5: FIFO age approximation - reduces age proportionally on consumption.
        When inventory is consumed, age is reduced proportionally to quantity sold,
        simulating FIFO where oldest units are sold first and their "age mass"
        is removed from the system.
        """
        if delta_tensor.shape != self.perceived_inventory.shape:
            raise ValueError(
                f"Shape mismatch: {delta_tensor.shape} "
                f"!= {self.perceived_inventory.shape}"
            )

        # v0.39.5 FIX: FIFO age approximation (SLOB bug fix)
        #
        # PROBLEM: Age incremented daily regardless of consumption:
        # - Slow-moving items that never fully deplete accumulate age indefinitely
        # - A C-item selling 1/day from 100 units would hit age > 120 days
        #   even though it's turning normally
        #
        # FIX: Reduce age proportionally when inventory is consumed.
        # This simulates FIFO where selling removes the "oldest" units,
        # taking their age with them.
        #
        # Example: 100 units at age 50, sell 50 units
        # - FIFO says oldest 50 sold, younger 50 remain
        # - Age approximation: 50 * (50/100) = 25 days
        #
        # This prevents age from growing unbounded on turning inventory.
        consumption_mask = delta_tensor < 0

        # Calculate age reduction for consumed inventory
        # Only apply to positions with positive inventory and consumption
        with np.errstate(divide='ignore', invalid='ignore'):
            # fraction_remaining = (current - consumed) / current
            # new_age = old_age * fraction_remaining
            old_qty = self.actual_inventory.copy()
            consumed_qty = np.abs(delta_tensor)

            # Compute fraction remaining (clamped to [0, 1])
            fraction_remaining = np.where(
                old_qty > 0,
                np.clip((old_qty - consumed_qty) / old_qty, 0.0, 1.0),
                0.0
            )

            # Reduce age proportionally where consumption occurred
            self.inventory_age = np.where(
                consumption_mask,
                self.inventory_age * fraction_remaining,
                self.inventory_age
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

    # =========================================================================
    # v0.38.0: Unmet Demand Tracking
    # =========================================================================
    # Tracks unfilled demand to improve replenishment signal accuracy.
    # When allocation can't fulfill an order fully, the unfilled qty is recorded.
    # Replenishment then includes this unmet demand in its signal, preventing
    # under-forecasting that occurs when shortages hide true demand.

    def record_unmet_demand(
        self, node_id: str, product_id: str, unfilled_qty: float
    ) -> None:
        """
        Record unfilled demand at a node for a product.

        Called by AllocationAgent when orders can't be fully filled.
        This captures the "hidden demand" that would otherwise be lost.

        Args:
            node_id: Source node where allocation occurred
            product_id: Product that couldn't be fully allocated
            unfilled_qty: Quantity that couldn't be fulfilled
        """
        n_idx = self.node_id_to_idx.get(node_id)
        p_idx = self.product_id_to_idx.get(product_id)

        if n_idx is not None and p_idx is not None and unfilled_qty > 0:
            self._unmet_demand[n_idx, p_idx] += unfilled_qty

    def record_unmet_demand_batch(self, unmet_matrix: np.ndarray) -> None:
        """
        Record unfilled demand for all nodes/products at once.

        More efficient than calling record_unmet_demand() repeatedly.

        Args:
            unmet_matrix: Shape [n_nodes, n_products] - unfilled quantities
        """
        if unmet_matrix.shape == self._unmet_demand.shape:
            self._unmet_demand += unmet_matrix

    def get_unmet_demand(self) -> np.ndarray:
        """
        Get accumulated unmet demand for all nodes/products.

        Returns:
            np.ndarray: Shape [n_nodes, n_products] - accumulated unmet demand
        """
        return self._unmet_demand.copy()

    def get_unmet_demand_by_node(self, node_id: str) -> np.ndarray:
        """
        Get unmet demand for a specific node (all products).

        Args:
            node_id: Node to query

        Returns:
            np.ndarray: Shape [n_products] - unmet demand per product
        """
        n_idx = self.node_id_to_idx.get(node_id)
        if n_idx is not None:
            return self._unmet_demand[n_idx, :].copy()
        return np.zeros(self.n_products, dtype=np.float64)

    def clear_unmet_demand(self, node_id: str | None = None) -> None:
        """
        Clear accumulated unmet demand.

        Call this after replenishment orders are placed to avoid double-counting.
        If node_id is specified, only clears for that node.

        Args:
            node_id: Optional - clear only for this node, else clear all
        """
        if node_id is not None:
            n_idx = self.node_id_to_idx.get(node_id)
            if n_idx is not None:
                self._unmet_demand[n_idx, :] = 0.0
        else:
            self._unmet_demand[:, :] = 0.0

    def decay_unmet_demand(self, decay_factor: float = 0.5) -> None:
        """
        Decay accumulated unmet demand over time.

        This prevents old unmet demand from causing over-ordering indefinitely.
        A decay factor of 0.5 means unmet demand halves each day.

        Args:
            decay_factor: Multiplier applied to unmet demand (0-1)
        """
        self._unmet_demand *= decay_factor

    # =========================================================================
    # v0.39.2: Inventory Age Tracking (SLOB Fix)
    # =========================================================================
    # Industry-standard SLOB calculation uses inventory AGE (how long inventory
    # has been sitting), not Days of Supply (how long it COULD last). A fresh
    # batch with 90 days supply is NOT obsolete - but inventory sitting for
    # 90 days IS obsolete.

    def age_inventory(self, days: int = 1) -> None:
        """
        Age all positive inventory by specified days.

        Called at the start of each simulation day to track how long
        inventory has been sitting. Only ages positive inventory.

        Args:
            days: Number of days to age inventory (default 1)
        """
        has_inventory = self.actual_inventory > 0
        self.inventory_age[has_inventory] += days

    def receive_inventory(
        self, node_idx: int, product_idx: int, qty: float
    ) -> None:
        """
        Receive fresh inventory with weighted average age blending.

        When fresh inventory (age 0) arrives, it blends with existing
        inventory to produce a new weighted average age:
            new_age = (old_qty × old_age + new_qty × 0) / total_qty

        This preserves age information while correctly modeling FIFO
        consumption at the aggregate level.

        Args:
            node_idx: Index of receiving node
            product_idx: Index of product
            qty: Quantity received (fresh, age 0)
        """
        current_qty = max(0.0, float(self.actual_inventory[node_idx, product_idx]))
        current_age = float(self.inventory_age[node_idx, product_idx])

        new_total = current_qty + qty
        if new_total > 0:
            # Weighted average: (old_qty × old_age + new_qty × 0) / total
            self.inventory_age[node_idx, product_idx] = (
                current_qty * current_age
            ) / new_total
        else:
            self.inventory_age[node_idx, product_idx] = 0.0

        # Update actual inventory
        self.actual_inventory[node_idx, product_idx] = new_total
        self.perceived_inventory[node_idx, product_idx] = new_total

    def receive_inventory_batch(self, delta_tensor: np.ndarray) -> None:
        """
        Receive inventory with weighted average age blending (batch version).

        Vectorized version of receive_inventory for processing multiple
        arrivals at once. More efficient than individual calls.

        Args:
            delta_tensor: Shape [n_nodes, n_products] - qty received per cell
        """
        # Current state
        current_qty = np.maximum(0.0, self.actual_inventory)
        current_age = self.inventory_age

        # New total after receipt
        new_total = current_qty + delta_tensor

        # Weighted average age calculation (vectorized)
        # new_age = (old_qty × old_age + new_qty × 0) / new_total
        # Simplifies to: new_age = (old_qty × old_age) / new_total
        with np.errstate(divide='ignore', invalid='ignore'):
            new_age = np.where(
                new_total > 0,
                (current_qty * current_age) / new_total,
                0.0
            )
        self.inventory_age = new_age

        # Update inventory tensors
        self.actual_inventory += delta_tensor
        self.perceived_inventory += delta_tensor
        np.maximum(self.actual_inventory, 0, out=self.actual_inventory)
        np.maximum(self.perceived_inventory, 0, out=self.perceived_inventory)

    def get_weighted_age_by_product(self) -> np.ndarray:
        """
        Get inventory-weighted average age per product across all nodes.

        Used for SLOB calculation: computes the weighted average age
        per SKU, where weights are the inventory quantities at each node.

        Returns:
            np.ndarray: Shape [n_products] - weighted average age per SKU
        """
        # Numerator: sum of (inventory × age) across nodes
        weighted_age_sum = np.sum(
            self.inventory_age * np.maximum(0, self.actual_inventory), axis=0
        )
        # Denominator: total inventory per product
        total_inv = np.sum(np.maximum(0, self.actual_inventory), axis=0)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_age = np.where(total_inv > 0, weighted_age_sum / total_inv, 0.0)

        return weighted_age
