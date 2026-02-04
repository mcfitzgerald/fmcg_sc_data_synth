import gzip
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from prism_sim.agents.allocation import AllocationAgent
from prism_sim.agents.replenishment import MinMaxReplenisher
from prism_sim.config.loader import load_manifest, load_simulation_config
from prism_sim.network.core import (
    Batch,
    Link,
    NodeType,
    Order,
    OrderLine,
    ProductionOrder,
    ProductionOrderStatus,
    Return,
    Shipment,
    ShipmentStatus,
)
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.demand import POSEngine
from prism_sim.simulation.logistics import LogisticsEngine
from prism_sim.simulation.monitor import (
    PhysicsAuditor,
    RealismMonitor,
    ResilienceTracker,
)
from prism_sim.simulation.mrp import MRPEngine
from prism_sim.simulation.quirks import QuirkManager
from prism_sim.simulation.risk_events import RiskEventManager
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.transform import TransformEngine
from prism_sim.simulation.writer import SimulationWriter

logger = logging.getLogger(__name__)

MemoryCallback = Callable[[str], None] | None


class Orchestrator:
    """The main time-stepper loop for the Prism Digital Twin."""

    def __init__(
        self,
        enable_logging: bool = False,
        output_dir: str = "data/output",
        streaming: bool | None = None,
        output_format: str | None = None,
        inventory_sample_rate: int | None = None,
        warm_start_path: str | None = None,
        skip_warm_start_hash_check: bool = False,
        auto_checkpoint: bool = True,
        memory_callback: MemoryCallback = None,
    ) -> None:
        # Store memory callback for periodic snapshots
        self._memory_callback = memory_callback
        # 1. Initialize World
        manifest = load_manifest()
        self.config = load_simulation_config()
        self._manifest = manifest  # Store for config hash computation

        # Merge static world definitions into config for Engines
        # that need them (e.g. POSEngine)
        self.config["promotions"] = manifest.get("promotions", [])
        self.config["packaging_types"] = manifest.get("packaging_types", [])

        self.builder = WorldBuilder(manifest)
        self.world = self.builder.build()

        # Load warm-start configuration
        sim_params = self.config.get("simulation_parameters", {})
        cal_config = sim_params.get("calibration", {})
        ws_config = cal_config.get("warm_start", {})
        default_burn_in = ws_config.get("default_burn_in_days", 0)

        # Compute config hash for checkpoint naming
        from prism_sim.simulation.snapshot import compute_config_hash

        self._config_hash = compute_config_hash(self.config, manifest)
        self._checkpoint_dir = Path("data/checkpoints")
        self._checkpoint_path = (
            self._checkpoint_dir / f"steady_state_{self._config_hash}.json.gz"
        )

        # Load warm-start snapshot: explicit path, auto-checkpoint, or none
        self._warm_start_snapshot: dict[str, Any] | None = None
        self._warm_start_path = warm_start_path
        self._needs_burn_in = False

        if warm_start_path:
            # Explicit warm-start path provided
            self._warm_start_snapshot = self._load_warm_start_snapshot(
                warm_start_path, skip_warm_start_hash_check
            )
        elif auto_checkpoint:
            # Auto-checkpoint: check if valid checkpoint exists
            if self._checkpoint_path.exists():
                try:
                    self._warm_start_snapshot = self._load_warm_start_snapshot(
                        str(self._checkpoint_path), skip_hash_check=False
                    )
                    print(f"Loaded checkpoint: {self._checkpoint_path.name}")
                except ValueError as e:
                    # Hash mismatch - checkpoint is stale
                    print(f"Checkpoint stale (config changed), will regenerate: {e}")
                    self._needs_burn_in = True
            else:
                # No checkpoint - need to run burn-in
                self._needs_burn_in = True

        # Set simulation start day: continues from burn-in for warm-start
        # Also set metrics start day to exclude burn-in from Triangle Report
        if self._warm_start_snapshot:
            burn_in_days = self._warm_start_snapshot["metadata"]["burn_in_days"]
            self._start_day = burn_in_days + 1
            self._metrics_start_day = self._start_day  # No burn-in in warm-start run
        elif self._needs_burn_in:
            # Cold-start WITH auto-checkpoint: will run burn-in, so exclude those days
            self._start_day = 1
            self._metrics_start_day = default_burn_in + 1 if default_burn_in > 0 else 1
        else:
            # Cold-start WITHOUT auto-checkpoint (--no-checkpoint):
            # start metrics immediately
            self._start_day = 1
            self._metrics_start_day = 1

        # Store burn-in config for later use
        self._default_burn_in_days = default_burn_in

        # ... (Initializing State and Engines) ...
        # 2. Initialize State
        self.state = StateManager(self.world)

        # 3. Initialize Engines & Agents
        # Initialize POS Engine first to get demand estimates for priming
        self.pos_engine = POSEngine(self.world, self.state, self.config)

        # Get equilibrium demand estimate for warm start
        # This prevents Day 1-2 bullwhip cascade from cold start
        warm_start_demand = self.pos_engine.get_average_demand_estimate()
        base_demand_matrix = self.pos_engine.get_base_demand_matrix()

        # Initialize Manufacturing Engines (Milestone 5)
        # MOVED UP: Initialize MRP Engine EARLY to get ABC classification
        # for inventory priming
        self.mrp_engine = MRPEngine(
            self.world, self.state, self.config, self.pos_engine, base_demand_matrix
        )

        # v0.48.0: Initialize DRP planner for forward-netting B/C production
        from prism_sim.simulation.drp import DRPPlanner

        self.drp_planner = DRPPlanner(
            self.world, self.state, self.config, self.pos_engine
        )
        self.mrp_engine.drp_planner = self.drp_planner

        # Initialize inventory: either from warm-start or cold-start priming
        if self._warm_start_snapshot:
            self._restore_inventory_from_snapshot()
        else:
            self._initialize_inventory()

        self.replenisher = MinMaxReplenisher(
            self.world,
            self.state,
            self.config,
            self.pos_engine,
            warm_start_demand=warm_start_demand,
            base_demand_matrix=base_demand_matrix,
        )

        # 7. Manufacturing State (initialized early for synthetic priming)
        self.active_production_orders: list[ProductionOrder] = []
        self.completed_batches: list[Batch] = []

        # v0.49.0: Synthetic steady-state priming (after replenisher init)
        # Eliminates 80+ days of burn-in by injecting realistic initial conditions
        if not self._warm_start_snapshot:
            self._prime_synthetic_steady_state()

        self.allocator = AllocationAgent(self.state, self.config)

        # v0.19.3: Set product velocity for ABC prioritization (Phase 1)
        # Sum base demand across all nodes to get total network velocity per SKU
        # This allows the Allocator to prioritize A-items (high velocity) when scarce
        product_velocity = np.sum(base_demand_matrix, axis=0)
        self.allocator.set_product_velocity(product_velocity)

        self.logistics = LogisticsEngine(self.world, self.state, self.config)

        self.transform_engine = TransformEngine(self.world, self.state, self.config)

        # v0.19.2: Pass base demand to transform engine for production prioritization
        self.transform_engine.set_base_demand(base_demand_matrix)

        # 5. Initialize Validation & Quirks (Milestone 6)
        sim_params = self.config.get("simulation_parameters", {})
        self.monitor = RealismMonitor(sim_params)
        self.auditor = PhysicsAuditor(self.state, self.world, sim_params)
        self.resilience = ResilienceTracker(self.state, self.world)
        self.quirks = QuirkManager(config=self.config)
        self.risks = RiskEventManager(sim_params)

        # Validation config (previously hardcoded thresholds)
        validation_config = sim_params.get("validation", {})
        self.slob_days_threshold = validation_config.get("slob_days_threshold", 60.0)
        self.mape_base = validation_config.get("mape_base", 0.30)
        self.mape_quirks_penalty = validation_config.get("mape_quirks_penalty", 0.15)

        # v0.26.0: ABC-differentiated SLOB thresholds and demand-based calculation
        slob_abc = validation_config.get("slob_abc_thresholds", {})
        self.slob_threshold_a = slob_abc.get("A", 30.0)
        self.slob_threshold_b = slob_abc.get("B", 60.0)
        self.slob_threshold_c = slob_abc.get("C", 120.0)
        self.slob_min_demand_velocity = validation_config.get(
            "slob_min_demand_velocity", 1.0
        )
        # Scope 3 emissions and perfect order penalty (v0.28.0 - moved from hardcode)
        self.scope_3_kg_co2_per_case = validation_config.get(
            "scope_3_kg_co2_per_case", 0.25
        )
        self.perfect_order_disruption_penalty = validation_config.get(
            "perfect_order_disruption_penalty", 0.5
        )

        # Cash-to-Cash config (v0.39.0 - moved from hardcode)
        c2c_config = sim_params.get("cash_to_cash", {})
        self.c2c_dso_days = c2c_config.get("dso_days", 30.0)
        self.c2c_dpo_days = c2c_config.get("dpo_days", 45.0)

        # Perfect Order config (v0.39.0 - real calculation, not just risk delays)
        po_config = sim_params.get("perfect_order", {})
        self.po_damage_rate = po_config.get("damage_rate", 0.02)
        self.po_documentation_error_rate = po_config.get(
            "documentation_error_rate", 0.005
        )
        self.po_on_time_tolerance_days = po_config.get("on_time_tolerance_days", 1)

        # Store base demand matrix for SLOB calculation (expected, not volatile)
        self._base_demand_matrix = base_demand_matrix

        # 6. Initialize Data Writer (Milestone 7)
        # Load writer config from simulation_config.json, allow CLI overrides
        writer_config = sim_params.get("writer", {})
        stream_mode = (
            streaming if streaming is not None
            else writer_config.get("streaming", False)
        )
        out_fmt = (
            output_format if output_format is not None
            else writer_config.get("output_format", "csv")
        )
        inv_sample = (
            inventory_sample_rate if inventory_sample_rate is not None
            else writer_config.get("inventory_sample_rate", 1)
        )
        self.writer = SimulationWriter(
            enable_logging=enable_logging,
            output_dir=output_dir,
            streaming=stream_mode,
            output_format=out_fmt,
            parquet_batch_size=writer_config.get("parquet_batch_size", 10000),
            inventory_sample_rate=inv_sample,
        )

        # 8. Finished Goods Mask for Metrics (excludes ingredients from inventory turns)
        self._fg_product_mask = self._build_finished_goods_mask()

        # v0.47.0: Diagnostic counter for push suppression (Fix 3 + Fix 5)
        self._push_suppression_count = 0

        # v0.45.0: Calculate deployment shares for production routing
        # Includes both RDCs and plant-direct DCs
        self.deployment_shares = self._calculate_deployment_shares()

        # v0.52.0: Identify plant-direct DCs and pre-compute their demand
        # for inventory-gated push in _create_plant_shipments
        self._plant_direct_dc_ids: set[str] = set()
        self._plant_direct_dc_demand: dict[str, np.ndarray] = {}
        self._precompute_plant_direct_dc_demand()

        # 9. Finalize warm-start if active
        # (restore production orders and history buffers)
        if self._warm_start_snapshot:
            self._finalize_warm_start_restoration()

    def _calculate_deployment_shares(self) -> dict[str, float]:
        """
        Calculate the share of global POS demand served by each deployment target.
        Deployment targets = RDCs + plant-direct DCs (DCs sourced from plants).
        Used to route production proportional to demand (physics-based flow).
        """
        # Build upstream and downstream maps
        upstream_map: dict[str, str] = {}
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            upstream_map[link.target_id] = link.source_id
            downstream_map.setdefault(link.source_id, []).append(link.target_id)

        # Deployment targets = RDCs + DCs sourced directly from plants
        deployment_targets: list[str] = []
        for n_id, n in self.world.nodes.items():
            if n.type == NodeType.DC and n_id.startswith("RDC-"):
                deployment_targets.append(n_id)
            elif (
                n.type == NodeType.DC
                and upstream_map.get(n_id, "").startswith("PLANT-")
            ):
                deployment_targets.append(n_id)

        target_demand: dict[str, float] = {}
        total_network_demand = 0.0

        base_demand_matrix = self.pos_engine.get_base_demand_matrix()

        for target_id in deployment_targets:
            # Recursive demand aggregation (target -> DCs -> Stores)
            target_total = 0.0
            visited: set[str] = set()
            stack = [target_id]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                children = downstream_map.get(current, [])
                for child_id in children:
                    child_node = self.world.nodes.get(child_id)
                    if child_node:
                        if child_node.type == NodeType.STORE:
                            # Aggregate POS demand
                            idx = self.state.node_id_to_idx.get(child_id)
                            if idx is not None:
                                target_total += float(
                                    np.sum(base_demand_matrix[idx, :])
                                )
                        else:
                            # Keep traversing logistics layer
                            stack.append(child_id)

            target_demand[target_id] = target_total
            total_network_demand += target_total

        # Convert to shares
        shares: dict[str, float] = {}
        if total_network_demand > 0:
            for tid, demand in target_demand.items():
                shares[tid] = demand / total_network_demand
        else:
            # Fallback to even split if no demand
            even = 1.0 / len(deployment_targets) if deployment_targets else 0.0
            shares = {tid: even for tid in deployment_targets}

        return shares

    def _precompute_plant_direct_dc_demand(self) -> None:
        """Pre-compute per-product daily demand for plant-direct DCs.

        v0.52.0: Used by _create_plant_shipments to gate push allocation
        based on DC inventory position (DOS cap). Sums POS base demand
        from all downstream stores for each plant-direct DC.
        """
        # Build upstream map to identify plant-direct DCs
        upstream_map: dict[str, str] = {}
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            upstream_map[link.target_id] = link.source_id
            downstream_map.setdefault(link.source_id, []).append(
                link.target_id
            )

        for n_id, n in self.world.nodes.items():
            if (
                n.type == NodeType.DC
                and not n_id.startswith("RDC-")
                and upstream_map.get(n_id, "").startswith("PLANT-")
            ):
                self._plant_direct_dc_ids.add(n_id)

        if not self._plant_direct_dc_ids:
            return

        base_demand = self.pos_engine.get_base_demand_matrix()

        for dc_id in self._plant_direct_dc_ids:
            dc_demand = np.zeros(self.state.n_products)
            # Sum POS demand from all downstream stores
            for child_id in downstream_map.get(dc_id, []):
                child_node = self.world.nodes.get(child_id)
                if child_node and child_node.type == NodeType.STORE:
                    store_idx = self.state.node_id_to_idx.get(child_id)
                    if store_idx is not None:
                        dc_demand += base_demand[store_idx, :]
            self._plant_direct_dc_demand[dc_id] = dc_demand

    # =========================================================================
    # Warm-Start Snapshot Methods (v0.33.0)
    # =========================================================================

    def _compute_config_hash(self) -> str:
        """Compute hash of current config for snapshot validation."""
        from prism_sim.simulation.snapshot import compute_config_hash

        return compute_config_hash(self.config, self._manifest)

    def _load_warm_start_snapshot(
        self, path: str, skip_hash_check: bool = False
    ) -> dict[str, Any]:
        """
        Load and validate a warm-start snapshot.

        Args:
            path: Path to snapshot file (JSON or gzipped JSON)
            skip_hash_check: If True, skip config hash validation

        Returns:
            Parsed snapshot dictionary

        Raises:
            ValueError: If snapshot is invalid or config hash mismatch
        """
        print(f"Loading warm-start snapshot from {path}...")

        # Load snapshot (handle gzip)
        snapshot: dict[str, Any]
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                snapshot = json.load(f)
        else:
            with open(path, encoding="utf-8") as f:
                snapshot = json.load(f)

        # Validate structure
        if "metadata" not in snapshot or "state" not in snapshot:
            raise ValueError("Invalid snapshot: missing metadata or state sections")

        metadata = snapshot["metadata"]
        print(f"  Snapshot from: {metadata.get('generated_at', 'unknown')}")
        print(f"  Burn-in days: {metadata.get('burn_in_days', 'unknown')}")

        # Validate config hash (unless skipped)
        if not skip_hash_check:
            snapshot_hash = metadata.get("config_hash", "")
            current_hash = self._compute_config_hash()
            if snapshot_hash != current_hash:
                raise ValueError(
                    f"Config hash mismatch! Snapshot: {snapshot_hash}, "
                    f"Current: {current_hash}. "
                    "Use --skip-hash-check to override (not recommended)."
                )
            print(f"  Config hash verified: {current_hash}")
        else:
            print("  Config hash check SKIPPED (--skip-hash-check)")

        return snapshot

    def _restore_inventory_from_snapshot(self) -> None:
        """
        Restore inventory tensors from warm-start snapshot.

        This replaces _initialize_inventory() when warm-start is active.
        """
        if not self._warm_start_snapshot:
            return

        print("Restoring inventory from warm-start snapshot...")
        state_data = self._warm_start_snapshot["state"]

        # Restore actual inventory
        actual_inv = state_data.get("actual_inventory", {})
        for node_id, products in actual_inv.items():
            node_idx = self.state.node_id_to_idx.get(node_id)
            if node_idx is None:
                continue
            for prod_id, qty in products.items():
                prod_idx = self.state.product_id_to_idx.get(prod_id)
                if prod_idx is not None:
                    self.state.actual_inventory[node_idx, prod_idx] = qty

        # Restore perceived inventory
        perceived_inv = state_data.get("perceived_inventory", {})
        for node_id, products in perceived_inv.items():
            node_idx = self.state.node_id_to_idx.get(node_id)
            if node_idx is None:
                continue
            for prod_id, qty in products.items():
                prod_idx = self.state.product_id_to_idx.get(prod_id)
                if prod_idx is not None:
                    self.state.perceived_inventory[node_idx, prod_idx] = qty

        # Restore active shipments
        shipments_data = state_data.get("active_shipments", [])
        for s_data in shipments_data:
            lines = [
                OrderLine(line["product_id"], line["quantity"])
                for line in s_data.get("lines", [])
            ]
            shipment = Shipment(
                id=s_data["id"],
                source_id=s_data["source_id"],
                target_id=s_data["target_id"],
                creation_day=s_data["creation_day"],
                arrival_day=s_data["arrival_day"],
                lines=lines,
                status=ShipmentStatus(s_data.get("status", "in_transit")),
                original_order_day=s_data.get("original_order_day"),
                total_weight_kg=s_data.get("total_weight_kg", 0.0),
                total_volume_m3=s_data.get("total_volume_m3", 0.0),
            )
            # Use batch add to update in-transit tensor correctly
            self.state.add_shipment(shipment)

        print(
            f"  Restored {len(actual_inv)} nodes with inventory, "
            f"{len(shipments_data)} active shipments"
        )

    def _finalize_warm_start_restoration(self) -> None:
        """
        Finalize warm-start by restoring production orders and history buffers.

        Called after all engines are initialized.
        """
        if not self._warm_start_snapshot:
            return

        print("Finalizing warm-start restoration...")
        state_data = self._warm_start_snapshot["state"]
        derived_data = self._warm_start_snapshot.get("derived", {})

        # Restore active production orders
        po_data = state_data.get("active_production_orders", [])
        for po in po_data:
            production_order = ProductionOrder(
                id=po["id"],
                plant_id=po["plant_id"],
                product_id=po["product_id"],
                quantity_cases=po["quantity_cases"],
                creation_day=po["creation_day"],
                due_day=po.get("due_day", po["creation_day"] + 7),
                status=ProductionOrderStatus(po.get("status", "planned")),
                produced_quantity=po.get("produced_quantity", 0.0),
            )
            self.active_production_orders.append(production_order)

        print(f"  Restored {len(self.active_production_orders)} production orders")

        # Initialize history buffers from derived parameters
        self._initialize_history_buffers_from_derived(derived_data)

    def _initialize_history_buffers_from_derived(
        self, derived_data: dict[str, Any]
    ) -> None:
        """
        Pre-populate history buffers with derived steady-state values.

        This enables safety stock calculations to work correctly from Day 1
        without waiting for history to accumulate.
        """
        # Get warm-start config for noise parameters
        sim_params = self.config.get("simulation_parameters", {})
        cal_config = sim_params.get("calibration", {})
        ws_config = cal_config.get("warm_start", {})
        hist_config = ws_config.get("history_buffer_init", {})

        demand_noise_cv = hist_config.get("demand_noise_cv", 0.1)
        lt_noise_cv = hist_config.get("lead_time_noise_cv", 0.1)
        use_derived_abc = hist_config.get("use_derived_abc", True)

        # 1. Initialize replenisher demand history buffer
        avg_demand = derived_data.get("avg_demand_by_product", {})
        if avg_demand and hasattr(self.replenisher, "demand_history_buffer"):
            print("  Initializing demand history buffer...")
            # Fill buffer with average demand + small noise
            for day_idx in range(self.replenisher.variance_lookback):
                for prod_id, avg_val in avg_demand.items():
                    prod_idx = self.state.product_id_to_idx.get(prod_id)
                    if prod_idx is not None:
                        # Add small noise to create realistic variance
                        noise = np.random.normal(1.0, demand_noise_cv)
                        value = avg_val * max(0.1, noise)
                        # Distribute across nodes proportionally (simplified)
                        for node_idx in range(self.state.n_nodes):
                            self.replenisher.demand_history_buffer[
                                day_idx, node_idx, prod_idx
                            ] = value / self.state.n_nodes

        # 2. Initialize replenisher lead time history (sparse storage)
        avg_lt = derived_data.get("avg_lead_time_by_link", {})
        if avg_lt:
            print("  Initializing lead time history (sparse)...")
            from collections import deque

            for target_id, sources in avg_lt.items():
                target_idx = self.state.node_id_to_idx.get(target_id)
                if target_idx is None:
                    continue
                for source_id, avg_val in sources.items():
                    source_idx = self.state.node_id_to_idx.get(source_id)
                    if source_idx is None:
                        continue
                    link_key = (target_idx, source_idx)
                    # Initialize deque with noisy samples
                    history: deque[float] = deque(
                        maxlen=self.replenisher.lt_history_len
                    )
                    for _ in range(self.replenisher.lt_history_len):
                        noise = np.random.normal(1.0, lt_noise_cv)
                        history.append(avg_val * max(0.5, noise))
                    self.replenisher._lt_history[link_key] = history
                    # Update sparse cache directly
                    self.replenisher._lt_mu_cache_sparse[link_key] = avg_val
                    lt_sigma = derived_data.get("lead_time_sigma_by_link", {})
                    if target_id in lt_sigma and source_id in lt_sigma[target_id]:
                        self.replenisher._lt_sigma_cache_sparse[link_key] = (
                            lt_sigma[target_id][source_id]
                        )
                    else:
                        self.replenisher._lt_sigma_cache_sparse[link_key] = 0.0

        # 3. Initialize ABC classifications and z-scores
        if use_derived_abc:
            abc_classes = derived_data.get("abc_classifications", {})
            z_scores = derived_data.get("z_scores_by_product", {})

            if abc_classes:
                print("  Initializing ABC classifications...")
                for prod_id, abc_class in abc_classes.items():
                    prod_idx = self.state.product_id_to_idx.get(prod_id)
                    if prod_idx is not None:
                        self.mrp_engine.abc_class[prod_idx] = abc_class

            if z_scores:
                for prod_id, z_val in z_scores.items():
                    prod_idx = self.state.product_id_to_idx.get(prod_id)
                    if prod_idx is not None:
                        self.replenisher.z_scores_vec[prod_idx] = z_val

        # 4. Initialize product volume history (for ABC updates)
        vol_history = derived_data.get("product_volume_history", {})
        if vol_history:
            for prod_id, vol in vol_history.items():
                prod_idx = self.state.product_id_to_idx.get(prod_id)
                if prod_idx is not None:
                    self.replenisher.product_volume_history[prod_idx] = vol

        print("  History buffer initialization complete")

    # =========================================================================
    # End Warm-Start Methods
    # =========================================================================

    # =========================================================================
    # Synthetic Steady-State Initialization (v0.49.0)
    # =========================================================================

    def _prime_synthetic_steady_state(self) -> None:
        """
        Inject synthetic steady-state conditions at day 0.

        Eliminates 80+ days of burn-in by pre-filling pipeline shipments,
        production WIP, history buffers, and inventory ages. Config-gated
        via calibration.warm_start.synthetic_steady_state flag.

        Must be called AFTER replenisher init (needs replenisher attributes)
        and AFTER _initialize_inventory() (needs on-hand inventory).
        """
        sim_params = self.config.get("simulation_parameters", {})
        cal_config = sim_params.get("calibration", {})
        ws_config = cal_config.get("warm_start", {})

        if not ws_config.get("synthetic_steady_state", False):
            return

        print("Priming synthetic steady-state...")

        # Build network topology maps (reused by multiple sub-methods)
        self._upstream_map: dict[str, str] = {}
        self._downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            self._upstream_map[link.target_id] = link.source_id
            self._downstream_map.setdefault(link.source_id, []).append(
                link.target_id
            )

        self._prime_history_buffers()
        self._prime_pipeline()
        self._prime_production_wip()
        self._prime_inventory_age()

        # Clean up temporary maps
        del self._upstream_map
        del self._downstream_map

        print("Synthetic steady-state priming complete.")

    def _get_day1_seasonal_factor(self) -> float:
        """Return the seasonal multiplier for day 1."""
        sim_params = self.config.get("simulation_parameters", {})
        demand_config = sim_params.get("demand", {})
        season_config = demand_config.get("seasonality", {})
        amplitude = season_config.get("amplitude", 0.12)
        phase_shift = season_config.get("phase_shift_days", 150)
        cycle_days = season_config.get("cycle_days", 365)
        return float(
            1.0 + amplitude * np.sin(2 * np.pi * (1 - phase_shift) / cycle_days)
        )

    def _aggregate_downstream_demand(
        self, node_id: str, base_demand: np.ndarray
    ) -> np.ndarray:
        """
        Recursively aggregate store-level base demand downstream of a node.

        Returns shape [n_products] with total daily demand from all
        downstream stores.
        """
        result = np.zeros(self.state.n_products)
        visited: set[str] = set()
        stack = [node_id]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for child_id in self._downstream_map.get(current, []):
                child_node = self.world.nodes.get(child_id)
                if child_node:
                    if child_node.type == NodeType.STORE:
                        idx = self.state.node_id_to_idx.get(child_id)
                        if idx is not None:
                            result += base_demand[idx, :]
                    else:
                        stack.append(child_id)
        return result

    def _estimate_link_flow(
        self, link: Link, base_demand: np.ndarray
    ) -> np.ndarray:
        """
        Estimate daily product flow on a network link.

        Routes demand appropriately per echelon:
        - DC→Store: flow = store's base demand
        - RDC→DC: flow = aggregate demand of DC's downstream stores
        - Plant→RDC/DC: flow = downstream demand * deployment share

        Returns shape [n_products].
        """
        tgt = self.world.nodes.get(link.target_id)
        if not tgt:
            return np.zeros(self.state.n_products)

        if tgt.type == NodeType.STORE:
            # DC→Store: flow is the store's own demand
            tgt_idx = self.state.node_id_to_idx.get(link.target_id)
            if tgt_idx is not None:
                return base_demand[tgt_idx, :].copy()
            return np.zeros(self.state.n_products)

        if tgt.type == NodeType.DC:
            # Plant→DC or RDC→DC: flow is aggregate downstream demand
            return self._aggregate_downstream_demand(link.target_id, base_demand)

        return np.zeros(self.state.n_products)

    def _prime_pipeline(self) -> None:
        """
        Create synthetic in-transit shipments on every link.

        For each link, creates one shipment per day of lead time, sized
        to match expected daily flow. This ensures the replenisher sees
        realistic Inventory Position (on-hand + in-transit) from day 1.
        """
        base_demand = self.pos_engine.get_base_demand_matrix()
        seasonal = self._get_day1_seasonal_factor()
        counter = 0

        for link in self.world.links.values():
            src = self.world.nodes.get(link.source_id)
            tgt = self.world.nodes.get(link.target_id)
            if not src or not tgt:
                continue

            tgt_idx = self.state.node_id_to_idx.get(link.target_id)
            if tgt_idx is None:
                continue

            # Calculate expected daily flow on this link
            daily_flow = self._estimate_link_flow(link, base_demand)
            if np.sum(daily_flow) < 1.0:
                continue

            lead_time = max(1, int(np.ceil(link.lead_time_days)))
            for day_offset in range(1, lead_time + 1):
                lines: list[OrderLine] = []
                for p_idx in range(self.state.n_products):
                    qty = float(daily_flow[p_idx] * seasonal)
                    if qty >= 1.0:
                        lines.append(
                            OrderLine(
                                self.state.product_idx_to_id[p_idx], qty
                            )
                        )
                if lines:
                    counter += 1
                    shipment = Shipment(
                        id=f"PRIME-{counter:06d}",
                        source_id=link.source_id,
                        target_id=link.target_id,
                        creation_day=0,
                        arrival_day=day_offset,
                        lines=lines,
                        status=ShipmentStatus.IN_TRANSIT,
                    )
                    self.state.add_shipment(shipment)

        print(f"  Pipeline primed: {counter} synthetic shipments")

    def _prime_production_wip(self) -> None:
        """
        Create synthetic production orders already mid-completion at plants.

        Seeds both in-progress POs (completing day 1-2) and finished goods
        buffer at plants (2 days of expected shipments). Eliminates the
        3-day production gap that occurs in cold start.
        """
        expected_demand = self.mrp_engine.expected_daily_demand
        lead_time = self.mrp_engine.production_lead_time
        po_count = 0

        for product_id in self.mrp_engine._finished_product_ids:
            p_idx = self.state.product_id_to_idx.get(product_id)
            if p_idx is None:
                continue
            daily_rate = float(expected_demand[p_idx])
            if daily_rate < 1.0:
                continue

            plant_id = self.mrp_engine._select_plant(product_id)
            batch_qty = daily_rate * lead_time  # lead_time days' worth

            po = ProductionOrder(
                id=f"PRIME-PO-{product_id}",
                plant_id=plant_id,
                product_id=product_id,
                quantity_cases=batch_qty,
                creation_day=-lead_time,
                due_day=1,
                status=ProductionOrderStatus.IN_PROGRESS,
                planned_start_day=-lead_time + 1,
                actual_start_day=-lead_time + 1,
                produced_quantity=batch_qty * 0.67,
            )
            self.active_production_orders.append(po)
            po_count += 1

        # Seed finished goods at plants (2 days of expected shipments)
        fg_count = 0
        for plant_id in self.mrp_engine._plant_ids:
            plant_idx = self.state.node_id_to_idx.get(plant_id)
            if plant_idx is None:
                continue
            fg_buffer = expected_demand * 2.0
            self.state.actual_inventory[plant_idx, :] += fg_buffer
            self.state.perceived_inventory[plant_idx, :] += fg_buffer
            fg_count += 1

        print(
            f"  WIP primed: {po_count} production orders, "
            f"{fg_count} plants with FG buffer"
        )

    def _prime_history_buffers(self) -> None:
        """
        Pre-fill replenisher demand and lead-time history buffers.

        Populates:
        - demand_history_buffer (28 days of synthetic noisy demand)
        - smoothed_demand (clean baseline)
        - _lt_history (lead time samples for every link)
        - _lt_mu_cache_sparse / _lt_sigma_cache_sparse
        - history_idx (marked full)

        Must be called AFTER replenisher init.
        """
        from collections import deque

        base_demand = self.pos_engine.get_base_demand_matrix()
        seasonal = self._get_day1_seasonal_factor()
        rng = np.random.default_rng(42)

        # 1. Demand history buffer: variance_lookback days of noisy demand
        for day_slot in range(self.replenisher.variance_lookback):
            noise = rng.normal(
                1.0, 0.1, size=base_demand.shape
            ).clip(0.5, 2.0)
            self.replenisher.demand_history_buffer[day_slot] = (
                base_demand * seasonal * noise
            )
        # Mark buffer as full so variance calculations activate
        self.replenisher.history_idx = self.replenisher.variance_lookback

        # 2. Smoothed demand: clean baseline (no noise)
        self.replenisher.smoothed_demand = (
            base_demand * seasonal
        ).astype(np.float32)

        # 3. Lead time history: fill for every link
        lt_len = self.replenisher.lt_history_len
        lt_count = 0
        for link in self.world.links.values():
            t_idx = self.state.node_id_to_idx.get(link.target_id)
            s_idx = self.state.node_id_to_idx.get(link.source_id)
            if t_idx is None or s_idx is None:
                continue
            key = (t_idx, s_idx)
            lt_noise = rng.normal(1.0, 0.08, size=lt_len).clip(0.7, 1.4)
            history: deque[float] = deque(
                (float(link.lead_time_days * n) for n in lt_noise),
                maxlen=lt_len,
            )
            self.replenisher._lt_history[key] = history
            self.replenisher._lt_mu_cache_sparse[key] = link.lead_time_days
            self.replenisher._lt_sigma_cache_sparse[key] = (
                link.lead_time_days * 0.08
            )
            lt_count += 1

        print(
            f"  History buffers primed: "
            f"{self.replenisher.variance_lookback} demand days, "
            f"{lt_count} link lead-time histories"
        )

    def _prime_inventory_age(self) -> None:
        """
        Set realistic FIFO ages on initial inventory based on ABC class.

        Steady-state average age approximates half the target days supply.
        Prevents SLOB discontinuity at day 60 when thresholds first apply.
        """
        abc = self.mrp_engine.abc_class
        # Steady-state avg age ≈ half the target days supply
        age_map = {0: 3.0, 1: 7.0, 2: 15.0}  # A, B, C

        for p_idx in range(self.state.n_products):
            target_age = age_map.get(int(abc[p_idx]), 7.0)
            mask = self.state.actual_inventory[:, p_idx] > 0
            self.state.inventory_age[mask, p_idx] = target_age

        print("  Inventory age seeded by ABC class")

    # =========================================================================
    # End Synthetic Steady-State Methods
    # =========================================================================

    def _initialize_inventory(self) -> None:
        """
        Seed initial inventory across the network (Priming).

        v0.19.12: RDCs only initialize with inventory if they have downstream
        demand. Demand-proportional priming is used for all nodes.
        """
        # Get manufacturing config for plant initial inventory
        sim_params = self.config.get("simulation_parameters", {})
        mfg_config = sim_params.get("manufacturing", {})
        mfg_config.get("initial_plant_inventory", {})

        # Get priming config
        inv_config = sim_params.get("inventory", {})
        init_config = inv_config.get("initialization", {})

        # CONFIG-DRIVEN priming (v0.26.0 fix - was previously hardcoded)
        store_days_supply = init_config.get("store_days_supply", 4.5)
        rdc_days_supply = init_config.get("rdc_days_supply", 7.5)
        customer_dc_days = init_config.get("customer_dc_days_supply", 7.5)

        # Get per-SKU demand matrix for demand-proportional priming
        base_demand_matrix = self.pos_engine.get_base_demand_matrix()

        # v0.26.0: ABC-based priming DERIVED FROM CONFIG, not hardcoded
        # Uses config values with ABC velocity factors to differentiate priming
        # A-items need slightly more buffer (higher service level target)
        # C-items need less (lower velocity, can tolerate longer replenishment)
        abc_velocity_cfg = init_config.get(
            "abc_velocity_factors", {"A": 1.2, "B": 1.0, "C": 0.8}
        )
        # Convert string keys to int keys (0=A, 1=B, 2=C)
        abc_velocity_factors = {
            0: abc_velocity_cfg.get("A", 1.2),
            1: abc_velocity_cfg.get("B", 1.0),
            2: abc_velocity_cfg.get("C", 0.8),
        }

        abc_target_dos = {
            abc_class: store_days_supply * factor
            for abc_class, factor in abc_velocity_factors.items()
        }

        # Same scaling for RDCs
        rdc_abc_target_dos = {
            abc_class: rdc_days_supply * factor
            for abc_class, factor in abc_velocity_factors.items()
        }

        # v0.28.0: Calculate seasonal factor for Day 1 (cold start adjustment)
        # Problem: Day 1 may be in a seasonal trough/peak, but priming uses expected
        # demand (no seasonality). This causes systematic over/under-stocking.
        # Solution: Adjust priming by the Day 1 seasonal factor.
        demand_config = sim_params.get("demand", {})
        season_config = demand_config.get("seasonality", {})
        amplitude = season_config.get("amplitude", 0.12)
        phase_shift = season_config.get("phase_shift_days", 150)
        cycle_days = season_config.get("cycle_days", 365)

        # Day 1 seasonal factor: matches demand.py seasonality calculation
        day_1_seasonal = 1.0 + amplitude * np.sin(
            2 * np.pi * (1 - phase_shift) / cycle_days
        )
        print(
            f"  Priming with seasonal adjustment:"
            f" Day 1 factor = {day_1_seasonal:.4f}"
        )

        # Seed finished goods at RDCs and Stores
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.STORE:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    node_demand = base_demand_matrix[node_idx, :]

                    # Apply ABC-based priming
                    # Vectorized: days supply vector based on ABC class
                    store_days_vec = np.array([
                        abc_target_dos.get(
                            self.mrp_engine.abc_class[p_idx], store_days_supply
                        )
                        for p_idx in range(self.state.n_products)
                    ])

                    # v0.28.0: Apply seasonal adjustment to match Day 1 actual demand
                    sku_levels = node_demand * store_days_vec * day_1_seasonal
                    self.state.perceived_inventory[node_idx, :] = sku_levels
                    self.state.actual_inventory[node_idx, :] = sku_levels

            elif node.type == NodeType.DC:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    if node_id.startswith("RDC-"):
                        # Aggregate downstream demand per SKU
                        rdc_downstream_demand = np.zeros(self.state.n_products)

                        # Use recursive discovery for RDC downstream demand
                        downstream_map: dict[str, list[str]] = {}
                        for link in self.world.links.values():
                            downstream_map.setdefault(
                                link.source_id, []
                            ).append(link.target_id)

                        visited = set()
                        stack = [node_id]
                        while stack:
                            current = stack.pop()
                            if current in visited:
                                continue
                            visited.add(current)
                            for child_id in downstream_map.get(current, []):
                                child_node = self.world.nodes.get(child_id)
                                if child_node:
                                    if child_node.type == NodeType.STORE:
                                        t_idx = self.state.node_id_to_idx.get(child_id)
                                        if t_idx is not None:
                                            rdc_downstream_demand += (
                                                base_demand_matrix[t_idx, :]
                                            )
                                    else:
                                        stack.append(child_id)

                        # Skip priming if no demand (Ghost RDC fix)
                        if rdc_downstream_demand.sum() == 0:
                            continue

                        # v0.26.0: Apply ABC-based priming to RDCs (config-driven)
                        rdc_days_vec = np.array([
                            rdc_abc_target_dos.get(
                                self.mrp_engine.abc_class[p_idx], rdc_days_supply
                            )
                            for p_idx in range(self.state.n_products)
                        ])
                        # v0.28.0: Apply seasonal adjustment
                        rdc_sku_levels = (
                            rdc_downstream_demand
                            * rdc_days_vec
                            * day_1_seasonal
                        )
                        self.state.perceived_inventory[node_idx, :] = rdc_sku_levels
                        self.state.actual_inventory[node_idx, :] = rdc_sku_levels
                    else:
                        # Customer DCs: aggregate downstream store demand per SKU
                        downstream_demand = np.zeros(self.state.n_products)
                        for link in self.world.links.values():
                            if link.source_id == node_id:
                                target_node = self.world.nodes.get(link.target_id)
                                if (
                                    target_node
                                    and target_node.type == NodeType.STORE
                                ):
                                    t_idx = self.state.node_id_to_idx.get(
                                        link.target_id
                                    )
                                    if t_idx is not None:
                                        downstream_demand += (
                                            base_demand_matrix[t_idx, :]
                                        )

                        # Skip priming if no demand
                        if downstream_demand.sum() == 0:
                            continue

                        # v0.25.0: Apply ABC-based priming to Customer DCs
                        # to reduce replenishment cascade at startup
                        dc_days_vec = np.array([
                            rdc_abc_target_dos.get(
                                self.mrp_engine.abc_class[p_idx], customer_dc_days
                            )
                            for p_idx in range(self.state.n_products)
                        ])
                        # v0.28.0: Apply seasonal adjustment
                        dc_levels = (
                            downstream_demand
                            * dc_days_vec
                            * day_1_seasonal
                        )
                        self.state.perceived_inventory[node_idx, :] = dc_levels
                        self.state.actual_inventory[node_idx, :] = dc_levels

            # Seed raw materials at Plants
            # v0.20.0: Sized for safety supply to ensure production stability.
            # v0.36.0: Moved to config to allow scaling.
            # v0.37.0: Demand-driven init (Velocity * Days)
            # to fix flat buffer issues
            elif node.type == NodeType.PLANT:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    # Calculate global demand per product
                    global_product_demand = np.sum(base_demand_matrix, axis=0)

                    # Derived ingredient demand: Product Demand @ Recipe Matrix
                    # How much of each ingredient needed globally per day
                    ingredient_demand = (
                        global_product_demand @ self.state.recipe_matrix
                    )

                    # Target days supply for ingredients (default 14 from policy)
                    target_days = 14.0
                    ing_policy = mfg_config.get(
                        "inventory_policies", {}
                    ).get("INGREDIENT")
                    if ing_policy:
                        target_days = ing_policy.get("target_days_supply", 14.0)

                    # Calculate target levels
                    target_levels = ingredient_demand * target_days

                    # Safety floor from config (plant_ingredient_buffer)
                    min_floor = init_config.get("plant_ingredient_buffer", 200000.0)

                    # Apply to state
                    for product in self.world.products.values():
                        if product.category == ProductCategory.INGREDIENT:
                            p_idx = self.state.product_id_to_idx.get(product.id)
                            if p_idx is not None:
                                # Target + Floor for lead time variance
                                qty = max(target_levels[p_idx], min_floor)
                                self.state.update_inventory(node_id, product.id, qty)

    def _build_finished_goods_mask(self) -> np.ndarray:
        """
        Build a boolean mask for finished goods products (excludes ingredients).

        Used to calculate inventory turns only on sellable products, not raw materials.
        Returns shape [n_products] where True = finished good, False = ingredient.
        """
        mask = np.zeros(self.state.n_products, dtype=bool)
        for p_id, product in self.world.products.items():
            if product.category != ProductCategory.INGREDIENT:
                p_idx = self.state.product_id_to_idx.get(p_id)
                if p_idx is not None:
                    mask[p_idx] = True
        return mask

    def run(self, days: int = 30) -> None:
        """
        Run simulation for `days` of steady-state data.

        When auto_checkpoint is enabled and no checkpoint exists, automatically
        runs burn-in phase first, saves checkpoint, then runs data collection.
        """
        # v0.39.4: Warn about memory usage for long runs with buffered logging
        buffered_logging_day_threshold = 100
        total_days = days + (self._default_burn_in_days if self._needs_burn_in else 0)
        if (
            total_days > buffered_logging_day_threshold
            and self.writer.enable_logging
            and not self.writer.streaming
        ):
            print(
                f"WARNING: Running {total_days} days with"
                f" buffered logging may use 10-20GB RAM.\n"
                f"  Consider: --streaming (writes"
                f" incrementally) or --no-logging (fastest)"
            )

        if self._needs_burn_in:
            # Phase 1: Run burn-in (no metrics recording)
            burn_in_days = self._default_burn_in_days
            # Removed hardcoded fallback to 90 days. Trust config.

            print(f"Running {burn_in_days}-day burn-in (checkpoint not found)...")

            for day in range(1, burn_in_days + 1):
                self._step(day)
                if day % 30 == 0:
                    print(f"  Burn-in day {day}/{burn_in_days} complete")

            # Memory snapshot after burn-in
            if self._memory_callback:
                self._memory_callback("after_burn_in")

            # Save checkpoint for future runs
            self._save_checkpoint(burn_in_days)

            # Update start day for data collection
            self._start_day = burn_in_days + 1
            self._metrics_start_day = self._start_day

            # Mark burn-in complete
            self._needs_burn_in = False

        # Phase 2: Run data collection
        print(f"Starting Simulation for {days} days...")
        end_day = self._start_day + days

        for day in range(self._start_day, end_day):
            self._step(day)
            # Memory snapshot every 10 simulation days
            if self._memory_callback and (day - self._start_day + 1) % 10 == 0:
                self._memory_callback(f"day_{day}")

        print("Simulation Complete.")

    def _save_checkpoint(self, burn_in_days: int) -> None:
        """Save current state as checkpoint for future runs."""
        from prism_sim.simulation.snapshot import save_snapshot

        save_snapshot(
            sim=self,
            burn_in_days=burn_in_days,
            config_hash=self._config_hash,
            output_path=self._checkpoint_path,
            auto_generated=True,
        )

    def _step(self, day: int) -> None:
        # 0. Start mass balance tracking
        self.auditor.start_day(day)

        # 0b. v0.39.2: Age all inventory by 1 day (SLOB tracking)
        # Must happen at start of day before any inventory movements
        self.state.age_inventory(1)

        # 0a. Risk & Quirks: Start of Day
        shrinkage_events = self._apply_pre_step_quirks(day)
        if shrinkage_events:
            self.auditor.record_shrinkage(shrinkage_events)

        # 1. Generate Demand (POS)
        daily_demand = self.pos_engine.generate_demand(day)
        daily_demand = self._apply_demand_quirks(daily_demand, day)

        # v0.16.0: Record demand in replenisher for variance tracking
        self.replenisher.record_demand(daily_demand)

        # 2. Consume Inventory (Sales) - Constrained to available inventory
        # Cannot sell more than what's on hand (lost sales model)
        available = np.maximum(0, self.state.actual_inventory)
        actual_sales = np.minimum(daily_demand, available)
        self.state.update_inventory_batch(-actual_sales)
        self.auditor.record_sales(actual_sales)

        # v0.39.3: Track TRUE unmet demand from store stockouts
        #
        # v0.39.2 BUG: unmet_demand was only recorded from allocation failures,
        # not from store-level stockouts. When a customer arrives at an empty
        # shelf, that lost sale was invisible to MRP - causing persistent
        # under-production for low-priority items (C-items).
        #
        # INDUSTRY REALITY: Lost sales at shelf = true unmet demand.
        # This should flow upstream to drive production planning.
        #
        # FIX: Record stockout-based unmet demand in state for MRP calibration.
        unmet_from_stockout = daily_demand - actual_sales
        self.state.record_unmet_demand_batch(unmet_from_stockout)

        # v0.39.2: Feed actual consumption back to MRP (SLOB fix)
        # This allows MRP to calibrate production to actual consumption,
        # preventing over-production when service level < 100%.
        self.mrp_engine.record_consumption(actual_sales)

        # 3. Replenishment Decision (The "Pull" Signal)
        raw_orders = self.replenisher.generate_orders(day, daily_demand)

        # v0.15.9: Record inflow (orders received) for true demand signal
        # This captures what was REQUESTED before allocation constrains it
        # Used by customer DCs to prevent demand signal attenuation
        self.replenisher.record_inflow(raw_orders)

        # v0.45.0: Pass order demand to MRP (pre-allocation, true demand)
        # Include orders TO RDCs + orders sourced from plants (plant-direct DCs)
        production_signal_orders = [
            o for o in raw_orders
            if o.source_id.startswith("RDC-") or o.source_id.startswith("PLANT-")
        ]
        self.mrp_engine.record_order_demand(production_signal_orders)

        # Generate Purchase Orders for Ingredients at Plants (Milestone 5.1 extension)
        # Uses production-based signal (active orders) instead of POS demand
        # to ensure ingredient replenishment matches actual consumption
        ing_orders = self.mrp_engine.generate_purchase_orders(
            day, self.active_production_orders
        )
        raw_orders.extend(ing_orders)

        # Capture Unconstrained Demand (before Allocator modifies in-place)
        unconstrained_demand_qty = sum(
            line.quantity for order in raw_orders for line in order.lines
        )

        # 4. Allocation (Milestone 4.1)
        allocation_result = self.allocator.allocate_orders(raw_orders)
        allocated_orders = allocation_result.allocated_orders
        self.auditor.record_allocation_out(allocation_result.allocation_matrix)

        # v0.15.4: Record allocation outflow for customer DC demand signal
        # This prevents bullwhip cascade by using actual outflow as demand
        self.replenisher.record_outflow(allocation_result.allocation_matrix)

        # v0.21.0: Removed pending order tracking (memory explosion fix)
        # Real retail systems use Inventory Position for reorder decisions,
        # not per-SKU pending order tracking. See replenishment.py for details.

        # 5. Logistics (Milestone 4.2)
        new_shipments = self.logistics.create_shipments(allocated_orders, day)
        self._apply_logistics_quirks_and_risks(new_shipments)
        # PERF: Use batch method to update in-transit tensor incrementally
        self.state.add_shipments_batch(new_shipments)

        # 6. Transit & Arrival (Milestone 4.3)
        # PERF: Use batch removal to update in-transit tensor incrementally
        _active, arrived = self.logistics.update_shipments(
            self.state.active_shipments, day
        )
        self.state.remove_arrived_shipments(arrived)

        # 7. Process Arrivals (Receive Inventory)
        self._process_arrivals(arrived)
        self.auditor.record_receipts(arrived)

        # 7a. Generate Returns (Phase 3d)
        new_returns = self.logistics.generate_returns_from_arrivals(arrived, day)

        # 8. Manufacturing: MRP (Milestone 5.1)
        # Filter for RDC -> Store shipments (Pull signal for MRP)
        rdc_store_shipments = [
            s
            for s in new_shipments
            if self.world.nodes[s.source_id].type == NodeType.DC
            and self.world.nodes[s.target_id].type == NodeType.STORE
        ]

        # v0.19.1: Pass POS demand to MRP as signal floor
        # This prevents demand signal collapse when orders decline
        new_production_orders = self.mrp_engine.generate_production_orders(
            day, rdc_store_shipments, self.active_production_orders, daily_demand
        )
        self.active_production_orders.extend(new_production_orders)

        # 9. Manufacturing: Production (Milestone 5.2)
        (
            updated_orders,
            new_batches,
            plant_oee,
            plant_teep,
        ) = self.transform_engine.process_production_orders(
            self.active_production_orders, day
        )
        self.auditor.record_production(new_batches)
        self.active_production_orders = [
            o for o in updated_orders if o.status.value != "complete"
        ]
        self.completed_batches.extend(new_batches)

        # v0.20.0: Production order cleanup - remove stale orders
        # Orders that haven't been fulfilled within timeout days are likely blocked
        # (material shortage, capacity issues) and should be dropped to prevent
        # unbounded backlog accumulation. MRP will regenerate if demand persists.
        sim_params = self.config.get("simulation_parameters", {})
        mfg_config = sim_params.get("manufacturing", {})
        production_order_timeout = mfg_config.get("production_order_timeout_days", 14)
        self.active_production_orders = [
            o for o in self.active_production_orders
            if day - o.creation_day <= production_order_timeout
        ]

        # v0.20.0: Memory cleanup - only retain recent batches for traceability
        # Keep last N days of batches, discard older ones to prevent unbounded growth
        batch_retention_days = mfg_config.get("batch_retention_days", 30)
        self.completed_batches = [
            b for b in self.completed_batches
            if day - b.production_day <= batch_retention_days
        ]

        # 10. Ship finished goods from Plants to RDCs
        plant_shipments = self._create_plant_shipments(new_batches, day)
        self.auditor.record_plant_shipments_out(plant_shipments)
        self._apply_logistics_quirks_and_risks(plant_shipments)
        # PERF: Use batch method to update in-transit tensor incrementally
        self.state.add_shipments_batch(plant_shipments)

        # 10a. v0.19.2: Push excess RDC inventory to Customer DCs
        # This breaks the negative feedback spiral where RDCs accumulate
        # inventory while downstream nodes starve.
        push_shipments = self._push_excess_rdc_inventory(day)
        if push_shipments:
            self.auditor.record_plant_shipments_out(push_shipments)
            self._apply_logistics_quirks_and_risks(push_shipments)
            # PERF: Use batch method to update in-transit tensor incrementally
            self.state.add_shipments_batch(push_shipments)

        # 11. Validation & Resilience
        self._apply_post_step_validation(day, arrived)

        # 11a. Mass Balance Audit
        self.auditor.end_day()
        mass_violations = self.auditor.check_mass_balance()
        if mass_violations:
            print("  MASS BALANCE VIOLATIONS DETECTED:")
            for v in mass_violations[:5]:
                print(f"    {v}")

        # 12. Monitors & Data Logging
        total_demand = np.sum(daily_demand)
        daily_shipments = new_shipments + plant_shipments + push_shipments

        total_shipped_qty = sum(
            line.quantity for s in daily_shipments for line in s.lines
        )
        shrinkage_qty = sum(e.quantity_lost for e in shrinkage_events)

        # Only record metrics and log data after burn-in period
        if day >= self._metrics_start_day:
            self._record_daily_metrics(
                daily_demand,
                daily_shipments,
                arrived,
                plant_oee,
                plant_teep,
                day,
                ordered_qty=unconstrained_demand_qty,
                shipped_qty=total_shipped_qty,
                shrinkage_qty=shrinkage_qty,
                actual_sales=actual_sales,
            )
            self._log_daily_data(
                raw_orders,
                new_shipments,
                plant_shipments,
                new_batches,
                new_production_orders,
                new_returns,
                day,
            )

        # 13. Logging / Metrics (Simple Print)
        daily_summary = {
            "demand": total_demand,
            "ordered": unconstrained_demand_qty,
            "shipped": sum(
                line.quantity for shipment in daily_shipments for line in shipment.lines
            ),
            "arrived": sum(
                line.quantity for shipment in arrived for line in shipment.lines
            ),
            "produced": sum(b.quantity_cases for b in new_batches),
        }
        self._print_daily_status(day, daily_summary)

    def _apply_pre_step_quirks(self, day: int) -> list[Any]:
        """Trigger risk events and apply inventory quirks."""
        triggered_risks = self.risks.check_triggers(day)
        if triggered_risks:
            print(
                f"Day {day:03}: RISK EVENTS TRIGGERED: "
                f"{[e.event_code for e in triggered_risks]}"
            )
        shrinkage_events = self.quirks.apply_shrinkage(self.state, day)
        self.quirks.process_discoveries(self.state, day)
        return shrinkage_events

    def _apply_demand_quirks(self, demand: np.ndarray, day: int) -> np.ndarray:
        """Apply optimism bias to generated demand."""
        product_ids = [
            self.state.product_idx_to_id[i] for i in range(self.state.n_products)
        ]
        return self.quirks.apply_optimism_bias(demand, product_ids, day)

    def _apply_logistics_quirks_and_risks(self, shipments: list[Shipment]) -> None:
        """Apply delays and risk multipliers to shipments."""
        if not shipments:
            return

        delay_multiplier = self.risks.get_logistics_delay_multiplier()
        self.quirks.apply_port_congestion(shipments)

        if delay_multiplier > 1.0:
            for shipment in shipments:
                original_duration = shipment.arrival_day - shipment.creation_day
                new_duration = original_duration * delay_multiplier
                shipment.arrival_day = shipment.creation_day + int(new_duration)

    def _apply_post_step_validation(self, day: int, arrived: list[Shipment]) -> None:
        """Check for risk recovery and run physics audit."""
        recovered = self.risks.check_recovery(day)
        if recovered:
            print(f"Day {day:03}: RISK RECOVERY: {recovered}")

        violations = self.auditor.check_kinematic_consistency(arrived, day)
        if violations:
            print(f"Day {day:03}: PHYSICS VIOLATIONS: {violations}")

    def _record_daily_metrics(
        self,
        daily_demand: np.ndarray,
        daily_shipments: list[Shipment],
        arrived: list[Shipment],
        plant_oee: dict[str, float],
        plant_teep: dict[str, float],
        day: int,
        ordered_qty: float = 0.0,
        shipped_qty: float = 0.0,
        shrinkage_qty: float = 0.0,
        actual_sales: np.ndarray | None = None,
    ) -> None:
        """Record simulation metrics for monitoring."""
        # ABC class codes (0=A, 1=B, 2=C)
        abc_class_a = 0
        abc_class_b = 1
        abc_class_c = 2
        # Minimum FG inventory threshold to avoid divide-by-small-number
        min_fg_inventory_threshold = 100.0

        # Record Service Level (Fill Rate)
        fill_rate = 1.0
        if ordered_qty > 0:
            fill_rate = shipped_qty / ordered_qty
        self.monitor.record_service_level(fill_rate)

        # Record Store Service Level (On-Shelf Availability proxy)
        # v0.40.0 FIX: Use pre-sales actual_sales from step 2 instead of
        # recalculating from end-of-day inventory (which overstates fill rate
        # because arrivals and production have replenished stock since step 2).
        total_demand_qty = np.sum(daily_demand)
        if total_demand_qty > 0:
            if actual_sales is not None:
                store_fill_rate = np.sum(actual_sales) / total_demand_qty
            else:
                available = np.maximum(0, self.state.actual_inventory)
                actual_sales = np.minimum(daily_demand, available)
                store_fill_rate = np.sum(actual_sales) / total_demand_qty
            self.monitor.record_store_service_level(store_fill_rate)

            # v0.26.0: ABC-class service level breakdown for mix analysis
            # Helps diagnose whether service level drift is due to A-item stockouts
            fg_demand = daily_demand[:, self._fg_product_mask]
            fg_sales = actual_sales[:, self._fg_product_mask]
            fg_abc_class = self.mrp_engine.abc_class[self._fg_product_mask]

            # Calculate fill rate by ABC class
            a_mask = fg_abc_class == abc_class_a
            b_mask = fg_abc_class == abc_class_b
            c_mask = fg_abc_class == abc_class_c

            a_demand = np.sum(fg_demand[:, a_mask])
            b_demand = np.sum(fg_demand[:, b_mask])
            c_demand = np.sum(fg_demand[:, c_mask])

            a_fill = np.sum(fg_sales[:, a_mask]) / a_demand if a_demand > 0 else 1.0
            b_fill = np.sum(fg_sales[:, b_mask]) / b_demand if b_demand > 0 else 1.0
            c_fill = np.sum(fg_sales[:, c_mask]) / c_demand if c_demand > 0 else 1.0

            self.monitor.record_abc_service_levels(a_fill, b_fill, c_fill)

        # Calculate Inventory Turns (Cash) - ONLY finished goods, not ingredients
        # Inventory turns = Annual Sales / Average Inventory (finished goods only)
        # v0.39.4 FIX: Guard against near-zero inventory causing extreme turns
        fg_inventory = self.state.actual_inventory[:, self._fg_product_mask]
        total_fg_inv = np.sum(np.maximum(0, fg_inventory))
        if total_fg_inv > min_fg_inventory_threshold:
            daily_turn_rate = total_demand_qty / total_fg_inv
            # Cap at 50x (2.5x industry max)
            annual_turns = min(daily_turn_rate * 365, 50.0)
            self.monitor.record_inventory_turns(annual_turns)

            # Cash-to-Cash (Est: DIO + DSO - DPO)
            # DIO = 365 / Turns, DSO and DPO from config
            dio = 365.0 / annual_turns
            c2c = dio + self.c2c_dso_days - self.c2c_dpo_days
            self.monitor.record_cash_to_cash(c2c)

            # Shrinkage Rate (on FG inventory only - raw materials tracked separately)
            shrink_rate = shrinkage_qty / total_fg_inv
            self.monitor.record_shrinkage_rate(shrink_rate)

            # v0.39.2: SLOB % (Age-based calculation - industry standard)
            #
            # Industry SLOB definition uses inventory AGE (how long sitting),
            # not Days of Supply (how long it COULD last). A fresh batch with
            # 90 days supply is NOT obsolete - but inventory sitting for 90
            # days IS obsolete.
            #
            # Config thresholds now represent AGE in days, not DOS:
            # - A-items: flag if sitting > threshold days (fast-turning)
            # - B-items: flag if sitting > threshold days
            # - C-items: flag if sitting > threshold days
            #   (slow-turning, higher threshold)
            fg_inv_per_sku = np.sum(fg_inventory, axis=0)  # Sum across nodes

            # Get inventory-weighted average age per product
            all_sku_age = self.state.get_weighted_age_by_product()
            fg_sku_age = all_sku_age[self._fg_product_mask]

            # ABC-differentiated AGE thresholds (from config)
            # Get ABC class for finished goods only (0=A, 1=B, 2=C)
            fg_abc_class = self.mrp_engine.abc_class[self._fg_product_mask]

            # Build threshold array per SKU based on ABC class
            # Config values are now AGE thresholds (days sitting), not DOS
            age_thresholds = np.where(
                fg_abc_class == abc_class_a,
                self.slob_threshold_a,
                np.where(
                    fg_abc_class == abc_class_b,
                    self.slob_threshold_b,
                    self.slob_threshold_c,
                ),
            )

            # Flag SKUs with AGE > their ABC-specific threshold as SLOB
            slob_mask = fg_sku_age > age_thresholds
            slob_inventory = fg_inv_per_sku[slob_mask].sum()

            slob_pct = slob_inventory / total_fg_inv if total_fg_inv > 0 else 0.0
            self.monitor.record_slob(slob_pct)

            # v0.39.2: SLOB diagnostic logging with age info
            a_mask = fg_abc_class == abc_class_a
            b_mask = fg_abc_class == abc_class_b
            c_mask = fg_abc_class == abc_class_c
            a_avg_age = np.mean(fg_sku_age[a_mask]) if np.any(a_mask) else 0.0
            b_avg_age = np.mean(fg_sku_age[b_mask]) if np.any(b_mask) else 0.0
            c_avg_age = np.mean(fg_sku_age[c_mask]) if np.any(c_mask) else 0.0
            logger.info(
                "SLOB Debug: total_fg=%.0f, slob_inv=%.0f, slob_pct=%.4f, "
                "a_age=%.1fd/%dd, b_age=%.1fd/%dd, c_age=%.1fd/%dd",
                total_fg_inv, slob_inventory, slob_pct,
                a_avg_age, self.slob_threshold_a,
                b_avg_age, self.slob_threshold_b,
                c_avg_age, self.slob_threshold_c
            )

            # v0.47.0: Per-echelon DOS diagnostics + suppression counters
            # Tracks DC inventory health by channel and measures fix effectiveness
            dc_dos_by_channel: dict[str, list[float]] = {}
            for n_id, node in self.world.nodes.items():
                if node.type != NodeType.DC or n_id.startswith("RDC-"):
                    continue
                n_idx = self.state.node_id_to_idx.get(n_id)
                if n_idx is None:
                    continue
                fg_inv = self.state.actual_inventory[n_idx, self._fg_product_mask]
                fg_demand = self._base_demand_matrix[n_idx, self._fg_product_mask]
                fg_demand_safe = np.maximum(fg_demand, 0.1)
                mean_dos = float(np.mean(fg_inv / fg_demand_safe))
                ch_name = (
                    node.channel.name if node.channel and hasattr(node.channel, "name")
                    else "UNKNOWN"
                )
                dc_dos_by_channel.setdefault(ch_name, []).append(mean_dos)

            channel_summaries = {
                ch: f"{np.mean(vals):.1f}" for ch, vals in dc_dos_by_channel.items()
            }

            unmet_magnitude = float(np.sum(self.state.get_unmet_demand()))

            # Pull suppression counts from replenisher (Fix 1) and push method (Fix 3)
            dc_suppress = self.replenisher._dc_order_suppression_count
            push_suppress = self._push_suppression_count

            logger.info(
                "v47 Diag: dc_dos=%s, order_suppress=%d, push_suppress=%d, "
                "unmet_mag=%.0f",
                channel_summaries,
                dc_suppress,
                push_suppress,
                unmet_magnitude,
            )

            # Reset daily counters
            self.replenisher._dc_order_suppression_count = 0
            self._push_suppression_count = 0

        log_config = self.config.get("simulation_parameters", {}).get("logistics", {})
        constraints = log_config.get("constraints", {})
        max_weight = constraints.get("truck_max_weight_kg", 20000.0)
        store_delivery_mode = log_config.get("store_delivery_mode", "FTL")

        # v0.26.0: Differentiate FTL vs LTL for truck fill metrics
        # FTL fill rate is the meaningful metric (target 85%)
        # LTL (store deliveries) are intentionally small - last-mile deliveries
        #
        # v0.35.0: Further separate inbound vs outbound FTL
        # - Inbound (Supplier->Plant): raw material deliveries, low fill expected
        # - Outbound (RDC->DC, Plant->RDC): finished goods, target 85%+
        for s in daily_shipments:
            fill_rate = min(1.0, s.total_weight_kg / max_weight)
            self.monitor.record_truck_fill(fill_rate)

            source_node = self.world.nodes.get(s.source_id)
            target_node = self.world.nodes.get(s.target_id)

            # Determine shipment type
            is_store_delivery = (
                target_node is not None
                and target_node.type == NodeType.STORE
                and store_delivery_mode == "LTL"
            )

            is_inbound = (
                source_node is not None
                and source_node.type == NodeType.SUPPLIER
            )

            if is_store_delivery:
                self.monitor.record_ltl_shipment()
            elif is_inbound:
                # Supplier->Plant: raw material inbound shipments
                self.monitor.record_inbound_fill(fill_rate)
            else:
                # Outbound FTL: RDC->DC, Plant->RDC (finished goods)
                self.monitor.record_outbound_ftl_fill(fill_rate)
                self.monitor.record_ftl_fill(fill_rate)  # Legacy compat

        # Record OEE and TEEP
        if plant_oee:
            avg_oee = sum(plant_oee.values()) / len(plant_oee)
            self.monitor.record_oee(avg_oee)
        if plant_teep:
            avg_teep = sum(plant_teep.values()) / len(plant_teep)
            self.monitor.record_teep(avg_teep)

        # Perfect Order Rate (v0.39.0 - real calculation)
        # Perfect Order = On-time AND Undamaged AND Correct Documentation
        # Note: "Complete" check handled by allocation -
        # shipments are what was allocated
        perfect_order_rate = self._calculate_perfect_order_rate(arrived, day)
        self.monitor.record_perfect_order(perfect_order_rate)

        # Scope 3 Emissions (config-driven)
        self.monitor.record_scope_3(self.scope_3_kg_co2_per_case)

        # MAPE
        # Baseline error + Optimism Bias penalty if active
        mape = self.mape_base
        if self.quirks.is_enabled("optimism_bias"):
            mape += self.mape_quirks_penalty
        self.monitor.record_mape(mape)

    def _calculate_perfect_order_rate(
        self, arrived_shipments: list[Shipment], day: int
    ) -> float:
        """
        Calculate Perfect Order Rate for arrived shipments.

        Perfect Order = Orders that are:
        1. On-time (arrived within expected lead time + tolerance)
        2. Undamaged (no quality issues - stochastic based on damage rate)
        3. Correct documentation (stochastic baseline)

        Note: "Complete" (100% fill rate) is handled by the allocation agent -
        shipments represent what was actually allocated and shipped.

        Returns: Fraction of perfect orders (0.0 to 1.0)
        """
        if not arrived_shipments:
            return 1.0  # No deliveries = no failures

        rng = np.random.default_rng(day * 12345)  # Deterministic per day
        perfect_count = 0

        for shipment in arrived_shipments:
            # 1. On-time check
            # Expected arrival = original_order_day + route_lead_time
            route = (shipment.source_id, shipment.target_id)
            link = self.logistics.route_map.get(route)
            expected_lead_time = link.lead_time_days if link else 3.0

            if shipment.original_order_day is not None:
                expected_arrival = (
                    shipment.original_order_day
                    + int(expected_lead_time)
                    + self.po_on_time_tolerance_days
                )
                is_on_time = shipment.arrival_day <= expected_arrival
            else:
                # Fall back to creation day if original not tracked
                expected_arrival = (
                    shipment.creation_day
                    + int(expected_lead_time)
                    + self.po_on_time_tolerance_days
                )
                is_on_time = shipment.arrival_day <= expected_arrival

            # 2. Damage check (stochastic based on config rate)
            is_undamaged = rng.random() > self.po_damage_rate

            # 3. Documentation check (stochastic baseline)
            has_correct_docs = rng.random() > self.po_documentation_error_rate

            # Perfect if all conditions met
            if is_on_time and is_undamaged and has_correct_docs:
                perfect_count += 1

        return perfect_count / len(arrived_shipments)

    def _log_daily_data(
        self,
        raw_orders: list[Order],
        new_shipments: list[Shipment],
        plant_shipments: list[Shipment],
        new_batches: list[Batch],
        new_production_orders: list[ProductionOrder],
        new_returns: list[Return],
        day: int,
    ) -> None:
        """Log data to the simulation writer."""
        self.writer.log_orders(raw_orders, day)
        self.writer.log_production_orders(new_production_orders, day)
        self.writer.log_shipments(new_shipments + plant_shipments, day)
        self.writer.log_batches(new_batches, day)
        self.writer.log_batch_ingredients(new_batches, day)
        self.writer.log_returns(new_returns, day)

        # v0.38.0: Log 14-day deterministic forecast (S&OP Export)
        # This represents the "Consensus Forecast" for the planning horizon
        forecast_vec = self.pos_engine.get_deterministic_forecast(
            start_day=day + 1, duration=14, aggregated=True
        )
        self.writer.log_forecasts(forecast_vec, self.state, day)

        self.writer.log_inventory(self.state, self.world, day)

    def _print_daily_status(
        self,
        day: int,
        summary: dict[str, float],
    ) -> None:
        """Print high-level daily simulation status."""
        total_demand = summary["demand"]
        total_ordered = summary["ordered"]
        total_shipped = summary["shipped"]
        total_arrived = summary["arrived"]
        total_produced = summary["produced"]

        # Debug: Inventory Stats
        # Only consider stores (first 4500 nodes approx, or just average all positive)
        # We know actual_inventory can be negative.
        mean_inv = np.mean(self.state.actual_inventory)

        # Calculate theoretical reorder point avg
        # RP = Demand * 3.0
        # This is approximate since we don't have the exact Replenisher view here easily
        # but daily_demand is what we passed.
        # We don't have access to the full daily_demand array here anymore,
        # so we'll just use the system total mean as a proxy.
        mean_demand = total_demand / self.state.n_products
        est_rp = mean_demand * 3.0

        print(
            f"Day {day:03}: Dmd={total_demand:.0f}, "
            f"Ord={total_ordered:.0f}, "
            f"Ship={total_shipped:.0f}, "
            f"Arr={total_arrived:.0f}, "
            f"Prod={total_produced:.0f}, "
            f"InvMean={mean_inv:.1f}, "
            f"EstRP={est_rp:.2f}"
        )

    def save_results(self) -> None:
        """Export all collected data."""
        report = self.monitor.get_report()
        self.writer.save(report)

    def generate_triangle_report(self) -> str:
        """
        Generate 'The Triangle Report': Service vs. Cost vs. Cash.
        [Task 7.3]
        """
        report = self.monitor.get_report()
        scoring_config = (
            self.config.get("simulation_parameters", {}).get("scoring", {})
        )

        # Scoring Weights
        truck_scale = scoring_config.get("truck_fill_scale", 100.0)
        oee_scale = scoring_config.get("oee_scale", 100.0)

        # Calculate Service (LIFR approx from backlogs)
        # Note: In our current simple state, negative inventory is backlog.
        # So we can look at actual vs perceived or just positive vs negative.
        total_backlog = np.sum(np.maximum(0, -self.state.actual_inventory))
        total_inventory = np.sum(np.maximum(0, self.state.actual_inventory))

        oee = report.get("oee", {}).get("mean", 0)
        teep = report.get("teep", {}).get("mean", 0)
        report.get("truck_fill", {}).get("mean", 0)

        # Use Store Service Level (Consumer OSA) for the Triangle Report
        # as it represents the actual "Service" delivered to customers.
        service_index = (
            report.get("store_service_level", {}).get("mean", 0.0) * 100.0
        )
        inv_turns = report.get("inventory_turns", {}).get("mean", 0)

        perfect_order = (
            report.get("perfect_order_rate", {}).get("mean", 0) * 100.0
        )
        c2c = report.get("cash_to_cash_days", {}).get("mean", 0)
        scope3 = report.get("scope_3_emissions", {}).get("mean", 0)
        mape = report.get("mape", {}).get("mean", 0) * 100.0
        shrink = report.get("shrinkage_rate", {}).get("mean", 0) * 100.0
        slob = report.get("slob", {}).get("mean", 0) * 100.0

        # v0.26.0: FTL fill rate and ABC service levels
        # v0.35.0: Separate inbound vs outbound FTL (outbound is the meaningful metric)
        outbound_ftl = report.get("outbound_ftl_fill", {}).get("mean", 0) * truck_scale
        inbound_fill = report.get("inbound_fill", {}).get("mean", 0) * truck_scale
        ltl_count = report.get("ltl_shipments", {}).get("count", 0)
        abc_svc = report.get("service_level_by_abc", {})
        svc_a = abc_svc.get("A", 0) * 100.0
        svc_b = abc_svc.get("B", 0) * 100.0
        svc_c = abc_svc.get("C", 0) * 100.0

        summary = [
            "==================================================",
            "        THE SUPPLY CHAIN TRIANGLE REPORT          ",
            "==================================================",
            f"1. SERVICE (Store Fill Rate):   {service_index:.2f}%",
            f"   - A-Items:                   {svc_a:.1f}%",
            f"   - B-Items:                   {svc_b:.1f}%",
            f"   - C-Items:                   {svc_c:.1f}%",
            f"2. CASH (Inventory Turns):      {inv_turns:.2f}x",
            f"3. COST (Truck Fill Rate):      {outbound_ftl:.1f}%",
            f"   - Inbound Fill (raw mat):    {inbound_fill:.1f}%",
            f"   - LTL Shipments:             {ltl_count:,}",
            "--------------------------------------------------",
            f"Manufacturing OEE:              {oee * oee_scale:.1f}%",
            f"TEEP (Total Utilization):       {teep * oee_scale:.1f}%",
            f"Perfect Order Rate:             {perfect_order:.1f}%",
            f"Cash-to-Cash Cycle:             {c2c:.1f} days",
            f"Scope 3 Emissions:              {scope3:.2f} kg/case",
            f"MAPE (Forecast):                {mape:.1f}%",
            f"Shrinkage Rate:                 {shrink:.2f}%",
            f"SLOB Inventory:                 {slob:.1f}%",
            f"Total System Inventory:         {total_inventory:,.0f} cases",
            f"Total Backlog:                  {total_backlog:,.0f} cases",
            "==================================================",
        ]
        return "\n".join(summary)

    def _process_arrivals(self, arrived_shipments: list[Shipment]) -> None:
        """
        PERF: Batch inventory updates instead of per-line calls.
        Reduces 3.6M update_inventory() calls to a single batch operation.

        v0.39.2: Uses receive_inventory_batch for age-aware receipt (SLOB fix).
        Fresh arrivals blend with existing inventory using weighted average age.
        """
        if not arrived_shipments:
            return

        # Build delta tensor for all arrivals
        delta = np.zeros(
            (self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        for shipment in arrived_shipments:
            # Physics Overhaul: Record realized lead time
            if shipment.original_order_day is not None:
                lead_time = float(shipment.arrival_day - shipment.original_order_day)
                self.replenisher.record_lead_time(
                    shipment.target_id, shipment.source_id, lead_time
                )

            target_idx = self.state.node_id_to_idx.get(shipment.target_id)
            if target_idx is None:
                continue

            for line in shipment.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    delta[target_idx, p_idx] += line.quantity

        # v0.39.2: Use age-aware batch receive (SLOB fix)
        # Fresh inventory blends with existing to update weighted average age
        self.state.receive_inventory_batch(delta)

    def _create_plant_shipments(
        self, batches: list[Batch], current_day: int
    ) -> list[Shipment]:
        """Create shipments from Plants to deployment targets.

        Distributes produced goods to RDCs (unconditionally) and plant-direct
        DCs (inventory-gated). v0.52.0: Plant-direct DCs only receive push
        up to a DOS-based soft taper. Gated excess is redirected to RDCs
        (not left at plants) to preserve production throughput.
        """
        shipments: list[Shipment] = []

        # Get default lead time from config
        default_lead_time = (
            self.config.get("simulation_parameters", {})
            .get("logistics", {})
            .get("default_lead_time_days", 3.0)
        )

        if not self.deployment_shares:
            return shipments

        # v0.52.0: Soft taper for plant-direct DC push
        sim_params = self.config.get("simulation_parameters", {})
        replen_params = sim_params.get("agents", {}).get("replenishment", {})
        push_floor_dos = float(
            replen_params.get("plant_push_floor_dos", 20.0)
        )
        push_cap_dos = float(
            replen_params.get("plant_push_cap_dos", 60.0)
        )
        taper_range = max(push_cap_dos - push_floor_dos, 1.0)

        # Cache in-transit tensor
        in_transit = self.state.get_in_transit_by_target()

        # Pre-compute RDC targets for excess redistribution
        rdc_shares = {
            k: v
            for k, v in self.deployment_shares.items()
            if k.startswith("RDC-")
        }
        total_rdc_share = sum(rdc_shares.values())

        shipment_counter = 0

        for batch in batches:
            if batch.status.value in {"hold", "rejected"}:
                continue

            prod_idx = self.state.product_id_to_idx.get(batch.product_id)

            # Track gated excess per batch for RDC redistribution
            excess_for_rdcs = 0.0

            for dc_id, share in self.deployment_shares.items():
                if share <= 0:
                    continue

                qty_for_dc = batch.quantity_cases * share

                # v0.52.0: Soft taper push for plant-direct DCs
                # Full push below floor_dos, linear taper to zero at cap_dos
                # Gated excess redirected to RDCs (not left at plant)
                if (
                    dc_id in self._plant_direct_dc_ids
                    and prod_idx is not None
                ):
                    dc_demand = self._plant_direct_dc_demand.get(dc_id)
                    dc_idx = self.state.node_id_to_idx.get(dc_id)
                    if dc_demand is not None and dc_idx is not None:
                        daily_d = max(float(dc_demand[prod_idx]), 0.01)
                        on_hand = self.state.actual_inventory[
                            dc_idx, prod_idx
                        ]
                        in_transit_qty = float(in_transit[dc_idx, prod_idx])
                        current_ip = on_hand + in_transit_qty
                        dos = current_ip / daily_d

                        if dos >= push_cap_dos:
                            excess_for_rdcs += qty_for_dc
                            qty_for_dc = 0.0
                        elif dos > push_floor_dos:
                            fraction = (push_cap_dos - dos) / taper_range
                            original = qty_for_dc
                            qty_for_dc = original * fraction
                            excess_for_rdcs += original - qty_for_dc

                if qty_for_dc < 0.01:
                    continue

                # Find the link from plant to deployment target
                link = self._find_link(batch.plant_id, dc_id)
                lead_time = link.lead_time_days if link else default_lead_time

                shipment_counter += 1
                shipment = Shipment(
                    id=(
                        f"SHIP-PLANT-{current_day:03d}"
                        f"-{shipment_counter:06d}"
                    ),
                    source_id=batch.plant_id,
                    target_id=dc_id,
                    creation_day=current_day,
                    arrival_day=current_day + int(lead_time),
                    lines=[OrderLine(batch.product_id, qty_for_dc)],
                    status=ShipmentStatus.IN_TRANSIT,
                )

                product = self.world.products.get(batch.product_id)
                if product:
                    shipment.total_weight_kg = product.weight_kg * qty_for_dc
                    shipment.total_volume_m3 = product.volume_m3 * qty_for_dc

                # Deduct from plant inventory
                self.state.update_inventory(
                    batch.plant_id, batch.product_id, -qty_for_dc
                )

                shipments.append(shipment)

            # Redirect gated excess to RDCs proportionally
            if excess_for_rdcs > 0.01 and total_rdc_share > 0:
                for rdc_id, rdc_share in rdc_shares.items():
                    rdc_qty = excess_for_rdcs * (rdc_share / total_rdc_share)
                    if rdc_qty < 0.01:
                        continue

                    link = self._find_link(batch.plant_id, rdc_id)
                    lead_time = (
                        link.lead_time_days if link else default_lead_time
                    )

                    shipment_counter += 1
                    shipment = Shipment(
                        id=(
                            f"SHIP-PLANT-{current_day:03d}"
                            f"-{shipment_counter:06d}"
                        ),
                        source_id=batch.plant_id,
                        target_id=rdc_id,
                        creation_day=current_day,
                        arrival_day=current_day + int(lead_time),
                        lines=[
                            OrderLine(batch.product_id, rdc_qty)
                        ],
                        status=ShipmentStatus.IN_TRANSIT,
                    )

                    product = self.world.products.get(batch.product_id)
                    if product:
                        shipment.total_weight_kg = (
                            product.weight_kg * rdc_qty
                        )
                        shipment.total_volume_m3 = (
                            product.volume_m3 * rdc_qty
                        )

                    self.state.update_inventory(
                        batch.plant_id, batch.product_id, -rdc_qty
                    )
                    shipments.append(shipment)

        return shipments

    def _find_link(self, source_id: str, target_id: str) -> Link | None:
        """Find the link between two nodes."""
        for link in self.world.links.values():
            if link.source_id == source_id and link.target_id == target_id:
                return link
        return None

    def _magic_fulfillment(self, orders: list[Order]) -> None:
        """Immediately fulfills orders for testing purposes."""
        for order in orders:
            target_idx = self.state.node_id_to_idx.get(order.target_id)
            if target_idx is None:
                continue
            for line in order.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    self.state.update_inventory(
                        order.target_id, line.product_id, line.quantity
                    )

    def _push_excess_rdc_inventory(self, day: int) -> list[Shipment]:
        """
        Push excess RDC inventory to Customer DCs when DOS > threshold.

        v0.19.2: Implements push-based allocation to break the negative feedback
        spiral. When RDCs accumulate inventory (because Customer DCs under-order),
        this pushes excess downstream to maintain flow.

        Returns:
            List of push shipments created
        """
        push_shipments: list[Shipment] = []

        # Get config
        sim_params = self.config.get("simulation_parameters", {})
        replen_params = sim_params.get("agents", {}).get("replenishment", {})
        push_threshold_dos = float(replen_params.get("push_threshold_dos", 30.0))
        push_enabled = replen_params.get("push_allocation_enabled", True)

        if not push_enabled:
            return push_shipments

        default_lead_time = (
            sim_params.get("logistics", {}).get("default_lead_time_days", 3.0)
        )

        # Get RDC IDs (manufacturer RDCs only)
        rdc_ids = [
            n_id
            for n_id, n in self.world.nodes.items()
            if n.type == NodeType.DC and n_id.startswith("RDC-")
        ]

        if not rdc_ids:
            return push_shipments

        # Build downstream map: RDC -> list of Customer DC IDs
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            if link.source_id in rdc_ids:
                target_node = self.world.nodes.get(link.target_id)
                # Only push to Customer DCs (not stores directly)
                if target_node and target_node.type == NodeType.DC:
                    downstream_map.setdefault(link.source_id, []).append(
                        link.target_id
                    )

        # Use POS-based demand (stable signal) instead of outflow demand
        # (which collapses)
        # This ensures push allocation doesn't under-push during the negative spiral
        base_demand_matrix = self.pos_engine.get_base_demand_matrix()

        shipment_counter = 0

        for rdc_id in rdc_ids:
            rdc_idx = self.state.node_id_to_idx.get(rdc_id)
            if rdc_idx is None:
                continue

            downstream_dcs = downstream_map.get(rdc_id, [])
            if not downstream_dcs:
                continue

            # Calculate RDC inventory and average outflow per product
            rdc_inventory = self.state.actual_inventory[rdc_idx, :]

            # Calculate expected daily demand for this RDC based on downstream POS
            # This is a stable signal that doesn't collapse with the spiral
            rdc_expected_demand = np.zeros(self.state.n_products)
            for dc_id in downstream_dcs:
                dc_idx = self.state.node_id_to_idx.get(dc_id)
                if dc_idx is not None:
                    # Get downstream stores for this DC and sum their base demand
                    for link in self.world.links.values():
                        if link.source_id == dc_id:
                            store_node = self.world.nodes.get(link.target_id)
                            if store_node and store_node.type == NodeType.STORE:
                                store_idx = self.state.node_id_to_idx.get(
                                    link.target_id
                                )
                                if store_idx is not None:
                                    rdc_expected_demand += base_demand_matrix[
                                        store_idx, :
                                    ]

            # Floor demand to avoid division by zero
            rdc_demand_safe = np.maximum(rdc_expected_demand, 0.1)

            # Calculate DOS per product based on expected demand
            dos_per_product = rdc_inventory / rdc_demand_safe

            # Find products with DOS > threshold
            excess_mask = dos_per_product > push_threshold_dos

            if not np.any(excess_mask):
                continue

            # Calculate excess inventory to push
            target_dos = push_threshold_dos
            target_inventory = rdc_demand_safe * target_dos
            excess_inventory = np.maximum(0, rdc_inventory - target_inventory)

            # Only push products with excess
            excess_inventory[~excess_mask] = 0

            # Skip if no significant excess
            total_excess = np.sum(excess_inventory)
            if total_excess < 100:  # noqa: PLR2004 (Min threshold)
                continue

            # Distribute excess proportionally to downstream DCs based on their
            # POS demand
            # Calculate each DC's share of downstream demand (using stable POS signal)
            dc_demands: dict[str, np.ndarray] = {}
            total_dc_demand = np.zeros(self.state.n_products)
            for dc_id in downstream_dcs:
                # Calculate POS-based demand for this DC's downstream stores
                dc_pos_demand = np.zeros(self.state.n_products)
                for link in self.world.links.values():
                    if link.source_id == dc_id:
                        store_node = self.world.nodes.get(link.target_id)
                        if store_node and store_node.type == NodeType.STORE:
                            store_idx = self.state.node_id_to_idx.get(link.target_id)
                            if store_idx is not None:
                                dc_pos_demand += base_demand_matrix[store_idx, :]
                dc_demands[dc_id] = dc_pos_demand
                total_dc_demand += dc_pos_demand

            total_dc_demand_safe = np.maximum(total_dc_demand, 0.1)

            # v0.47.0 Fix 3: DOS cap for push receive — suppress push for
            # products where the target DC already has sufficient stock
            push_receive_cap = float(
                replen_params.get("push_receive_dos_cap", 15.0)
            )

            for dc_id, dc_demand in dc_demands.items():
                # Calculate this DC's share (proportional to demand)
                share_ratio = dc_demand / total_dc_demand_safe
                dc_push_qty = excess_inventory * share_ratio

                # v0.47.0 Fix 3: Check target DC inventory before pushing
                dc_idx = self.state.node_id_to_idx.get(dc_id)
                if dc_idx is not None:
                    dc_inv = self.state.actual_inventory[dc_idx, :]
                    dc_demand_safe = np.maximum(dc_demand, 0.1)
                    dc_dos = dc_inv / dc_demand_safe
                    # Suppress push for products where DC already has enough
                    dc_over_cap = dc_dos >= push_receive_cap
                    suppressed = int(np.sum((dc_push_qty >= 10) & dc_over_cap))  # noqa: PLR2004
                    self._push_suppression_count += suppressed
                    dc_push_qty = np.where(dc_over_cap, 0.0, dc_push_qty)

                # Create order lines for products with significant push qty
                lines = []
                for p_idx in range(self.state.n_products):
                    qty = dc_push_qty[p_idx]
                    if qty >= 10:  # noqa: PLR2004 (Min 10 cases)
                        p_id = self.state.product_idx_to_id[p_idx]
                        lines.append(OrderLine(p_id, qty))

                if not lines:
                    continue

                # Find link for lead time
                link_obj = self._find_link(rdc_id, dc_id)
                lead_time = (
                    link_obj.lead_time_days if link_obj else default_lead_time
                )

                shipment_counter += 1
                shipment = Shipment(
                    id=f"PUSH-{day:03d}-{rdc_id}-{shipment_counter:04d}",
                    source_id=rdc_id,
                    target_id=dc_id,
                    creation_day=day,
                    arrival_day=day + int(lead_time),
                    lines=lines,
                    status=ShipmentStatus.IN_TRANSIT,
                )

                # Deduct from RDC inventory
                for line in lines:
                    self.state.update_inventory(rdc_id, line.product_id, -line.quantity)

                push_shipments.append(shipment)

        return push_shipments


if __name__ == "__main__":
    sim = Orchestrator()
    sim.run(days=30)
