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


class Orchestrator:
    """The main time-stepper loop for the Prism Digital Twin."""

    def __init__(
        self,
        enable_logging: bool = False,
        output_dir: str = "data/output",
        streaming: bool | None = None,
        output_format: str | None = None,
        inventory_sample_rate: int | None = None,
    ) -> None:
        # 1. Initialize World
        manifest = load_manifest()
        self.config = load_simulation_config()

        # Merge static world definitions into config for Engines that need them (e.g. POSEngine)
        self.config["promotions"] = manifest.get("promotions", [])
        self.config["packaging_types"] = manifest.get("packaging_types", [])

        self.builder = WorldBuilder(manifest)
        self.world = self.builder.build()

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
        # MOVED UP: Initialize MRP Engine EARLY to get ABC classification for inventory priming
        self.mrp_engine = MRPEngine(
            self.world, self.state, self.config, base_demand_matrix
        )

        self._initialize_inventory()

        self.replenisher = MinMaxReplenisher(
            self.world,
            self.state,
            self.config,
            warm_start_demand=warm_start_demand,
            base_demand_matrix=base_demand_matrix,
        )
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

        # 7. Manufacturing State
        self.active_production_orders: list[ProductionOrder] = []
        self.completed_batches: list[Batch] = []

        # 8. Finished Goods Mask for Metrics (excludes ingredients from inventory turns)
        self._fg_product_mask = self._build_finished_goods_mask()

        # v0.19.12: Calculate RDC Demand Shares for production routing
        self.rdc_demand_shares = self._calculate_rdc_demand_shares()

    def _calculate_rdc_demand_shares(self) -> dict[str, float]:
        """
        Calculate the share of global POS demand served by each RDC.
        Used to route production proportional to demand (physics-based flow).
        """
        rdc_demand = {}
        total_network_demand = 0.0

        # Get RDC IDs
        rdc_ids = [n_id for n_id, n in self.world.nodes.items()
                   if n.type == NodeType.DC and n_id.startswith("RDC-")]

        # Build downstream map: source_id -> [target_ids]
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            downstream_map.setdefault(link.source_id, []).append(link.target_id)

        base_demand_matrix = self.pos_engine.get_base_demand_matrix()

        for rdc_id in rdc_ids:
            # Recursive demand aggregation
            # RDC -> DCs -> Stores
            # RDC -> Stores
            rdc_total = 0.0
            visited = set()
            stack = [rdc_id]

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
                                rdc_total += np.sum(base_demand_matrix[idx, :])
                        else:
                            # Keep traversing logistics layer
                            stack.append(child_id)

            rdc_demand[rdc_id] = rdc_total
            total_network_demand += rdc_total

        # Convert to shares
        shares = {}
        if total_network_demand > 0:
            for rdc_id, demand in rdc_demand.items():
                shares[rdc_id] = demand / total_network_demand
        else:
            # Fallback to even split if no demand (shouldn't happen with GIS)
            even = 1.0 / len(rdc_ids) if rdc_ids else 0.0
            shares = {rid: even for rid in rdc_ids}

        return shares

    def _initialize_inventory(self) -> None:  # noqa: PLR0912, PLR0915
        """
        Seed initial inventory across the network (Priming).

        v0.19.12: RDCs only initialize with inventory if they have downstream
        demand. Demand-proportional priming is used for all nodes.
        """
        # Get manufacturing config for plant initial inventory
        sim_params = self.config.get("simulation_parameters", {})
        mfg_config = sim_params.get("manufacturing", {})
        initial_plant_inv = mfg_config.get("initial_plant_inventory", {})

        # Get priming config
        inv_config = sim_params.get("inventory", {})
        init_config = inv_config.get("initialization", {})

        # Defaults if config missing
        store_days_default = init_config.get("store_days_supply", 14.0)
        rdc_days = init_config.get("rdc_days_supply", 21.0)
        customer_dc_days = init_config.get("customer_dc_days_supply", 21.0)

        # Get per-SKU demand matrix for demand-proportional priming
        base_demand_matrix = self.pos_engine.get_base_demand_matrix()

        # Phase 2.3: Target DOS by ABC class
        abc_target_dos = {
            0: 21.0,   # A-items: 3 weeks
            1: 14.0,   # B-items: 2 weeks
            2: 7.0,    # C-items: 1 week
        }

        # Seed finished goods at RDCs and Stores
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.STORE:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    node_demand = base_demand_matrix[node_idx, :]

                    # Apply ABC-based priming
                    # Vectorized operation: Create a vector of days supply based on product ABC class
                    store_days_vec = np.array([
                        abc_target_dos.get(
                            self.mrp_engine.abc_class[p_idx], store_days_default
                        )
                        for p_idx in range(self.state.n_products)
                    ])

                    sku_levels = node_demand * store_days_vec
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
                            downstream_map.setdefault(link.source_id, []).append(link.target_id)

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
                                            rdc_downstream_demand += base_demand_matrix[t_idx, :]
                                    else:
                                        stack.append(child_id)

                        # Skip priming if no demand (Ghost RDC fix)
                        if rdc_downstream_demand.sum() == 0:
                            continue

                        # RDC inventory = downstream demand x rdc_days_supply
                        rdc_sku_levels = rdc_downstream_demand * rdc_days
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

                        # DC inventory = aggregated downstream demand x days supply
                        dc_levels = downstream_demand * customer_dc_days
                        self.state.perceived_inventory[node_idx, :] = dc_levels
                        self.state.actual_inventory[node_idx, :] = dc_levels

            # Seed raw materials at Plants
            # v0.20.0: Sized for ~120 days supply to ensure 90-day production stability.
            # Some ingredients are consumed much faster due to recipe clustering.
            # Using 50M per ingredient to provide sufficient buffer for high-demand
            # ingredients while MRP ingredient ordering catches up.
            # NOTE: Long-term fix needed in MRP ingredient ordering to maintain supply.
            elif node.type == NodeType.PLANT:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    for product in self.world.products.values():
                        if product.category == ProductCategory.INGREDIENT:
                            qty = initial_plant_inv.get(product.id, 50000000.0)
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
        print(f"Starting Simulation for {days} days...")

        for day in range(1, days + 1):
            self._step(day)

        print("Simulation Complete.")

    def _step(self, day: int) -> None:
        # 0. Start mass balance tracking
        self.auditor.start_day(day)

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
        # Note: lost_sales tracked implicitly via fill rate metrics

        # 3. Replenishment Decision (The "Pull" Signal)
        raw_orders = self.replenisher.generate_orders(day, daily_demand)

        # v0.15.9: Record inflow (orders received) for true demand signal
        # This captures what was REQUESTED before allocation constrains it
        # Used by customer DCs to prevent demand signal attenuation
        self.replenisher.record_inflow(raw_orders)

        # v0.15.9: Pass order demand to MRP (pre-allocation, true demand)
        # Filter for orders TO RDCs (from customer DCs) - these drive production
        rdc_orders = [
            o for o in raw_orders if o.source_id.startswith("RDC-")
        ]
        self.mrp_engine.record_order_demand(rdc_orders)

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
        active, arrived = self.logistics.update_shipments(
            self.state.active_shipments, day
        )
        self.state.remove_arrived_shipments(arrived)

        # 7. Process Arrivals (Receive Inventory)
        self._process_arrivals(arrived)
        self.auditor.record_receipts(arrived)

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
        ) = self.transform_engine.process_production_orders(
            self.active_production_orders, day
        )
        self.auditor.record_production(new_batches)
        self.active_production_orders = [
            o for o in updated_orders if o.status.value != "complete"
        ]
        self.completed_batches.extend(new_batches)

        # v0.20.0: Production order cleanup - remove stale orders
        # Orders that haven't been fulfilled within 14 days are likely blocked
        # (material shortage, capacity issues) and should be dropped to prevent
        # unbounded backlog accumulation. MRP will regenerate if demand persists.
        production_order_timeout = 14
        self.active_production_orders = [
            o for o in self.active_production_orders
            if day - o.creation_day <= production_order_timeout
        ]

        # v0.20.0: Memory cleanup - only retain recent batches for traceability
        # Keep last 30 days of batches, discard older ones to prevent unbounded growth
        batch_retention_days = 30
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

        self._record_daily_metrics(
            daily_demand,
            daily_shipments,
            arrived,
            plant_oee,
            day,
            ordered_qty=unconstrained_demand_qty,
            shipped_qty=total_shipped_qty,
            shrinkage_qty=shrinkage_qty,
        )
        self._log_daily_data(
            raw_orders, new_shipments, plant_shipments, new_batches, day
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
        day: int,
        ordered_qty: float = 0.0,
        shipped_qty: float = 0.0,
        shrinkage_qty: float = 0.0,
    ) -> None:
        """Record simulation metrics for monitoring."""
        # Record Service Level (Fill Rate)
        fill_rate = 1.0
        if ordered_qty > 0:
            fill_rate = shipped_qty / ordered_qty
        self.monitor.record_service_level(fill_rate)

        # Record Store Service Level (On-Shelf Availability proxy)
        total_demand_qty = np.sum(daily_demand)
        if total_demand_qty > 0:
            available = np.maximum(0, self.state.actual_inventory)
            actual_sales = np.minimum(daily_demand, available)
            store_fill_rate = np.sum(actual_sales) / total_demand_qty
            self.monitor.record_store_service_level(store_fill_rate)

        # Calculate Inventory Turns (Cash) - ONLY finished goods, not ingredients
        # Inventory turns = Annual Sales / Average Inventory (finished goods only)
        fg_inventory = self.state.actual_inventory[:, self._fg_product_mask]
        total_fg_inv = np.sum(np.maximum(0, fg_inventory))
        if total_fg_inv > 0:
            daily_turn_rate = total_demand_qty / total_fg_inv  # Sales / Avg FG Inv
            annual_turns = daily_turn_rate * 365
            self.monitor.record_inventory_turns(annual_turns)

            # Cash-to-Cash (Est: DIO + DSO - DPO)
            # DIO = 365 / Turns
            # DSO ~ 30, DPO ~ 45
            dio = 365.0 / annual_turns
            c2c = dio + 30.0 - 45.0
            self.monitor.record_cash_to_cash(c2c)

            # Shrinkage Rate (on FG inventory only - raw materials tracked separately)
            shrink_rate = shrinkage_qty / total_fg_inv
            self.monitor.record_shrinkage_rate(shrink_rate)

            # SLOB % (Per-SKU calculation)
            # Only finished goods can be "slow/obsolete" for this metric
            # Calculate per-SKU days of supply across all nodes
            fg_inv_per_sku = np.sum(fg_inventory, axis=0)  # Sum across nodes
            fg_demand_per_sku = np.sum(
                daily_demand[:, self._fg_product_mask], axis=0
            )

            # Avoid division by zero - use small floor for zero-demand SKUs
            demand_per_sku_safe = np.maximum(fg_demand_per_sku, 0.01)
            sku_dos = fg_inv_per_sku / demand_per_sku_safe

            # Flag SKUs with DOS > threshold as SLOB
            slob_mask = sku_dos > self.slob_days_threshold
            slob_inventory = fg_inv_per_sku[slob_mask].sum()

            slob_pct = slob_inventory / total_fg_inv if total_fg_inv > 0 else 0.0
            self.monitor.record_slob(slob_pct)

        log_config = self.config.get("simulation_parameters", {}).get("logistics", {})
        constraints = log_config.get("constraints", {})
        max_weight = constraints.get("truck_max_weight_kg", 20000.0)

        # v0.15.5: Measure truck fill for all shipments
        # Note: With LTL for stores, many shipments are intentionally small.
        # The metric now reflects actual truck utilization across the network.
        # For FMCG products that "cube out" before "weighting out", fill rates
        # of 30-50% are realistic (light but bulky products).
        for s in daily_shipments:
            fill_rate = min(1.0, s.total_weight_kg / max_weight)
            self.monitor.record_truck_fill(fill_rate)

        # Record OEE
        if plant_oee:
            avg_oee = sum(plant_oee.values()) / len(plant_oee)
            self.monitor.record_oee(avg_oee)

        # Perfect Order Rate
        # Check active risk delays
        delay_mult = self.risks.get_logistics_delay_multiplier()
        is_perfect = 1.0 if delay_mult == 1.0 else 0.5
        self.monitor.record_perfect_order(is_perfect)

        # Scope 3 Emissions (0.25 kg CO2 per case placeholder)
        self.monitor.record_scope_3(0.25)

        # MAPE
        # Baseline error + Optimism Bias penalty if active
        mape = self.mape_base
        if self.quirks.is_enabled("optimism_bias"):
            mape += self.mape_quirks_penalty
        self.monitor.record_mape(mape)

    def _log_daily_data(
        self,
        raw_orders: list[Order],
        new_shipments: list[Shipment],
        plant_shipments: list[Shipment],
        new_batches: list[Batch],
        day: int,
    ) -> None:
        """Log data to the simulation writer."""
        self.writer.log_orders(raw_orders, day)
        self.writer.log_shipments(new_shipments + plant_shipments, day)
        self.writer.log_batches(new_batches, day)
        if day % 7 == 0:
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
        truck_fill = report.get("truck_fill", {}).get("mean", 0)

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

        summary = [
            "==================================================",
            "        THE SUPPLY CHAIN TRIANGLE REPORT          ",
            "==================================================",
            f"1. SERVICE (Store Fill Rate):   {service_index:.2f}%",
            f"2. CASH (Inventory Turns):      {inv_turns:.2f}x",
            f"3. COST (Truck Fill Rate):      {truck_fill * truck_scale:.1f}%",
            "--------------------------------------------------",
            f"Manufacturing OEE:              {oee * oee_scale:.1f}%",
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

        # Single batch update instead of 3.6M individual calls
        self.state.update_inventory_batch(delta)

    def _create_plant_shipments(
        self, batches: list[Batch], current_day: int
    ) -> list[Shipment]:
        """
        Create shipments from Plants to RDCs for completed production batches.

        Distributes produced goods evenly across RDCs (simplified logic).
        """
        shipments: list[Shipment] = []

        # Get default lead time from config
        default_lead_time = (
            self.config.get("simulation_parameters", {})
            .get("logistics", {})
            .get("default_lead_time_days", 3.0)
        )

        # Get list of manufacturer RDC IDs only (RDC-* prefix)
        # Excludes customer DCs (RET-DC-*, DIST-DC-*, ECOM-FC-*, DTC-FC-*)
        rdc_ids = [
            n_id
            for n_id, n in self.world.nodes.items()
            if n.type == NodeType.DC and n_id.startswith("RDC-")
        ]

        if not rdc_ids:
            return shipments

        shipment_counter = 0

        for batch in batches:
            if batch.status.value in {"hold", "rejected"}:
                # Don't ship held/rejected batches
                continue

            # Distribute batch quantity based on demand shares
            # v0.19.12: Demand-proportional routing (physics-based flow)
            for rdc_id, share in self.rdc_demand_shares.items():
                if share <= 0:
                    continue

                qty_for_rdc = batch.quantity_cases * share

                # Find the link from plant to RDC
                link = self._find_link(batch.plant_id, rdc_id)
                lead_time = link.lead_time_days if link else default_lead_time

                shipment_counter += 1
                shipment = Shipment(
                    id=f"SHIP-PLANT-{current_day:03d}-{shipment_counter:06d}",
                    source_id=batch.plant_id,
                    target_id=rdc_id,
                    creation_day=current_day,
                    arrival_day=current_day + int(lead_time),
                    lines=[OrderLine(batch.product_id, qty_for_rdc)],
                    status=ShipmentStatus.IN_TRANSIT,
                )

                # Deduct from plant inventory
                self.state.update_inventory(
                    batch.plant_id, batch.product_id, -qty_for_rdc
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

    def _push_excess_rdc_inventory(self, day: int) -> list[Shipment]:  # noqa: PLR0912, PLR0915
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

            for dc_id, dc_demand in dc_demands.items():
                # Calculate this DC's share (proportional to demand)
                share_ratio = dc_demand / total_dc_demand_safe
                dc_push_qty = excess_inventory * share_ratio

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
