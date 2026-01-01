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

        self._initialize_inventory()

        # Get equilibrium demand estimate for warm start
        # This prevents Day 1-2 bullwhip cascade from cold start
        warm_start_demand = self.pos_engine.get_average_demand_estimate()

        self.replenisher = MinMaxReplenisher(
            self.world, self.state, self.config, warm_start_demand=warm_start_demand
        )
        self.allocator = AllocationAgent(self.state, self.config)
        self.logistics = LogisticsEngine(self.world, self.state, self.config)

        # 4. Initialize Manufacturing Engines (Milestone 5)
        self.mrp_engine = MRPEngine(self.world, self.state, self.config)
        self.transform_engine = TransformEngine(self.world, self.state, self.config)

        # 5. Initialize Validation & Quirks (Milestone 6)
        sim_params = self.config.get("simulation_parameters", {})
        self.monitor = RealismMonitor(sim_params)
        self.auditor = PhysicsAuditor(self.state, self.world, sim_params)
        self.resilience = ResilienceTracker(self.state, self.world)
        self.quirks = QuirkManager(config=self.config)
        self.risks = RiskEventManager(sim_params)

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

    def _initialize_inventory(self) -> None:
        """Seed initial inventory across the network (Priming)."""
        # Get manufacturing config for plant initial inventory
        sim_params = self.config.get("simulation_parameters", {})
        mfg_config = sim_params.get("manufacturing", {})
        initial_plant_inv = mfg_config.get("initial_plant_inventory", {})

        # Get priming config
        inv_config = sim_params.get("inventory", {})
        init_config = inv_config.get("initialization", {})

        # Defaults if config missing
        store_days = init_config.get("store_days_supply", 14.0)
        rdc_days = init_config.get("rdc_days_supply", 21.0)
        rdc_multiplier = init_config.get("rdc_store_multiplier", 1500.0)
        customer_dc_days = init_config.get("customer_dc_days_supply", 21.0)

        # Base demand proxy (calculated from POSEngine)
        base_demand = self.pos_engine.get_average_demand_estimate()
        print(f"Orchestrator: Priming inventory with base_demand={base_demand:.2f}")

        # Calculate Levels
        store_level = base_demand * store_days
        rdc_level = base_demand * rdc_multiplier * rdc_days

        # Seed finished goods at RDCs and Stores
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.STORE:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    self.state.perceived_inventory[node_idx, :] = store_level
                    self.state.actual_inventory[node_idx, :] = store_level

            elif node.type == NodeType.DC:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    if node_id.startswith("RDC-"):
                        # Manufacturer RDCs use fixed multiplier
                        dc_level = rdc_level
                    else:
                        # Customer DCs: scale by downstream store count
                        # Count downstream stores for this DC
                        downstream_stores = sum(
                            1
                            for link in self.world.links.values()
                            if link.source_id == node_id
                            and self.world.nodes.get(link.target_id)
                            and self.world.nodes[link.target_id].type
                            == NodeType.STORE
                        )
                        # Customer DC level = base_demand × store_count × target_days
                        # v0.15.8: Use config value for customer DC days supply
                        dc_level = base_demand * max(downstream_stores, 1) * customer_dc_days

                    self.state.perceived_inventory[node_idx, :] = dc_level
                    self.state.actual_inventory[node_idx, :] = dc_level

            # Seed raw materials at Plants
            elif node.type == NodeType.PLANT:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    # Initialize ALL ingredients to prevent cold start starvation
                    for product in self.world.products.values():
                        if product.category == ProductCategory.INGREDIENT:
                            # Use config override if present, else robust default (5M units)
                            # 5M units covers ~20 days of production at 230k run rate
                            qty = initial_plant_inv.get(product.id, 5000000.0)
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

        # 2. Consume Inventory (Sales) - Constrained to available inventory
        # Cannot sell more than what's on hand (lost sales model)
        available = np.maximum(0, self.state.actual_inventory)
        actual_sales = np.minimum(daily_demand, available)
        lost_sales = daily_demand - actual_sales
        self.state.update_inventory_batch(-actual_sales)
        self.auditor.record_sales(actual_sales)
        # Note: lost_sales tracked implicitly via fill rate metrics

        # 3. Replenishment Decision (The "Pull" Signal)
        raw_orders = self.replenisher.generate_orders(day, daily_demand)

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

        # 5. Logistics (Milestone 4.2)
        new_shipments = self.logistics.create_shipments(allocated_orders, day)
        self._apply_logistics_quirks_and_risks(new_shipments)
        self.state.active_shipments.extend(new_shipments)

        # 6. Transit & Arrival (Milestone 4.3)
        active, arrived = self.logistics.update_shipments(
            self.state.active_shipments, day
        )
        self.state.active_shipments = active

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
        
        new_production_orders = self.mrp_engine.generate_production_orders(
            day, rdc_store_shipments, self.active_production_orders
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

        # 10. Ship finished goods from Plants to RDCs
        plant_shipments = self._create_plant_shipments(new_batches, day)
        self.auditor.record_plant_shipments_out(plant_shipments)
        self._apply_logistics_quirks_and_risks(plant_shipments)
        self.state.active_shipments.extend(plant_shipments)

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
        daily_shipments = new_shipments + plant_shipments
        
        total_shipped_qty = sum(line.quantity for s in daily_shipments for line in s.lines)
        shrinkage_qty = sum(e.quantity_lost for e in shrinkage_events)
        
        self._record_daily_metrics(
            daily_demand, daily_shipments, arrived, plant_oee, day,
            ordered_qty=unconstrained_demand_qty,
            shipped_qty=total_shipped_qty,
            shrinkage_qty=shrinkage_qty
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

            # SLOB % (Simplification: FG Inventory > 60 days of demand)
            # Only finished goods can be "slow/obsolete" for this metric
            global_dos = total_fg_inv / max(total_demand_qty, 1.0)
            is_slob = 1.0 if global_dos > 60.0 else 0.0
            self.monitor.record_slob(is_slob)

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
        # 30% baseline error + Optimism Bias if active
        mape = 0.30
        if self.quirks.is_enabled("optimism_bias"):
             mape += 0.15
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
        scoring_config = self.config.get("simulation_parameters", {}).get("scoring", {})

        # Scoring Weights
        base_service = scoring_config.get("service_index_base", 100.0)
        backlog_penalty = scoring_config.get("backlog_penalty_divisor", 1000.0)
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
        service_index = report.get("store_service_level", {}).get("mean", 0.0) * 100.0
        inv_turns = report.get("inventory_turns", {}).get("mean", 0)
        
        perfect_order = report.get("perfect_order_rate", {}).get("mean", 0) * 100.0
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
        for shipment in arrived_shipments:
            target_idx = self.state.node_id_to_idx.get(shipment.target_id)
            if target_idx is None:
                continue

            for line in shipment.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    # Update both perceived and actual inventory
                    self.state.update_inventory(
                        shipment.target_id, line.product_id, line.quantity
                    )

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

        # Get list of RDC IDs
        rdc_ids = [
            n_id for n_id, n in self.world.nodes.items() if n.type == NodeType.DC
        ]

        if not rdc_ids:
            return shipments

        shipment_counter = 0

        for batch in batches:
            if batch.status.value in {"hold", "rejected"}:
                # Don't ship held/rejected batches
                continue

            # Distribute batch quantity evenly across RDCs
            qty_per_rdc = batch.quantity_cases / len(rdc_ids)

            for rdc_id in rdc_ids:
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
                    lines=[OrderLine(batch.product_id, qty_per_rdc)],
                    status=ShipmentStatus.IN_TRANSIT,
                )

                # Deduct from plant inventory (both perceived and actual)
                plant_idx = self.state.node_id_to_idx.get(batch.plant_id)
                prod_idx = self.state.product_id_to_idx.get(batch.product_id)
                if plant_idx is not None and prod_idx is not None:
                    self.state.update_inventory(
                        batch.plant_id, batch.product_id, -qty_per_rdc
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


if __name__ == "__main__":
    sim = Orchestrator()
    sim.run(days=30)
