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

    def __init__(self) -> None:
        # 1. Initialize World
        manifest = load_manifest()
        self.config = load_simulation_config()
        self.builder = WorldBuilder(manifest)
        self.world = self.builder.build()

        # 2. Initialize State
        self.state = StateManager(self.world)
        self._initialize_inventory()

        # 3. Initialize Engines & Agents
        self.pos_engine = POSEngine(self.world, self.state, self.config)
        self.replenisher = MinMaxReplenisher(self.world, self.state, self.config)
        self.allocator = AllocationAgent(self.state)
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
        self.writer = SimulationWriter()

        # 7. Manufacturing State
        self.active_production_orders: list[ProductionOrder] = []
        self.completed_batches: list[Batch] = []

    def _initialize_inventory(self) -> None:
        """
        Seed initial inventory across the network.

        - RDCs and Stores: Initial finished goods stock
        - Plants: Initial raw material inventory
        """
        # Get manufacturing config for plant initial inventory
        sim_params = self.config.get("simulation_parameters", {})
        mfg_config = sim_params.get("manufacturing", {})
        initial_plant_inv = mfg_config.get("initial_plant_inventory", {})

        # Get initial FG level from config
        inv_config = sim_params.get("inventory", {})
        initial_fg_level = inv_config.get("initial_fg_level", 100.0)

        # Seed finished goods at RDCs and Stores
        for node_id, node in self.world.nodes.items():
            if node.type in (NodeType.DC, NodeType.STORE):
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    # Set all finished goods
                    # Initialize both Perceived (System) and Actual (Physical) inventory
                    self.state.perceived_inventory[node_idx, :] = initial_fg_level
                    self.state.actual_inventory[node_idx, :] = initial_fg_level

            # Seed raw materials at Plants
            elif node.type == NodeType.PLANT:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    for ing_id, qty in initial_plant_inv.items():
                        ing_idx = self.state.product_id_to_idx.get(ing_id)
                        if ing_idx is not None:
                            # Use update_inventory to set both
                            self.state.update_inventory(node_id, ing_id, qty)

    def run(self, days: int = 30) -> None:
        print(f"Starting Simulation for {days} days...")

        for day in range(1, days + 1):
            self._step(day)

        print("Simulation Complete.")

    def _step(self, day: int) -> None:
        # --- Milestone 6: Risk & Quirks Start ---
        # 0. Trigger Risk Events
        triggered_risks = self.risks.check_triggers(day)
        if triggered_risks:
            print(
                f"Day {day:03}: RISK EVENTS TRIGGERED: "
                f"{[e.event_code for e in triggered_risks]}"
            )

        # Track active disruptions for Resilience
        for _ in triggered_risks:
            # Assuming event implies disruption at some node?
            # For now, generic disruption start.
            pass

        # Apply Phantom Inventory Shrinkage & Discovery
        self.quirks.apply_shrinkage(self.state, day)
        self.quirks.process_discoveries(self.state, day)
        # ----------------------------------------

        # 1. Generate Demand (POS)
        # Shape: [Nodes, Products]
        daily_demand = self.pos_engine.generate_demand(day)

        # --- Milestone 6: Optimism Bias ---
        product_ids = [
            self.state.product_idx_to_id[i] for i in range(self.state.n_products)
        ]
        daily_demand = self.quirks.apply_optimism_bias(daily_demand, product_ids, day)
        # ----------------------------------

        # 2. Consume Inventory (Sales)
        # Note: In real life, you can only sell what you have (OSA).
        # For now, we allow negative inventory (Backlog)
        self.state.update_inventory_batch(-daily_demand)

        # 3. Replenishment Decision (The "Pull" Signal)
        # Agents look at current state and place orders
        raw_orders = self.replenisher.generate_orders(day, daily_demand)

        # 4. Allocation (Milestone 4.1)
        # Check source inventory, apply rationing
        allocated_orders = self.allocator.allocate_orders(raw_orders)

        # 5. Logistics (Milestone 4.2)
        # Create shipments (Tetris)
        new_shipments = self.logistics.create_shipments(allocated_orders, day)

        # --- Milestone 6: Logistics Quirks & Risks ---
        delay_multiplier = self.risks.get_logistics_delay_multiplier()
        
        # Apply Port Congestion (adds days)
        self.quirks.apply_port_congestion(new_shipments)

        for shipment in new_shipments:
            # Apply Risk Multiplier to transit time
            if delay_multiplier > 1.0:
                original_duration = shipment.arrival_day - shipment.creation_day
                new_duration = original_duration * delay_multiplier
                shipment.arrival_day = shipment.creation_day + int(new_duration)
        # ---------------------------------------------

        self.state.active_shipments.extend(new_shipments)

        # 6. Transit & Arrival (Milestone 4.3)
        # Advance shipment states
        active, arrived = self.logistics.update_shipments(
            self.state.active_shipments, day
        )
        self.state.active_shipments = active

        # 7. Process Arrivals (Receive Inventory)
        self._process_arrivals(arrived)

        # 8. Manufacturing: MRP (Milestone 5.1)
        # Generate Production Orders based on RDC inventory
        new_production_orders = self.mrp_engine.generate_production_orders(
            day, daily_demand
        )
        self.active_production_orders.extend(new_production_orders)

        # 9. Manufacturing: Production (Milestone 5.2)
        # Process Production Orders with finite capacity and changeover
        updated_orders, new_batches = self.transform_engine.process_production_orders(
            self.active_production_orders, day
        )

        # Filter out completed orders
        self.active_production_orders = [
            o for o in updated_orders if o.status.value != "complete"
        ]
        self.completed_batches.extend(new_batches)

        # 10. Ship finished goods from Plants to RDCs
        plant_shipments = self._create_plant_shipments(new_batches, day)

        # --- Milestone 6: Plant Shipment Delays ---
        # Apply Port Congestion (adds days)
        self.quirks.apply_port_congestion(plant_shipments)
        
        for shipment in plant_shipments:
             # Risk multiplier
             if delay_multiplier > 1.0:
                 original_duration = shipment.arrival_day - shipment.creation_day
                 new_duration = original_duration * delay_multiplier
                 shipment.arrival_day = shipment.creation_day + int(new_duration)
        # ------------------------------------------

        self.state.active_shipments.extend(plant_shipments)

        # --- Milestone 6: Validation & Resilience ---
        # 11. Resilience Check
        # Check active risks recovery
        recovered = self.risks.check_recovery(day)
        if recovered:
             print(f"Day {day:03}: RISK RECOVERY: {recovered}")

        # TTS/TTR (Placeholder logic for now, needs Node disruption state mapping)
        # self.resilience.check_survival(day)

        # 12. Monitors
        # Truck Fill
        log_config = self.config.get("simulation_parameters", {}).get("logistics", {})
        max_weight = log_config.get("constraints", {}).get("truck_max_weight_kg", 20000.0)

        for s in new_shipments + plant_shipments:
             # Calculate fill rate (Weight)
             fill_rate = min(1.0, s.total_weight_kg / max_weight)
             self.monitor.record_truck_fill(fill_rate)

        # Physics Audit
        violations = self.auditor.check_kinematic_consistency(arrived, day)
        if violations:
            print(f"Day {day:03}: PHYSICS VIOLATIONS: {violations}")

        # --------------------------------------------

        # 13. Data Logging (Milestone 7)
        self.writer.log_orders(raw_orders, day)
        self.writer.log_shipments(new_shipments + plant_shipments, day)
        self.writer.log_batches(new_batches, day)
        # Log inventory every 7 days to keep file size manageable
        if day % 7 == 0:
            self.writer.log_inventory(self.state, self.world, day)

        # 14. Logging / Metrics (Simple Print)
        total_demand = np.sum(daily_demand)
        total_ordered = sum(
            line.quantity for order in raw_orders for line in order.lines
        )
        total_shipped = sum(
            line.quantity for shipment in new_shipments for line in shipment.lines
        )
        total_arrived = sum(
            line.quantity for shipment in arrived for line in shipment.lines
        )
        total_produced = sum(b.quantity_cases for b in new_batches)

        print(
            f"Day {day:03}: Demand={total_demand:.1f}, "
            f"Ordered={total_ordered:.1f}, "
            f"Shipped={total_shipped:.1f}, "
            f"Arrived={total_arrived:.1f}, "
            f"Produced={total_produced:.1f}, "
            f"InTransit={len(self.state.active_shipments)} trucks"
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
        
        # Calculate Service (LIFR approx from backlogs)
        # Note: In our current simple state, negative inventory is backlog.
        # So we can look at actual vs perceived or just positive vs negative.
        total_backlog = np.sum(np.maximum(0, -self.state.actual_inventory))
        total_inventory = np.sum(np.maximum(0, self.state.actual_inventory))
        
        oee = report.get("oee", {}).get("mean", 0)
        truck_fill = report.get("truck_fill", {}).get("mean", 0)
        
        summary = [
            "==================================================",
            "        THE SUPPLY CHAIN TRIANGLE REPORT          ",
            "==================================================",
            f"1. SERVICE (Fill Rate Index):   {100.0 - (total_backlog/1000.0):.2f}%", # Arbitrary scaling for demo
            f"2. CASH (Inventory Turns):      {report.get('inventory_turns', {}).get('mean', 0):.2f}x",
            f"3. COST (Truck Fill Rate):      {truck_fill*100.0:.1f}%",
            "--------------------------------------------------",
            f"Manufacturing OEE:              {oee*100.0:.1f}%",
            f"Total System Inventory:         {total_inventory:,.0f} cases",
            f"Total Backlog:                  {total_backlog:,.0f} cases",
            "=================================================="
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
                    self.state.update_inventory(shipment.target_id, line.product_id, line.quantity)

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
                    self.state.update_inventory(batch.plant_id, batch.product_id, -qty_per_rdc)

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
                    self.state.update_inventory(order.target_id, line.product_id, line.quantity)


if __name__ == "__main__":
    sim = Orchestrator()
    sim.run(days=30)
