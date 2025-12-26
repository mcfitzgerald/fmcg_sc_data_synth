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
from prism_sim.simulation.mrp import MRPEngine
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.transform import TransformEngine


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

        # 5. Manufacturing State
        self.active_production_orders: list[ProductionOrder] = []
        self.completed_batches: list[Batch] = []

    def _initialize_inventory(self) -> None:
        """
        Seed initial inventory across the network.

        - RDCs and Stores: Initial finished goods stock
        - Plants: Initial raw material inventory
        """
        # Get manufacturing config for plant initial inventory
        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        initial_plant_inv = mfg_config.get("initial_plant_inventory", {})

        # Seed finished goods at RDCs and Stores (100 cases each)
        for node_id, node in self.world.nodes.items():
            if node.type in (NodeType.DC, NodeType.STORE):
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    # Set all finished goods to 100
                    self.state.inventory[node_idx, :] = 100.0

            # Seed raw materials at Plants
            elif node.type == NodeType.PLANT:
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    for ing_id, qty in initial_plant_inv.items():
                        ing_idx = self.state.product_id_to_idx.get(ing_id)
                        if ing_idx is not None:
                            self.state.inventory[node_idx, ing_idx] = qty

    def run(self, days: int = 30) -> None:
        print(f"Starting Simulation for {days} days...")

        for day in range(1, days + 1):
            self._step(day)

        print("Simulation Complete.")

    def _step(self, day: int) -> None:
        # 1. Generate Demand (POS)
        # Shape: [Nodes, Products]
        daily_demand = self.pos_engine.generate_demand(day)

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
        self.state.active_shipments.extend(plant_shipments)

        # 11. Logging / Metrics (Simple Print)
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

    def _process_arrivals(self, arrived_shipments: list[Shipment]) -> None:
        for shipment in arrived_shipments:
            target_idx = self.state.node_id_to_idx.get(shipment.target_id)
            if target_idx is None:
                continue

            for line in shipment.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    self.state.inventory[target_idx, p_idx] += line.quantity

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

                # Deduct from plant inventory
                plant_idx = self.state.node_id_to_idx.get(batch.plant_id)
                prod_idx = self.state.product_id_to_idx.get(batch.product_id)
                if plant_idx is not None and prod_idx is not None:
                    self.state.inventory[plant_idx, prod_idx] -= qty_per_rdc

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
                    self.state.inventory[target_idx, p_idx] += line.quantity


if __name__ == "__main__":
    sim = Orchestrator()
    sim.run(days=30)
