from typing import List
import numpy as np
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.demand import POSEngine
from prism_sim.agents.replenishment import MinMaxReplenisher
from prism_sim.agents.allocation import AllocationAgent
from prism_sim.simulation.logistics import LogisticsEngine
from prism_sim.config.loader import load_manifest
from prism_sim.network.core import Shipment


class Orchestrator:
    """
    The main time-stepper loop for the Prism Digital Twin.
    """

    def __init__(self):
        # 1. Initialize World
        manifest = load_manifest()
        self.builder = WorldBuilder(manifest)
        self.world = self.builder.build()

        # 2. Initialize State
        self.state = StateManager(self.world)
        self._initialize_inventory()

        # 3. Initialize Engines & Agents
        self.pos_engine = POSEngine(self.world, self.state)
        self.replenisher = MinMaxReplenisher(self.world, self.state)
        self.allocator = AllocationAgent(self.state)
        self.logistics = LogisticsEngine(self.world, self.state)

    def _initialize_inventory(self):
        # Seed some initial inventory to avoid massive day 1 orders
        # For now, just set everything to 100 cases
        self.state.inventory.fill(100.0)

    def run(self, days: int = 30):
        print(f"Starting Simulation for {days} days...")

        for day in range(1, days + 1):
            self._step(day)

        print("Simulation Complete.")

    def _step(self, day: int):
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

        # 8. Logging / Metrics (Simple Print)
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

        print(
            f"Day {day:03}: Demand={total_demand:.1f}, Ordered={total_ordered:.1f}, Shipped={total_shipped:.1f}, Arrived={total_arrived:.1f}, InTransit={len(self.state.active_shipments)} trucks"
        )

    def _process_arrivals(self, arrived_shipments: List[Shipment]):
        for shipment in arrived_shipments:
            target_idx = self.state.node_id_to_idx.get(shipment.target_id)
            if target_idx is None:
                continue

            for line in shipment.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    self.state.inventory[target_idx, p_idx] += line.quantity


if __name__ == "__main__":
    sim = Orchestrator()
    sim.run(days=30)
