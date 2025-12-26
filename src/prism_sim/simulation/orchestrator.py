from typing import List
import numpy as np
from prism_sim.simulation.world import World
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.demand import POSEngine
from prism_sim.agents.replenishment import MinMaxReplenisher
from prism_sim.config.loader import load_manifest

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
        orders = self.replenisher.generate_orders(day, daily_demand)
        
        # 4. Magic Fulfillment (Milestone 3 Shortcut)
        # Instant delivery to close the loop and prevent infinite ordering.
        # In Milestone 4, this will be replaced by Allocation/Logistics/Transit.
        self._magic_fulfillment(orders)
        
        # 5. Logging / Metrics (Simple Print)
        total_demand = np.sum(daily_demand)
        total_ordered = sum(line.quantity for order in orders for line in order.lines)
        print(f"Day {day:03}: Demand={total_demand:.1f}, Orders={total_ordered:.1f} (Bullwhip Ratio: {total_ordered/total_demand if total_demand > 0 else 0:.2f})")

    def _magic_fulfillment(self, orders: List):
        # Instantly add ordered quantity to target node's inventory
        for order in orders:
            target_idx = self.state.node_id_to_idx[order.target_id]
            for line in order.lines:
                prod_idx = self.state.product_id_to_idx[line.product_id]
                self.state.inventory[target_idx, prod_idx] += line.quantity

if __name__ == "__main__":
    sim = Orchestrator()
    sim.run(days=30)
