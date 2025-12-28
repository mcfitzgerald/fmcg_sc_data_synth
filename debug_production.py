
import time
import os
import json
from prism_sim.simulation.orchestrator import Orchestrator
from prism_sim.config.loader import load_simulation_config

class DebugOrchestrator(Orchestrator):
    """Orchestrator with overridden config for Cold Start debugging."""
    def __init__(self):
        # Initialize normally first (loads default config)
        super().__init__(enable_logging=True)
        
        # Override inventory config to force immediate production
        # RDC Inventory (8 days) is close to Reorder Point (7 days)
        self.config["simulation_parameters"]["inventory"]["initialization"]["rdc_days_supply"] = 8.0
        
        # Re-initialize inventory with new settings
        # We need to reset the state first
        self.state.actual_inventory.fill(0)
        self.state.perceived_inventory.fill(0)
        self._initialize_inventory()

def run_debug_sim():
    print("Initializing Debug Simulation (Cold Start)...")
    sim = DebugOrchestrator()
    
    print("Starting 10-Day Run (Expect Production on Day 1 or 2)...")
    sim.run(days=10)
    
    # Save results
    sim.save_results()
    print("\n" + sim.generate_triangle_report())

if __name__ == "__main__":
    run_debug_sim()
