import time
import os
from prism_sim.simulation.orchestrator import Orchestrator

def run_365_days():
    print("Initializing Prism Digital Twin...")
    # Disable CSV logging for benchmark speed (In-Memory Validation Mode)
    sim = Orchestrator(enable_logging=False)
    
    print("Starting 365-Day 'Deep NAM' Simulation Run...")
    start_time = time.time()
    
    # Run the simulation
    sim.run(days=365)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nSimulation completed in {duration:.2f} seconds.")
    
    # Generate Reports
    print("\nGenerating Artifacts...")
    
    # 1. Save SCOR-DS Data (CSV/JSON)
    sim.save_results()
    
    # 2. Generate and Print Triangle Report
    report = sim.generate_triangle_report()
    print("\n" + report + "\n")
    
    # 3. Save Triangle Report to file
    output_dir = sim.writer.output_dir
    report_path = os.path.join(output_dir, "triangle_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Triangle Report saved to {report_path}")

if __name__ == "__main__":
    run_365_days()