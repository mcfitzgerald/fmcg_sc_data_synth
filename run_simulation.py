import argparse
import os
import time
from prism_sim.simulation.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Prism Digital Twin Simulation Runner")
    parser.add_argument("--days", type=int, default=90, help="Number of simulation days (default: 90)")
    parser.add_argument("--no-logging", action="store_true", help="Disable CSV/JSON logging (faster)")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory for output artifacts")
    
    args = parser.parse_args()
    
    enable_logging = not args.no_logging
    
    print(f"Initializing Prism Digital Twin (Days={args.days}, Logging={'Enabled' if enable_logging else 'Disabled'})...")
    
    sim = Orchestrator(enable_logging=enable_logging, output_dir=args.output_dir)
    
    print(f"Starting Simulation Run...")
    start_time = time.time()
    
    # Run the simulation
    sim.run(days=args.days)
    
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
    if enable_logging:
        report_path = os.path.join(sim.writer.output_dir, "triangle_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Triangle Report saved to {report_path}")

if __name__ == "__main__":
    main()
