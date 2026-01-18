#!/usr/bin/env python3
"""
Standard Simulation Runner for Prism Sim.
Enforces the codified workflow: Config -> World Check -> Warm Start -> Run.

Workflow:
1. Load Config & Manifest.
2. Check if static world exists (data/output/static_world).
3. Compute Config Hash.
4. Check for Checkpoint (data/checkpoints/steady_state_{hash}.json.gz).
5. Run Simulation (Orchestrator handles burn-in vs warm-start internally).
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prism_sim.simulation.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run a standard Prism Sim workflow.")
    parser.add_argument("--days", type=int, default=365, help="Steady-state data days to simulate.")
    parser.add_argument("--output-dir", type=str, default="data/output/standard_run", help="Output directory.")
    parser.add_argument("--no-logging", action="store_true", help="Disable CSV/Parquet logging for speed.")
    parser.add_argument("--inventory-sample-rate", type=int, default=7, help="Log inventory every N days (default weekly).")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate the static world topology.")
    parser.add_argument("--recalibrate", action="store_true", help="Run the physics calibration script.")
    
    args = parser.parse_args()

    print("==================================================")
    print("        PRISM SIM: STANDARD WORKFLOW RUNNER       ")
    print("==================================================")

    import subprocess

    # 1. (Optional) Rebuild World
    if args.rebuild:
        print(">>> Step 1: Regenerating Static World...")
        subprocess.run(["poetry", "run", "python", "scripts/generate_static_world.py"], check=True)

    # 2. (Optional) Recalibrate
    if args.recalibrate:
        print(">>> Step 2: Running Physics Calibration...")
        subprocess.run(["poetry", "run", "python", "scripts/calibrate_config.py", "--apply"], check=True)

    print(">>> Step 3: Initializing Simulation...")
    # 3. Initialize Orchestrator
    # Note: Orchestrator internally handles config hash, checkpoint loading, 
    # and burn-in if needed.
    sim = Orchestrator(
        enable_logging=not args.no_logging,
        output_dir=args.output_dir,
        inventory_sample_rate=args.inventory_sample_rate,
        auto_checkpoint=True
    )

    # 2. Execute Run
    # If checkpoint exists, it runs for 'days'.
    # If not, it runs 'burn-in' + 'days'.
    sim.run(days=args.days)

    # 3. Finalize
    sim.save_results()
    print("\n" + sim.generate_triangle_report())
    print("\nWorkflow Complete.")


if __name__ == "__main__":
    main()
