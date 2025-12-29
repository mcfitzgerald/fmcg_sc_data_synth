"""
Prism Digital Twin Simulation Runner.

Usage:
    poetry run python run_simulation.py                    # Default 90-day run
    poetry run python run_simulation.py --days 365         # Full year
    poetry run python run_simulation.py --streaming        # Enable streaming export
    poetry run python run_simulation.py --no-logging       # Fast mode (no export)
"""

import argparse
import os
import time

from prism_sim.simulation.orchestrator import Orchestrator


def main() -> None:
    """Run the Prism Digital Twin simulation."""
    parser = argparse.ArgumentParser(
        description="Prism Digital Twin Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python run_simulation.py --days 30 --no-logging  # Fast test
  poetry run python run_simulation.py --days 365 --streaming  # Full year
  poetry run python run_simulation.py --streaming --format parquet
        """,
    )

    # Core simulation parameters
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of simulation days (default: 90)",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable CSV/JSON logging (faster)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Directory for output artifacts",
    )

    # Streaming writer parameters (Task 7.3)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode for large runs (writes incrementally to disk)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default=None,
        help="Output format: csv (default) or parquet (requires pyarrow)",
    )
    parser.add_argument(
        "--inventory-sample-rate",
        type=int,
        default=None,
        help="Log inventory every N days (1=daily, 7=weekly). Reduces data volume.",
    )

    args = parser.parse_args()

    enable_logging = not args.no_logging

    # Build mode description string
    mode_parts = []
    mode_parts.append(f"Days={args.days}")
    mode_parts.append(f"Logging={'Enabled' if enable_logging else 'Disabled'}")
    if enable_logging and args.streaming:
        mode_parts.append("Streaming=On")
        if args.format:
            mode_parts.append(f"Format={args.format}")

    print(f"Initializing Prism Digital Twin ({', '.join(mode_parts)})...")

    sim = Orchestrator(
        enable_logging=enable_logging,
        output_dir=args.output_dir,
        streaming=args.streaming if enable_logging else False,
        output_format=args.format,
        inventory_sample_rate=args.inventory_sample_rate,
    )

    print("Starting Simulation Run...")
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
        report_path = os.path.join(str(sim.writer.output_dir), "triangle_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Triangle Report saved to {report_path}")


if __name__ == "__main__":
    main()
