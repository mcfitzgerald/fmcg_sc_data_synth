"""
Prism Digital Twin Simulation Runner.

Auto-Checkpoint Behavior (v0.33.0):
    By default, the runner uses automatic checkpointing. On first run, it performs
    a burn-in phase (default 90 days) and saves a checkpoint. Subsequent runs with
    the same config load the checkpoint and skip burn-in.

    The `--days N` flag specifies N days of steady-state data (post burn-in).

Usage:
    poetry run python run_simulation.py                    # Default 90-day data run
    poetry run python run_simulation.py --days 365         # Full year of data
    poetry run python run_simulation.py --streaming        # Enable streaming export
    poetry run python run_simulation.py --no-logging       # Fast mode (no export)
    poetry run python run_simulation.py --no-checkpoint    # Force cold-start (no auto-checkpoint)
    poetry run python run_simulation.py --warm-start path  # Use explicit snapshot
"""

import argparse
import os
import time
from pathlib import Path

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

    # Warm-start parameters (v0.33.0)
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to warm-start snapshot file (from generate_warm_start.py)",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip config hash validation for warm-start (use with caution)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable automatic checkpointing (always cold-start, no checkpoint saved)",
    )

    args = parser.parse_args()

    enable_logging = not args.no_logging

    # Validate warm-start file if provided
    warm_start_path = None
    if args.warm_start:
        warm_start_path = Path(args.warm_start)
        if not warm_start_path.exists():
            print(f"ERROR: Warm-start file not found: {warm_start_path}")
            return

    # Build mode description string
    mode_parts = []
    mode_parts.append(f"Days={args.days}")
    mode_parts.append(f"Logging={'Enabled' if enable_logging else 'Disabled'}")
    if enable_logging and args.streaming:
        mode_parts.append("Streaming=On")
        if args.format:
            mode_parts.append(f"Format={args.format}")
    if warm_start_path:
        mode_parts.append("WarmStart=On")
    if args.no_checkpoint:
        mode_parts.append("AutoCheckpoint=Off")

    print(f"Initializing Prism Digital Twin ({', '.join(mode_parts)})...")

    sim = Orchestrator(
        enable_logging=enable_logging,
        output_dir=args.output_dir,
        streaming=args.streaming if enable_logging else False,
        output_format=args.format,
        inventory_sample_rate=args.inventory_sample_rate,
        warm_start_path=str(warm_start_path) if warm_start_path else None,
        skip_warm_start_hash_check=args.skip_hash_check,
        auto_checkpoint=not args.no_checkpoint,
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
