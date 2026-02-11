"""
Prism Digital Twin Simulation Runner.

Usage:
    poetry run python run_simulation.py                    # Default 365-day data run
    poetry run python run_simulation.py --days 365         # Full year of data
    poetry run python run_simulation.py --streaming        # Enable streaming export
    poetry run python run_simulation.py --no-logging       # Fast mode (no export)
    poetry run python run_simulation.py --profile-memory   # Enable memory profiling
"""

import argparse
import json
import os
import time
import tracemalloc
from typing import Any

from prism_sim.simulation.orchestrator import Orchestrator

# =============================================================================
# Memory Profiling (Temporary - Remove after analysis)
# =============================================================================

# Use a mutable container to avoid global statement
_profiling_state: dict[str, Any] = {
    "enabled": False,
    "snapshots": [],
}


def log_memory(label: str) -> None:
    """Capture memory snapshot with label (only if profiling enabled)."""
    if not _profiling_state["enabled"]:
        return

    current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")[:10]

    _profiling_state["snapshots"].append(
        {
            "label": label,
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024,
            "top_allocations": [str(stat) for stat in top_stats[:5]],
        }
    )
    current_mb = current / 1024 / 1024
    peak_mb = peak / 1024 / 1024
    print(f"[MEM] {label}: {current_mb:.1f} MB (peak: {peak_mb:.1f} MB)")


def save_memory_profile(output_dir: str) -> None:
    """Save memory profile to JSON file."""
    if not _profiling_state["enabled"] or not _profiling_state["snapshots"]:
        return

    profile_path = os.path.join(output_dir, "memory_profile.json")
    with open(profile_path, "w") as f:
        json.dump(_profiling_state["snapshots"], f, indent=2)
    print(f"Memory profile saved to {profile_path}")


# =============================================================================
# End Memory Profiling
# =============================================================================


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
        help="Number of steady-state data days (default: 90)",
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

    # Streaming writer parameters
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
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling with tracemalloc (outputs memory_profile.json)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Write final-day state snapshot for warm-start (works with --no-logging).",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Load initial state from a prior run's parquet output directory. "
            "Requires: inventory.parquet, shipments.parquet, production_orders.parquet."
        ),
    )

    args = parser.parse_args()

    # Enable memory profiling if requested
    if args.profile_memory:
        _profiling_state["enabled"] = True
        tracemalloc.start()
        log_memory("startup")

    enable_logging = not args.no_logging

    # Build mode description string
    mode_parts = []
    mode_parts.append(f"Days={args.days}")
    mode_parts.append(f"Logging={'Enabled' if enable_logging else 'Disabled'}")
    if args.warm_start:
        mode_parts.append(f"WarmStart={args.warm_start}")
    if args.snapshot:
        mode_parts.append("Snapshot=On")
    if enable_logging and args.streaming:
        mode_parts.append("Streaming=On")
        if args.format:
            mode_parts.append(f"Format={args.format}")

    print(f"Initializing Prism Digital Twin ({', '.join(mode_parts)})...")

    sim = Orchestrator(
        enable_logging=enable_logging,
        output_dir=args.output_dir,
        streaming=args.streaming if args.streaming else None,
        output_format=args.format,
        inventory_sample_rate=args.inventory_sample_rate,
        memory_callback=log_memory if args.profile_memory else None,
        warm_start_dir=args.warm_start,
    )

    log_memory("after_world_build")

    print("Starting Simulation Run...")
    start_time = time.time()

    # Run the simulation
    sim.run(days=args.days)

    end_time = time.time()
    duration = end_time - start_time
    log_memory("after_simulation")
    print(f"\nSimulation completed in {duration:.2f} seconds.")

    # Save snapshot if requested (before reports, works with --no-logging)
    if args.snapshot:
        snapshot_dir = os.path.join(args.output_dir, "snapshot")
        sim.save_snapshot(snapshot_dir)

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

    # 4. Save memory profile if enabled
    if args.profile_memory:
        log_memory("final")
        save_memory_profile(args.output_dir)


if __name__ == "__main__":
    main()
