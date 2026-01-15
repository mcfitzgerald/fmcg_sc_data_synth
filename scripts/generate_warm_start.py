"""
Warm-Start Snapshot Generator for Prism Sim.

Runs a burn-in simulation to capture steady-state inventory and derive
initialization parameters that eliminate cold-start artifacts.

Usage:
    poetry run python scripts/generate_warm_start.py
    poetry run python scripts/generate_warm_start.py --burn-in-days 90
    poetry run python scripts/generate_warm_start.py --output snapshot.json

The snapshot contains:
1. Minimal State (must be exact):
   - actual_inventory tensor
   - perceived_inventory tensor
   - active_shipments list
   - active_production_orders list

2. Derived Parameters (for history buffer initialization):
   - avg_demand_by_product
   - avg_lead_time_by_link
   - abc_classifications
   - demand_sigma_by_product

Note: Core snapshot functions are in src/prism_sim/simulation/snapshot.py
for sharing with the Orchestrator's auto-checkpoint system.
"""

import argparse
import gzip
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism_sim.config.loader import load_manifest, load_simulation_config
from prism_sim.simulation.orchestrator import Orchestrator
from prism_sim.simulation.snapshot import (
    capture_minimal_state,
    compute_config_hash,
    derive_initialization_params,
)


def run_burn_in(burn_in_days: int, silent: bool = True) -> Orchestrator:
    """
    Run a burn-in simulation to reach steady state.

    Args:
        burn_in_days: Number of days to run (typically 90)
        silent: If True, suppress daily output

    Returns:
        Orchestrator instance at steady state
    """
    print(f"Running {burn_in_days}-day burn-in simulation...")
    start_time = time.time()

    # Create orchestrator with logging disabled for speed
    sim = Orchestrator(
        enable_logging=False,
        output_dir="data/output/burn_in_temp",
    )

    # Run simulation
    for day in range(1, burn_in_days + 1):
        sim._step(day)
        if not silent and day % 30 == 0:
            print(f"  Day {day}/{burn_in_days} complete")

    elapsed = time.time() - start_time
    print(f"Burn-in completed in {elapsed:.1f}s ({elapsed/burn_in_days:.2f}s/day)")

    return sim


def generate_snapshot(
    burn_in_days: int = 90,
    output_path: str | None = None,
    compress: bool = True,
) -> str:
    """
    Generate a warm-start snapshot.

    Args:
        burn_in_days: Number of days for burn-in simulation
        output_path: Path for output file (default: data/snapshots/)
        compress: If True, gzip compress the output

    Returns:
        Path to the generated snapshot file
    """
    # Load configs for hash
    config = load_simulation_config()
    manifest = load_manifest()
    config_hash = compute_config_hash(config, manifest)

    # Run burn-in
    sim = run_burn_in(burn_in_days)

    print("Capturing steady-state...")

    # Capture minimal state
    minimal_state = capture_minimal_state(sim)

    # Derive initialization parameters
    derived_params = derive_initialization_params(sim, burn_in_days)

    # Build snapshot
    snapshot = {
        "metadata": {
            "version": "0.33.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "burn_in_days": burn_in_days,
            "config_hash": config_hash,
            "n_nodes": sim.state.n_nodes,
            "n_products": sim.state.n_products,
            "n_shipments": len(minimal_state["active_shipments"]),
            "n_production_orders": len(minimal_state["active_production_orders"]),
        },
        "state": minimal_state,
        "derived": derived_params,
    }

    # Determine output path
    if output_path is None:
        output_dir = Path("data/snapshots")
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".json.gz" if compress else ".json"
        output_path = str(output_dir / f"warm_start_{burn_in_days}d{suffix}")

    # Write snapshot
    print(f"Writing snapshot to {output_path}...")

    if compress:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(snapshot, f, separators=(",", ":"))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

    # Report size
    file_size = Path(output_path).stat().st_size
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{file_size / 1024:.1f} KB"
    print(f"Snapshot size: {size_str}")

    # Report summary
    print("\nSnapshot Summary:")
    print(f"  Config hash: {config_hash}")
    print(f"  Burn-in days: {burn_in_days}")
    print(f"  Nodes: {snapshot['metadata']['n_nodes']}")
    print(f"  Products: {snapshot['metadata']['n_products']}")
    print(f"  Active shipments: {snapshot['metadata']['n_shipments']}")
    print(f"  Active production orders: {snapshot['metadata']['n_production_orders']}")
    n_demand_products = len(derived_params["avg_demand_by_product"])
    print(f"  Derived params: {n_demand_products} products with demand")

    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate warm-start snapshot for Prism Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python scripts/generate_warm_start.py
  poetry run python scripts/generate_warm_start.py --burn-in-days 120
  poetry run python scripts/generate_warm_start.py --output snapshot.json --no-compress
        """,
    )

    parser.add_argument(
        "--burn-in-days",
        type=int,
        default=90,
        help="Number of days for burn-in simulation (default: 90)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for snapshot file",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression (larger but human-readable)",
    )

    args = parser.parse_args()

    output_path = generate_snapshot(
        burn_in_days=args.burn_in_days,
        output_path=args.output,
        compress=not args.no_compress,
    )

    print(f"\nSnapshot saved to: {output_path}")
    print(f"Use with: poetry run python run_simulation.py --warm-start {output_path}")


if __name__ == "__main__":
    main()
