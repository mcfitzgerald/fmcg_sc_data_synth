import json
from pathlib import Path
from typing import Any


def load_manifest(manifest_path: str | None = None) -> dict[str, Any]:
    """
    Loads the benchmark manifest configuration.
    If no path is provided, looks for benchmark_manifest.json in the config directory.
    """
    if manifest_path is None:
        # Default to the file next to this script
        final_path = Path(__file__).parent / "benchmark_manifest.json"
    else:
        final_path = Path(manifest_path)

    with open(final_path) as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict from {final_path}, got {type(data)}")
        return data


def load_simulation_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Loads the simulation runtime configuration.
    If no path is provided, looks for simulation_config.json in the config directory.
    """
    if config_path is None:
        # Default to the file next to this script
        final_path = Path(__file__).parent / "simulation_config.json"
    else:
        final_path = Path(config_path)

    with open(final_path) as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict from {final_path}, got {type(data)}")
        return data


def load_world_definition(config_path: str | None = None) -> dict[str, Any]:
    """
    Loads the static world definition (Products, Network, Recipes).
    If no path is provided, looks for world_definition.json in the config directory.
    """
    if config_path is None:
        # Default to the file next to this script
        final_path = Path(__file__).parent / "world_definition.json"
    else:
        final_path = Path(config_path)

    with open(final_path) as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict from {final_path}, got {type(data)}")
        return data
