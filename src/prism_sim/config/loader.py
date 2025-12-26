import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the benchmark manifest configuration.
    If no path is provided, looks for benchmark_manifest.json in the config directory.
    """
    if manifest_path is None:
        # Default to the file next to this script
        final_path = Path(__file__).parent / "benchmark_manifest.json"
    else:
        final_path = Path(manifest_path)

    with open(final_path, "r") as f:
        return json.load(f)
