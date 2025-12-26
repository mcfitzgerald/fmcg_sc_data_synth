import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the benchmark manifest configuration.
    If no path is provided, looks for benchmark_manifest.json in the config directory.
    """
    if manifest_path is None:
        # Default to the file next to this script
        current_dir = Path(__file__).parent
        manifest_path = current_dir / "benchmark_manifest.json"
    
    with open(manifest_path, 'r') as f:
        return json.load(f)
