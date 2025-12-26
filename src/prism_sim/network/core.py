from dataclasses import dataclass, field
from typing import Optional, List, Dict
import enum

class NodeType(enum.Enum):
    PLANT = "plant"
    DC = "dc"
    STORE = "store"
    PORT = "port"
    SUPPLIER = "supplier"

@dataclass
class Node:
    """
    Abstract base representing a location in the supply chain network.
    """
    id: str
    name: str
    type: NodeType
    location: str  # e.g., "Chicago, IL"
    
    # Capacity constraints
    throughput_capacity: float = float('inf')  # Units per day
    storage_capacity: float = float('inf')     # Pallets
    
    # State (will be vectorized later, but kept here for object clarity)
    current_inventory: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Node ID cannot be empty")

@dataclass
class Link:
    """
    Abstract base representing a logistics route between two nodes.
    """
    id: str
    source_id: str
    target_id: str
    mode: str = "truck" # truck, rail, ocean, air
    
    # Physics
    distance_km: float = 0.0
    lead_time_days: float = 0.0
    variability_sigma: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Link ID cannot be empty")
