import enum
from dataclasses import dataclass, field


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
    throughput_capacity: float = float("inf")  # Units per day
    storage_capacity: float = float("inf")  # Pallets

    # State (will be vectorized later, but kept here for object clarity)
    current_inventory: float = 0.0

    def __post_init__(self) -> None:
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
    mode: str = "truck"  # truck, rail, ocean, air

    # Physics
    distance_km: float = 0.0
    lead_time_days: float = 0.0
    variability_sigma: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Link ID cannot be empty")


@dataclass
class OrderLine:
    product_id: str
    quantity: float


@dataclass
class Order:
    id: str
    source_id: str  # Who is shipping (Supplier/RDC)
    target_id: str  # Who is ordering (Store/RDC)
    creation_day: int
    lines: list[OrderLine] = field(default_factory=list)
    status: str = "OPEN"  # OPEN, IN_TRANSIT, CLOSED


class ShipmentStatus(enum.Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"


@dataclass
class Shipment:
    id: str
    source_id: str
    target_id: str
    creation_day: int
    arrival_day: int
    lines: list[OrderLine] = field(default_factory=list)
    status: ShipmentStatus = ShipmentStatus.PENDING

    # Physicals
    total_weight_kg: float = 0.0
    total_volume_m3: float = 0.0
    truck_count: int = 1
