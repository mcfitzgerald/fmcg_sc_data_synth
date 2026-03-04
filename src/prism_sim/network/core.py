import enum
from dataclasses import dataclass, field

import numpy as np


class NodeType(enum.Enum):
    PLANT = "plant"
    DC = "dc"
    STORE = "store"
    PORT = "port"
    SUPPLIER = "supplier"


class CustomerChannel(enum.Enum):
    MASS_RETAIL = "mass_retail"    # Big retailers (Walmart DC, Target DC) - FTL
    GROCERY = "grocery"            # Traditional grocery (Kroger, Albertsons) - FTL
    CLUB = "club"                  # Club stores (Costco, Sam's) - FTL pallets
    PHARMACY = "pharmacy"          # Pharmacy chains (CVS, Walgreens) - FTL
    DISTRIBUTOR = "distributor"    # 3P Distributors (consolidate for small retailers)
    ECOMMERCE = "ecommerce"       # Amazon, pure-play digital
    DTC = "dtc"                    # Direct to consumer (parcel)


class StoreFormat(enum.Enum):
    RETAILER_DC = "retailer_dc"    # Walmart/Target distribution center
    HYPERMARKET = "hypermarket"    # Big box store
    SUPERMARKET = "supermarket"    # Traditional grocery
    CLUB = "club"                  # Costco, Sam's Club
    CONVENIENCE = "convenience"   # Small format
    PHARMACY = "pharmacy"         # CVS, Walgreens
    DISTRIBUTOR_DC = "distributor_dc"  # 3P distributor warehouse
    ECOM_FC = "ecom_fc"           # E-commerce fulfillment center
    DTC_FC = "dtc_fc"             # Direct-to-consumer fulfillment center


@dataclass
class Node:
    """Abstract base representing a location in the supply chain network."""

    id: str
    name: str
    type: NodeType
    location: str  # e.g., "Chicago, IL"
    lat: float = 0.0
    lon: float = 0.0

    # Capacity constraints
    throughput_capacity: float = float("inf")  # Units per day

    # State (will be vectorized later, but kept here for object clarity)
    current_inventory: float = 0.0

    # Customer segmentation
    channel: CustomerChannel | None = None
    store_format: StoreFormat | None = None
    parent_account_id: str | None = None  # Which retail account owns this

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Node ID cannot be empty")


@dataclass
class Link:
    """Abstract base representing a logistics route between two nodes."""

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


class OrderPriority(enum.IntEnum):
    RUSH = 1
    HIGH = 2
    STANDARD = 5
    LOW = 10


class ABCClass(enum.StrEnum):
    A = "A"
    B = "B"
    C = "C"


class OrderType(enum.Enum):
    STANDARD = "standard"      # Normal replenishment (70%)
    RUSH = "rush"              # Expedited handling (10%)
    BACKORDER = "backorder"    # Fill when available (10%)
    PROMOTIONAL = "promotional" # Promo-driven (10%)


@dataclass(slots=True)
class OrderLine:
    product_id: str
    quantity: float
    product_idx: int = -1
    unit_price: float = 0.0


@dataclass(slots=True)
class Order:
    id: str
    source_id: str  # Who is shipping (Supplier/RDC)
    target_id: str  # Who is ordering (Store/RDC)
    creation_day: int
    lines: list[OrderLine] = field(default_factory=list)
    status: str = "OPEN"  # OPEN, IN_TRANSIT, CLOSED

    # Advanced attributes
    order_type: OrderType = OrderType.STANDARD
    promo_id: str | None = None
    priority: OrderPriority = OrderPriority.STANDARD  # 1=highest, 10=lowest
    requested_date: int | None = None  # Day number

    # PERF v0.69.3: Cached integer indices to avoid repeated dict lookups
    source_idx: int = -1
    target_idx: int = -1


@dataclass(slots=True)
class OrderBatch:
    """Vectorized order representation — parallel arrays, one element per order-line.

    Replaces list[Order] for the hot path from replenishment → allocation.
    All pre-allocation consumers (record_inflow, record_order_demand,
    unconstrained demand sum, allocation demand vector) only need
    (source_idx, product_idx, quantity) which are direct array reads.

    After allocation, call materialize_orders() to get list[Order] for
    logistics (which still needs Order objects for held-order tracking).

    PERF v0.86.0: Eliminates ~50K OrderLine + ~6.5K Order object creation.
    String IDs are deferred to materialization (only for surviving lines).
    """

    # Per order-line arrays  (N = total non-zero entries from batched_qty)
    source_idx: np.ndarray      # int32[N]
    target_idx: np.ndarray      # int32[N]
    product_idx: np.ndarray     # int32[N]
    quantity: np.ndarray        # float64[N] — mutable (allocation modifies in-place)
    unit_price: np.ndarray      # float64[N]
    order_type: np.ndarray      # int8[N]  (OrderType enum → int mapping)
    priority: np.ndarray        # int8[N]  (OrderPriority int value)
    creation_day: int
    requested_date: np.ndarray  # int32[N]
    promo_ids: dict[int, str]   # target_idx → promo_id (sparse, most are None)
    # Per-target (unique) data — indexed by group-index from np.unique
    _tgt_source_ids: list[str]   # group_idx → source_id string
    _tgt_order_ids: list[str]    # group_idx → order ID string
    _tgt_inverse: np.ndarray     # int[N] — maps line → group_idx

    @property
    def n_lines(self) -> int:
        return len(self.quantity)

    def total_quantity(self) -> float:
        return float(np.sum(self.quantity))



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

    # Physics Overhaul: Track effective lead time
    original_order_day: int | None = None

    # Physicals
    total_weight_kg: float = 0.0
    total_volume_m3: float = 0.0
    total_cases: float = 0.0  # PERF v0.86.0: cached line-qty sum
    truck_count: int = 1
    emissions_kg: float = 0.0

    # PERF v0.69.3: Cached integer indices to avoid repeated dict lookups
    source_idx: int = -1
    target_idx: int = -1


# --- Manufacturing Primitives ---


class ProductionOrderStatus(enum.Enum):
    PLANNED = "planned"
    RELEASED = "released"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass
class ProductionOrder:
    """
    A request to produce a specific quantity of a product at a plant.
    Generated by MRP based on DRP requirements.
    """

    id: str
    plant_id: str
    product_id: str
    quantity_cases: float
    creation_day: int
    due_day: int
    status: ProductionOrderStatus = ProductionOrderStatus.PLANNED

    # Tracking
    planned_start_day: int | None = None
    actual_start_day: int | None = None
    actual_end_day: int | None = None
    produced_quantity: float = 0.0


class BatchStatus(enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    HOLD = "hold"
    REJECTED = "rejected"


@dataclass
class Batch:
    """
    A physical batch of produced goods with genealogy tracking.
    Links to the Production Order and tracks ingredient consumption.
    """

    id: str
    production_order_id: str
    product_id: str
    plant_id: str
    production_day: int
    quantity_cases: float
    status: BatchStatus = BatchStatus.COMPLETE

    # Genealogy - Map of ingredient_id -> quantity consumed
    ingredients_consumed: dict[str, float] = field(default_factory=dict)

    # Quality
    yield_percent: float = 100.0
    notes: str = ""


# --- Returns Primitives ---


class ReturnStatus(enum.Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    RECEIVED = "received"
    PROCESSED = "processed"
    REJECTED = "rejected"


@dataclass
class ReturnLine:
    product_id: str
    quantity_cases: float
    disposition: str = "restock"  # restock, scrap, donate


@dataclass
class Return:
    """A return request (RMA) from a store/customer back to a DC."""

    id: str
    rma_number: str
    order_id: str | None
    source_id: str  # Store (Returning)
    target_id: str  # DC (Receiving)
    creation_day: int
    lines: list[ReturnLine] = field(default_factory=list)
    status: ReturnStatus = ReturnStatus.REQUESTED

    received_day: int | None = None
    processed_day: int | None = None
    credit_amount: float = 0.0
