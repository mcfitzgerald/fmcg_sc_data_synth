import enum
from dataclasses import dataclass


class ProductCategory(enum.Enum):
    ORAL_CARE = "oral_care"  # High value-density, cubes out
    PERSONAL_WASH = "personal_wash"  # The Brick, weighs out
    HOME_CARE = "home_care"  # Fluid Heavyweight, damage risk
    INGREDIENT = "ingredient"  # Raw material (leaf node)
    BULK_INTERMEDIATE = "bulk_intermediate"  # Semi-finished (compounded bulk)


class ContainerType(enum.Enum):
    TUBE = "tube"           # Toothpaste
    BOTTLE = "bottle"       # Dish soap, body wash
    PUMP = "pump"           # Premium body wash
    POUCH = "pouch"         # Sachets, refills
    GLASS = "glass"         # Premium


class ValueSegment(enum.Enum):
    PREMIUM = "premium"     # Glass, large pump bottles
    MAINSTREAM = "mainstream"  # Standard sizes
    VALUE = "value"         # Large refills, bulk
    TRIAL = "trial"         # Sachets, travel sizes


@dataclass
class PackagingType:
    code: str               # PKG-TUBE-100
    name: str               # "100ml Tube"
    container: ContainerType
    size_ml: int
    material: str           # "plastic", "glass"
    recyclable: bool
    units_per_case: int     # Critical for logistics
    segment: ValueSegment


@dataclass
class Product:
    """Represents a SKU (Stock Keeping Unit) with physical supply chain attributes."""

    id: str
    name: str
    category: ProductCategory

    # Physical Dimensions (Per Case)
    weight_kg: float
    length_cm: float
    width_cm: float
    height_cm: float

    # Pallet Configuration
    cases_per_pallet: int
    pallet_stacking_limit: int = 1  # How many pallets can be stacked

    # Financials
    cost_per_case: float = 0.0
    price_per_case: float = 0.0

    # Extended Attributes (Fix 0B)
    brand: str | None = None
    packaging_type_id: str | None = None

    # Derived from packaging
    units_per_case: int = 1
    # case_weight_kg is already present as weight_kg

    # Segment/positioning
    value_segment: ValueSegment | None = None

    # Sustainability
    recyclable: bool = False
    material: str | None = None

    # BOM hierarchy level (0=finished SKU, 1=bulk intermediate, 2=raw material)
    bom_level: int = 0

    # PERF: Cached volume calculation (was 6M property calls)
    _volume_m3_cached: float | None = None

    @property
    def is_finished_good(self) -> bool:
        """True if this is a sellable SKU (has consumer demand, flows to stores)."""
        return self.category not in (
            ProductCategory.INGREDIENT,
            ProductCategory.BULK_INTERMEDIATE,
        )

    @property
    def is_manufacturable(self) -> bool:
        """True if this product is produced at plants (SKUs and bulk intermediates)."""
        return self.category != ProductCategory.INGREDIENT

    @property
    def volume_m3(self) -> float:
        """Returns cached volume in cubic meters."""
        if self._volume_m3_cached is None:
            self._volume_m3_cached = (
                self.length_cm * self.width_cm * self.height_cm
            ) / 1_000_000
        return self._volume_m3_cached

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Product ID cannot be empty")
        # Pre-compute volume on creation
        self._volume_m3_cached = (
            self.length_cm * self.width_cm * self.height_cm
        ) / 1_000_000


@dataclass
class Recipe:
    """Bill of Materials (BOM) for producing a Product."""

    product_id: str
    # Map of Ingredient ID -> Quantity Required per Case
    ingredients: dict[str, float]

    # Manufacturing Physics
    run_rate_cases_per_hour: float
    changeover_time_hours: float = 0.0
