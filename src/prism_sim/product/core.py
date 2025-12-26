from dataclasses import dataclass
from typing import Dict
import enum


class ProductCategory(enum.Enum):
    ORAL_CARE = "oral_care"  # High value-density, cubes out
    PERSONAL_WASH = "personal_wash"  # The Brick, weighs out
    HOME_CARE = "home_care"  # Fluid Heavyweight, damage risk
    INGREDIENT = "ingredient"  # Raw material


@dataclass
class Product:
    """
    Represents a SKU (Stock Keeping Unit) with physical supply chain attributes.
    """

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

    @property
    def volume_m3(self) -> float:
        """Calculates volume in cubic meters."""
        return (self.length_cm * self.width_cm * self.height_cm) / 1_000_000

    def __post_init__(self):
        if not self.id:
            raise ValueError("Product ID cannot be empty")


@dataclass
class Recipe:
    """
    Bill of Materials (BOM) for producing a Product.
    """

    product_id: str
    # Map of Ingredient ID -> Quantity Required per Case
    ingredients: Dict[str, float]

    # Manufacturing Physics
    run_rate_cases_per_hour: float
    changeover_time_hours: float = 0.0
