"""Generators for Product and Location hierarchies."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from prism_sim.generators.distributions import zipf_weights
from prism_sim.product.core import Product, ProductCategory, Recipe

if TYPE_CHECKING:
    from numpy.random import Generator


class ProductArchetype(enum.Enum):
    PREMIUM = "premium"
    MAIN_STREAM = "mainstream"
    VALUE = "value"


@dataclass
class CategoryProfile:
    """Physical characteristics profile for a product category."""

    category: ProductCategory
    avg_weight_kg: float
    avg_volume_cc: float  # Cubic centimeters
    cases_per_pallet_range: tuple[int, int]
    cost_range: tuple[float, float]
    # Dimensions (L, W, H) in cm - approximate
    dim_ratios: tuple[float, float, float] = (1.0, 0.8, 0.5)


PROFILES = {
    ProductCategory.ORAL_CARE: CategoryProfile(
        category=ProductCategory.ORAL_CARE,
        avg_weight_kg=1.2,  # 12 pack toothpaste
        avg_volume_cc=3000,  # 20x15x10
        cases_per_pallet_range=(100, 140),
        cost_range=(20.0, 40.0),
        dim_ratios=(2.0, 1.5, 1.0)
    ),
    ProductCategory.PERSONAL_WASH: CategoryProfile(
        category=ProductCategory.PERSONAL_WASH,
        avg_weight_kg=8.5,  # 48 pack soap
        avg_volume_cc=9000,  # 30x20x15
        cases_per_pallet_range=(60, 100),
        cost_range=(15.0, 25.0),
        dim_ratios=(2.0, 1.3, 1.0)
    ),
    ProductCategory.HOME_CARE: CategoryProfile(
        category=ProductCategory.HOME_CARE,
        avg_weight_kg=6.0,  # 6 pack detergent
        avg_volume_cc=26000, # 35x25x30
        cases_per_pallet_range=(30, 50),
        cost_range=(10.0, 18.0),
        dim_ratios=(1.2, 0.8, 1.0)
    ),
}


class ProductGenerator:
    """Generates a realistic product portfolio using Zipfian popularity."""

    def __init__(self, seed: int = 42) -> None:
        self.rng: Generator = np.random.default_rng(seed)

    def generate_products(self, n_skus: int = 50) -> list[Product]:
        """
        Generate a list of products distributed across categories.

        Args:
            n_skus: Total number of SKUs to generate.

        Returns:
            List of Product objects.
        """
        products: list[Product] = []

        # Split SKUs across categories (approximate thirds)
        cats = [
            ProductCategory.ORAL_CARE,
            ProductCategory.PERSONAL_WASH,
            ProductCategory.HOME_CARE
        ]

        # Generate popularity weights for the entire portfolio
        zipf_weights(n_skus)

        for i in range(n_skus):
            # Round-robin category assignment
            category = cats[i % len(cats)]
            profile = PROFILES[category]

            # ID and Name
            sku_id = f"SKU-{category.name.split('_')[0]}-{i+1:03d}"
            name = f"Prism {category.name.replace('_', ' ').title()} {i+1}"

            # Physicals (add some noise)
            noise = self.rng.normal(1.0, 0.1)
            weight = profile.avg_weight_kg * noise

            # Calculate dimensions from volume
            vol_cc = profile.avg_volume_cc * noise
            # Solve for dims: L*W*H = vol, L=r1*x, W=r2*x, H=r3*x
            r1, r2, r3 = profile.dim_ratios
            x = (vol_cc / (r1 * r2 * r3)) ** (1/3)

            length = r1 * x
            width = r2 * x
            height = r3 * x

            # Rounding
            cases_pal = int(self.rng.integers(*profile.cases_per_pallet_range))

            # Financials (correlated with popularity/premiumness?)
            # Let's say less popular items are slightly more expensive (niche)
            # or more popular are cheaper. Using random for now within range.
            cost = self.rng.uniform(*profile.cost_range)
            price = cost * self.rng.uniform(1.3, 1.8)  # 30-80% margin

            products.append(Product(
                id=sku_id,
                name=name,
                category=category,
                weight_kg=round(weight, 2),
                length_cm=round(length, 1),
                width_cm=round(width, 1),
                height_cm=round(height, 1),
                cases_per_pallet=cases_pal,
                cost_per_case=round(cost, 2),
                price_per_case=round(price, 2)
            ))

        return products

    def generate_recipes(
        self,
        products: list[Product],
        ingredients: list[Product]
    ) -> list[Recipe]:
        """
        Generate recipes for the given products.

        Args:
            products: List of finished goods.
            ingredients: List of raw materials.

        Returns:
            List of Recipe objects.
        """
        recipes: list[Recipe] = []

        # Identify common ingredients (Water, Surfactant)
        base_ing = next((i for i in ingredients if "BASE" in i.id), ingredients[0])
        active_ing = next((i for i in ingredients if "SURF" in i.id), ingredients[-1])

        for p in products:
            # Simple recipe logic: 90% base, 10% active
            # Adjusted by product weight

            # Assuming ingredients are fungible/bulk for this generator level
            # We map 1 case of FG to X units of Ingredients.
            # Usually recipes are defined per unit output (e.g. per case)

            # Let's assume ingredients are consumed in 'units' where
            # 1 unit ~ 1 kg for simplicity
            # or strictly follow the schema: map ingredient ID to Quantity

            qty_base = p.weight_kg * 0.9
            qty_active = p.weight_kg * 0.1

            # Run rates vary by category
            if p.category == ProductCategory.ORAL_CARE:
                rate = 1500
                changeover = 2.0
            elif p.category == ProductCategory.PERSONAL_WASH:
                rate = 1800
                changeover = 3.0
            else:
                rate = 1200
                changeover = 4.0

            recipes.append(Recipe(
                product_id=p.id,
                ingredients={
                    base_ing.id: round(qty_base, 3),
                    active_ing.id: round(qty_active, 3)
                },
                run_rate_cases_per_hour=rate,
                changeover_time_hours=changeover
            ))

        return recipes
