"""Generators for Product and Location hierarchies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from prism_sim.product.core import Product, ProductCategory, Recipe

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass
class CategoryProfile:
    """Physical characteristics profile for a product category."""

    category: ProductCategory
    avg_weight_kg: float
    avg_volume_cc: float  # Cubic centimeters
    cases_per_pallet_range: tuple[int, int]
    cost_range: tuple[float, float]
    dim_ratios: tuple[float, float, float]
    run_rate: float
    changeover: float


class ProductGenerator:
    """Generates a realistic product portfolio using Zipfian popularity."""

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.config = config
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load category profiles from config."""
        profile_root = self.config.get("generation_profiles", {})
        profiles_config = profile_root.get("categories", {})
        self.profiles: dict[ProductCategory, CategoryProfile] = {}

        # Mapping string keys to Enum
        cat_map = {
            "ORAL_CARE": ProductCategory.ORAL_CARE,
            "PERSONAL_WASH": ProductCategory.PERSONAL_WASH,
            "HOME_CARE": ProductCategory.HOME_CARE,
        }

        for key, p in profiles_config.items():
            cat = cat_map.get(key)
            if cat:
                self.profiles[cat] = CategoryProfile(**p)

    def generate_products(self, n_skus: int = 50) -> list[Product]:
        """Generate a list of products distributed across categories."""
        products: list[Product] = []
        cats = list(self.profiles.keys())

        for i in range(n_skus):
            category = cats[i % len(cats)]
            profile = self.profiles[category]

            sku_id = f"SKU-{category.name.split('_')[0]}-{i + 1:03d}"
            name = f"Prism {category.name.replace('_', ' ').title()} {i + 1}"

            noise = self.rng.normal(1.0, 0.1)
            weight = profile.avg_weight_kg * noise
            vol_cc = profile.avg_volume_cc * noise

            r1, r2, r3 = profile.dim_ratios
            x = (vol_cc / (r1 * r2 * r3)) ** (1 / 3)

            length = r1 * x
            width = r2 * x
            height = r3 * x

            cases_pal = int(self.rng.integers(*profile.cases_per_pallet_range))
            cost = self.rng.uniform(*profile.cost_range)
            price = cost * self.rng.uniform(1.3, 1.8)

            products.append(
                Product(
                    id=sku_id,
                    name=name,
                    category=category,
                    weight_kg=round(weight, 2),
                    length_cm=round(length, 1),
                    width_cm=round(width, 1),
                    height_cm=round(height, 1),
                    cases_per_pallet=cases_pal,
                    cost_per_case=round(cost, 2),
                    price_per_case=round(price, 2),
                )
            )

        return products

    def generate_recipes(
        self, products: list[Product], ingredients: list[Product]
    ) -> list[Recipe]:
        """Generate recipes for the given products."""
        recipes: list[Recipe] = []

        base_ing = next((i for i in ingredients if "BASE" in i.id), ingredients[0])
        active_ing = next((i for i in ingredients if "SURF" in i.id), ingredients[-1])

        for p in products:
            qty_base = p.weight_kg * 0.9
            qty_active = p.weight_kg * 0.1

            profile = self.profiles.get(p.category)
            rate = profile.run_rate if profile else 1000
            changeover = profile.changeover if profile else 1.0

            recipes.append(
                Recipe(
                    product_id=p.id,
                    ingredients={
                        base_ing.id: round(qty_base, 3),
                        active_ing.id: round(qty_active, 3),
                    },
                    run_rate_cases_per_hour=rate,
                    changeover_time_hours=changeover,
                )
            )

        return recipes
