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
    run_rate_cases_per_hour: float
    changeover_time_hours: float


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
                self.profiles[cat] = CategoryProfile(category=cat, **p)

        self.ingredient_profiles = self.config.get("ingredient_profiles", {})
        self.bom_complexity = self.config.get("bom_complexity", {})
        self.recipe_logic = self.config.get("recipe_logic", {})

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

    def generate_ingredients(self, n_per_type: int = 5) -> list[Product]:
        """Generate a pool of ingredients based on profiles."""
        ingredients: list[Product] = []
        
        # 1. Packaging Components
        pkg_profile = self.ingredient_profiles.get("PACKAGING", {})
        pkg_types = pkg_profile.get("types", [])
        pkg_prefix = pkg_profile.get("prefix", "PKG")
        
        for p_type in pkg_types:
            for i in range(n_per_type):
                ing_id = f"{pkg_prefix}-{p_type}-{i + 1:03d}"
                name = f"Std {p_type.title()} Type {i + 1}"
                
                weight = self.rng.uniform(*pkg_profile.get("weight_range", [0.01, 0.05]))
                cost = self.rng.uniform(*pkg_profile.get("cost_range", [0.05, 0.20]))
                
                ingredients.append(Product(
                    id=ing_id,
                    name=name,
                    category=ProductCategory.INGREDIENT,
                    weight_kg=round(weight, 3),
                    length_cm=10, width_cm=10, height_cm=10, # Generic dims
                    cases_per_pallet=1000,
                    cost_per_case=round(cost, 3)
                ))

        # 2. Active Chemicals
        act_profile = self.ingredient_profiles.get("ACTIVE_CHEM", {})
        act_prefix = act_profile.get("prefix", "ACT")
        for i in range(n_per_type * 2): # More chemicals for variety
            ing_id = f"{act_prefix}-CHEM-{i + 1:03d}"
            name = f"Active Agent {i + 1}"
            
            weight = self.rng.uniform(*act_profile.get("weight_range", [0.1, 0.5]))
            cost = self.rng.uniform(*act_profile.get("cost_range", [10.0, 50.0]))
            
            ingredients.append(Product(
                id=ing_id,
                name=name,
                category=ProductCategory.INGREDIENT,
                weight_kg=round(weight, 3),
                length_cm=20, width_cm=20, height_cm=30,
                cases_per_pallet=100,
                cost_per_case=round(cost, 3)
            ))
            
        # 3. Bulk Base
        blk_profile = self.ingredient_profiles.get("BASE_BULK", {})
        blk_prefix = blk_profile.get("prefix", "BLK")
        base_types = ["WATER", "OIL", "SILICATE"]
        for b_type in base_types:
            for i in range(3): # Fewer bulk types needed
                ing_id = f"{blk_prefix}-{b_type}-{i + 1:03d}"
                name = f"Bulk {b_type.title()} Grade {i + 1}"
                
                weight = self.rng.uniform(*blk_profile.get("weight_range", [1000, 1000]))
                cost = self.rng.uniform(*blk_profile.get("cost_range", [100.0, 100.0]))
                
                ingredients.append(Product(
                    id=ing_id,
                    name=name,
                    category=ProductCategory.INGREDIENT,
                    weight_kg=weight,
                    length_cm=100, width_cm=100, height_cm=100,
                    cases_per_pallet=1,
                    cost_per_case=cost
                ))

        return ingredients

    def generate_recipes(
        self, products: list[Product], ingredients: list[Product]
    ) -> list[Recipe]:
        """Generate recipes using logic-driven semantic BOM rules."""
        recipes: list[Recipe] = []

        # Index ingredients by type/tag for fast lookup
        ing_map: dict[str, list[Product]] = {}
        for ing in ingredients:
            # Parse type from ID structure: PREFIX-TYPE-NUMBER
            parts = ing.id.split("-")
            if len(parts) >= 2:
                key = parts[1] # e.g. BOTTLE, CAP, CHEM, WATER
                if key not in ing_map:
                    ing_map[key] = []
                ing_map[key].append(ing)

        # Fallback for safety
        if not ing_map:
            return []

        for i, p in enumerate(products):
            bom: dict[str, float] = {}
            logic = self.recipe_logic.get(p.category.name, self.recipe_logic.get("DEFAULT", {}))
            base_pct = logic.get("base_pct", 0.9)
            act_pct = logic.get("active_pct", 0.1)
            
            # 1. Packaging Logic
            if p.category == ProductCategory.ORAL_CARE:
                # Tube + Cap + Box
                self._add_random_component(bom, ing_map.get("TUBE", ing_map.get("BOTTLE", [])), 1)
                self._add_random_component(bom, ing_map.get("CAP", []), 1)
                self._add_random_component(bom, ing_map.get("BOX", []), 1)
                
            elif p.category == ProductCategory.PERSONAL_WASH:
                # Wrapper + Box
                self._add_random_component(bom, ing_map.get("WRAPPER", []), 1)
                self._add_random_component(bom, ing_map.get("BOX", []), 1)
                
            elif p.category == ProductCategory.HOME_CARE:
                # Bottle + Cap + Label
                self._add_random_component(bom, ing_map.get("BOTTLE", []), 1)
                self._add_random_component(bom, ing_map.get("CAP", []), 1)
                self._add_random_component(bom, ing_map.get("LABEL", []), 1)
                
            else:
                # Fallback generic
                self._add_random_component(bom, ing_map.get("BOX", []), 1)

            # 2. Chemical Logic
            # Base (Bulk) - consume in fractions of a tote (which is 1000kg)
            # We need to convert p.weight_kg into units of the ingredient
            # If ingredient is 1000kg tote, and we need 0.9kg: usage = 0.0009
            
            base_candidates = ing_map.get("WATER", []) + ing_map.get("OIL", []) + ing_map.get("SILICATE", [])
            if base_candidates:
                base_ing = self.rng.choice(base_candidates) # type: ignore
                qty_needed_kg = p.weight_kg * base_pct
                qty_units = qty_needed_kg / base_ing.weight_kg
                bom[base_ing.id] = float(round(qty_units, 6))

            # Active (Chem)
            chem_candidates = ing_map.get("CHEM", [])
            spof_id = "ACT-CHEM-001"
            spof_ing = next((x for x in chem_candidates if x.id == spof_id), None)
            
            # Remove SPOF from general pool so it isn't picked randomly
            general_chem_candidates = [x for x in chem_candidates if x.id != spof_id]

            if general_chem_candidates:
                # Determine if this SKU gets the SPOF ingredient
                # Target: ~20% of portfolio (Premium Oral Care)
                # We can use the loop index 'i' (enumerated)
                is_premium_oral = (p.category == ProductCategory.ORAL_CARE) and (i % 5 == 0)
                
                if is_premium_oral and spof_ing:
                    # This product DEPENDS on the SPOF
                    actives = [spof_ing]
                    # Maybe add a second minor active for realism
                    if len(general_chem_candidates) > 0:
                        actives.append(self.rng.choice(general_chem_candidates)) # type: ignore
                else:
                    # Standard product - use general pool
                    n_actives = self.rng.integers(1, 3)
                    actives = self.rng.choice(general_chem_candidates, size=n_actives, replace=False) # type: ignore
                
                total_active_kg = p.weight_kg * act_pct
                for act in actives:
                    # Split weight evenly
                    qty_kg = total_active_kg / len(actives)
                    # Active ingredients defined as small units (e.g. 0.5kg bag) or drum?
                    # In generate_ingredients, Actives are 0.1-0.5kg per "case"? 
                    # Let's assume the ingredient unit is a "bag" or "drum"
                    # If generated weight is 0.5kg, and we need 0.1kg -> 0.2 units.
                    qty_units = qty_kg / act.weight_kg
                    bom[act.id] = float(round(qty_units, 6))

            # 3. Create Recipe
            profile = self.profiles.get(p.category)
            rate = profile.run_rate_cases_per_hour if profile else 1000
            changeover = profile.changeover_time_hours if profile else 1.0

            recipes.append(
                Recipe(
                    product_id=p.id,
                    ingredients=bom,
                    run_rate_cases_per_hour=rate,
                    changeover_time_hours=changeover,
                )
            )

        return recipes

    def _add_random_component(self, bom: dict[str, float], candidates: list[Product], qty: float) -> None:
        """Helper to safely add a random component from list to BOM."""
        if candidates:
            ing = self.rng.choice(candidates) # type: ignore
            bom[ing.id] = qty