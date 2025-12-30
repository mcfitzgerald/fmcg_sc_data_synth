"""Generators for Product and Location hierarchies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from prism_sim.product.core import (
    ContainerType,
    PackagingType,
    Product,
    ProductCategory,
    Recipe,
    ValueSegment,
)

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
    """Generates a realistic product portfolio with Packaging Hierarchy and SKU variants."""

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.config = config
        self._load_profiles()
        self._load_packaging()

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

    def _load_packaging(self) -> None:
        """Load packaging definitions."""
        self.packaging_types: list[PackagingType] = []
        raw_pkgs = self.config.get("packaging_types", [])
        
        for p in raw_pkgs:
            self.packaging_types.append(PackagingType(
                code=p["code"],
                name=p.get("name", p["code"]),
                container=ContainerType(p["container"]),
                size_ml=p["size_ml"],
                material=p.get("material", "plastic"),
                recyclable=p.get("recyclable", True),
                units_per_case=p["units_per_case"],
                segment=ValueSegment(p["segment"])
            ))

    def generate_products(self, n_skus: int = 50) -> list[Product]:
        """
        Generate SKUs based on Brand-Pack Hierarchy.
        (n_skus is ignored in favor of exhaustive generation or limited by it)
        """
        products: list[Product] = []
        
        # Brand-Category Mapping
        brands = {
            ProductCategory.ORAL_CARE: "PrismWhite",
            ProductCategory.PERSONAL_WASH: "AquaPure",
            ProductCategory.HOME_CARE: "ClearWave",
        }
        
        # Packaging Suitability
        # Which packaging types fit which category?
        pack_map = {
            ProductCategory.ORAL_CARE: [ContainerType.TUBE, ContainerType.PUMP],
            ProductCategory.PERSONAL_WASH: [ContainerType.BOTTLE, ContainerType.PUMP, ContainerType.POUCH],
            ProductCategory.HOME_CARE: [ContainerType.BOTTLE, ContainerType.POUCH],
        }

        counter = 0
        
        for category, brand in brands.items():
            valid_containers = pack_map.get(category, [])
            valid_packs = [p for p in self.packaging_types if p.container in valid_containers]
            
            # For each valid packaging type, generate 1-3 variants (flavors/scents)
            for pkg in valid_packs:
                n_variants = self.rng.integers(1, 3) # 1 or 2 variants
                
                for v in range(n_variants):
                    counter += 1
                    # Stop if we exceed requested n_skus drastically, but usually we want full range
                    if n_skus > 0 and counter > n_skus * 1.5: 
                         break 

                    sku_id = f"SKU-{category.name.split('_')[0]}-{counter:03d}"
                    variant_name = f"{brand} {pkg.size_ml}ml {pkg.container.value.title()}"
                    if v > 0:
                        variant_name += f" Var-{v+1}"

                    profile = self.profiles[category]
                    
                    # Calculate weight based on size_ml and specific gravity
                    # Water = 1g/ml. Paste ~1.3. Soap ~1.1.
                    specific_gravity = 1.0
                    if category == ProductCategory.ORAL_CARE: specific_gravity = 1.3
                    elif category == ProductCategory.PERSONAL_WASH: specific_gravity = 1.05
                    
                    net_weight_kg = (pkg.size_ml * specific_gravity) / 1000.0
                    case_weight_kg = net_weight_kg * pkg.units_per_case * 1.05 # +5% packaging weight
                    
                    # Dimensions from profile scaled by size
                    # Simplified: scale base profile by cube root of volume ratio
                    base_vol = profile.avg_volume_cc
                    target_vol = pkg.size_ml * pkg.units_per_case * 1.2 # Packing factor
                    scale = (target_vol / base_vol) ** (1/3)
                    
                    r1, r2, r3 = profile.dim_ratios
                    x = (target_vol / (r1 * r2 * r3)) ** (1/3)
                    
                    products.append(Product(
                        id=sku_id,
                        name=variant_name,
                        category=category,
                        brand=brand,
                        packaging_type_id=pkg.code,
                        units_per_case=pkg.units_per_case,
                        value_segment=pkg.segment,
                        recyclable=pkg.recyclable,
                        material=pkg.material,
                        
                        weight_kg=round(case_weight_kg, 2),
                        length_cm=round(r1 * x, 1),
                        width_cm=round(r2 * x, 1),
                        height_cm=round(r3 * x, 1),
                        
                        cases_per_pallet=profile.cases_per_pallet_range[0], # Simplified
                        cost_per_case=round(self.rng.uniform(*profile.cost_range), 2),
                        price_per_case=round(self.rng.uniform(*profile.cost_range) * 1.5, 2),
                    ))

        return products[:n_skus] if n_skus > 0 else products

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
                    length_cm=10, width_cm=10, height_cm=10,
                    cases_per_pallet=1000,
                    cost_per_case=round(cost, 3)
                ))

        # 2. Active Chemicals
        act_profile = self.ingredient_profiles.get("ACTIVE_CHEM", {})
        act_prefix = act_profile.get("prefix", "ACT")
        for i in range(n_per_type * 2):
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
            for i in range(3):
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

        # Index ingredients by type/tag
        ing_map: dict[str, list[Product]] = {}
        for ing in ingredients:
            parts = ing.id.split("-")
            if len(parts) >= 2:
                key = parts[1] # e.g. BOTTLE, CAP, CHEM, WATER
                if key not in ing_map:
                    ing_map[key] = []
                ing_map[key].append(ing)

        if not ing_map:
            return []

        for i, p in enumerate(products):
            bom: dict[str, float] = {}
            logic = self.recipe_logic.get(p.category.name, self.recipe_logic.get("DEFAULT", {}))
            base_pct = logic.get("base_pct", 0.9)
            act_pct = logic.get("active_pct", 0.1)
            
            # 1. Packaging Logic
            # Use packaging_type_id to guide BOM
            # e.g., PKG-TUBE-100 -> Needs TUBE, CAP, BOX
            
            pkg_code = p.packaging_type_id
            container_type = None
            if pkg_code:
                # Find the packaging def (inefficient linear search but fine for setup)
                pkg_def = next((pk for pk in self.packaging_types if pk.code == pkg_code), None)
                if pkg_def:
                    container_type = pkg_def.container

            # Add primary container
            if container_type == ContainerType.TUBE:
                self._add_random_component(bom, ing_map.get("TUBE", []), 1)
                self._add_random_component(bom, ing_map.get("CAP", []), 1)
                self._add_random_component(bom, ing_map.get("BOX", []), 1)
            elif container_type == ContainerType.BOTTLE:
                self._add_random_component(bom, ing_map.get("BOTTLE", []), 1)
                self._add_random_component(bom, ing_map.get("CAP", []), 1)
                self._add_random_component(bom, ing_map.get("LABEL", []), 1)
            elif container_type == ContainerType.PUMP:
                self._add_random_component(bom, ing_map.get("BOTTLE", []), 1) # Use bottle as base
                self._add_random_component(bom, ing_map.get("CAP", []), 1) # Assume pump cap
            elif container_type == ContainerType.POUCH:
                 # We don't have POUCH in default ingredient types (BOTTLE, CAP, BOX...)
                 # Use WRAPPER as proxy or BOTTLE if missing
                 self._add_random_component(bom, ing_map.get("WRAPPER", ing_map.get("BOTTLE", [])), 1)
            else:
                 # Fallback
                 self._add_random_component(bom, ing_map.get("BOX", []), 1)

            # 2. Chemical Logic
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
            general_chem_candidates = [x for x in chem_candidates if x.id != spof_id]

            if general_chem_candidates:
                is_premium_oral = (p.category == ProductCategory.ORAL_CARE) and (p.value_segment == ValueSegment.PREMIUM)
                
                if is_premium_oral and spof_ing:
                    actives = [spof_ing]
                    if len(general_chem_candidates) > 0:
                        actives.append(self.rng.choice(general_chem_candidates)) # type: ignore
                else:
                    n_actives = self.rng.integers(1, 3)
                    actives = self.rng.choice(general_chem_candidates, size=n_actives, replace=False) # type: ignore
                
                total_active_kg = p.weight_kg * act_pct
                for act in actives:
                    qty_kg = total_active_kg / len(actives)
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
