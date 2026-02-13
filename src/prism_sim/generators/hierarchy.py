"""Generators for Product and Location hierarchies.

Supports 3-level BOM:
  Level 2 — Raw Materials (purchased leaf nodes)
  Level 1 — Bulk Intermediates (compounded semi-finished goods)
  Level 0 — Finished SKUs (packed, shippable cases)
"""

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


# Physical dimension defaults for raw material categories
_RM_DIMENSIONS: dict[str, dict[str, int]] = {
    "BASE_BULK": {"length": 100, "width": 100, "height": 100, "pallets": 1},
    "ACTIVE_CHEM": {"length": 20, "width": 20, "height": 30, "pallets": 100},
    "PKG_PRIMARY": {"length": 10, "width": 10, "height": 10, "pallets": 1000},
    "PKG_SECONDARY": {
        "length": 15, "width": 12, "height": 8, "pallets": 1000,
    },
    "PKG_TERTIARY": {
        "length": 60, "width": 40, "height": 30, "pallets": 100,
    },
}

# Category short codes for bulk intermediate naming
_CAT_SHORT: dict[ProductCategory, str] = {
    ProductCategory.ORAL_CARE: "OC",
    ProductCategory.PERSONAL_WASH: "PW",
    ProductCategory.HOME_CARE: "HC",
}


class ProductGenerator:
    """Generates a realistic product portfolio with 3-level BOM."""

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.config = config
        self._load_profiles()
        self._load_packaging()

        # Populated by generate_bulk_intermediates()
        self._sku_to_bulk: dict[str, str] = {}
        self._bulk_to_category: dict[str, ProductCategory] = {}

    def _load_profiles(self) -> None:
        """Load category profiles from config."""
        profile_root = self.config.get("generation_profiles", {})
        profiles_config = profile_root.get("categories", {})
        self.profiles: dict[ProductCategory, CategoryProfile] = {}

        cat_map = {
            "ORAL_CARE": ProductCategory.ORAL_CARE,
            "PERSONAL_WASH": ProductCategory.PERSONAL_WASH,
            "HOME_CARE": ProductCategory.HOME_CARE,
        }

        for key, p in profiles_config.items():
            cat = cat_map.get(key)
            if cat:
                self.profiles[cat] = CategoryProfile(category=cat, **p)

        self.ingredient_profiles = self.config.get(
            "ingredient_profiles", {}
        )
        self.bom_complexity = self.config.get("bom_complexity", {})
        self.recipe_logic = self.config.get("recipe_logic", {})
        self.bulk_config = self.config.get(
            "bulk_intermediate_config", {}
        )

    def _load_packaging(self) -> None:
        """Load packaging definitions."""
        self.packaging_types: list[PackagingType] = []
        raw_pkgs = self.config.get("packaging_types", [])

        for p in raw_pkgs:
            self.packaging_types.append(
                PackagingType(
                    code=p["code"],
                    name=p.get("name", p["code"]),
                    container=ContainerType(p["container"]),
                    size_ml=p["size_ml"],
                    material=p.get("material", "plastic"),
                    recyclable=p.get("recyclable", True),
                    units_per_case=p["units_per_case"],
                    segment=ValueSegment(p["segment"]),
                )
            )

    # ------------------------------------------------------------------
    # SKU Generation (unchanged from v0.69)
    # ------------------------------------------------------------------

    def generate_products(
        self, n_skus: int = 500
    ) -> list[Product]:
        """Generate SKUs based on Brand-Pack-Variant Hierarchy."""
        products: list[Product] = []

        sim_params = self.config.get("simulation_parameters", {})
        demand_config = sim_params.get("demand", {})
        cat_profiles = demand_config.get("category_profiles", {})

        proportions = {
            ProductCategory.ORAL_CARE: 0.33,
            ProductCategory.PERSONAL_WASH: 0.33,
            ProductCategory.HOME_CARE: 0.33,
        }

        for cat_name, profile in cat_profiles.items():
            if cat_name in ProductCategory.__members__:
                cat = ProductCategory[cat_name]
                if "sku_proportion" in profile:
                    proportions[cat] = profile["sku_proportion"]

        total_prop = sum(proportions.values())
        if total_prop > 0:
            for k in proportions:
                proportions[k] /= total_prop

        targets: dict[ProductCategory, int] = {}
        allocated = 0
        categories = list(proportions.keys())

        for i, cat in enumerate(categories):
            if i == len(categories) - 1:
                targets[cat] = n_skus - allocated
            else:
                count = int(n_skus * proportions[cat])
                targets[cat] = count
                allocated += count

        gen_profiles = self.config.get("generation_profiles", {})
        variant_palettes = gen_profiles.get("variant_palettes", {})

        brands = {
            ProductCategory.ORAL_CARE: [
                "PrismWhite", "FreshSmile", "DentEx",
            ],
            ProductCategory.PERSONAL_WASH: [
                "AquaPure", "SilkTouch", "CleanEssence",
            ],
            ProductCategory.HOME_CARE: [
                "ClearWave", "HomeGuard", "PureShine",
            ],
        }

        pack_map = {
            ProductCategory.ORAL_CARE: [
                ContainerType.TUBE, ContainerType.PUMP,
            ],
            ProductCategory.PERSONAL_WASH: [
                ContainerType.BOTTLE,
                ContainerType.PUMP,
                ContainerType.POUCH,
            ],
            ProductCategory.HOME_CARE: [
                ContainerType.BOTTLE, ContainerType.POUCH,
            ],
        }

        global_counter = 0

        for category, target_count in targets.items():
            if target_count <= 0:
                continue

            valid_containers = pack_map.get(category, [])
            valid_packs = [
                p
                for p in self.packaging_types
                if p.container in valid_containers
            ]

            if not valid_packs:
                continue

            cat_products: list[Product] = []
            variants = variant_palettes.get(category.name, ["Original"])
            cat_brands = brands.get(category, ["Generic"])

            attempts = 0
            max_attempts = target_count * 10

            while (
                len(cat_products) < target_count
                and attempts < max_attempts
            ):
                attempts += 1

                product, global_counter = self._create_single_sku(
                    category,
                    cat_brands,
                    valid_packs,
                    variants,
                    cat_products,
                    global_counter,
                )

                if product:
                    products.append(product)
                    cat_products.append(product)

        return products

    def _create_single_sku(
        self,
        category: ProductCategory,
        brands: list[str],
        valid_packs: list[PackagingType],
        variants: list[str],
        existing_products: list[Product],
        global_counter: int,
    ) -> tuple[Product | None, int]:
        """Try to create a single unique SKU."""
        brand = self.rng.choice(brands)

        zipf_idx = self.rng.zipf(1.5) - 1
        pkg_idx = zipf_idx % len(valid_packs)
        pkg = valid_packs[pkg_idx]

        variant = self.rng.choice(variants)

        variant_name = (
            f"{brand} {pkg.size_ml}ml"
            f" {pkg.container.value.title()} - {variant}"
        )

        if any(p.name == variant_name for p in existing_products):
            return None, global_counter

        global_counter += 1
        sku_id = (
            f"SKU-{category.name.split('_')[0]}-{global_counter:03d}"
        )

        profile = self.profiles[category]
        specific_gravity = 1.0
        if category == ProductCategory.ORAL_CARE:
            specific_gravity = 1.3
        elif category == ProductCategory.PERSONAL_WASH:
            specific_gravity = 1.05

        net_weight_kg = (pkg.size_ml * specific_gravity) / 1000.0
        case_weight_kg = net_weight_kg * pkg.units_per_case * 1.05

        target_vol = pkg.size_ml * pkg.units_per_case * 1.2

        r1, r2, r3 = profile.dim_ratios
        x = (target_vol / (r1 * r2 * r3)) ** (1 / 3)

        product = Product(
            id=sku_id,
            name=variant_name,
            category=category,
            brand=str(brand),
            packaging_type_id=pkg.code,
            units_per_case=pkg.units_per_case,
            value_segment=pkg.segment,
            recyclable=pkg.recyclable,
            material=pkg.material,
            weight_kg=round(case_weight_kg, 2),
            length_cm=round(r1 * x, 1),
            width_cm=round(r2 * x, 1),
            height_cm=round(r3 * x, 1),
            cases_per_pallet=profile.cases_per_pallet_range[0],
            cost_per_case=round(
                self.rng.uniform(*profile.cost_range), 2
            ),
            price_per_case=round(
                self.rng.uniform(*profile.cost_range) * 1.5, 2
            ),
            bom_level=0,
        )
        return product, global_counter

    # ------------------------------------------------------------------
    # Raw Material Generation (Level 2)
    # ------------------------------------------------------------------

    def generate_ingredients(self) -> list[Product]:
        """Generate raw material products from ingredient_profiles."""
        ingredients: list[Product] = []

        for profile_key, profile in self.ingredient_profiles.items():
            prefix = profile.get("prefix", profile_key)
            types = profile.get("types", [])
            count = int(profile.get("count_per_type", 2))
            weight_range = profile.get("weight_range", [0.1, 1.0])
            cost_range = profile.get("cost_range", [1.0, 10.0])
            bom_level = int(profile.get("bom_level", 2))
            dims = _RM_DIMENSIONS.get(
                profile_key, _RM_DIMENSIONS["PKG_PRIMARY"]
            )

            for type_name in types:
                for i in range(count):
                    ing_id = f"{prefix}-{type_name}-{i + 1:03d}"
                    pretty = type_name.replace("_", " ").title()
                    name = f"{pretty} Grade {i + 1}"

                    weight = self.rng.uniform(*weight_range)
                    cost = self.rng.uniform(*cost_range)

                    ingredients.append(
                        Product(
                            id=ing_id,
                            name=name,
                            category=ProductCategory.INGREDIENT,
                            bom_level=bom_level,
                            weight_kg=round(float(weight), 3),
                            length_cm=dims["length"],
                            width_cm=dims["width"],
                            height_cm=dims["height"],
                            cases_per_pallet=dims["pallets"],
                            cost_per_case=round(float(cost), 3),
                        )
                    )

        return ingredients

    # ------------------------------------------------------------------
    # Bulk Intermediate Generation (Level 1)
    # ------------------------------------------------------------------

    def generate_bulk_intermediates(
        self, products: list[Product]
    ) -> list[Product]:
        """Generate bulk intermediates by grouping SKUs into families.

        Groups SKUs by (category, variant) so that products sharing a
        formula base (e.g. all "Mint" oral care) use the same
        intermediate. Populates ``self._sku_to_bulk`` and
        ``self._bulk_to_category`` for downstream recipe generation.

        Returns list of BULK_INTERMEDIATE Product objects.
        """
        prefix = self.bulk_config.get("prefix", "BULK")
        weight = float(self.bulk_config.get("weight_per_unit_kg", 25.0))
        cost_mult_range = self.bulk_config.get(
            "cost_multiplier_range", [0.4, 0.7]
        )

        # Group SKUs by (category, variant)
        families: dict[
            tuple[ProductCategory, str], list[Product]
        ] = {}
        for p in products:
            variant = self._extract_variant(p.name)
            key = (p.category, variant)
            if key not in families:
                families[key] = []
            families[key].append(p)

        bulk_products: list[Product] = []
        self._sku_to_bulk = {}
        self._bulk_to_category = {}
        counter = 0

        for (category, variant), sku_list in sorted(
            families.items(), key=lambda kv: (kv[0][0].name, kv[0][1])
        ):
            counter += 1
            cat_code = _CAT_SHORT.get(category, "XX")
            variant_key = (
                variant.upper().replace(" ", "_")[:12]
            )
            bulk_id = f"{prefix}-{cat_code}-{variant_key}-{counter:03d}"

            avg_cost = sum(s.cost_per_case for s in sku_list) / len(
                sku_list
            )
            cost_mult = float(self.rng.uniform(*cost_mult_range))
            bulk_cost = round(avg_cost * cost_mult, 2)

            cat_name = category.name.replace("_", " ").title()
            bulk_products.append(
                Product(
                    id=bulk_id,
                    name=f"{variant} {cat_name} Compound",
                    category=ProductCategory.BULK_INTERMEDIATE,
                    bom_level=1,
                    weight_kg=weight,
                    length_cm=50,
                    width_cm=50,
                    height_cm=50,
                    cases_per_pallet=10,
                    cost_per_case=bulk_cost,
                )
            )

            self._bulk_to_category[bulk_id] = category
            for sku in sku_list:
                self._sku_to_bulk[sku.id] = bulk_id

        return bulk_products

    # ------------------------------------------------------------------
    # Recipe Generation (3-level BOM)
    # ------------------------------------------------------------------

    def generate_recipes(
        self,
        products: list[Product],
        bulk_intermediates: list[Product],
        ingredients: list[Product],
    ) -> list[Recipe]:
        """Generate 3-level BOM recipes.

        Creates recipes for:
        1. Bulk intermediates (Level 1): RM → BULK
        2. Finished SKUs (Level 0): BULK + packaging → SKU
        """
        recipes: list[Recipe] = []
        ing_index = self._build_ingredient_index(ingredients)

        mfg = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        spof_id = mfg.get("spof", {}).get("ingredient_id", "")

        # Stage 1: Bulk intermediate recipes
        for bulk in bulk_intermediates:
            bom = self._create_bulk_recipe(bulk, ing_index, spof_id)
            profile = self.profiles.get(
                self._bulk_to_category.get(
                    bulk.id, ProductCategory.ORAL_CARE
                )
            )
            rate = (
                profile.run_rate_cases_per_hour if profile else 1000
            )
            changeover = (
                profile.changeover_time_hours if profile else 1.0
            )
            run_rate_mult = float(
                self.bulk_config.get("run_rate_multiplier", 1.0)
            )
            recipes.append(
                Recipe(
                    product_id=bulk.id,
                    ingredients=bom,
                    run_rate_cases_per_hour=rate * run_rate_mult,
                    changeover_time_hours=changeover,
                )
            )

        # Stage 2: SKU recipes
        for sku in products:
            bom = self._create_sku_recipe(sku, ing_index)
            profile = self.profiles.get(sku.category)
            rate = (
                profile.run_rate_cases_per_hour if profile else 1000
            )
            changeover = (
                profile.changeover_time_hours if profile else 1.0
            )
            recipes.append(
                Recipe(
                    product_id=sku.id,
                    ingredients=bom,
                    run_rate_cases_per_hour=rate,
                    changeover_time_hours=changeover,
                )
            )

        return recipes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ingredient_index(
        self, ingredients: list[Product]
    ) -> dict[str, dict[str, list[Product]]]:
        """Build 2-level index: profile_key → type → [Product].

        Derives the profile key and type from each ingredient's ID
        prefix, matching against ``self.ingredient_profiles``.
        """
        # Build prefix → profile_key lookup
        prefix_to_profile: dict[str, str] = {}
        for prof_key, prof in self.ingredient_profiles.items():
            prefix_to_profile[prof.get("prefix", prof_key)] = prof_key

        index: dict[str, dict[str, list[Product]]] = {}

        for ing in ingredients:
            parts = ing.id.split("-")
            if len(parts) < 3:  # noqa: PLR2004
                continue

            # Reconstruct prefix: may be multi-segment (PKG-PRI, PKG-SEC)
            # Try 2-segment prefix first, then 1-segment
            prof_key = None
            type_name = None

            if len(parts) >= 4:  # noqa: PLR2004
                two_seg = f"{parts[0]}-{parts[1]}"
                if two_seg in prefix_to_profile:
                    prof_key = prefix_to_profile[two_seg]
                    type_name = parts[2]

            if prof_key is None:
                one_seg = parts[0]
                if one_seg in prefix_to_profile:
                    prof_key = prefix_to_profile[one_seg]
                    type_name = parts[1]

            if prof_key is None or type_name is None:
                continue

            if prof_key not in index:
                index[prof_key] = {}
            if type_name not in index[prof_key]:
                index[prof_key][type_name] = []
            index[prof_key][type_name].append(ing)

        return index

    def _create_bulk_recipe(
        self,
        bulk: Product,
        ing_index: dict[str, dict[str, list[Product]]],
        spof_id: str,
    ) -> dict[str, float]:
        """Create recipe for a bulk intermediate (Level 1 → Level 2)."""
        bom: dict[str, float] = {}

        category = self._bulk_to_category.get(
            bulk.id, ProductCategory.ORAL_CARE
        )
        logic = self.recipe_logic.get(
            category.name, self.recipe_logic.get("DEFAULT", {})
        )
        base_pct = float(logic.get("base_pct", 0.9))
        active_pct = float(logic.get("active_pct", 0.1))
        preferred_bases: list[str] = logic.get("preferred_bases", [])
        preferred_actives: list[str] = logic.get(
            "preferred_actives", []
        )

        base_range = self.bulk_config.get(
            "base_components_per_formula", [3, 5]
        )
        active_range = self.bulk_config.get(
            "active_components_per_formula", [4, 7]
        )

        n_bases = int(self.rng.integers(base_range[0], base_range[1] + 1))
        n_actives = int(
            self.rng.integers(active_range[0], active_range[1] + 1)
        )

        # Select base materials (prefer category-specific)
        all_bases = ing_index.get("BASE_BULK", {})
        selected_bases = self._select_from_index(
            all_bases, preferred_bases, n_bases
        )

        total_base_wt = bulk.weight_kg * base_pct
        if selected_bases:
            for base in selected_bases:
                share = total_base_wt / len(selected_bases)
                qty = share / max(base.weight_kg, 0.001)
                bom[base.id] = round(qty, 6)

        # Select active chemicals (prefer category-specific)
        all_actives = ing_index.get("ACTIVE_CHEM", {})

        # SPOF logic: oral care bulks include the SPOF ingredient
        spof_ing = None
        if category == ProductCategory.ORAL_CARE and spof_id:
            for type_list in all_actives.values():
                for ing in type_list:
                    if ing.id == spof_id:
                        spof_ing = ing
                        break
                if spof_ing:
                    break

        selected_actives = self._select_from_index(
            all_actives, preferred_actives, n_actives, exclude_id=spof_id
        )

        if spof_ing:
            selected_actives = [
                spof_ing, *selected_actives[: n_actives - 1]
            ]

        total_active_wt = bulk.weight_kg * active_pct
        if selected_actives:
            for act in selected_actives:
                share = total_active_wt / len(selected_actives)
                qty = share / max(act.weight_kg, 0.001)
                bom[act.id] = round(qty, 6)

        return bom

    def _create_sku_recipe(
        self,
        sku: Product,
        ing_index: dict[str, dict[str, list[Product]]],
    ) -> dict[str, float]:
        """Create recipe for a finished SKU (Level 0 → Level 1 + pkg).

        Each SKU references:
        - 1 bulk intermediate (1 unit per case)
        - Primary packaging (container-type specific)
        - Secondary packaging (carton + label)
        - Tertiary packaging (shipper + shrink, fractional)
        """
        bom: dict[str, float] = {}

        # 1. Bulk intermediate
        bulk_id = self._sku_to_bulk.get(sku.id)
        if bulk_id:
            bom[bulk_id] = 1.0

        container_type = self._get_container_type(sku)
        pri = ing_index.get("PKG_PRIMARY", {})
        sec = ing_index.get("PKG_SECONDARY", {})
        ter = ing_index.get("PKG_TERTIARY", {})

        # 2. Primary packaging (varies by container type)
        if container_type == ContainerType.TUBE:
            self._add_random_component(bom, pri.get("TUBE", []), 1.0)
            self._add_random_component(bom, pri.get("CAP", []), 1.0)
            self._add_random_component(bom, pri.get("SEAL", []), 1.0)
        elif container_type == ContainerType.BOTTLE:
            self._add_random_component(
                bom, pri.get("BOTTLE", []), 1.0
            )
            self._add_random_component(bom, pri.get("CAP", []), 1.0)
        elif container_type == ContainerType.PUMP:
            self._add_random_component(
                bom, pri.get("BOTTLE", []), 1.0
            )
            self._add_random_component(bom, pri.get("PUMP", []), 1.0)
        elif container_type == ContainerType.POUCH:
            self._add_random_component(
                bom, pri.get("POUCH", []), 1.0
            )
            self._add_random_component(bom, pri.get("SEAL", []), 1.0)
        elif container_type == ContainerType.GLASS:
            self._add_random_component(
                bom, pri.get("BOTTLE", []), 1.0
            )
            self._add_random_component(bom, pri.get("CAP", []), 1.0)

        # 3. Secondary packaging
        self._add_random_component(bom, sec.get("CARTON", []), 1.0)
        self._add_random_component(bom, sec.get("LABEL", []), 1.0)

        # 4. Tertiary packaging (fractional — shared across cases)
        cases_per_shipper = 12.0
        self._add_random_component(
            bom,
            ter.get("SHIPPER", []),
            round(1.0 / cases_per_shipper, 6),
        )
        self._add_random_component(
            bom,
            ter.get("SHRINK_WRAP", []),
            round(1.0 / cases_per_shipper, 6),
        )

        return bom

    def _select_from_index(
        self,
        type_index: dict[str, list[Product]],
        preferred_types: list[str],
        n_select: int,
        *,
        exclude_id: str = "",
    ) -> list[Product]:
        """Select n components, preferring specified types."""
        preferred_pool: list[Product] = []
        for t in preferred_types:
            preferred_pool.extend(type_index.get(t, []))

        other_pool: list[Product] = []
        preferred_ids = {p.id for p in preferred_pool}
        for type_list in type_index.values():
            for p in type_list:
                if p.id not in preferred_ids:
                    other_pool.append(p)

        if exclude_id:
            preferred_pool = [
                p for p in preferred_pool if p.id != exclude_id
            ]
            other_pool = [
                p for p in other_pool if p.id != exclude_id
            ]

        selected: list[Product] = []

        # Pick from preferred first
        if preferred_pool:
            n_from_preferred = min(n_select, len(preferred_pool))
            chosen = self.rng.choice(  # type: ignore[arg-type]
                preferred_pool,
                size=n_from_preferred,
                replace=False,
            )
            selected.extend(list(chosen))

        # Fill remainder from other pool
        remaining = n_select - len(selected)
        if remaining > 0 and other_pool:
            n_from_other = min(remaining, len(other_pool))
            chosen = self.rng.choice(  # type: ignore[arg-type]
                other_pool,
                size=n_from_other,
                replace=False,
            )
            selected.extend(list(chosen))

        return selected[:n_select]

    def _get_container_type(self, sku: Product) -> ContainerType | None:
        """Resolve container type from SKU's packaging_type_id."""
        if not sku.packaging_type_id:
            return None
        pkg_def = next(
            (
                pk
                for pk in self.packaging_types
                if pk.code == sku.packaging_type_id
            ),
            None,
        )
        return pkg_def.container if pkg_def else None

    @staticmethod
    def _extract_variant(sku_name: str) -> str:
        """Extract variant name from SKU name.

        SKU names follow: "Brand SIZEml Container - Variant"
        """
        if " - " in sku_name:
            return sku_name.rsplit(" - ", 1)[-1]
        return "Original"

    def _add_random_component(
        self,
        bom: dict[str, float],
        candidates: list[Product],
        qty: float,
    ) -> None:
        """Add a random component from candidate list to BOM."""
        if candidates:
            ing = self.rng.choice(candidates)  # type: ignore
            bom[ing.id] = qty
