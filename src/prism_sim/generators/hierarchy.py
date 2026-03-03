"""Generators for Product and Location hierarchies.

Supports N-level BOM (2-4 levels):
  Level 2+ — Raw Materials + Sub-Intermediates (premixes)
  Level 1  — Bulk Intermediates (compounded semi-finished goods)
  Level 0  — Finished SKUs (packed, shippable cases)

Variable BOM depth (v0.84.0):
  ~70% of SKUs: 2-level (SKU → bulk → RM)
  ~20% of SKUs: 3-level (SKU → bulk → premix → RM), with diamond dependencies
  ~10% of SKUs: multi-intermediate (SKU → bulk₁ + bulk₂ → RM)
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
    specific_gravity: float = 1.0  # Density relative to water


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
    """Generates a realistic product portfolio with variable-depth BOM."""

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.config = config
        self._load_profiles()
        self._load_packaging()

        # Populated by generate_bulk_intermediates()
        self._sku_to_bulk: dict[str, str] = {}
        self._bulk_to_category: dict[str, ProductCategory] = {}

        # Populated by generate_sub_intermediates()
        self._bulk_to_premixes: dict[str, list[str]] = {}
        self._sku_secondary_bulks: dict[str, str] = {}
        self._group_b_bulk_ids: set[str] = set()
        self._group_c_bulk_ids: set[str] = set()

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
        tare_factor = self.bom_complexity.get("tare_weight_factor", 1.05)
        vol_overhead = self.bom_complexity.get("volume_overhead_factor", 1.2)

        net_weight_kg = (pkg.size_ml * profile.specific_gravity) / 1000.0
        case_weight_kg = net_weight_kg * pkg.units_per_case * tare_factor

        target_vol = pkg.size_ml * pkg.units_per_case * vol_overhead

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

            display_map: dict[str, list[str]] = profile.get("display_names", {})
            for type_name in types:
                type_names = display_map.get(type_name, [])
                for i in range(count):
                    ing_id = f"{prefix}-{type_name}-{i + 1:03d}"
                    if i < len(type_names):
                        name = type_names[i]
                    else:
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
    # Sub-Intermediate Generation (Variable BOM Depth)
    # ------------------------------------------------------------------

    def generate_sub_intermediates(
        self,
        bulk_intermediates: list[Product],
        ingredients: list[Product],
    ) -> list[Product]:
        """Generate sub-intermediates for variable BOM depth.

        Partitions bulk families into 3 groups:
        - Group A (~70%): 2-level depth, no change
        - Group B (~20%): 3-level depth via premixes (bulk → premix → RM)
        - Group C (~10%): Multi-intermediate (SKU → primary + secondary bulk)

        Returns list of new PREMIX and secondary bulk Product objects.
        """
        mlb_config = self.config.get("multi_level_bom", {})
        if not mlb_config:
            return []

        # Separate RNG stream to avoid perturbing existing generation
        sub_rng = self.rng.spawn(1)[0]

        depth_dist = mlb_config.get("depth_distribution", {})
        pct_3level = depth_dist.get("3_level", 0.20)
        pct_multi = depth_dist.get("multi_intermediate", 0.10)

        premix_prefix = mlb_config.get(
            "sub_intermediate_prefix", "PREMIX"
        )
        premix_weight = float(
            mlb_config.get("weight_per_unit_kg", 15.0)
        )
        premix_cost_range = mlb_config.get(
            "cost_multiplier_range", [0.2, 0.4]
        )

        # Deterministic partition of bulk families
        sorted_bulk_ids = sorted(b.id for b in bulk_intermediates)
        n_bulks = len(sorted_bulk_ids)
        n_3level = max(1, round(n_bulks * pct_3level))
        n_multi = max(1, round(n_bulks * pct_multi))

        # Shuffle deterministically, then assign groups
        indices = list(range(n_bulks))
        sub_rng.shuffle(indices)

        group_b_indices = set(indices[:n_3level])
        group_c_indices = set(
            indices[n_3level : n_3level + n_multi]
        )

        self._group_b_bulk_ids = {
            sorted_bulk_ids[i] for i in group_b_indices
        }
        self._group_c_bulk_ids = {
            sorted_bulk_ids[i] for i in group_c_indices
        }

        new_products: list[Product] = []
        bulk_lookup = {b.id: b for b in bulk_intermediates}
        premix_counter = 0

        # --- Group B: Generate premixes for 3-level depth ---
        for bulk_id in sorted(self._group_b_bulk_ids):
            bulk = bulk_lookup[bulk_id]
            category = self._bulk_to_category.get(
                bulk_id, ProductCategory.ORAL_CARE
            )
            cat_code = _CAT_SHORT.get(category, "XX")
            variant = bulk.name.split(" ")[0]
            variant_key = (
                variant.upper().replace(" ", "_")[:12]
            )

            n_premixes = int(sub_rng.integers(1, 3))  # 1-2
            premix_ids: list[str] = []

            for pm_idx in range(n_premixes):
                premix_counter += 1
                premix_id = (
                    f"{premix_prefix}-{cat_code}-{variant_key}"
                    f"-{premix_counter:03d}"
                )

                cost_mult = float(
                    sub_rng.uniform(*premix_cost_range)
                )
                premix_cost = round(
                    bulk.cost_per_case * cost_mult, 2
                )

                suffixes = ["Base Premix", "Active Premix"]
                name = (
                    f"{variant} {suffixes[pm_idx % len(suffixes)]}"
                )

                new_products.append(
                    Product(
                        id=premix_id,
                        name=name,
                        category=ProductCategory.BULK_INTERMEDIATE,
                        bom_level=2,
                        weight_kg=premix_weight,
                        length_cm=40,
                        width_cm=40,
                        height_cm=40,
                        cases_per_pallet=15,
                        cost_per_case=premix_cost,
                    )
                )
                premix_ids.append(premix_id)
                self._bulk_to_category[premix_id] = category

            self._bulk_to_premixes[bulk_id] = premix_ids

        # --- Group C: Secondary bulks for multi-intermediate SKUs ---
        secondary_counter = 0
        bulk_weight = float(
            self.bulk_config.get("weight_per_unit_kg", 25.0)
        )

        for bulk_id in sorted(self._group_c_bulk_ids):
            bulk = bulk_lookup[bulk_id]
            category = self._bulk_to_category.get(
                bulk_id, ProductCategory.ORAL_CARE
            )
            cat_code = _CAT_SHORT.get(category, "XX")
            variant = bulk.name.split(" ")[0]
            variant_key = (
                variant.upper().replace(" ", "_")[:12]
            )

            secondary_counter += 1
            sec_id = (
                f"BULK-{cat_code}-{variant_key}"
                f"-BLEND-{secondary_counter:03d}"
            )

            cost_mult = float(
                sub_rng.uniform(*premix_cost_range)
            )
            sec_cost = round(bulk.cost_per_case * cost_mult, 2)

            cat_name = category.name.replace("_", " ").title()
            new_products.append(
                Product(
                    id=sec_id,
                    name=f"{variant} {cat_name} Blend",
                    category=ProductCategory.BULK_INTERMEDIATE,
                    bom_level=1,
                    weight_kg=bulk_weight,
                    length_cm=50,
                    width_cm=50,
                    height_cm=50,
                    cases_per_pallet=10,
                    cost_per_case=sec_cost,
                )
            )
            self._bulk_to_category[sec_id] = category

            # Map SKUs in this family to the secondary bulk
            for sku_id, primary_bulk in self._sku_to_bulk.items():
                if primary_bulk == bulk_id:
                    self._sku_secondary_bulks[sku_id] = sec_id

        return new_products

    # ------------------------------------------------------------------
    # Recipe Generation (N-level BOM)
    # ------------------------------------------------------------------

    def generate_recipes(
        self,
        products: list[Product],
        bulk_intermediates: list[Product],
        ingredients: list[Product],
        sub_intermediates: list[Product] | None = None,
    ) -> list[Recipe]:
        """Generate N-level BOM recipes.

        Creates recipes for:
        Stage 0: Sub-intermediates (premixes → RM, secondary bulks → RM)
        Stage 1: Bulk intermediates (RM + premixes → BULK)
        Stage 2: Finished SKUs (BULK + packaging → SKU)
        """
        recipes: list[Recipe] = []
        ing_index = self._build_ingredient_index(ingredients)

        mfg = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        spof_id = mfg.get("spof", {}).get("ingredient_id", "")

        # Spawned RNG for new product recipes (avoids perturbing
        # existing bulk/SKU recipe generation downstream)
        sub_rng = self.rng.spawn(1)[0]

        # --- Stage 0: Sub-intermediate recipes ---
        sub_lookup: dict[str, Product] = {}
        if sub_intermediates:
            sub_lookup = {p.id: p for p in sub_intermediates}
            mlb_config = self.config.get("multi_level_bom", {})
            run_rate_mult_sub = float(
                mlb_config.get("run_rate_multiplier", 1.5)
            )

            for product in sub_intermediates:
                if product.bom_level == 2:  # noqa: PLR2004
                    # Premix: consumes raw materials
                    bom = self._create_premix_recipe(
                        product, ing_index, mlb_config, sub_rng
                    )
                else:
                    # Secondary bulk (bom_level=1): consumes RM
                    bom = self._create_bulk_recipe(
                        product, ing_index, spof_id="",
                        rng=sub_rng,
                    )

                profile = self.profiles.get(
                    self._bulk_to_category.get(
                        product.id, ProductCategory.ORAL_CARE
                    )
                )
                rate = (
                    profile.run_rate_cases_per_hour
                    if profile else 1000
                )
                changeover = (
                    profile.changeover_time_hours
                    if profile else 1.0
                )
                recipes.append(
                    Recipe(
                        product_id=product.id,
                        ingredients=bom,
                        run_rate_cases_per_hour=rate
                        * run_rate_mult_sub,
                        changeover_time_hours=changeover,
                    )
                )

        # --- Stage 1: Bulk intermediate recipes ---
        for bulk in bulk_intermediates:
            bom = self._create_bulk_recipe(
                bulk, ing_index, spof_id
            )

            # Group B: inject premix references
            premix_ids = self._bulk_to_premixes.get(
                bulk.id, []
            )
            if premix_ids:
                bom = self._inject_premix_refs(
                    bulk, bom, premix_ids, sub_lookup
                )

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

        # --- Stage 2: SKU recipes ---
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
        *,
        rng: Generator | None = None,
    ) -> dict[str, float]:
        """Create recipe for a bulk intermediate (Level 1 → Level 2)."""
        bom: dict[str, float] = {}
        _rng = rng or self.rng

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

        n_bases = int(_rng.integers(base_range[0], base_range[1] + 1))
        n_actives = int(
            _rng.integers(active_range[0], active_range[1] + 1)
        )

        # Select base materials (prefer category-specific)
        all_bases = ing_index.get("BASE_BULK", {})
        selected_bases = self._select_from_index(
            all_bases, preferred_bases, n_bases, rng=_rng
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
            all_actives, preferred_actives, n_actives,
            exclude_id=spof_id, rng=_rng,
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
        - 1-2 bulk intermediates (primary + optional secondary for Group C)
        - Primary packaging (container-type specific)
        - Secondary packaging (carton + label)
        - Tertiary packaging (shipper + shrink, fractional)
        """
        bom: dict[str, float] = {}

        # 1. Bulk intermediate(s)
        bulk_id = self._sku_to_bulk.get(sku.id)
        sec_bulk_id = self._sku_secondary_bulks.get(sku.id)

        if bulk_id:
            if sec_bulk_id:
                # Group C: co-blended (primary 0.7 + secondary 0.3)
                bom[bulk_id] = 0.7
                bom[sec_bulk_id] = 0.3
            else:
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
        cases_per_shipper = self.bom_complexity.get("cases_per_shipper", 12.0)
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
        rng: Generator | None = None,
    ) -> list[Product]:
        """Select n components, preferring specified types."""
        _rng = rng or self.rng

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
            chosen = _rng.choice(  # type: ignore[arg-type]
                preferred_pool,
                size=n_from_preferred,
                replace=False,
            )
            selected.extend(list(chosen))

        # Fill remainder from other pool
        remaining = n_select - len(selected)
        if remaining > 0 and other_pool:
            n_from_other = min(remaining, len(other_pool))
            chosen = _rng.choice(  # type: ignore[arg-type]
                other_pool,
                size=n_from_other,
                replace=False,
            )
            selected.extend(list(chosen))

        return selected[:n_select]

    def _create_premix_recipe(
        self,
        premix: Product,
        ing_index: dict[str, dict[str, list[Product]]],
        mlb_config: dict[str, Any],
        rng: Generator,
    ) -> dict[str, float]:
        """Create recipe for a sub-intermediate premix (RM → premix).

        Premixes blend active chemicals and some base materials.
        Diamond dependencies emerge when premix and parent bulk share RMs.
        """
        bom: dict[str, float] = {}

        components_range = mlb_config.get(
            "components_per_sub_intermediate", [2, 4]
        )
        n_components = int(
            rng.integers(
                components_range[0], components_range[1] + 1
            )
        )

        category = self._bulk_to_category.get(
            premix.id, ProductCategory.ORAL_CARE
        )
        logic = self.recipe_logic.get(
            category.name, self.recipe_logic.get("DEFAULT", {})
        )

        preferred_actives = logic.get("preferred_actives", [])
        all_actives = ing_index.get("ACTIVE_CHEM", {})
        preferred_bases = logic.get("preferred_bases", [])
        all_bases = ing_index.get("BASE_BULK", {})

        # ~60% actives, ~40% bases for diamond dependency potential
        n_actives = max(1, round(n_components * 0.6))
        n_bases = n_components - n_actives

        selected_actives = self._select_from_index(
            all_actives, preferred_actives, n_actives, rng=rng
        )
        selected_bases = (
            self._select_from_index(
                all_bases, preferred_bases, n_bases, rng=rng
            )
            if n_bases > 0
            else []
        )

        all_selected = selected_actives + selected_bases
        if all_selected:
            share = premix.weight_kg / len(all_selected)
            for comp in all_selected:
                qty = share / max(comp.weight_kg, 0.001)
                bom[comp.id] = round(qty, 6)

        return bom

    def _inject_premix_refs(
        self,
        bulk: Product,
        bom: dict[str, float],
        premix_ids: list[str],
        premix_lookup: dict[str, Product],
    ) -> dict[str, float]:
        """Replace some active chemical inputs with premix references.

        For Group B bulks, routes ~50% of the active chemical weight
        through premixes. Diamond dependencies emerge naturally from
        shared raw materials between premix and remaining direct inputs.
        """
        category = self._bulk_to_category.get(
            bulk.id, ProductCategory.ORAL_CARE
        )
        logic = self.recipe_logic.get(
            category.name, self.recipe_logic.get("DEFAULT", {})
        )
        active_pct = float(logic.get("active_pct", 0.1))

        # Identify active chemical entries in BOM
        active_keys = [k for k in bom if k.startswith("ACT-")]
        if not active_keys:
            return bom

        # Remove ~50% of active entries
        n_to_remove = max(1, len(active_keys) // 2)
        keys_to_remove = active_keys[-n_to_remove:]

        for key in keys_to_remove:
            del bom[key]

        # Weight budget routed through premixes: ~50% of active portion
        routed_weight = bulk.weight_kg * active_pct * 0.5

        # Add premix references (qty = mass / premix_weight / n_premixes)
        for premix_id in premix_ids:
            premix = premix_lookup.get(premix_id)
            p_weight = premix.weight_kg if premix else 15.0
            qty = routed_weight / (p_weight * len(premix_ids))
            bom[premix_id] = round(qty, 6)

        return bom

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
