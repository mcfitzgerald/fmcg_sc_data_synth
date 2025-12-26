"""
Bottleneck Fixes - Integration layer for O(1) lookups and batch Faker sampling.

This module provides ready-to-use components that integrate StaticDataPool
and LookupBuilder to eliminate the critical bottlenecks in generate_data.py.

Critical Bottlenecks Fixed:
| Location     | Problem                            | Operations Before | After |
|--------------|------------------------------------|--------------------|-------|
| Lines ~2380  | GR lines scan PO lines per GR      | 20K × 75K = 1.5B  | O(1)  |
| Lines ~2409  | WO materials scan formula_ings     | 50K × 1.5K = 75M  | O(1)  |
| Lines ~2523  | Batch ingredients same pattern     | 50K × 1.5K = 75M  | O(1)  |
| Lines ~2770  | Orders scan retail_locations       | 200K × 10K = 2B   | O(1)  |
| ~30 Faker    | Per-row string generation          | ~800K calls       | Batch |

Usage:
    # Initialize once at generator startup
    cache = LookupCache()
    faker = PooledFaker(seed=42)

    # Build indices when data is available
    cache.build_from_data(generator.data)

    # Use O(1) lookups instead of list comprehensions
    po_lines = cache.po_lines_by_po.get(gr["po_id"])
    formula_ings = cache.formula_ings_by_formula.get(wo["formula_id"])
    locations = cache.locations_by_account.get(account_id)

    # Use batch Faker sampling
    names = faker.names(1000)
    cities = faker.cities(500)
"""

from __future__ import annotations

from typing import Any

from .lookup_builder import LookupBuilder, LookupIndex
from .static_pool import StaticDataPool


class LookupCache:
    """
    Pre-built lookup indices for O(1) FK access during generation.

    Centralizes all lookup indices needed by generate_data.py to eliminate
    O(N) list comprehensions. Build once after each level's data is ready,
    then use O(1) lookups for all subsequent references.

    Attributes:
        po_lines_by_po: PO lines indexed by po_id (Level 6)
        formula_ings_by_formula: Formula ingredients by formula_id (Level 6, 7)
        formulas_by_id: Formulas by id (Level 6, 7)
        locations_by_account: Retail locations by account_id (Level 8)
        accounts_by_id: Retail accounts by id (Level 8)
        order_lines_by_order: Order lines by order_id (Level 9, 10)
        batches_by_sku: Batches by sku_id (Level 10, 11)
        batches_by_formula: Batches by formula_id (Level 7)
        promotions_by_date_range: Promotions indexed for date lookup (Level 8)
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        # Level 5-6: Procurement
        self.po_lines_by_po: LookupIndex | None = None
        self.gr_lines_by_gr: LookupIndex | None = None

        # Level 2, 6-7: Formulas and ingredients
        self.formula_ings_by_formula: LookupIndex | None = None
        self.formulas_by_id: dict[int, dict] | None = None

        # Level 3, 8: Locations and accounts
        self.locations_by_account: LookupIndex | None = None
        self.accounts_by_id: dict[int, dict] | None = None

        # Level 8-9: Orders
        self.order_lines_by_order: LookupIndex | None = None

        # Level 6-7, 10-11: Batches
        self.batches_by_sku: LookupIndex | None = None
        self.batches_by_formula: LookupIndex | None = None

        # Level 4: Promotions
        self.promotions_list: list[dict] | None = None

        # Track what's been built
        self._built_indices: set[str] = set()

    def build_from_data(self, data: dict[str, list[dict]]) -> None:
        """
        Build all available indices from generator data.

        Call this after each level completes to make indices available
        for subsequent levels.

        Args:
            data: The generator's self.data dict
        """
        # Level 5-6: PO lines
        if "purchase_order_lines" in data and "po_lines_by_po" not in self._built_indices:
            self.po_lines_by_po = LookupBuilder.build_po_lines_by_po_id(
                data["purchase_order_lines"]
            )
            self._built_indices.add("po_lines_by_po")

        # Level 6: GR lines
        if "goods_receipt_lines" in data and "gr_lines_by_gr" not in self._built_indices:
            self.gr_lines_by_gr = LookupBuilder.build_gr_lines_by_gr_id(
                data["goods_receipt_lines"]
            )
            self._built_indices.add("gr_lines_by_gr")

        # Level 2: Formula ingredients
        if "formula_ingredients" in data and "formula_ings_by_formula" not in self._built_indices:
            self.formula_ings_by_formula = LookupBuilder.build_formula_ings_by_formula_id(
                data["formula_ingredients"]
            )
            self._built_indices.add("formula_ings_by_formula")

        # Level 2: Formulas (unique index)
        if "formulas" in data and "formulas_by_id" not in self._built_indices:
            self.formulas_by_id = LookupBuilder.build_unique(data["formulas"], "id")
            self._built_indices.add("formulas_by_id")

        # Level 3: Retail locations by account
        if "retail_locations" in data and "locations_by_account" not in self._built_indices:
            self.locations_by_account = LookupBuilder.build_locations_by_account_id(
                data["retail_locations"]
            )
            self._built_indices.add("locations_by_account")

        # Level 3: Retail accounts (unique index)
        if "retail_accounts" in data and "accounts_by_id" not in self._built_indices:
            self.accounts_by_id = LookupBuilder.build_unique(data["retail_accounts"], "id")
            self._built_indices.add("accounts_by_id")

        # Level 9: Order lines
        if "order_lines" in data and "order_lines_by_order" not in self._built_indices:
            self.order_lines_by_order = LookupBuilder.build_order_lines_by_order_id(
                data["order_lines"]
            )
            self._built_indices.add("order_lines_by_order")

        # Level 6-7: Batches
        if "batches" in data:
            if "batches_by_sku" not in self._built_indices:
                self.batches_by_sku = LookupBuilder.build_batches_by_sku_id(
                    data["batches"]
                )
                self._built_indices.add("batches_by_sku")

            if "batches_by_formula" not in self._built_indices:
                self.batches_by_formula = LookupBuilder.build(
                    data["batches"], "formula_id"
                )
                self._built_indices.add("batches_by_formula")

        # Level 4: Promotions (keep as list for date filtering)
        if "promotions" in data and "promotions_list" not in self._built_indices:
            self.promotions_list = data["promotions"]
            self._built_indices.add("promotions_list")

    def rebuild_index(self, index_name: str, data: dict[str, list[dict]]) -> None:
        """
        Rebuild a specific index (useful after data modifications).

        Args:
            index_name: Name of index to rebuild
            data: The generator's self.data dict
        """
        if index_name in self._built_indices:
            self._built_indices.remove(index_name)
        self.build_from_data(data)

    def get_po_lines(self, po_id: int) -> list[dict]:
        """Get PO lines for a purchase order (O(1))."""
        if self.po_lines_by_po is None:
            raise RuntimeError("po_lines_by_po index not built - call build_from_data first")
        return self.po_lines_by_po.get(po_id)

    def get_formula_ingredients(self, formula_id: int) -> list[dict]:
        """Get formula ingredients for a formula (O(1))."""
        if self.formula_ings_by_formula is None:
            raise RuntimeError("formula_ings_by_formula index not built")
        return self.formula_ings_by_formula.get(formula_id)

    def get_formula(self, formula_id: int) -> dict | None:
        """Get formula by ID (O(1))."""
        if self.formulas_by_id is None:
            raise RuntimeError("formulas_by_id index not built")
        return self.formulas_by_id.get(formula_id)

    def get_account_locations(self, account_id: int) -> list[dict]:
        """Get retail locations for an account (O(1))."""
        if self.locations_by_account is None:
            raise RuntimeError("locations_by_account index not built")
        return self.locations_by_account.get(account_id)

    def get_account(self, account_id: int) -> dict | None:
        """Get retail account by ID (O(1))."""
        if self.accounts_by_id is None:
            raise RuntimeError("accounts_by_id index not built")
        return self.accounts_by_id.get(account_id)

    def get_order_lines(self, order_id: int) -> list[dict]:
        """Get order lines for an order (O(1))."""
        if self.order_lines_by_order is None:
            raise RuntimeError("order_lines_by_order index not built")
        return self.order_lines_by_order.get(order_id)

    def get_active_promotions(self, order_date) -> list[dict]:
        """Get promotions active on a given date."""
        if self.promotions_list is None:
            return []
        return [
            p for p in self.promotions_list
            if p["start_date"] <= order_date <= p["end_date"]
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        stats = {"built_indices": list(self._built_indices)}

        if self.po_lines_by_po:
            stats["po_lines_by_po"] = {
                "unique_keys": len(self.po_lines_by_po),
                "total_values": self.po_lines_by_po.total_values(),
            }

        if self.formula_ings_by_formula:
            stats["formula_ings_by_formula"] = {
                "unique_keys": len(self.formula_ings_by_formula),
                "total_values": self.formula_ings_by_formula.total_values(),
            }

        if self.locations_by_account:
            stats["locations_by_account"] = {
                "unique_keys": len(self.locations_by_account),
                "total_values": self.locations_by_account.total_values(),
            }

        return stats


class PooledFaker:
    """
    Batch Faker sampling wrapper for efficient string generation.

    Wraps StaticDataPool with a simpler API matching common Faker patterns.
    Pre-generates pools at initialization, then provides O(1) batch sampling.

    Usage:
        faker = PooledFaker(seed=42)

        # Batch sampling (efficient)
        names = faker.names(1000)
        cities = faker.cities(500)

        # Iterator for loop compatibility
        for name in faker.name_iterator(1000):
            row["name"] = name
    """

    def __init__(
        self,
        seed: int = 42,
        pool_sizes: dict[str, int] | None = None,
    ) -> None:
        """
        Initialize pooled faker.

        Args:
            seed: Random seed for reproducibility
            pool_sizes: Optional custom pool sizes
        """
        self._pool = StaticDataPool(seed=seed, pool_sizes=pool_sizes)

    def names(self, n: int) -> list[str]:
        """Sample n full names."""
        return self._pool.sample_names(n)

    def companies(self, n: int) -> list[str]:
        """Sample n company names."""
        return self._pool.sample_companies(n)

    def cities(self, n: int) -> list[str]:
        """Sample n city names."""
        return self._pool.sample_cities(n)

    def emails(self, n: int) -> list[str]:
        """Sample n email addresses."""
        return self._pool.sample_emails(n)

    def addresses(self, n: int) -> list[str]:
        """Sample n street addresses."""
        return self._pool.sample_addresses(n)

    def phone_numbers(self, n: int) -> list[str]:
        """Sample n phone numbers."""
        return self._pool.sample_phone_numbers(n)

    def first_names(self, n: int) -> list[str]:
        """Sample n first names."""
        return self._pool.sample_first_names(n)

    def last_names(self, n: int) -> list[str]:
        """Sample n last names."""
        return self._pool.sample_last_names(n)

    def name_iterator(self, n: int):
        """Yield n names one at a time (for loop compatibility)."""
        yield from self._pool.sample_names(n)

    def company_iterator(self, n: int):
        """Yield n company names one at a time."""
        yield from self._pool.sample_companies(n)

    def city_iterator(self, n: int):
        """Yield n city names one at a time."""
        yield from self._pool.sample_cities(n)

    def reset(self, seed: int | None = None) -> None:
        """Reset the random number generator."""
        self._pool.reset_rng(seed)

    @property
    def pool(self) -> StaticDataPool:
        """Access underlying StaticDataPool."""
        return self._pool


# =============================================================================
# Integration Examples
# =============================================================================

def example_gr_lines_fix(data: dict, cache: LookupCache) -> None:
    """
    Example: Fix GR lines generation (Lines ~2378-2402).

    BEFORE (O(N×M) = 20K × 75K = 1.5B operations):
        for gr in self.data["goods_receipts"]:
            po_lines = [pl for pl in self.data["purchase_order_lines"]
                        if pl["po_id"] == gr["po_id"]]
            for pl in po_lines:
                # generate GR line...

    AFTER (O(N) = 20K operations):
        cache.build_from_data(self.data)  # Once after Level 5
        for gr in self.data["goods_receipts"]:
            po_lines = cache.get_po_lines(gr["po_id"])  # O(1)
            for pl in po_lines:
                # generate GR line...
    """
    # Build cache (do this once after Level 5 completes)
    cache.build_from_data(data)

    # Then in the GR lines loop:
    for gr in data.get("goods_receipts", []):
        po_lines = cache.get_po_lines(gr["po_id"])  # O(1) lookup
        for pl in po_lines:
            # Generate GR line using pl data...
            pass


def example_wo_materials_fix(data: dict, cache: LookupCache) -> None:
    """
    Example: Fix WO materials generation (Lines ~2406-2429).

    BEFORE (O(N×M) = 50K × 1.5K = 75M operations):
        for wo in self.data["work_orders"]:
            formula_ings = [fi for fi in self.data["formula_ingredients"]
                           if fi["formula_id"] == wo["formula_id"]]

    AFTER (O(N) = 50K operations):
        for wo in self.data["work_orders"]:
            formula_ings = cache.get_formula_ingredients(wo["formula_id"])  # O(1)
    """
    cache.build_from_data(data)

    for wo in data.get("work_orders", []):
        formula = cache.get_formula(wo["formula_id"])
        formula_ings = cache.get_formula_ingredients(wo["formula_id"])
        for fi in formula_ings[:5]:  # Limit to 5 key materials
            # Generate WO material using fi data...
            pass


def example_orders_fix(data: dict, cache: LookupCache) -> None:
    """
    Example: Fix orders generation (Lines ~2770-2777).

    BEFORE (O(N×M) = 200K × 10K = 2B operations):
        for _ in range(200000):
            mega_locs = [l["id"] for l in self.data["retail_locations"]
                        if l["retail_account_id"] == mega_account_id]
            acct_locs = [l["id"] for l in self.data["retail_locations"]
                        if l["retail_account_id"] == acct_id]

    AFTER (O(N) = 200K operations):
        # Pre-extract location IDs by account
        mega_loc_ids = [l["id"] for l in cache.get_account_locations(mega_account_id)]
        for _ in range(200000):
            if is_mega:
                loc_id = random.choice(mega_loc_ids)  # O(1)
            else:
                acct_loc_ids = [l["id"] for l in cache.get_account_locations(acct_id)]
    """
    import random

    cache.build_from_data(data)

    # Pre-cache location ID lists for hot accounts
    mega_account_id = 1  # ACCT-MEGA-001
    mega_loc_ids = [loc["id"] for loc in cache.get_account_locations(mega_account_id)]

    # In the orders loop:
    for _ in range(1000):  # Example with fewer iterations
        if random.random() < 0.25:
            loc_id = random.choice(mega_loc_ids) if mega_loc_ids else 1
        else:
            acct_id = random.randint(1, 100)
            acct_locs = cache.get_account_locations(acct_id)
            loc_ids = [l["id"] for l in acct_locs]
            loc_id = random.choice(loc_ids) if loc_ids else 1


def example_faker_fix(faker: PooledFaker) -> None:
    """
    Example: Fix Faker loops (scattered throughout).

    BEFORE (~800K individual Faker calls):
        for i in range(200):
            row["name"] = self.fake.company()
            row["city"] = self.fake.city()
            row["email"] = self.fake.email()

    AFTER (3 batch calls):
        companies = faker.companies(200)
        cities = faker.cities(200)
        emails = faker.emails(200)
        for i in range(200):
            row["name"] = companies[i]
            row["city"] = cities[i]
            row["email"] = emails[i]

    OR with iterators:
        for name, city, email in zip(
            faker.name_iterator(200),
            faker.city_iterator(200),
            faker.email_iterator(200)
        ):
            row["name"] = name
            row["city"] = city
            row["email"] = email
    """
    # Batch approach
    n = 200
    companies = faker.companies(n)
    cities = faker.cities(n)
    emails = faker.emails(n)

    rows = []
    for i in range(n):
        rows.append({
            "name": companies[i],
            "city": cities[i],
            "email": emails[i],
        })

    return rows
