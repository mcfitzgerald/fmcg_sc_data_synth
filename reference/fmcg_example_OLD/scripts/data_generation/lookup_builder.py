"""
LookupIndex and LookupBuilder - O(1) FK lookups for data generation.

Replaces O(N) list comprehensions with O(1) dict-based lookups.

Before (O(N) per lookup, O(N*M) total):
    po_lines = [pl for pl in self.data["purchase_order_lines"] if pl["po_id"] == gr["po_id"]]

After (O(1) per lookup, O(N+M) total):
    po_lines_idx = LookupBuilder.build(data["purchase_order_lines"], "po_id")
    po_lines = po_lines_idx.get(gr["po_id"])

Usage:
    builder = LookupBuilder()

    # Build index from list of dicts
    po_lines_by_po = builder.build(data["purchase_order_lines"], "po_id")

    # O(1) lookup
    lines = po_lines_by_po.get(123)  # Returns list of matching rows

    # Composite key lookup
    inv_by_loc_sku = builder.build_composite(
        data["inventory"],
        ["location_type", "location_id", "sku_id"]
    )
    inv = inv_by_loc_sku.get(("DC", 5, 42))
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Generic, Hashable, TypeVar

# Type variables for generic index
K = TypeVar("K", bound=Hashable)  # Key type (must be hashable)
V = TypeVar("V")  # Value type (typically dict for row data)


class LookupIndex(Generic[K, V]):
    """
    Generic O(1) lookup index for grouped data.

    Wraps a dict mapping keys to lists of values, providing a clean API
    for lookups with sensible defaults for missing keys.

    Type Parameters:
        K: Key type (must be hashable)
        V: Value type (typically dict for row data)

    Attributes:
        _index: Internal dict mapping keys to lists of values
        _key_name: Name of the key field(s) for debugging
    """

    __slots__ = ("_index", "_key_name")

    def __init__(
        self,
        index: dict[K, list[V]],
        key_name: str | tuple[str, ...] = "key",
    ) -> None:
        """
        Initialize the lookup index.

        Args:
            index: Pre-built dict mapping keys to value lists
            key_name: Name of key field(s) for debugging
        """
        self._index: dict[K, list[V]] = index
        self._key_name = key_name

    def get(self, key: K, default: list[V] | None = None) -> list[V]:
        """
        Get all values for a key.

        Args:
            key: Lookup key
            default: Default value if key not found (empty list if None)

        Returns:
            List of values matching the key
        """
        if default is None:
            default = []
        return self._index.get(key, default)

    def get_first(self, key: K, default: V | None = None) -> V | None:
        """
        Get the first value for a key (useful for 1:1 relationships).

        Args:
            key: Lookup key
            default: Default value if key not found

        Returns:
            First matching value or default
        """
        values = self._index.get(key)
        if values:
            return values[0]
        return default

    def get_single(self, key: K) -> V:
        """
        Get exactly one value for a key (asserts uniqueness).

        Args:
            key: Lookup key

        Returns:
            The single matching value

        Raises:
            KeyError: If key not found
            ValueError: If multiple values exist for key
        """
        values = self._index.get(key)
        if not values:
            raise KeyError(f"No value found for {self._key_name}={key}")
        if len(values) > 1:
            raise ValueError(
                f"Expected single value for {self._key_name}={key}, "
                f"found {len(values)}"
            )
        return values[0]

    def __contains__(self, key: K) -> bool:
        """Check if key exists in index."""
        return key in self._index

    def __len__(self) -> int:
        """Return number of unique keys."""
        return len(self._index)

    def keys(self):
        """Return all keys in the index."""
        return self._index.keys()

    def values(self):
        """Return all value lists in the index."""
        return self._index.values()

    def items(self):
        """Return all (key, value_list) pairs."""
        return self._index.items()

    def count(self, key: K) -> int:
        """Return number of values for a key."""
        return len(self._index.get(key, []))

    def total_values(self) -> int:
        """Return total number of values across all keys."""
        return sum(len(v) for v in self._index.values())

    def __repr__(self) -> str:
        return (
            f"LookupIndex(key={self._key_name}, "
            f"unique_keys={len(self)}, total_values={self.total_values()})"
        )


class LookupBuilder:
    """
    Factory for building LookupIndex instances from lists of row dicts.

    Provides both generic building methods and pre-defined builders for
    common FMCG data patterns (PO lines, formula ingredients, etc.).
    """

    @staticmethod
    def build(
        rows: list[dict[str, Any]],
        key_field: str,
    ) -> LookupIndex[Any, dict[str, Any]]:
        """
        Build a lookup index for a single-column key.

        Args:
            rows: List of row dicts to index
            key_field: Field name to use as key

        Returns:
            LookupIndex mapping key values to lists of rows

        Example:
            po_lines_by_po = LookupBuilder.build(po_lines, "po_id")
            lines = po_lines_by_po.get(123)
        """
        index: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = row.get(key_field)
            if key is not None:
                index[key].append(row)
        return LookupIndex(dict(index), key_field)

    @staticmethod
    def build_composite(
        rows: list[dict[str, Any]],
        key_fields: list[str] | tuple[str, ...],
    ) -> LookupIndex[tuple, dict[str, Any]]:
        """
        Build a lookup index for a composite (multi-column) key.

        Args:
            rows: List of row dicts to index
            key_fields: List/tuple of field names to combine as key

        Returns:
            LookupIndex mapping tuple keys to lists of rows

        Example:
            inv_idx = LookupBuilder.build_composite(
                inventory, ["location_type", "location_id", "sku_id"]
            )
            inv = inv_idx.get(("DC", 5, 42))
        """
        index: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        key_fields_tuple = tuple(key_fields)

        for row in rows:
            key = tuple(row.get(f) for f in key_fields_tuple)
            # Only index if all key parts are present
            if None not in key:
                index[key].append(row)

        return LookupIndex(dict(index), key_fields_tuple)

    @staticmethod
    def build_unique(
        rows: list[dict[str, Any]],
        key_field: str,
    ) -> dict[Any, dict[str, Any]]:
        """
        Build a simple dict for unique keys (1:1 relationship).

        Unlike build(), this returns a flat dict rather than a LookupIndex.
        Raises ValueError if duplicate keys are found.

        Args:
            rows: List of row dicts to index
            key_field: Field name to use as key

        Returns:
            Dict mapping key values directly to rows

        Raises:
            ValueError: If duplicate keys are found

        Example:
            sku_by_code = LookupBuilder.build_unique(skus, "code")
            sku = sku_by_code["SKU-001"]
        """
        index: dict[Any, dict[str, Any]] = {}
        for row in rows:
            key = row.get(key_field)
            if key is not None:
                if key in index:
                    raise ValueError(
                        f"Duplicate key found: {key_field}={key}"
                    )
                index[key] = row
        return index

    # =========================================================================
    # Pre-defined builders for common FMCG data patterns
    # =========================================================================

    @classmethod
    def build_po_lines_by_po_id(
        cls,
        po_lines: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of purchase order lines by PO ID.

        Used in Level 6 (goods receipt lines) to find PO lines to receive.
        """
        return cls.build(po_lines, "po_id")

    @classmethod
    def build_formula_ings_by_formula_id(
        cls,
        formula_ingredients: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of formula ingredients by formula ID.

        Used in Level 6-7 (work order materials, batch ingredients) to
        find ingredients needed for a formula.
        """
        return cls.build(formula_ingredients, "formula_id")

    @classmethod
    def build_locations_by_account_id(
        cls,
        retail_locations: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of retail locations by retail_account_id.

        Used in Level 8 (orders) to find stores for an account.
        """
        return cls.build(retail_locations, "retail_account_id")

    @classmethod
    def build_order_lines_by_order_id(
        cls,
        order_lines: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of order lines by order ID.

        Used in Level 9-11 (allocations, shipments) to find lines for an order.
        """
        return cls.build(order_lines, "order_id")

    @classmethod
    def build_batches_by_sku_id(
        cls,
        batches: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of batches by SKU ID.

        Used in Level 10-11 (shipment lines) to find batches to ship.
        """
        return cls.build(batches, "sku_id")

    @classmethod
    def build_shipments_by_order_id(
        cls,
        shipments: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of shipments by order ID.

        Used in Level 11 (shipment lines) to allocate batches to shipments.
        """
        return cls.build(shipments, "order_id")

    @classmethod
    def build_work_orders_by_formula_id(
        cls,
        work_orders: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of work orders by formula ID.

        Used in Level 6 (work order materials) to find WOs for a formula.
        """
        return cls.build(work_orders, "formula_id")

    @classmethod
    def build_gr_lines_by_gr_id(
        cls,
        gr_lines: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of goods receipt lines by goods receipt ID.

        Used in Level 7 for inventory updates from receipts.
        """
        return cls.build(gr_lines, "goods_receipt_id")

    @classmethod
    def build_supplier_ings_by_ingredient_id(
        cls,
        supplier_ingredients: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of supplier-ingredient links by ingredient ID.

        Used to find suppliers for an ingredient (SPOF detection).
        """
        return cls.build(supplier_ingredients, "ingredient_id")

    @classmethod
    def build_supplier_ings_by_supplier_id(
        cls,
        supplier_ingredients: list[dict[str, Any]],
    ) -> LookupIndex[int, dict[str, Any]]:
        """
        Build index of supplier-ingredient links by supplier ID.

        Used to find ingredients from a supplier.
        """
        return cls.build(supplier_ingredients, "supplier_id")
