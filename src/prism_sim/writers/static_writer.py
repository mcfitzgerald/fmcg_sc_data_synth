"""Static data writer for dumping the generated world to CSVs."""

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prism_sim.network.core import Link, Node, NodeType
from prism_sim.product.core import Product, Recipe
from prism_sim.writers.base import BaseWriter


class StaticWriter(BaseWriter):
    """Writes static world definitions (Products, Locations, Partners) to CSV files."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, data: Any, destination: str) -> None:
        """Generic write not implemented; use specific methods."""
        raise NotImplementedError("Use specific write_xyz methods")

    def _write_csv(self, filename: str, items: list[Any]) -> None:
        """Helper to write a list of dataclasses to CSV."""
        if not items:
            return

        filepath = self.output_dir / filename
        # Convert first item to dict to get headers
        # Note: This is a shallow conversion. Nested objects (like Recipe ingredients)
        # might need special handling.

        # We'll use asdict but we might need to flatten or stringify complex types
        fieldnames = list(asdict(items[0]).keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in items:
                row = asdict(item)
                # Simple serialization for dicts/lists
                for k, v in row.items():
                    if isinstance(v, (dict, list)):
                        row[k] = str(v)
                writer.writerow(row)

    def write_products(self, products: list[Product]) -> None:
        """Write products to products.csv."""
        self._write_csv("products.csv", products)

    def write_recipes(self, recipes: list[Recipe]) -> None:
        """Write recipes to recipes.csv."""
        self._write_csv("recipes.csv", recipes)

    def write_locations(self, nodes: list[Node]) -> None:
        """Write all nodes to locations.csv."""
        self._write_csv("locations.csv", nodes)

    def write_partners(self, nodes: list[Node]) -> None:
        """
        Write supplier partners to partners.csv.

        Note: currently maps suppliers from nodes.
        """
        partners = [n for n in nodes if n.type == NodeType.SUPPLIER]
        self._write_csv("partners.csv", partners)

    def write_links(self, links: list[Link]) -> None:
        """Write network topology to links.csv."""
        self._write_csv("links.csv", links)
