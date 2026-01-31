"""Dense Matrix representation of Bill of Materials (BOM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prism_sim.product.core import Product, Recipe


class RecipeMatrixBuilder:
    """Converts product recipes into a dense dependency matrix."""

    def __init__(self, products: list[Product], recipes: list[Recipe]) -> None:
        self.products = products
        self.recipes = recipes
        self.product_id_to_idx: dict[str, int] = {
            p.id: i for i, p in enumerate(products)
        }
        self.n_products = len(products)

    def build_matrix(self) -> NDArray[np.float64]:
        """Builds the Recipe Matrix R.

        Rows (i): Output Product Index (The Finished Good)
        Cols (j): Input Product Index (The Ingredient)
        Value (R_ij): Quantity of j required to make 1 unit of i
        """
        # Initialize dense matrix
        matrix = np.zeros((self.n_products, self.n_products), dtype=np.float64)

        # Map recipes for fast lookup
        recipe_map = {r.product_id: r for r in self.recipes}

        for i, product in enumerate(self.products):
            if product.id in recipe_map:
                recipe = recipe_map[product.id]
                for ing_id, qty in recipe.ingredients.items():
                    j = self.product_id_to_idx.get(ing_id)
                    if j is not None:
                        matrix[i, j] = qty
                    else:
                        # In a strict simulation, this might be an error.
                        # For now, we assume ingredients might be
                        # outside the list if filtering happened.
                        pass

        return matrix

    def get_id_mapping(self) -> dict[str, int]:
        """Returns the mapping from Product ID to Matrix Index."""
        return self.product_id_to_idx
