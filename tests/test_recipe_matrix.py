"""Unit tests for RecipeMatrixBuilder."""

import numpy as np

from prism_sim.network.recipe_matrix import RecipeMatrixBuilder
from prism_sim.product.core import Product, ProductCategory, Recipe


def test_matrix_builder_simple() -> None:
    """Test building a simple 3-item BOM matrix."""
    # 1. Setup Products
    p_fg = Product("FG-001", "Finished Good", ProductCategory.ORAL_CARE, 1.0, 10, 10, 10, 100)
    p_ing1 = Product("ING-001", "Ingredient 1", ProductCategory.INGREDIENT, 0.1, 5, 5, 5, 1000)
    p_ing2 = Product("ING-002", "Ingredient 2", ProductCategory.INGREDIENT, 0.2, 5, 5, 5, 1000)

    products = [p_fg, p_ing1, p_ing2]

    # 2. Setup Recipe
    # FG needs 2 units of ING-1 and 0.5 units of ING-2
    recipe = Recipe(
        product_id="FG-001",
        ingredients={"ING-001": 2.0, "ING-002": 0.5},
        run_rate_cases_per_hour=100
    )

    # 3. Build Matrix
    builder = RecipeMatrixBuilder(products, [recipe])
    matrix = builder.build_matrix()
    mapping = builder.get_id_mapping()

    # 4. Assertions
    idx_fg = mapping["FG-001"]
    idx_ing1 = mapping["ING-001"]
    idx_ing2 = mapping["ING-002"]

    assert matrix.shape == (3, 3)

    # Check dependencies
    assert matrix[idx_fg, idx_ing1] == 2.0
    assert matrix[idx_fg, idx_ing2] == 0.5

    # Check zeros (Ingredients don't consume anything)
    assert matrix[idx_ing1, idx_fg] == 0.0
    assert matrix[idx_ing1, idx_ing2] == 0.0
    assert np.sum(matrix[idx_ing1, :]) == 0.0

def test_matrix_builder_missing_ingredient() -> None:
    """Test robustness when a recipe references a missing product."""
    p_fg = Product("FG-001", "Finished Good", ProductCategory.ORAL_CARE, 1.0, 10, 10, 10, 100)
    products = [p_fg]

    recipe = Recipe(
        product_id="FG-001",
        ingredients={"MISSING-ID": 5.0},
        run_rate_cases_per_hour=100
    )

    builder = RecipeMatrixBuilder(products, [recipe])
    matrix = builder.build_matrix()

    # Should be all zeros, no crash
    assert np.all(matrix == 0.0)

def test_vectorized_operation() -> None:
    """Test a mock MRP calculation using the matrix."""
    # Scenario: 2 FGs sharing 1 common ingredient
    p_fg1 = Product("FG-1", "A", ProductCategory.ORAL_CARE, 1, 1, 1, 1, 1)
    p_fg2 = Product("FG-2", "B", ProductCategory.ORAL_CARE, 1, 1, 1, 1, 1)
    p_com = Product("ING-C", "Common", ProductCategory.INGREDIENT, 1, 1, 1, 1, 1)
    p_uniq = Product("ING-U", "Unique", ProductCategory.INGREDIENT, 1, 1, 1, 1, 1)

    products = [p_fg1, p_fg2, p_com, p_uniq]

    r1 = Recipe("FG-1", {"ING-C": 2.0, "ING-U": 1.0}, 100)
    r2 = Recipe("FG-2", {"ING-C": 3.0}, 100)

    builder = RecipeMatrixBuilder(products, [r1, r2])
    R = builder.build_matrix()
    idx = builder.get_id_mapping()

    # Demand Vector: We want to make 10 units of FG-1 and 20 units of FG-2
    demand = np.zeros(4)
    demand[idx["FG-1"]] = 10.0
    demand[idx["FG-2"]] = 20.0

    # Calculate Requirement: req = demand @ R
    # (Note: In standard linear algebra, if Rows=Output, Cols=Input, then:
    #  Total Input j = Sum_i (Output i * R_ij)
    #  This is vector-matrix multiplication: d * R

    requirements = demand @ R

    # Expected:
    # ING-C: (10 * 2.0) + (20 * 3.0) = 20 + 60 = 80
    # ING-U: (10 * 1.0) + (20 * 0) = 10

    assert requirements[idx["ING-C"]] == 80.0
    assert requirements[idx["ING-U"]] == 10.0
    assert requirements[idx["FG-1"]] == 0.0 # FGs don't consume themselves
