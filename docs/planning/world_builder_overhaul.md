# World Builder Overhaul & MRP Vectorization Plan

## 1. Executive Summary
**Objective:** Transition the Prism Sim environment from a simplified, hardcoded "2-ingredient" model to a robust, procedurally generated supply chain with realistic Bills of Materials (BOMs).
**Constraint:** Maintain or improve simulation performance (`tick()` speed) despite a 10x-50x increase in the number of tracked SKU/Ingredient interactions.
**Solution:** Implement **Procedural Ingredient Profiles** in configuration and refactor the MRP/Transform engines to use **Dense Matrix Algebra (Vectorization)** instead of iterative object loops.

## 2. Prime Directives for Implementation
All code changes must strictly adhere to the following mandates:
1.  **No Hardcodes:** All ingredient profiles, weights, costs, and BOM complexity parameters must live in `world_definition.json`. Use `semgrep` to validate absence of literals before every commit.
2.  **Modularity:** The "Recipe Matrix" logic should be encapsulated in a dedicated module (`src/prism_sim/network/recipe_matrix.py`), separate from the engine logic.
3.  **Vectorization First:** Do not write loops over products or orders. Always design for $O(1)$ numpy operations over the entire state tensor.
4.  **Type Safety:** All new data structures must be strictly typed (`numpy.typing.NDArray`), and `mypy` must pass strict mode.

---

## 3. Phase 1: Configuration & Generation (The "Realism" Layer)

### 3.1. `world_definition.json` Expansion
Move away from implicit hardcodes to explicit definition. Add a new top-level section:

```json
{
  "ingredient_profiles": {
    "PACKAGING": {
      "prefix": "PKG",
      "types": ["BOTTLE", "CAP", "BOX"],
      "weight_range": [0.01, 0.05],
      "cost_range": [0.05, 0.20],
      "sourcing": "REGIONAL" 
    },
    "ACTIVE_CHEM": {
      "prefix": "ACT",
      "weight_range": [0.1, 0.5],
      "cost_range": [10.0, 50.0],
      "sourcing": "GLOBAL" // Implies longer lead times
    },
    "BASE_BULK": {
      "prefix": "BLK",
      "weight_range": [100, 1000],
      "sourcing": "LOCAL"
    }
  },
  "bom_complexity": {
    "ingredients_per_sku_min": 3,
    "ingredients_per_sku_max": 8
  }
}
```

### 3.2. Procedural Generators (`generators/hierarchy.py`)
Update `ProductGenerator` to support a `generate_ingredients=True` mode.
*   **Logic:** Iterate through profiles, generate $N$ unique ingredients per profile type.
*   **Linking:** Store these in the `World` object as standard `Product` entities but with `category=INGREDIENT`.

### 3.3. Logic-Driven Recipe Builder
Refactor `generate_recipes` to build semantic BOMs rather than random ones.
*   *Rule:* Every "Liquid" SKU needs: 1 Bottle + 1 Cap + 1 Label + X kg Base + Y kg Active.
*   *Rule:* Every "Bar Soap" SKU needs: 1 Wrapper + 1 Box + X kg Base.

---

## 4. Phase 2: The Recipe Matrix (The "Performance" Layer)

### 4.1. Technical Implementation Strategy
**Decision:** Use **Dense NumPy Arrays** (`numpy.array`), NOT `scipy.sparse`.

**Rationale:** 
Even in a "Deep NAM" scenario with 100 Finished Goods and 500 Ingredients, the Recipe Matrix is $100 \times 600 = 60,000$ elements. This fits in L1 CPU cache. The overhead of Sparse Matrix formats (CSR/CSC) is only justified when density is $< 1\%$ and $N > 10,000$. Dense arrays allow for faster SIMD/AVX vectorization by the CPU.

### 4.2. The Recipe Matrix Structure ($\mathbf{R}$)
We will convert the BOM structure into a dense matrix $\mathbf{R}$ of shape $[N_{Products}, N_{Products}]$.

*   **Rows ($i$):** Output Product Index (The Finished Good).
*   **Cols ($j$):** Input Product Index (The Ingredient).
*   **Value ($R_{ij}$):** Quantity of $j$ required to make 1 unit of $i$.
*   **Property:** Strictly triangular (assuming no cycles). $R_{ij} > 0$ implies $i$ consumes $j$.

### 4.3. State Manager Updates
*   **Caching:** The `StateManager` must initialize this matrix *once* at startup (`_init_recipe_matrix`).
*   **Mapping:** Ensure `product_id_to_idx` mapping remains immutable during the simulation run to guarantee matrix validity.

---

## 5. Phase 3: Vectorized Execution

### 5.1. Vectorized MRP (Planning)
Instead of looping through orders to find requirements:

1.  **Demand Vector ($\mathbf{d}$):** Aggregated production demand for all SKUs at a Plant (Shape: $[N_{Products}]$).
2.  **Requirement Calculation:**
    $$ \mathbf{req} = \mathbf{d} \cdot \mathbf{R} $$
    Result $\mathbf{req}$ is a vector $[N_{Products}]$ containing the *total* gross requirement for every ingredient instantly.
3.  **Netting:**
    $$ \mathbf{net} = \mathbf{req} - \mathbf{Inventory}_{plant} $$
4.  **Ordering:** Generate POs where $\mathbf{net} > 0$.

### 5.2. Vectorized Transform (Execution)
1.  **Feasibility Check:**
    $$ \mathbf{Possible} = \mathbf{Inventory}_{plant} \oslash \mathbf{R}_{sku} $$
    (Element-wise division to find the limiting reagent).
2.  **Batch Deduction:**
    $$ \mathbf{Inventory}_{new} = \mathbf{Inventory}_{old} - (\mathbf{ActualOutput} \times \mathbf{R}_{sku}) $$

This moves the heavy lifting from Python loops to C-optimized NumPy BLAS operations.

---

## 6. Implementation Roadmap

### Step 1: Config & Generators (Milestone 8.1)
*   Modify `world_definition.json`.
*   Update `HierarchyGenerator` to produce ingredients.
*   Update `generate_static_world.py` to use the new generator (remove hardcodes).
*   **Check:** Run `semgrep` to ensure no new hardcodes were introduced.

### Step 2: The Matrix Builder (Milestone 8.2)
*   Create a utility class `RecipeMatrixBuilder` in `src/prism_sim/network/recipe_matrix.py`.
*   Implement unit tests to verify the matrix correctly represents the dictionary-based recipes.

### Step 3: Engine Refactoring (Milestone 8.3)
*   Rewrite `MRPEngine.generate_purchase_orders` to use matrix multiplication.
*   Rewrite `TransformEngine._check_material_availability` to use vector comparisons.
*   **Check:** Verify `mypy` strict compliance for new vector operations.

### Step 4: Verification
*   Run the "Deep NAM" benchmark.
*   **Success Criteria:** Simulation speed (days/sec) remains within 10% of current baseline despite BOM complexity increasing from 2 to ~8 items.
