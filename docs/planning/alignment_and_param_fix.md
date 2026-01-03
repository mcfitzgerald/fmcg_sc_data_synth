# ABC Alignment and Parameter Fix Plan

## Problem Statement

The implementation of ABC Prioritization (Phases 1-4) resulted in a 365-day Service Level of **71.02%** and SLOB of **78.0%**. This is a regression from the v0.19.2 baseline (76% Service Level, 73% SLOB).

### Root Cause Analysis

The degradation is likely caused by **inconsistent ABC classification** across different simulation engines.

1.  **Replenisher:** Uses `product_volume_history` which accumulates actual `daily_demand` (including Zipfian popularity). This is the **source of truth**.
2.  **MRP Engine:** Uses `expected_daily_demand` derived from `simulation_config.json` category profiles (e.g., `base_daily_demand: 7.0` for all ORAL_CARE products). This **ignores Zipfian popularity**, causing popular A-items to be misclassified as B or C if their base rate is standard.
3.  **Transform Engine:** Uses `base_demand` passed from `Orchestrator`, which is the static `pos_engine.get_base_demand_matrix()`. This is **correct** (includes Zipfian), but it is static and doesn't adapt to dynamic shifts.

### The Misalignment Gap

*   **Scenario:** A Zipfian "head" product (Product X) has 100x demand of a "tail" product (Product Y).
*   **Replenisher:** Sees Product X as "A" (Z=2.33) and Product Y as "C" (Z=1.28). Orders aggressively for X.
*   **MRP Engine:** Sees Product X and Y as having equal `expected_daily_demand` (7.0). Classifies both as "B". Applies neutral ROP multiplier (1.0x) to both.
*   **Result:** Replenisher orders huge quantities of X to meet 99% service level. MRP plans production for X using a standard ROP, causing stockouts. Meanwhile, it over-plans for Y relative to its actual low demand (but Replenisher correctly orders less for Y).

## Proposed Fixes

### 1. Unified ABC Signal (Architecture)

Ensure all engines use the same demand signal for classification.

*   **Source of Truth:** `POSEngine`'s `base_demand_matrix` (which includes Zipfian weights) OR the `Replenisher`'s dynamic volume history.
*   **Strategy:** Inject `base_demand_matrix` into `MRPEngine` during initialization, replacing the config-based `expected_daily_demand` estimation.

#### Implementation Steps

1.  **Orchestrator:** Pass `base_demand_matrix` to `MRPEngine` constructor (already doing this for Replenisher and Transform).
2.  **MRP Engine:**
    *   Update `__init__` to accept `base_demand_matrix`.
    *   Replace `_build_expected_demand_vector` logic: instead of iterating config profiles, sum `base_demand_matrix` across all Store nodes to get network-wide expected demand per product.
    *   Re-run `_classify_products_abc` using this accurate vector.

### 2. Parameter Tuning (Calibration)

The aggressive Z-score for A-items (2.33 = 99%) combined with the "Fill or Kill" logic might be creating instability.

*   **Hypothesis:** A-items deplete faster than they can be replenished because the ROP multiplier (1.2x) isn't aggressive enough to counter the lead time variance, OR the Z-score target is creating inventory targets that exceed plant capacity.
*   **Adjustment:**
    *   **Phase 1:** Fix alignment first. Run 90-day validation.
    *   **Phase 2:** If Service Level < 90%, increase A-item ROP multiplier (1.2 -> 1.5) to buffer against manufacturing lead time.
    *   **Phase 3:** If SLOB is high, decrease C-item ROP multiplier (0.8 -> 0.5) and reduce Z-score (1.28 -> 1.0).

### 3. Dynamic Re-classification (Advanced - Optional)

If static alignment isn't enough, `MRPEngine` should update its ABC class based on `demand_history` (rolling average of actual shipments), similar to how `Replenisher` does.

## Execution Plan

1.  **Refactor MRPEngine:**
    *   File: `src/prism_sim/simulation/mrp.py`
    *   Method: `__init__`, `_build_expected_demand_vector`
    *   Action: Use `base_demand_matrix` for expected demand calculation.

2.  **Verify Alignment:**
    *   Add logging to print the number of A, B, and C items identified by Replenisher vs MRP at startup. They must match.

3.  **Run Validation:**
    *   Execute 90-day simulation.
    *   Success Metric: Service Level > 85% (recovery to near baseline).

4.  **Tune Parameters (Iterative):**
    *   Adjust `simulation_config.json` ABC multipliers based on results.
