# Pickup: Zero Orders Debugging

## Current Status
We are investigating a critical issue where **Store Orders drop to zero** or remain near-zero despite inventory depletion.

### Key Findings
1.  **Performance Fix:** The simulation was slowing down due to O(N*L) supplier lookup in `MinMaxReplenisher`. We implemented O(1) caching, which resolved the slowness.
2.  **Physics Fix:** We reduced `base_daily_demand` to ~1.0 case/day and `initial_fg_level` to 5.0 to create a realistic NAM scenario.
3.  **The "Zero Orders" Anomaly:**
    *   `InvMean` drops deeply negative (e.g., -9.1).
    *   `EstRP` (Reorder Point) is positive (~3.0).
    *   `NeedsOrderCount` is high (e.g., 225,000 cells need orders).
    *   `MinMaxReplenisher` confirms it is creating `Order` objects with non-zero lines.
    *   **However, `Orchestrator` reports `Ord=0` (or very low).**

### Root Cause Hypothesis
The issue lies in **`AllocationAgent.allocate_orders`**.

*   We suspect `AllocationAgent` is modifying `raw_orders` in-place.
*   If Source Inventory (RDC) is empty (which happens quickly with `Inv=5.0` and no production), the `Allocator` sets `line.quantity = 0`.
*   The `Orchestrator` prints the sum of quantities *after* `allocate_orders` is called (even though it uses the `raw_orders` variable, if the objects are modified in place, the sum reflects the *allocated* quantity, not the *demanded* quantity).
*   This creates a misleading log where "Demand" (from POS) is high, but "Ordered" (Unconstrained replenishment signal) appears to be zero.

## Next Steps

1.  **Verify In-Place Modification:**
    *   Inspect `src/prism_sim/agents/allocation.py`.
    *   Confirm if `line.quantity` is mutated.
2.  **Fix Reporting:**
    *   Update `Orchestrator` to calculate `total_ordered` *before* passing `raw_orders` to the allocator.
    *   This will allow us to see the "Unconstrained Demand" signal separate from the "Fulfilled Orders".
3.  **Fix Supply:**
    *   Investigate why `Prod=0`. Plants should be producing to refill RDCs.
    *   If RDCs are empty, Allocator cuts orders to 0.
    *   If Plants don't produce, RDCs stay empty.
    *   `MRPEngine` might be seeing "Perceived Inventory" as high? Or lacking a signal?
4.  **Restore Benchmark:**
    *   Once validated, revert `run_benchmark.py` to 365 days (already done).
    *   Remove or silence debug prints in `Replenishment.py` and `Orchestrator.py`.

## Commands to Resume
```bash
poetry run python run_benchmark.py
```
