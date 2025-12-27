# Next Steps: Simulation Tuning & Optimization

## Current Status (v0.9.3)
The simulation infrastructure is functionally complete. We have:
*   **Deep NAM Network:** 4,500 stores, RDCs, Plants, and Suppliers fully integrated.
*   **Physics:** Replenishment, Allocation, Logistics (Bin-Packing), MRP, and Production (Finite Capacity) engines are active.
*   **Priming:** We can "warm start" the system with configurable inventory levels (Stores @ 14 days, RDCs @ 21 days).
*   **Reporting:** The "Triangle Report" correctly tracks Service, Cost, Cash, and OEE.

## The Problem: Structural Deficit
Despite "priming" the system with healthy inventory (InvMean ~378), the simulation drifts into a massive backlog over 365 days (Service Index drops to 0.00%, Backlog ~72M cases).

*   **Symptoms:**
    *   Inventory drains steadily from Day 1 to Day 365.
    *   **OEE is Low (~5.5%):** Plants are not producing enough volume to sustain the network, yet they report low utilization. This is a paradox.
    *   **Demand > Supply:** The network consumes ~200k cases/day, but production often lags or is "lumpy" (due to batching constraints).

## Hypotheses & Action Plan

### 1. Fix Low OEE / Under-Production
**Hypothesis:** The MRP engine is generating orders, but the `TransformEngine` might be rejecting them or deferring them inefficiently due to strict constraints, or the `min_production_qty` is too small relative to the demand, causing excessive changeovers (though OEE calculation suggests we aren't even using the capacity for changeovers).
**Action:**
*   **Audit `TransformEngine`:** Why is OEE 5%? Are plants idle waiting for orders? Or are they constrained by something else (e.g., ingredients)?
*   **Tune MRP:** Does `MRPEngine` reorder logic (Reorder Point vs. Target Stock) trigger frequently enough?
*   **Increase Throughput:** Verify `run_rate_cases_per_hour` in recipes vs. `production_hours_per_day`. We increased hours to 100, but output didn't scale proportionally.

### 2. Balance the Triangle
**Hypothesis:** The logistics and allocation logic might be "trapping" inventory or creating friction.
**Action:**
*   **Review `AllocationAgent`:** Is "Fair Share" logic effectively starving everyone when supply is tight?
*   **Review `LogisticsEngine`:** Are trucks being filled? (Truck Fill is low, ~0.7%). This suggests shipments are too small/fragmented.
*   **Consolidation:** We need to batch orders better or force full trucks.

### 3. Simulation Configuration Tuning
We need to find the "Sweet Spot" parameters for:
*   `production_hours_per_day` (Capacity)
*   `min_production_qty` (Batching)
*   `recipes.run_rate` (Speed)
*   `replenishment.batch_size_cases` (Order Aggregation)

## Goals for Next Session
1.  **Solve the OEE Paradox:** Get OEE to a realistic 65-85% while meeting demand.
2.  **Stabilize Inventory:** Achieve a "steady state" where inventory flattens out rather than crashing to zero.
3.  **Positive Service Index:** Maintain >95% service level for the full 365-day run.
