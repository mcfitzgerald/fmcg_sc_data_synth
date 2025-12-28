# Session Pickup: Prism-Sim Structural Deficit Debug

## Current Status
- **World Scale:** "Deep NAM" (4,500 nodes, 4 Plants, 4 RDCs, 50 SKUs).
- **The Issue:** Systemic Inventory Collapse. Inventory drains monotonically from Day 1.
- **Key Metric:** 
    - **Demand:** ~230k cases/day.
    - **Observed Production:** Capped at ~160k cases/day.
    - **Theoretical Capacity:** ~232k cases/day (after doubling run rates to 3k/hr).
- **Bottleneck:** Not Ingredients (Plants have 9M+ units). Not theoretical run-rate. 

## Technical State
1. **Config Paradigm:** Restored. `world_definition.json` now controls `run_rate` and `changeover_time`.
2. **Current Settings:**
    - **MOQ:** Increased to 25,000 cases (to amortize changeovers).
    - **Changeovers:** Reduced to 0.5 - 1.5 hours.
    - **Run Rates:** 2,400 - 3,600 cases/hr.
    - **Logging:** Enabled in `run_benchmark.py`.
    - **Horizon:** Reduced to 90 days for faster iteration.

## Active Hypotheses
1. **Changeover Death Spiral:** Even with 25k MOQ, the plant is switching too often, or the `TransformEngine` is losing more time than calculated.
2. **MRP "Inventory Position" Logic Error:** The `MRPEngine` includes *all* planned production in its supply calculation. If the plant queue grows, the MRP stops ordering because it "perceives" the massive queue as available supply, even if the plant won't hit those orders for weeks.
3. **Allocation/DRP silence:** Verify if RDCs are actually shipping to Stores or if inventory is stuck at RDCs (Backlog analysis needed).

## Next Steps for Fresh Session
1. **Step 1:** Run the 90-day simulation and IMMEDIATELY analyze `batches.csv` vs `orders.csv`. 
2. **Step 2:** Audit `src/prism_sim/simulation/mrp.py`. Refactor the `inventory_position` calculation to use a "Look-ahead Horizon" (e.g., only count production finishing in next 7 days).
3. **Step 3:** Perform a "Stress Test" by setting `efficiency_factor: 1.0` and `changeover_time: 0.0`. if it *still* collapses, the bug is in the State Manager or Logistics transit logic, not the plants.
