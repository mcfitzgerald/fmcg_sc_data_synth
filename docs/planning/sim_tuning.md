# Prism Simulation Tuning & Recovery Plan

**Status:** The "Fill or Kill" logic has stabilized the engine's computational performance, but the supply chain is physically failing. Production collapses due to material starvation, resulting in a 0% Service Level.

**Goal:** Tune the physics parameters to achieve a stable, realistic "Deep NAM" simulation where disruptions (like SPOF) hurt specific areas but do not crash the entire network.

---

## 1. The SPOF Refinement (Targeted Vulnerability)
**Current Issue:** `ACT-CHEM-001` (SPOF) appears to be a bottleneck for a significant portion of the portfolio. Analysis of `recipes.csv` confirms it is used across **all** categories (`ORAL`, `HOME`, `PERSONAL`), making it a universal dependency rather than a specific vulnerability. If this runs out, the entire network crashes.

**Strategy:**
*   **Pareto Isolation:** Ensure the SPOF ingredient is used in high-value but not *all* high-volume products. It should impact ~20% of the portfolio volume (e.g., Premium Oral Care), not the "Bread and Butter" items (Standard Bar Soap).
*   **Action:**
    *   Audit `src/prism_sim/generators/hierarchy.py` to ensure `ACT-CHEM-001` is assigned conditionally (e.g., only to `ORAL_CARE` or `PREMIUM` tier).
    *   Verify `world_definition.json` ingredient profiles.

## 2. Material Replenishment (The "Starvation" Fix)
**Current Issue:** The global `reorder_point_days: 14` is insufficient for imported or constrained ingredients. A single shipping delay causes a stockout.

**Strategy:**
*   **Tiered Buffers:** Differentiate Inventory Policies based on ingredient type.
    *   **Commodities (Water, Cardboard):** Short Lead Time, Low ROP (7 days).
    *   **Specialty/Imported (SPOF):** Long Lead Time, High ROP (30-45 days).
*   **Action:**
    *   Update `simulation_config.json` to support granular `reorder_point_days` by Category or Ingredient ID, rather than a global manufacturing default.
    *   Modify `MRPEngine` to respect these granular configs.

## 3. Supplier Capacity & Throughput
**Current Issue:** Suppliers might be mathematically incapable of keeping up with the "Deep NAM" demand (230k cases/day ~ huge ingredient volume).

**Strategy:**
*   **Uncap Key Suppliers:** Ensure "Bulk" suppliers (Water, Base Oil) have effectively infinite throughput.
*   **Constrain SPOF:** Keep the SPOF supplier constrained to force the "Allocation" logic to kick in during the risk event, but ensure the *steady state* capacity is sufficient (Capacity > Demand * 1.2).
*   **Action:**
    *   Audit `NetworkGenerator` in `src/prism_sim/generators/network.py` to ensure Supplier nodes are initialized with sufficient `throughput_capacity`.

## 4. Service Level Metric (Reporting Reality)
**Current Issue:** The "Fill Rate Index" uses a penalty based on backlog size. Since we now "Kill" orders, the backlog is technically empty (or just accounting artifacts), but the *Index* logic is likely misinterpreting the lack of filled orders.

**Strategy:**
*   **Switch to LIFR:** Calculate `Line Item Fill Rate` (Orders Shipped / Orders Placed).
*   **Action:**
    *   Update `Orchestrator._record_daily_metrics` to calculate `Daily Fill Rate = Shipped_Qty / Ordered_Qty`.
    *   This provides a 0-100% percentage that is physically meaningful.

## 5. Execution Plan (Next Session)
1.  **Refine Recipes:** Re-run `generate_static_world.py` with tighter constraints on SPOF usage.
2.  **Tune Config:** Update `simulation_config.json` with Tiered ROPs (30 days for Specialty).
3.  **Fix Metric:** Rewrite the Service Level calculation in `Orchestrator`.
4.  **Run:** Execute 365-day run. Expecting:
    *   Service Level > 90% (Steady State).
    *   Drop to ~85% during SPOF Risk Event (Day 120/200).
    *   Recovery within 30 days.
