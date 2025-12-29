# Prism Simulation Tuning & Recovery Plan

**Status:** **CRITICAL FAILURE - THE BULLWHIP COLLAPSE**
While the simulation now successfully runs for 365 days with stabilized OEE (78%) and Mass Balance integrity, the network suffers a catastrophic "Bullwhip" collapse.
- **Service Level:** 3.4% (Target >95%)
- **Daily Orders:** Exploded from ~460k to **66,000,000** cases/day (Panic Ordering).
- **Backlog:** 58M cases.
- **Root Cause:** "Fill or Kill" logic + Zero Inventory = Infinite Reordering Loop. Stores order 1000, get 0, order 1000 again next day.

**Goal:** Implement dampeners to stop the panic ordering spiral and stabilize the inventory signal.

---

## 1. The SPOF Refinement (Targeted Vulnerability) - **IMPLEMENTED**
**Strategy:**
*   **Pareto Isolation:** `ACT-CHEM-001` is now isolated to ~20% of the portfolio (Premium Oral Care) in `hierarchy.py`.
*   **Capacity:** SPOF Supplier (`SUP-001`) constrained to 500k units/day.

## 2. Material Replenishment (The "Starvation" Fix) - **IMPLEMENTED**
**Strategy:**
*   **Tiered Buffers:** `MRPEngine` now uses vectorized Inventory Policies:
    *   **Commodities:** ROP 7 / Target 14
    *   **Specialty/SPOF:** ROP 30 / Target 45
*   **Initialization:** Plant ingredient inventory seeded to 5M units to prevent cold-start starvation.

## 3. Supplier Capacity & Throughput - **IMPLEMENTED**
**Strategy:**
*   **Uncap Key Suppliers:** `NetworkGenerator` sets `throughput_capacity = inf` for all suppliers except the SPOF source.
*   **Allocation:** `AllocationAgent` now respects supplier capacity limits (Fair Share) instead of assuming infinite supply.

## 4. Service Level Metric (Reporting Reality) - **IMPLEMENTED**
**Strategy:**
*   **LIFR:** `Orchestrator` and `RealismMonitor` now track `Daily Fill Rate = Shipped / Ordered`.
*   **Reporting:** Triangle Report uses the monitored Service Level mean instead of the backlog approximation.

## 5. The Bullwhip Crisis (Current Focus)
**Current Issue:** The "Fill or Kill" logic works technically (orders close), but physically fails because the Replenisher immediately re-orders the deficit the next day. This creates a feedback loop where demand signal is amplified 150x (460k -> 66M).

**Recovery Strategy:**
1.  **Allocation Minimum Threshold:** Modify `AllocationAgent`. If we can't fill at least X% (e.g., 50%) of an order, **ship nothing** to that destination. This prevents "dust" shipments (Truck Fill 0.7%) and saves logistics capacity.
2.  **Replenishment Dampener:** Modify `MinMaxReplenisher`.
    *   **Max Order Cap:** Cap daily orders at 200% of average demand.
    *   **Order Smoothing:** Don't re-order the full deficit immediately if it exceeds capacity.

## 6. Execution Plan
1.  **Dampen:** Implement `max_order_cap` in `MinMaxReplenisher`.
2.  **Threshold:** Implement `min_fill_threshold` in `AllocationAgent`.
3.  **Run:** Execute 365-day run. Expecting:
    *   Orders stabilize at ~460k/day (+ seasonality).
    *   Service Level rises as phantom demand disappears.
