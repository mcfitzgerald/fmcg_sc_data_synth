# Fix Plan: MRP Starvation & Service Level Regression

## Problem Statement

A regression was observed in the 365-day simulation (v0.19.8-dev), where Store Service Level dropped to **49.36%** (from a previous high of ~86%).

### Symptoms
1.  **Service Level Collapse:** Dropped from ~86% to 49%.
2.  **Low Production:** Inventory levels likely drained due to insufficient production output.
3.  **Self-Reinforcing Failure:** The system appears to have entered a "starvation loop" where low production triggered lower ingredient ordering, which further constrained production.

## Root Cause Analysis

The regression was caused by a change in `MRPEngine.generate_purchase_orders` (committed locally but not yet released) intended to smooth ingredient ordering.

### The Flawed Logic
```python
avg_production = np.mean(self.production_order_history)
if avg_production > 0:
    # ...
    daily_production = mix * avg_production
```

**The Feedback Loop:**
1.  Ingredient ordering was coupled to `avg_production` (historical realized output).
2.  If production dropped for *any* reason (e.g., temporary bottleneck, shipping delay, initial ramp-up), `avg_production` decreased.
3.  This caused the MRP engine to order *fewer* ingredients.
4.  Fewer ingredients led to material shortages, forcing production to drop further.
5.  The system spiraled down to a low-equilibrium state (starvation).

**Correct Physics:**
Ingredient replenishment should be driven by **Scheduled Demand** (Active Production Orders) constrained by **Plant Capacity**, *not* by historical throughput. If we have a backlog of orders, we must order enough ingredients to run the plants at max capacity to catch up.

## Proposed Fix

### 1. Decouple Ingredient Demand from Historical Flow
Change `generate_purchase_orders` to calculate ingredient requirements based on **Active Production Orders** (the backlog) scheduled for the immediate future.

### 2. Cap by Plant Capacity (Not History)
Instead of scaling by `avg_production`, scale the daily ingredient needs by the **Maximum Daily Capacity** of the plants.
*   If `Backlog > Capacity`, order ingredients for `Capacity` (run full speed).
*   If `Backlog < Capacity`, order ingredients for `Backlog`.

This breaks the feedback loop: even if yesterday's production was 0, today's ingredient order will be for "Max Capacity" if the backlog is full.

### Implementation Details

**File:** `src/prism_sim/simulation/mrp.py`

**Method:** `generate_purchase_orders`

**Logic:**
1.  Calculate `active_backlog_qty` per product from `active_production_orders`.
2.  Calculate `max_daily_capacity` for the network (sum of run rates * 24h).
3.  `target_daily_production = min(active_backlog_qty, max_daily_capacity)`
4.  Use `target_daily_production` to drive `ingredient_reqs`.

## Validation Plan

1.  **Unit Test:** Verify that ingredient orders max out at plant capacity when backlog is infinite, even if history is zero.
2.  **30-Day Run:** Verify no immediate crash.
3.  **365-Day Run:** Target Service Level > 85%.
