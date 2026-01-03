# Debug Plan: Foxtrot - MRP Starvation & Service Level Degradation

## Problem Statement

365-day simulation degrades from ~90% service level (30-day) to ~50% (365-day). Production collapses from ~400K/day to ~120-200K/day despite having adequate plant capacity (543K/day).

## What We Know

### Capacity Analysis
- **Expected daily demand:** 424,540 cases
- **Max daily capacity:** 543,855 cases (1.28x demand)
- **OEE in degraded state:** 60-75% (plants are NOT fully utilized)
- **Conclusion:** Capacity is NOT the bottleneck

### The Original Starvation Loop (Partially Fixed)
The original issue in `generate_purchase_orders`:
```python
avg_production = np.mean(self.production_order_history)
daily_production = mix * avg_production  # PROBLEM: couples to historical flow
```

**Fix applied:** Decouple ingredient ordering from historical production, use expected demand as baseline.

### The Smoothing Cap Loop (Partially Fixed)
The C.5 smoothing in `generate_production_orders`:
```python
if total_orders_today > avg_recent * 1.5:  # PROBLEM: cap drops with history
    scale_factor = ...
```

**Fix applied:** Use `max(avg_recent, expected_production)` as baseline.

## Key Hypothesis: Demand Anticipation Gap

The system may not be **anticipating demand far enough ahead** in its planning:

### Current Flow
1. Stores order when inventory drops below ROP
2. Customer DCs order when their inventory drops
3. RDCs order from Plants when RDC inventory drops
4. MRP generates production orders when RDC DOS < ROP
5. MRP generates ingredient orders based on active production backlog

### The Problem
Each step is **reactive**, not **anticipatory**:
- Ingredient orders are based on **current** production backlog
- Production orders are based on **current** RDC inventory position
- There's no forward-looking demand signal

### Evidence
- Days 1-30: System runs well (warm-started inventory buffers demand)
- Days 30-60: Production starts dropping as buffers drain
- Days 60+: System enters low-equilibrium state

## Verification Steps (Next Session)

### Step 1: Verify Ingredient Ordering Fix
Add logging to `generate_purchase_orders` to confirm:
- [ ] `daily_production` is using expected demand (not historical)
- [ ] Ingredient orders are being generated at expected levels
- [ ] Orders are actually reaching plants (check arrivals)

### Step 2: Verify Production Order Generation
Add logging to `generate_production_orders` to confirm:
- [ ] DOS calculation is correct (not inflated)
- [ ] ROP triggers are firing when expected
- [ ] Production orders are being created at expected levels
- [ ] Smoothing cap is using expected as baseline

### Step 3: Trace Demand Signal Flow
Track demand signal through the system:
- [ ] POS demand → Replenisher demand signal
- [ ] Replenisher → Order generation
- [ ] Orders → MRP demand history
- [ ] MRP demand → Production orders
- [ ] Production orders → Ingredient orders

### Step 4: Check Anticipation Horizons
Verify planning horizons are adequate:
- [ ] Production lead time vs. ROP buffer
- [ ] Ingredient lead time vs. ingredient ROP buffer
- [ ] Order-to-delivery vs. customer inventory targets

## Potential Root Causes to Investigate

### 1. DOS Inflation
The Days of Supply calculation includes:
- On-hand inventory
- In-transit inventory
- In-production inventory

If any of these are inflated (e.g., stale in-production orders), DOS appears high → no orders generated.

### 2. Mix Mismatch
When backlog is low, we use expected demand mix for ingredients. But if the backlog mix doesn't match expected mix, we may:
- Over-order some ingredients (SLOB)
- Under-order others (shortage)

### 3. Warm Start Decay
The warm starts for demand_history and production_order_history are set once at init. Over time, actual values replace warm values. If the system enters a low state early, the warm start benefit is lost.

### 4. Horizon Mismatch
- Ingredient orders use 7-14 day ROP/Target
- Production orders use 7-21 day ROP/Target
- But lead times may not align with these buffers

## Configuration Parameters to Tune

```json
{
  "production_floor_pct": 0.5,        // Minimum production as % of expected
  "min_production_cap_pct": 0.7,      // Minimum ingredient ordering as % of expected
  "reorder_point_days": 21.0,         // FG ROP at RDCs
  "target_days_supply": 28.0,         // FG target at RDCs
  "ingredient_rop_days": 7.0,         // Ingredient ROP at plants
  "ingredient_target_days": 14.0      // Ingredient target at plants
}
```

## Proposed Fix: Anticipatory Planning

Instead of reactive planning, implement forward-looking demand:

### Option A: Extended Demand Horizon
Use POS demand forecast (not just history) to drive:
- Production orders (produce to expected demand, not just ROP trigger)
- Ingredient orders (order for expected production, not just backlog)

### Option B: Safety Stock Multipliers
Increase ROP multipliers to create larger buffers:
- Higher buffers → earlier ordering → more production → stable inventory

### Option C: Rate-Based Production
Instead of DOS-triggered ordering, use rate-based:
- Always produce at rate = expected demand
- Use inventory to absorb variability
- Only throttle when inventory exceeds max threshold

## Files Modified in This Session

- `src/prism_sim/simulation/mrp.py`
  - Added `_calculate_max_daily_capacity()` method
  - Fixed `generate_purchase_orders()` to use expected demand as baseline
  - Fixed production smoothing cap to use `max(avg_recent, expected)` as baseline

- `src/prism_sim/config/simulation_config.json`
  - Tuned `production_floor_pct` and `min_production_cap_pct`

## Results Summary

| Configuration | 30-day SL | 90-day SL | 365-day SL | OEE |
|---------------|-----------|-----------|------------|-----|
| Baseline (broken) | 93% | 88% | 50% | 60% |
| Floor=70% | - | - | 57% | 71% |
| Floor=100% | - | - | 58% | 75% |

The fixes improve 365-day service level from 50% to 58%, but still far from the 85%+ target.

## Next Steps

1. Add diagnostic logging to trace demand/order/production flow
2. Implement anticipatory planning (Option A or C)
3. Consider longer warm-start period or dynamic warm-start adjustment
4. Validate with 365-day simulation after each change
