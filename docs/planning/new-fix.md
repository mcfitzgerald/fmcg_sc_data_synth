# Service Level Degradation Fix Plan

## Problem Statement

365-day simulation shows service level degradation from ~99% (first 30 days) to ~73% (full year). Inventory accumulates at MFG RDCs (93% of total) while downstream stores starve (4.5% of total).

## Root Cause Analysis

### The Negative Feedback Spiral

```
Store inventory depletes
       ↓
Stores order less (IP < ROP triggers less often)
       ↓
Customer DCs see weaker demand signal
       ↓
Customer DCs order less from RDCs (only 54% of actual demand)
       ↓
RDCs accumulate inventory (no one pulling it)
       ↓
MRP sees high RDC inventory → produces less
       ↓
Less inventory enters system, but outflow also drops
       ↓
Repeat... stores continue to starve
```

### Key Metrics (v0.19.1 - 365 days)

| Metric | Value | Issue |
|--------|-------|-------|
| Service Level | 73% | Target >90% |
| SLOB | 65% | Target <30% |
| MFG RDC Inventory | 92.8% of total | Should be ~40% |
| Store Inventory | 4.5% of total | Should be ~20% |
| Customer DC Orders | 54% of demand | Should be ~100% |

### What v0.19.1 Fixed (Partial)

1. **MEIO IP Bug**: Customer DCs now use Local IP instead of Echelon IP
2. **MRP Signal Collapse**: MRP now uses POS demand as floor

These fixes helped marginally (+2% service level) but didn't break the spiral.

## Proposed Fixes

### Fix 1: Daily Ordering for Customer DCs (HIGH PRIORITY)

**File:** `src/prism_sim/agents/replenishment.py`

**Current Behavior:** Customer DCs order every 3 days (order_cycle_days=3)
```python
# Line ~632-634
order_day = hash(n_id) % order_cycle_days
if day % order_cycle_days != order_day:
    continue  # Skip this DC today
```

**Problem:** On non-order days, demand signals accumulate but orders don't flow. When a DC finally orders, it may still under-order if its IP hasn't dropped enough.

**Fix:** Set `order_cycle_days=1` for Customer DCs, or remove the cycle restriction entirely for Customer DCs using echelon logic.

```python
# In generate_orders(), for Customer DCs in echelon logic:
# ALWAYS allow ordering (remove cycle restriction)
if t_idx in self._customer_dc_indices:
    pass  # Always process, don't skip based on order_day
```

### Fix 2: Increase Customer DC Target Days (MEDIUM PRIORITY)

**File:** `src/prism_sim/agents/replenishment.py` or `src/prism_sim/config/simulation_config.json`

**Current:** Customer DCs have 21-day target, 14-day ROP

**Problem:** With 21-day target and echelon demand of ~6,600/day:
- Target = 138,600 cases
- If Local IP = 100,000 → Order only 38,600
- This is insufficient to cover downstream demand plus rebuild buffer

**Fix:** Increase to 35-day target, 21-day ROP for Customer DCs

```json
// In simulation_config.json, channel_profiles section:
"B2M_LARGE": {
    "target_days": 35.0,
    "reorder_point_days": 21.0
},
"B2M_DISTRIBUTOR": {
    "target_days": 35.0,
    "reorder_point_days": 21.0
}
```

### Fix 3: Push-Based Allocation from RDCs (MEDIUM PRIORITY)

**File:** `src/prism_sim/simulation/orchestrator.py` (new logic)

**Current:** RDCs only ship in response to Customer DC orders (pure pull)

**Problem:** When Customer DCs under-order, RDCs accumulate inventory with no mechanism to push it downstream.

**Fix:** Add push allocation when RDC DOS exceeds threshold:

```python
# In orchestrator.py, after order processing:
def _push_excess_rdc_inventory(self, day: int) -> list[Order]:
    """Push excess RDC inventory to Customer DCs when DOS > threshold."""
    push_orders = []
    threshold_dos = 30  # days

    for rdc_id in self._rdc_ids:
        rdc_idx = self.state.node_id_to_idx[rdc_id]
        rdc_inv = self.state.inventory[rdc_idx]

        # Calculate DOS based on recent outflow
        avg_outflow = self._get_rdc_avg_outflow(rdc_id)
        dos = rdc_inv / avg_outflow if avg_outflow > 0 else float('inf')

        if dos > threshold_dos:
            # Push excess to downstream Customer DCs proportionally
            excess = rdc_inv - (avg_outflow * threshold_dos)
            # Create push orders to Customer DCs...
```

### Fix 4: Safety Stock Multiplier for Echelon Logic (LOW PRIORITY)

**File:** `src/prism_sim/agents/replenishment.py`

**Current:** Echelon target = Echelon_Demand × Target_Days

**Problem:** This doesn't account for variance in demand or lead time at each echelon level.

**Fix:** Add multiplier to echelon target:

```python
# In echelon logic section:
echelon_safety_multiplier = 1.3  # 30% buffer
echelon_target = current_e_demand * dc_target_days * echelon_safety_multiplier
```

## Implementation Order

1. **Fix 1 (Daily Ordering)** - Quick config change, high impact
2. **Fix 2 (Increase Target Days)** - Config change, medium impact
3. **Fix 3 (Push Allocation)** - New code, medium complexity
4. **Fix 4 (Safety Multiplier)** - Simple code change, low impact

## Validation

After each fix:

```bash
# Run 90-day validation
poetry run python run_simulation.py --days 90 --output-dir data/results/fix_test

# Check triangle report
cat data/results/fix_test/triangle_report.txt

# Run diagnostics
poetry run python scripts/analysis/diagnose_service_level.py data/results/fix_test
poetry run python scripts/analysis/diagnose_slob.py data/results/fix_test
```

**Success Criteria:**
- Service Level > 90%
- SLOB < 30%
- MFG RDC Inventory < 50% of total
- Store Inventory > 15% of total

## Files to Modify

| File | Changes |
|------|---------|
| `src/prism_sim/agents/replenishment.py` | Daily ordering, safety multiplier |
| `src/prism_sim/config/simulation_config.json` | Target days increase |
| `src/prism_sim/simulation/orchestrator.py` | Push allocation (if needed) |

## Diagnostic Scripts

Use these to analyze results:

```bash
# Service level analysis
poetry run python scripts/analysis/diagnose_service_level.py data/results/<run> --csv

# SLOB/inventory distribution analysis
poetry run python scripts/analysis/diagnose_slob.py data/results/<run> --csv
```

## Background

- Issue first identified in v0.18.0
- v0.19.0 added MEIO (echelon inventory) but used wrong IP calculation
- v0.19.1 fixed IP calculation and MRP signal, marginal improvement
- This plan addresses remaining architectural issues
