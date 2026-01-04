# Product Mix Optimization Plan

## Context

After implementing rate-based production (v0.19.9), the 365-day service level improved from 50% to 67.5%. However, there's still a 17.5pp gap to the 85%+ target.

**Key insight from diagnostics:** Production rate is no longer the bottleneck. The remaining gap is due to **product mix mismatch** - the system produces the right total quantity but the wrong product mix.

### Evidence

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Production Rate | 412K/day (97% of expected) | Not bottlenecked |
| OEE | 88% | Plants running well |
| SLOB | 70% | 70% of inventory is slow-moving |
| 30-day SL | 93% | Short-term works well |
| 365-day SL | 67.5% | Long-term degrades |

### Root Cause Analysis

The rate-based production uses a **fixed expected mix** derived from `base_demand_matrix` at initialization. This creates issues:

1. **Seasonality Ignored**: Demand has ±12% seasonal variation, but expected mix is static
2. **C-Item Accumulation**: C-items produced at expected rate accumulate as SLOB
3. **A-Item Stockouts**: A-items may stockout during demand spikes while C-items pile up

## Proposed Solutions

### Option 1: Dynamic Mix from Actual Demand

Replace fixed expected mix with rolling actual demand signal.

**Implementation:**
```python
# In _generate_rate_based_orders():

# Current (static mix):
baseline_qty = expected_demand  # From base_demand_matrix at init

# Proposed (dynamic mix):
# Use 7-day rolling average of actual demand for mix
actual_demand_avg = np.mean(self.demand_history, axis=0)
total_actual = np.sum(actual_demand_avg)
total_expected = np.sum(self.expected_daily_demand)

# Scale to expected RATE but use actual MIX
if total_actual > 0:
    actual_mix = actual_demand_avg / total_actual
    baseline_qty = actual_mix[p_idx] * total_expected
else:
    baseline_qty = expected_demand
```

**Pros:**
- Automatically adjusts to seasonal patterns
- Responds to demand shifts within 7 days
- No manual seasonal calibration needed

**Cons:**
- Could amplify bullwhip if demand signal is noisy
- Delayed response (7-day lag)
- May still under-produce during sudden demand spikes

### Option 2: Seasonal Adjustment Factor

Apply known seasonal multiplier to expected demand.

**Implementation:**
```python
# In generate_production_orders():

# Calculate seasonal factor for current day
seasonal_config = self.config.get("simulation_parameters", {}).get("demand", {}).get("seasonality", {})
amplitude = seasonal_config.get("amplitude", 0.12)
phase_shift = seasonal_config.get("phase_shift_days", 150)
cycle_days = seasonal_config.get("cycle_days", 365)

# Seasonal multiplier: 1 + amplitude * sin(2*pi*(day - phase)/cycle)
import math
seasonal_factor = 1.0 + amplitude * math.sin(
    2 * math.pi * (current_day - phase_shift) / cycle_days
)

# Apply to expected demand
adjusted_expected = self.expected_daily_demand * seasonal_factor
```

**Pros:**
- Proactive (anticipates demand, not reactive)
- No lag - adjusts immediately
- Smooth, predictable production schedule

**Cons:**
- Only handles known seasonality, not unexpected demand shifts
- Requires accurate seasonal parameters
- Doesn't address product-level mix variations

### Option 3: ABC-Prioritized Production Scheduling (Recommended)

Ensure A-items always get produced first, with C-items only when capacity allows.

**Implementation in TransformEngine:**
```python
# In process_production_orders():

# Current: Sort by ABC priority then due date
# This is already implemented but may not be aggressive enough

# Proposed: Reserve capacity for A-items
def _allocate_production_capacity(self, orders: list[ProductionOrder], day: int):
    # Calculate available capacity per plant
    plant_capacity = self._get_available_capacity(day)

    # Phase 1: Allocate capacity to A-items (up to 80% of capacity)
    a_items = [o for o in orders if self._get_abc_class(o.product_id) == 'A']
    a_capacity_limit = {p: cap * 0.80 for p, cap in plant_capacity.items()}
    a_allocated = self._allocate_orders(a_items, a_capacity_limit)

    # Phase 2: Allocate remaining capacity to B-items
    remaining = {p: cap - used for p, (cap, used) in ...}
    b_items = [o for o in orders if self._get_abc_class(o.product_id) == 'B']
    b_allocated = self._allocate_orders(b_items, remaining)

    # Phase 3: C-items get whatever is left
    c_items = [o for o in orders if self._get_abc_class(o.product_id) == 'C']
    c_allocated = self._allocate_orders(c_items, remaining)

    return a_allocated + b_allocated + c_allocated
```

**Pros:**
- Guarantees A-items get priority (80% of volume)
- Naturally reduces C-item SLOB
- Works regardless of demand signal quality

**Cons:**
- C-items may stockout during unusual demand patterns
- Requires careful capacity reservation tuning
- More complex implementation

### Option 4: Demand-Driven Production with Safety Stock Buffers

Different production strategy per ABC class.

**Implementation:**
```python
# A-items: Produce at MAX(expected, actual_demand) - never stockout
# B-items: Produce at expected rate
# C-items: Produce at MIN(expected, actual_demand) - minimize SLOB

if abc_class == 'A':
    baseline_qty = max(expected_demand, actual_demand_avg)
elif abc_class == 'B':
    baseline_qty = expected_demand
else:  # C-items
    baseline_qty = min(expected_demand, actual_demand_avg) * 0.8  # 80% of lower
```

**Pros:**
- Tailored strategy per product class
- Naturally prioritizes A-items
- Reduces C-item SLOB aggressively

**Cons:**
- C-items may experience frequent stockouts
- May over-produce A-items (increases inventory cost)

## Recommended Implementation Order

### Phase 1: Quick Win - Dynamic Mix (Option 1)
**Effort:** Low (modify `_generate_rate_based_orders`)
**Impact:** Medium (addresses seasonal mix shift)

1. Add rolling 7-day demand history for mix calculation
2. Use actual mix, scaled to expected total rate
3. Add config flag `dynamic_mix_enabled`

### Phase 2: ABC Capacity Reservation (Option 3)
**Effort:** Medium (modify TransformEngine)
**Impact:** High (guarantees A-item availability)

1. Add `a_item_capacity_reservation_pct` config (default 0.8)
2. Modify `process_production_orders` to reserve capacity
3. Track ABC-level fill rates in diagnostics

### Phase 3: Seasonal Pre-Adjustment (Option 2)
**Effort:** Low (add seasonal factor calculation)
**Impact:** Medium (proactive seasonal response)

1. Calculate seasonal factor from config
2. Apply to expected demand in rate-based mode
3. Add `seasonal_adjustment_enabled` config flag

## Files to Modify

| File | Changes |
|------|---------|
| `src/prism_sim/simulation/mrp.py` | Dynamic mix, seasonal adjustment |
| `src/prism_sim/simulation/transform.py` | ABC capacity reservation |
| `src/prism_sim/config/simulation_config.json` | New config parameters |
| `docs/planning/debug_foxtrot.md` | Update with results |

## Success Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| 365-day Service Level | 67.5% | 85%+ | Primary goal |
| SLOB | 70% | <30% | Product mix indicator |
| A-Item Fill Rate | Unknown | 95%+ | Track separately |
| C-Item Fill Rate | Unknown | 80%+ | Acceptable lower |

## Testing Strategy

1. **30-day regression test**: Ensure no degradation (>90% SL)
2. **365-day validation**: Target 85%+ SL
3. **ABC breakdown**: Track fill rate by ABC class
4. **SLOB tracking**: Monitor weekly SLOB trend

## Risk Considerations

1. **Bullwhip Amplification**: Dynamic mix could amplify demand signal noise
   - Mitigation: Use longer rolling window (14 days vs 7)

2. **C-Item Stockouts**: Aggressive SLOB reduction may cause stockouts
   - Mitigation: Maintain minimum C-item production floor (50% of expected)

3. **Capacity Starvation**: ABC reservation might leave capacity unused
   - Mitigation: Allow B/C items to use unreserved A-item capacity

## Related Documentation

- `docs/planning/debug_foxtrot.md` - Starvation loop analysis
- `docs/planning/alignment_and_param_fix.md` - ABC alignment
- `CHANGELOG.md` - Version history
