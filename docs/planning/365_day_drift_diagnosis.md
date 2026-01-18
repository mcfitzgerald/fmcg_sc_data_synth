# 365-Day Service Level Drift - Diagnosis Plan

## Problem Statement

Service level degrades over simulation duration:
- 30-day: 96-99% SL (healthy)
- 90-day: ~90% SL (degrading)
- 365-day: ~80% SL (broken)

This is NOT a config regression - it's a structural simulation issue that has never been solved.

## Current Metrics (365-day run with "working" config)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Service Level | 80.72% | 95%+ | ❌ |
| Inventory Turns | 6.04x | ~6x | ✅ |
| SLOB | 1.9% | <15% | ✅ |
| OEE | 7.4% | ~15% | ⚠️ Low but expected |

## Documented Root Cause (from llm_context.md §19.7)

> **Metrics healthy at 30 days but degrade at 365 days?**
> - **SLOB drift:** C-items accumulating (slow movers don't sell)
> - **Service drift:** SLOB throttling reducing production
> - **Key diagnostic:** Compare ABC class inventory at day 30 vs day 365

## Hypothesis

1. C-items (slow movers) accumulate inventory over time
2. SLOB throttling logic sees high DOS and reduces production
3. Reduced production eventually starves A-items (fast movers)
4. Service level drops because A-items (80% of volume) can't be fulfilled

## Key Files to Investigate

### SLOB Throttling Logic
- `src/prism_sim/simulation/mrp.py` - Look for SLOB-related production throttling
- Search for: `slob`, `throttle`, `dos`, `inventory_cap`

### ABC Production Floors
- `src/prism_sim/simulation/mrp.py` - ABC production floor logic
- Config: `simulation_config.json` → `mrp_thresholds.abc_production_enabled`
- The floor should protect A-items but may not be working correctly

### Campaign Batching
- `src/prism_sim/simulation/mrp.py` → `campaign_batching` section
- Config: `trigger_dos_a/b/c` = 14/10/5
- Products only produce when DOS drops below trigger

### Replenishment
- `src/prism_sim/agents/replenishment.py` - Order generation logic
- May be under-ordering for A-items or over-ordering for C-items

## Diagnostic Steps

### Step 1: Add ABC-level metrics tracking
Modify the triangle report or add logging to show:
- Service level BY ABC class (A/B/C separately)
- Inventory DOS BY ABC class over time
- Production volume BY ABC class

### Step 2: Compare day 30 vs day 365 snapshots
Run simulation and capture:
```python
# At day 30 and day 365, log:
# - Inventory by ABC class
# - DOS by ABC class
# - Production orders by ABC class
# - Fulfilled vs unfulfilled demand by ABC class
```

### Step 3: Trace SLOB throttling
In `mrp.py`, find where SLOB throttling is applied and add logging:
- What DOS threshold triggers throttling?
- Is it applied per-SKU or globally?
- Does it respect ABC floors?

### Step 4: Check A-item starvation
If A-items are being starved:
- Is production capacity being consumed by C-items?
- Is the ABC priority system working in allocation?
- Are A-item orders being generated but not fulfilled?

## Potential Solutions to Explore

### Option A: Disable SLOB throttling entirely
- Config: Set `inventory_cap_dos` very high (999)
- Risk: Inventory explosion, but would prove the hypothesis

### Option B: ABC-aware SLOB throttling
- Only throttle C-items for SLOB, never A-items
- A-items should always produce at floor level regardless of DOS

### Option C: Separate production pools
- Dedicate % of capacity to each ABC class
- Prevents C-item production from crowding out A-items

### Option D: Dynamic C-item production reduction
- Reduce C-item production over time if SLOB accumulating
- But maintain A/B production at stable levels

## Config Reference (current working values)

```json
{
  "campaign_batching": {
    "trigger_dos_a": 14,
    "trigger_dos_b": 10,
    "trigger_dos_c": 5
  },
  "abc_production_enabled": true,
  "a_production_buffer": 1.2,
  "c_production_factor": 0.35,
  "c_demand_factor": 0.6,
  "inventory_cap_dos": 45.0
}
```

## Commands to Run

```bash
# Clear checkpoint and run fresh
rm -f data/checkpoints/steady_state_*.json.gz

# Short run to verify baseline
poetry run python run_simulation.py --days 30

# Full run to reproduce drift
poetry run python run_simulation.py --days 365

# Run tests
poetry run pytest tests/ -v
```

## Success Criteria

A proper fix should achieve:
- 365-day Service Level: ≥95%
- 365-day Inventory Turns: ~6x (not degraded)
- 365-day SLOB: <15%
- Stable metrics from day 30 through day 365 (no drift)

## Related Commits

- `2639064` - Restored "working" config (but only tested 30-day)
- `ca30083` - v0.35.5 noted drift as "pre-existing structural issue"
- `e64801c` - v0.23.0 campaign batching (may have introduced drift)

## Files to Read First

1. `docs/llm_context.md` - Full architecture context
2. `src/prism_sim/simulation/mrp.py` - MRP and production logic
3. `src/prism_sim/config/simulation_config.json` - All tunable parameters
