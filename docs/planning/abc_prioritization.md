# ABC Prioritization Implementation Plan

## Problem Statement

The v0.19.2 simulation achieves **91.84% service level at 90 days** but degrades to **76% at 365 days**. Root cause analysis reveals a **product mix problem**, not a flow problem.

### Key Metrics (v0.19.2)
| Metric | 90-day | 365-day | Target |
|--------|--------|---------|--------|
| Service Level | 91.84% | 76% | >90% |
| SLOB | 54% | 73% | <30% |
| Inventory Turns | 6.12x | 4.69x | 6-14x |

### Demand Distribution Analysis
```
Total daily demand: 424,540 cases
Top 10 SKUs: 252,582 cases (59.5% of volume)

A-items (80% of volume): 16 SKUs
B-items (next 15%): 5 SKUs
C-items (bottom 5%): 47 SKUs (mostly ingredients with 0 demand)
```

### The Paradox
- **High SLOB (73%)** = inventory accumulating
- **Low Service Level (76%)** = stockouts occurring
- **Conclusion**: Wrong products are in stock. A-items stock out while C-items accumulate.

## Root Cause

### Why A-Items Stock Out
1. MRP produces all products equally (no demand weighting)
2. Allocation is FIFO by order priority, not by product velocity
3. When capacity is constrained, C-items consume resources that A-items need
4. Over 365 days, small imbalances compound into large inventory misallocation

### Why C-Items Accumulate
1. MRP minimum batch = 7 days of demand (still large for slow movers)
2. Low-velocity products get produced but don't sell
3. Batch production creates lumpy supply vs. smooth demand
4. Result: C-items sit in inventory for months (SLOB)

  Completed:
   1. Phase 1: ABC-Prioritized Allocation (agents/allocation.py)
       * Logic: When inventory is scarce at an RDC, orders are now sorted by (Order Priority, -Product Velocity).
   2. Phase 2: ABC-Prioritized MRP (simulation/mrp.py)
       * Logic: Production planning now applies dynamic Reorder Point (ROP) multipliers based on product classification.
   3. Phase 3: ABC-Aware Replenishment (agents/replenishment.py):
       * Logic: MinMaxReplenisher now uses config-driven thresholds (80/95%) for dynamic ABC classification.
       * Mechanism: Updates Z-scores based on product volume history using configurable thresholds.
   4. Phase 4: Production Capacity Reservation (simulation/transform.py):
       * Logic: Production orders are sorted by ABC priority (A > B > C) then Due Date.
       * Mechanism: TransformEngine classifies products using base demand and prioritizes A-item orders.

  Status:
   * All phases (1-4) completed.
   * Logic implemented and verified with 90-day simulation.
   * Service Level at 90 days: ~85% (Needs tuning, down from 91% baseline but ABC logic is active).
   * Ready for parameter tuning.

## Configuration Parameters

Add to `simulation_config.json`:

```json
{
  "simulation_parameters": {
    "agents": {
      "abc_prioritization": {
        "enabled": true,
        "a_threshold_pct": 0.80,
        "b_threshold_pct": 0.95,
        "a_rop_multiplier": 1.2,
        "b_rop_multiplier": 1.0,
        "c_rop_multiplier": 0.8,
        "a_allocation_priority": 1,
        "b_allocation_priority": 2,
        "c_allocation_priority": 3
      }
    }
  }
}
```

## Implementation Order

1. **Phase 1: ABC-Prioritized Allocation** (High impact, low complexity)
   - Files: `allocation.py`, `orchestrator.py`
   - Estimated changes: ~50 lines
   - Test: Run 90-day, verify A-item service level improves

2. **Phase 2: ABC-Prioritized MRP** (Medium impact, medium complexity)
   - Files: `mrp.py`
   - Estimated changes: ~80 lines
   - Test: Run 365-day, verify SLOB decreases

3. **Phase 3: ABC-Aware Replenishment** (Medium impact, higher complexity)
   - Files: `replenishment.py`
   - Estimated changes: ~60 lines
   - Test: Run 365-day, verify A-item inventory stability

4. **Phase 4: Production Capacity Reservation** (Lower impact, medium complexity)
   - Files: `transform.py`
   - Estimated changes: ~40 lines
   - Test: Run 365-day under capacity-constrained scenario

## Validation Criteria

After implementing ABC prioritization:

| Metric | Current (365-day) | Target |
|--------|-------------------|--------|
| Service Level | 76% | >90% |
| SLOB | 73% | <30% |
| A-item Service Level | ~70% (estimated) | >95% |
| C-item SLOB | ~90% (estimated) | <50% |

## Key Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/prism_sim/agents/allocation.py` | Order allocation | `allocate_orders()`, `_prioritize_orders()` |
| `src/prism_sim/simulation/mrp.py` | Production planning | `generate_production_orders()` |
| `src/prism_sim/agents/replenishment.py` | Order generation | `generate_orders()`, `_compute_targets()` |
| `src/prism_sim/simulation/transform.py` | Production execution | `process_production_orders()` |
| `src/prism_sim/simulation/orchestrator.py` | Main loop | `_step()`, initializers |
| `src/prism_sim/simulation/demand.py` | POS demand | `get_base_demand_matrix()` |

## Context for Fresh Session

### What Was Tried in v0.19.2
1. Daily ordering for Customer DCs - helps signal flow
2. Increased DC/Store targets (35/21 days) - provides buffer
3. Echelon safety multiplier (1.3x) - helps throughput
4. Push allocation from RDCs - moves inventory but doesn't fix product mix
5. Demand-proportional MRP batches - reduces SLOB slightly
6. Production prioritization by demand - made things worse (removed)

### What Works
- 90-day simulation: 91.84% service level
- Initial inventory priming provides good starting point
- Echelon logic correctly uses Local IP for Customer DCs
- MRP uses POS demand as floor for signal collapse

### What Doesn't Work
- 365-day service level degrades to 76%
- SLOB accumulates to 73%
- Product mix drifts over time (A-items deplete, C-items accumulate)
- Push allocation helps flow but doesn't address product priority

### Key Insight
The problem is not "how much" inventory flows, but "which products" get priority. The system needs to treat A-items (60% of demand) differently from C-items (5% of demand).

## Diagnostic Commands

```bash
# Run 90-day validation
poetry run python run_simulation.py --days 90 --output-dir data/results/abc_test

# Run 365-day validation
poetry run python run_simulation.py --days 365 --output-dir data/results/abc_test_365 --no-logging

# Check demand distribution
poetry run python -c "
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.demand import POSEngine
from prism_sim.simulation.state import StateManager
from prism_sim.config.loader import load_manifest, load_simulation_config
import numpy as np

manifest = load_manifest()
config = load_simulation_config()
config['promotions'] = manifest.get('promotions', [])
builder = WorldBuilder(manifest)
world = builder.build()
state = StateManager(world)
pos = POSEngine(world, state, config)

demand = np.sum(pos.get_base_demand_matrix(), axis=0)
sorted_idx = np.argsort(demand)[::-1]
cumsum = np.cumsum(demand[sorted_idx])
total = np.sum(demand)

print(f'A-items (80%): {np.searchsorted(cumsum, 0.8*total)} SKUs')
print(f'B-items (95%): {np.searchsorted(cumsum, 0.95*total)} SKUs')
print(f'Top 10 SKUs: {100*np.sum(demand[sorted_idx[:10]])/total:.1f}% of demand')
"
```
