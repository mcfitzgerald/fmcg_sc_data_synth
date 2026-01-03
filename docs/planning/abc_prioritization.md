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

## Proposed Solution: ABC Prioritization

### Phase 1: ABC-Prioritized Allocation (HIGH PRIORITY)

**File:** `src/prism_sim/agents/allocation.py`

**Current Behavior:** Orders are sorted by `order_type` priority (RUSH > PROMO > STANDARD), then processed FIFO.

**Problem:** When inventory is scarce at a source (DC/RDC), all products compete equally. A C-item order might get filled before an A-item order.

**Fix:** Add product velocity as a secondary sort key within each priority tier.

```python
# In AllocationAgent._prioritize_orders()
def _prioritize_orders(self, orders: list[Order]) -> list[Order]:
    """
    Sort orders by:
    1. Order type priority (RUSH=1, PROMO=2, STANDARD=5)
    2. Product velocity (A-items first within same priority)
    """
    def order_sort_key(order: Order) -> tuple[int, float]:
        priority = order.priority

        # Calculate order velocity (sum of line velocities)
        order_velocity = 0.0
        for line in order.lines:
            p_idx = self.state.product_id_to_idx.get(line.product_id)
            if p_idx is not None and self._product_velocity is not None:
                order_velocity += self._product_velocity[p_idx]

        # Return (priority, -velocity) so high velocity = earlier processing
        return (priority, -order_velocity)

    return sorted(orders, key=order_sort_key)
```

**Implementation Steps:**
1. Add `_product_velocity: np.ndarray` attribute to `AllocationAgent`
2. Add `set_product_velocity(velocity: np.ndarray)` method
3. Call from Orchestrator after POS engine initialization
4. Update `_prioritize_orders()` to use velocity as secondary sort

### Phase 2: ABC-Prioritized MRP (MEDIUM PRIORITY)

**File:** `src/prism_sim/simulation/mrp.py`

**Current Behavior:** All products are checked for replenishment equally. Production orders are created when DOS < ROP.

**Problem:** C-items trigger production orders just as readily as A-items, consuming plant capacity.

**Fix:** Apply different service levels per ABC class:
- A-items: z-score 2.33 (99% service level)
- B-items: z-score 1.65 (95% service level)
- C-items: z-score 1.28 (90% service level)

```python
# In MRPEngine.generate_production_orders()
# After calculating avg_daily_demand_vec

# Classify products by velocity
velocity = np.sum(self.expected_daily_demand)  # Or from POS
sorted_indices = np.argsort(avg_daily_demand_vec)[::-1]
cumsum = np.cumsum(avg_daily_demand_vec[sorted_indices])
total = np.sum(avg_daily_demand_vec)

# Build ABC mask
a_cutoff = np.searchsorted(cumsum, 0.80 * total)
b_cutoff = np.searchsorted(cumsum, 0.95 * total)

# Apply class-specific ROP multipliers
rop_multiplier = np.ones(self.state.n_products)
for i, idx in enumerate(sorted_indices):
    if i < a_cutoff:
        rop_multiplier[idx] = 1.2  # A-items: order earlier
    elif i < b_cutoff:
        rop_multiplier[idx] = 1.0  # B-items: standard
    else:
        rop_multiplier[idx] = 0.8  # C-items: order later
```

**Implementation Steps:**
1. Add ABC classification method `_classify_products_abc()`
2. Store ABC class per product: `self._abc_class: np.ndarray`
3. Apply class-specific ROP multipliers in `generate_production_orders()`
4. Add config parameters for ABC thresholds and multipliers

### Phase 3: ABC-Aware Replenishment (MEDIUM PRIORITY)

**File:** `src/prism_sim/agents/replenishment.py`

**Current Behavior:** ABC segmentation exists (z-scores A=2.33, B=1.65, C=1.28) but is applied to individual nodes, not products.

**Problem:** The segmentation is per-node, not per-product. A high-velocity product at a low-velocity store gets the store's z-score.

**Fix:** Apply product-level ABC classification to safety stock calculations.

```python
# In MinMaxReplenisher._compute_targets()
# Use product ABC class for z-score, not node ABC class

# Current (node-based):
z_score = self.z_scores_vec[node_idx]

# Proposed (product-based):
product_z_scores = self._get_product_z_scores()  # [n_products]
z_score = product_z_scores  # Use for safety stock calc
```

### Phase 4: Production Capacity Reservation (LOW PRIORITY)

**File:** `src/prism_sim/simulation/transform.py`

**Current Behavior:** Production orders processed by due date. No capacity reservation.

**Problem:** When plant capacity is constrained, C-item orders might consume capacity needed for A-items.

**Fix:** Reserve capacity for A-items before processing C-items.

```python
# In TransformEngine.process_production_orders()
# Split orders by ABC class, process A first

a_orders = [o for o in orders if self._get_abc_class(o.product_id) == 'A']
b_orders = [o for o in orders if self._get_abc_class(o.product_id) == 'B']
c_orders = [o for o in orders if self._get_abc_class(o.product_id) == 'C']

# Process in priority order
for order_batch in [a_orders, b_orders, c_orders]:
    sorted_batch = sorted(order_batch, key=lambda o: o.due_day)
    for order in sorted_batch:
        # ... existing processing logic
```

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
