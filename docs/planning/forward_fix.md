# Forward Fix Plan: Death Spiral & Performance

## Executive Summary

The simulation suffers from two interrelated problems:
1. **Object Explosion**: 316M+ orders generated per day causing memory blowup
2. **Death Spiral**: Stores stockout → panic ordering → fragmented shipments → more stockouts

Current fixes (rate multiplier, changeover reduction) are **bandaids** that don't address root causes.

---

## Root Cause Analysis

### 1. Demand-Production Mismatch (Structural)

| Metric | Value |
|--------|-------|
| Daily consumer demand | ~8M cases |
| Theoretical plant capacity | ~8.4M cases (with 15x multiplier) |
| Actual production (before fix) | 2.6M cases |
| Actual production (after changeover fix) | 5.7M cases |

**Why production < theoretical:**
- 500 SKUs × 0.5h changeover = 250h needed per plant per day
- Plants only have 17-20h available
- Even with 0.1x changeover multiplier: 25h needed, still exceeds capacity

**Real fix needed:** The model conflates "production lines" with "plants". A real plant has multiple parallel lines. The rate_multiplier is a hack to simulate this.

### 2. Order Explosion (Algorithmic)

The replenishment agent generates **one order per (store, SKU, day)** when below ROP:
- 4,000+ stores × 500 SKUs = 2M potential orders/day
- When stores stockout, backlog multiplier inflates quantities
- Orders cascade through DC tiers, multiplying further

**Current state:** Day 1 generates 316M orders (cumulative across all tiers)

**Real fix needed:** Order aggregation at source. Stores should generate consolidated orders, not per-SKU orders.

### 3. Feedback Loop (Systemic)

```
Production < Demand
       ↓
Stores stockout
       ↓
Panic ordering (backlog_penalty_factor)
       ↓
Order explosion (memory)
       ↓
LTL fragmentation (low truck fill)
       ↓
Slow replenishment
       ↓
More stockouts
       ↓
[REPEAT]
```

---

## Proposed Solutions

### Phase 1: Stop the Bleeding (Quick Fixes)

These are already partially implemented but need validation:

1. **Increase production capacity** ✅ (rate_multiplier: 15→25)
2. **Reduce changeover time** ✅ (changeover_time_multiplier: 0.1)
3. **Group production orders** ✅ (sort by plant+product)

### Phase 2: Order Aggregation (Medium-Term)

**Goal:** Reduce order count from millions to thousands

#### 2.1 Store-Level Consolidation
- Instead of one order per SKU, create one order per store containing all SKUs
- Change `Order` to have multiple `OrderLine` items (already structured this way)
- Modify `ReplenishmentAgent.generate_orders()` to batch by destination

#### 2.2 DC-Level Consolidation
- RDCs should aggregate downstream demand before ordering from plants
- Implement order batching window (e.g., consolidate orders within 1-day window)

#### 2.3 Order Deduplication
- Track pending orders per (source, destination, product)
- Don't create new order if one already exists for same route

### Phase 3: Capacity Model Overhaul (Long-Term)

**Goal:** Properly model multi-line plants

#### 3.1 Production Line Entity
```python
@dataclass
class ProductionLine:
    id: str
    plant_id: str
    supported_categories: list[ProductCategory]
    run_rate_cases_per_hour: float
    current_product_id: str | None
```

#### 3.2 Line Assignment Logic
- MRP assigns production orders to specific lines
- Lines can run in parallel
- Changeover only affects individual line, not whole plant

#### 3.3 Capacity Planning
- Each plant has N lines (configurable)
- Total capacity = sum of line capacities
- Changeover is per-line, not per-plant

### Phase 4: Demand Scaling (Alternative)

If production model is too complex to fix, scale demand down:

1. Reduce store count (4000 → 1000)
2. Reduce SKU count (500 → 100)
3. Reduce base demand per store

This maintains simulation fidelity while fitting within capacity.

---

## Implementation Priority

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Order aggregation in replenishment | Medium | High - fixes memory explosion |
| P1 | Validate capacity with current multipliers | Low | Medium - confirms stability |
| P2 | Production line model | High | High - proper physics |
| P3 | Demand scaling fallback | Low | Medium - quick workaround |

---

## Metrics for Success

1. **Order count**: < 100K orders/day (currently 316M)
2. **Memory usage**: < 4GB for 365-day run
3. **Runtime**: < 5 minutes for 365-day run
4. **Service level**: > 90% sustained
5. **Production/Demand ratio**: > 1.0

---

## Files to Modify

### Order Aggregation (P0)
- `src/prism_sim/agents/replenishment.py` - Batch orders by destination
- `src/prism_sim/simulation/orchestrator.py` - Track pending orders

### Capacity Model (P2)
- `src/prism_sim/network/core.py` - Add ProductionLine entity
- `src/prism_sim/simulation/transform.py` - Multi-line scheduling
- `src/prism_sim/simulation/mrp.py` - Line-aware order generation
- `src/prism_sim/config/simulation_config.json` - Line configuration

---

## Current Bandaid Summary

Applied but not solving root cause:
- `production_rate_multiplier: 25.0` - Artificially inflates capacity
- `changeover_time_multiplier: 0.1` - Reduces changeover from 30min to 3min
- Order sorting by product - Minimizes changeover count

These allow ~9.5M cases/day production (vs 8M demand) but don't fix the 316M order explosion.
