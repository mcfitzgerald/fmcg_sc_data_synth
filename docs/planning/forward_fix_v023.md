# Forward Fix List: Prism Sim v0.23.0

> **Purpose**: This document captures remaining issues and proposed fixes for a fresh planning session.
> **Context**: Death spiral is FIXED. Production stable at ~8M/day over 365 days.

---

## Current State (v0.23.0)

### What Works
- Campaign batching production (real FMCG-style scheduling)
- 365-day runs complete without memory explosion
- Mass balance, capacity constraints, physics all correct
- Production matches demand (~8M/day stable)

### Metrics Off-Target

| Metric | Current | Target | Severity |
|--------|---------|--------|----------|
| Service Level | 80.5% | >85% | Medium |
| Truck Fill | 43% | >85% | **High** |
| SLOB | 82% | <30% | **High** |
| OEE | 40% | 65-85% | Low* |

*OEE is intentionally low - we're only producing what's needed, not using excess capacity.

---

## Issue P0: Order Explosion → Low Truck Fill

### Symptom
- 300M+ orders generated on Day 1
- Truck fill rate 43% (target >85%)
- Many small LTL shipments instead of consolidated FTL

### Root Cause
```
Current: 6000 stores × 500 SKUs × daily reorder = millions of orders
         Each (store, SKU) pair generates separate order when below ROP

Real World: Store generates ONE order with multiple SKUs
            DC consolidates orders into FTL shipments
```

### Proposed Fix: Warm Start + Order Consolidation

**Option A: Warm Start (Quick Win)**
- Start simulation at Day 30 instead of Day 1
- Pre-populate in-transit inventory from a baseline run
- Avoids "cold start" order explosion
- Files: `simulation/orchestrator.py` (add warm start mode)

**Option B: Order Aggregation (Proper Fix)**
- Store-level consolidation: One order per store, multiple SKU lines
- DC-level batching: Aggregate orders within 1-day window before shipping
- Files: `agents/replenishment.py`, `simulation/logistics.py`

**Expected Impact**:
- Orders: 300M → ~100K/day
- Truck fill: 43% → 70%+
- Memory: Significantly reduced

---

## Issue P1: SLOB Inventory (82% vs <30%)

### Symptom
- 82% of inventory has >60 days supply
- Capital tied up in slow-moving items
- Warehouses full of wrong products

### Root Cause
```
Initial Seeding Problem:
- store_days_supply: 4.5 days
- rdc_days_supply: 7.5 days
- But seeded UNIFORMLY across all SKUs
- C-items (low velocity) get same DOS as A-items
- C-item inventory sits for months

Production Mismatch:
- Campaign batching produces 10 days' worth per SKU
- But C-items only sell ~100 units/day
- Creates SLOB accumulation over time
```

### Proposed Fix: ABC-Differentiated Seeding

**Option A: Reduce Initial Inventory**
```json
"initialization": {
  "store_days_supply_a": 7.0,   // A-items: higher DOS
  "store_days_supply_b": 4.0,   // B-items: medium
  "store_days_supply_c": 2.0,   // C-items: lean
}
```
Files: `simulation/orchestrator.py:_initialize_store_inventory()`

**Option B: SLOB Liquidation**
- Add logic to reduce production when DOS > 45 days
- Already partially implemented in campaign batching (SLOB throttling)
- May need stronger throttling for C-items

**Option C: Smaller C-Item Batches**
```python
# In campaign batching, reduce horizon for C-items
if abc == 2:  # C-item
    batch_qty = sustainable_demand * 5  # 5 days vs 10
```

**Expected Impact**: SLOB 82% → 30-40%

---

## Issue P2: Service Level (80.5% vs 85%)

### Symptom
- 4.5% gap to target
- Stockouts during production cycles

### Root Cause
```
Campaign batching creates gaps:
- A-item triggers at DOS < 14, gets 10 days' worth
- But lead time is 3 days
- Gap: DOS drops to ~4 before production arrives
- Stockouts during the gap
```

### Proposed Fix: Trigger Earlier + Safety Stock

**Option A: Increase Trigger Thresholds**
```json
"campaign_batching": {
  "trigger_dos_a": 17,  // was 14
  "trigger_dos_b": 12,  // was 10
  "trigger_dos_c": 9,   // was 7
}
```

**Option B: Increase Store Safety Stock**
```json
"replenishment": {
  "min_safety_stock_days": 5.0,  // was 3.0
}
```

**Expected Impact**: Service 80.5% → 85%+

---

## Recommended Approach for Fresh Session

### Phase 1: Quick Wins (30 min)
1. Tune campaign batching triggers (P2 fix)
2. Reduce C-item batch sizes (P1 partial fix)
3. Run 365-day validation

### Phase 2: Warm Start Implementation (1-2 hours)
1. Add warm start mode to orchestrator
2. Pre-populate in-transit shipments from baseline
3. Skip first 14-30 days of simulation

### Phase 3: Order Consolidation (if needed, 2-4 hours)
1. Refactor replenishment to generate store-level orders
2. Add consolidation window to logistics
3. Update order data structures

---

## Key Files

| File | What to Change |
|------|----------------|
| `simulation/orchestrator.py` | Warm start, inventory seeding |
| `simulation/mrp.py` | Campaign batching parameters |
| `agents/replenishment.py` | Order aggregation |
| `simulation/logistics.py` | Shipment consolidation |
| `config/simulation_config.json` | All tuning parameters |

---

## Quick Reference: Current Campaign Config
```json
"campaign_batching": {
  "enabled": true,
  "production_horizon_days": 10,
  "trigger_dos_a": 14,
  "trigger_dos_b": 10,
  "trigger_dos_c": 7,
  "max_skus_per_plant_per_day": 60
}
```
