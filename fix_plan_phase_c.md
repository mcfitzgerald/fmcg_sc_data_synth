# Fix Plan Phase C: System Stabilization

**Created:** 2025-12-30
**Status:** PENDING
**Prerequisite:** Read `docs/llm_context.md` for architecture overview

---

## Context for New Session

This plan addresses critical bugs discovered in a 365-day baseline simulation (no quirks/events enabled). The simulation collapses around day 22-27.

### How to Reproduce

```bash
poetry run python run_simulation.py --days 365 --output-dir data/results/baseline_365
```

### Simulation Results (Pre-Fix)

| Metric | Result | Target |
|--------|--------|--------|
| Service Level (OSA) | **8.81%** | >93% |
| OEE | **1.8%** | 75-85% |
| Inventory Turns | **0.17x** | 8-12x |
| SLOB % | **100%** | <30% |
| Cash-to-Cash | **2196 days** | 40-50 days |

**Key Observations from Console Output:**
- Days 1-11: System warming up, shipments start
- Day 12: Production starts (363k cases)
- Days 12-22: Production declining (444k → 80k → 0)
- Day 23+: Complete collapse (Ship=0, Arr=0, Prod=0)
- Orders continue growing: 355k → 4.7M → 6.7M (bullwhip)

---

## Root Cause Analysis

### The Death Spiral Mechanism

```
1. Initial inventory drains (14+ days of DOS consumed)
         ↓
2. RDCs can't ship to Stores (no inventory)
         ↓
3. MRP gets no RDC→Store shipment signal (orchestrator.py:224-232)
         ↓
4. MRP generates no production orders
         ↓
5. No production → No finished goods at Plants
         ↓
6. No replenishment to RDCs
         ↓
7. Loop continues → Complete collapse
```

### Production Capacity Deficit

- **Daily demand:** ~390k cases/day
- **Actual production (days 12-22):** 2.8M total = ~255k/day average
- **Production rate:** Only 65% of demand
- Initial inventory couldn't bridge the gap

### Ingredient Supplier Routing Bug

In `mrp.py:_find_supplier_for_ingredient()`:
- SPOF ingredient (`ACT-CHEM-001`) routes ALL plants to `SUP-001`
- But `SUP-001` only has link to `PLANT-CA` per `data/output/static_world/links.csv`
- PLANT-OH, PLANT-TX, PLANT-GA can't get the SPOF ingredient

---

## Key Files to Modify

| File | Purpose |
|------|---------|
| `src/prism_sim/simulation/mrp.py` | MRP engine - production order generation |
| `src/prism_sim/simulation/state.py` | State manager - may need expected_demand |
| `src/prism_sim/config/simulation_config.json` | Runtime parameters |
| `src/prism_sim/config/world_definition.json` | Static world config (run rates) |

---

## Critical Fixes (Priority Order)

### C.1: MRP Demand Fallback (CRITICAL - Prevents Death Spiral)

**Problem:** MRP uses RDC→Store shipments as signal (`rdc_store_shipments` in orchestrator.py:224-232). When shipments stop, the 7-day moving average drops to near-zero, and no production orders are generated.

**File:** `src/prism_sim/simulation/mrp.py`

**Current Code (lines 142-152):**
```python
def generate_production_orders(self, current_day, rdc_shipments, active_production_orders):
    # 1. Update Demand History with daily shipment volume
    self._update_demand_history(current_day, rdc_shipments)

    # Calculate Moving Average Demand
    avg_daily_demand_vec = np.mean(self.demand_history, axis=0)
    # ... continues to use avg_daily_demand_vec for ROP checks
```

**Fix Strategy:**
1. Track expected daily demand (from `POSEngine.get_average_demand_estimate()` or config)
2. If shipment-based signal drops below 10% of expected, fall back to expected demand
3. This prevents the death spiral while maintaining the "lumpy signal" benefit during normal ops

**Implementation:**
```python
def __init__(self, world, state, config):
    # ... existing init ...

    # Add: Cache expected daily demand for fallback
    demand_config = config.get("simulation_parameters", {}).get("demand", {})
    cat_profiles = demand_config.get("category_profiles", {})
    # Simple estimate: base_demand * n_stores * n_products_per_category
    # More accurate: get from POSEngine if available
    self.expected_daily_demand = np.ones(self.state.n_products) * 100.0  # Placeholder

def generate_production_orders(self, current_day, rdc_shipments, active_production_orders):
    self._update_demand_history(current_day, rdc_shipments)
    avg_daily_demand_vec = np.mean(self.demand_history, axis=0)

    # FIX: Fallback to prevent death spiral
    total_signal = np.sum(avg_daily_demand_vec)
    expected_total = np.sum(self.expected_daily_demand)
    if total_signal < expected_total * 0.1:  # Signal dropped below 10%
        avg_daily_demand_vec = np.maximum(avg_daily_demand_vec, self.expected_daily_demand)

    # ... rest of method unchanged
```

---

### C.2: Fix Supplier-Plant Routing (CRITICAL)

**Problem:** `_find_supplier_for_ingredient()` returns `SUP-001` for ALL plants needing SPOF ingredient, but only PLANT-CA has a link to SUP-001.

**File:** `src/prism_sim/simulation/mrp.py`

**Current Code (lines 344-362):**
```python
def _find_supplier_for_ingredient(self, plant_id: str, ing_id: str) -> str | None:
    # Special case for SPOF
    spof_config = mfg_config.get("spof", {})
    if ing_id == spof_config.get("ingredient_id"):
        val = spof_config.get("primary_supplier_id")
        return str(val) if val else None  # BUG: Returns SUP-001 even if no link!

    # Generic case - this works correctly
    for link in self.world.links.values():
        if link.target_id == plant_id:
            # ...
```

**Fix:**
```python
def _find_supplier_for_ingredient(self, plant_id: str, ing_id: str) -> str | None:
    mfg_config = self.config.get("simulation_parameters", {}).get("manufacturing", {})
    spof_config = mfg_config.get("spof", {})

    # Check SPOF ingredient - but verify link exists!
    if ing_id == spof_config.get("ingredient_id"):
        primary = spof_config.get("primary_supplier_id")
        if primary:
            # Only use primary if valid link exists
            has_link = any(
                l.source_id == primary and l.target_id == plant_id
                for l in self.world.links.values()
            )
            if has_link:
                return primary
            # Fall through to generic case if no link

    # Generic case: find any supplier linked to this plant
    for link in self.world.links.values():
        if link.target_id == plant_id:
            source_node = self.world.nodes.get(link.source_id)
            if source_node and source_node.type == NodeType.SUPPLIER:
                return source_node.id

    return None
```

---

### C.3: Increase Production Capacity (HIGH)

**Problem:** Capacity (255k actual) < Demand (390k) = 65% coverage

**Current Config Values (`simulation_config.json`):**
```json
"production_hours_per_day": 20.0,
"plant_parameters": {
  "PLANT-OH": { "efficiency_factor": 0.78 },
  "PLANT-TX": { "efficiency_factor": 0.88 },
  "PLANT-CA": { "efficiency_factor": 0.82 },
  "PLANT-GA": { "efficiency_factor": 0.80 }
}
```

**Current Run Rates (`world_definition.json`):**
- ORAL: 7500 cases/hour
- PERSONAL: 9000 cases/hour
- HOME: 6000 cases/hour

**Capacity Math (Current):**
```
PLANT-OH (HOME):     6000 × 20h × 0.78 = 93,600/day
PLANT-TX (ORAL):     7500 × 20h × 0.88 = 132,000/day
PLANT-CA (ORAL+PERS): avg 8250 × 20h × 0.82 = 135,300/day
PLANT-GA (PERS+HOME): avg 7500 × 20h × 0.80 = 120,000/day
─────────────────────────────────────────────────
TOTAL THEORETICAL: ~481k/day
ACTUAL (with changeover, material constraints): ~255k/day (53%)
```

**Fix Options:**
1. **Increase production hours to 24h** (3-shift 24/7 operation)
   - New capacity: 481k × 24/20 = 577k/day theoretical

2. **Increase run rates by 25%** in `world_definition.json`
   - Must regenerate static world: `poetry run python scripts/generate_static_world.py`

3. **Reduce changeover times** in category profiles

**Recommended:** Option 1 (simplest, config-only change):
```json
"production_hours_per_day": 24.0
```

---

### C.4: Reduce Initial Inventory (HIGH - Adds Realism)

**Problem:** Initial 930M cases masks issues and creates unrealistic scenario.

**Current Config:**
```json
"initialization": {
  "store_days_supply": 14.0,
  "rdc_days_supply": 21.0,
  "rdc_store_multiplier": 100.0
},
"initial_plant_inventory": {
  "ACT-CHEM-001": 10000000.0,
  "BLK-WATER-001": 10000000.0
}
```

**Fix:** Reduce to realistic levels:
```json
"initialization": {
  "store_days_supply": 7.0,
  "rdc_days_supply": 14.0,
  "rdc_store_multiplier": 50.0
},
"initial_plant_inventory": {
  "ACT-CHEM-001": 500000.0,
  "BLK-WATER-001": 200000.0
}
```

**Note:** Only apply AFTER C.1-C.3 are working to avoid immediate collapse.

---

### C.5: Production Order Smoothing (MEDIUM)

**Problem:** Production swings wildly (0 → 444k → 80k → 0)

**File:** `src/prism_sim/simulation/mrp.py`

**Fix:** Add production order history and cap increases:
```python
def __init__(self, ...):
    # ... existing ...
    self.production_order_history = np.zeros(7)  # 7-day history
    self._prod_hist_ptr = 0

def generate_production_orders(self, ...):
    # ... after calculating order_qty ...

    # Smooth production orders
    avg_recent = np.mean(self.production_order_history)
    if avg_recent > 0:
        max_increase = avg_recent * 1.5  # Cap at 50% increase
        order_qty = min(order_qty, max_increase)

    # Update history
    self.production_order_history[self._prod_hist_ptr] = total_orders_today
    self._prod_hist_ptr = (self._prod_hist_ptr + 1) % 7
```

---

## Verification Criteria

After implementing fixes, run:
```bash
poetry run python run_simulation.py --days 365 --output-dir data/results/phase_c_test
```

**Expected Results:**

| Metric | Pre-Fix | Target |
|--------|---------|--------|
| Service Level | 8.8% | >90% |
| OEE | 1.8% | 75-85% |
| Inventory Turns | 0.17x | 8-12x |
| Production (day 365) | 0 | >300k |
| Shipments (day 365) | 0 | >300k |
| Order Amplification | 20x | 3-9x |
| System Collapse | Day 27 | Never |

---

## Implementation Order

1. **C.1: MRP Fallback** - Prevents death spiral (do first!)
2. **C.2: Supplier Routing** - Ensures ingredient flow to all plants
3. **C.3: Capacity Increase** - Matches production to demand
4. **Test 30-day run** to verify system stabilizes
5. **C.4: Reduce Inventory** - Creates realistic scenario
6. **C.5: Production Smoothing** - Reduces volatility
7. **Test 365-day run** for final validation

---

## Future Enhancements (From fix_plan_v2.md)

### D.1: Backorder Handling (Optional)

Instead of Fill-or-Kill, allow partial fills to carry forward:
```python
# In allocation.py
if fill_rate < 1.0:
    unfilled_qty = order_qty * (1 - fill_rate)
    # Create backorder for next day instead of killing
```

### E.1: Scenario Capacity (Low Priority)

Add capacity multiplier config for "what-if" analysis:
```json
"capacity_scenarios": {
  "enabled": false,
  "multipliers": {
    "PLANT-OH": 1.0,
    "CONTRACT_MFG": 0.0  // 0 = disabled, 0.15 = +15% capacity
  }
}
```

---

## Related Documents

- `docs/llm_context.md` - Architecture overview and physics laws
- `fix_plan_v2.md` - Phase A & B implementation notes (superseded, historical reference)
- `docs/planning/intent.md` - Technical spec
- `CHANGELOG.md` - Version history

---

## Test Commands

```bash
# Quick 30-day validation
poetry run python run_simulation.py --days 30 --no-logging

# Full 365-day with logging
poetry run python run_simulation.py --days 365 --output-dir data/results/phase_c

# Analyze results
poetry run python scripts/analyze_bullwhip.py data/results/phase_c
```
