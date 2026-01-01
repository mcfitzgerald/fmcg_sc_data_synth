# Forward Tasks - Phase C Continuation

## IMPORTANT: Validation Requirements

**Always run 365-day simulations for validation.** Short runs (30-90 days) mask critical issues:
- v0.15.5 showed 92.8% service level at 90 days, but collapsed to 69.95% at 365 days
- Production collapse (days 252-279) only visible in full-year runs
- Bullwhip cascades take 200+ days to fully manifest

```bash
# REQUIRED validation command
poetry run python run_simulation.py --days 365 --streaming --output-dir data/results/validation
```

---

## Session Summary (Jan 1, 2025 - v0.15.7) - COMPLETED

### Fixes Completed

#### 1. Inventory Turns Calculation - FIXED (v0.15.7)
**Root Cause:** Inventory turns was calculated using ALL inventory including 523M units of raw materials at plants. This inflated the denominator and showed 0.23x turns instead of actual ~6x.

**Fix:** Created finished goods mask to exclude INGREDIENT category products from inventory turns, SLOB, and shrinkage calculations.

**Files:** `orchestrator.py`

#### 2. MRP Demand Signal Dampening - FIXED (v0.15.6)
**Root Cause:** In 365-day simulations, production collapsed to zero during days 252-279. When stores had sufficient inventory, they ordered less, reducing the MRP shipment signal. The previous 10% collapse threshold didn't catch the gradual decline (signal was at 40-50% of expected). MRP calculated high Days-of-Supply and stopped production. Eventually stores depleted, triggering massive bullwhip (35M orders on day 281).

**Fix (Three-Pronged Approach):**
1. **Raised signal threshold** from 10% → 40% of expected demand
2. **Added velocity tracking** - detects week-over-week declining trends (week1 < 60% of week2)
3. **Added 30% production floor** - ensures minimum production regardless of signal

**Files:** `mrp.py`

#### 2. Customer DC Bullwhip Cascade - FIXED (v0.15.4)
**Root Cause:** Customer DCs (RET-DC, DIST-DC, ECOM-FC) used POS demand signal (=0, floored to 0.1) instead of derived demand. When stores ordered on Day 1, customer DCs saw inventory drop below their tiny ROP and mass-ordered on Day 2 (272M orders).

**Fix (MRP Derived Demand Theory):**
- Customer DCs now track allocation outflow (orders fulfilled to downstream stores) as their demand signal
- Warm start from `POSEngine.get_average_demand_estimate()` × downstream store count
- Order staggering: Customer DCs order on 5-day cycle (stores use 3-day)
- Inventory priming: Customer DC initial inventory scaled by downstream store count

**Files:** `replenishment.py`, `orchestrator.py`

#### 3. Fragmented Store Orders - FIXED (v0.15.5)
**Root Cause:** Stores ordered 20-40 cases but FTL required 300-1200 cases (5-20 pallets). Orders were held in `held_orders` for FTL consolidation, causing service level issues and low truck fill when they eventually shipped.

**Fix (LTL for Store Deliveries):**
- Differentiated shipping modes: FTL for DC-to-DC, LTL for DC-to-Store
- LTL shipments ship immediately without pallet minimum (min 10 cases)
- FTL shipments maintain pallet minimums for consolidation

**Files:** `logistics.py`, `simulation_config.json`

### Results (v0.15.7 - 365-day simulation)

| Metric | v0.15.6 | v0.15.7 | Target |
|--------|---------|---------|--------|
| Store Service Level | 73.34% | **75.32%** | 98.5% |
| Manufacturing OEE | 78.6% | **82.0%** | 75-85% ✓ |
| Inventory Turns | 0.23x | **6.18x** | 6-14x ✓ |
| SLOB | 100% | 60.3% | <30% |

---

## Remaining Issues (Priority Order)

### 1. Service Level (75.32% vs 98.5% target) - HIGH PRIORITY
**Status:** Improved from 73.34%, but still ~23pp below target

**Root Cause Identified (v0.15.7 investigation):**
- Stores have only ~17 cases/SKU (2.4 days supply) vs 70 target (10 days)
- Store inventory declining steadily over time
- Stores ordered 95M cases but needed ~143M (66% coverage)
- ECOM FCs holding 18M cases, RET-DCs holding 13M cases
- Finished goods inventory stuck at intermediate DCs

**Potential Fixes:**
1. Increase store replenishment policy (target_days: 10→14, ROP: 7→10)
2. Higher initial store inventory (store_days_supply: 14→21)
3. Reduce order staggering cycle (currently every 3 days)
4. Improve DC→Store flow

### 2. SLOB Inventory (60.3% vs <30% target) - MEDIUM PRIORITY
**Status:** Improved from 100%, but still above target

**Analysis:** Finished goods inventory is accumulating at:
- ECOM FCs: 18.3M cases (no downstream stores - B2C only)
- RET-DCs: 13.2M cases
- Customer DCs not pushing inventory to stores fast enough

### 3. Truck Fill Rate (3.6% vs 85% target) - LOW PRIORITY
**Status:** Metric needs re-evaluation for LTL mode

**Recommendations:**
1. Track FTL and LTL fill rates separately
2. Use volume-based fill rate for FMCG products
3. Consider adjusting target to 30-50% for realistic FMCG operations

---

## Future Work: Documentation Deep Dive

### Simulation Theory & Physics Documentation (EARMARKED)

Create comprehensive documentation covering the theoretical foundations of the simulation. Target location: `docs/theory/` or expand `docs/llm_context.md`.

**Topics to cover:**

1. **MRP Theory**
   - Independent vs Derived Demand
   - Inventory Position calculation (On-Hand + In-Transit + In-Production)
   - Days of Supply / Reorder Point / Target Level relationships
   - Bill of Materials (BOM) explosion and recipe matrices

2. **Bullwhip Effect**
   - Causes: demand signal processing, order batching, price fluctuations, rationing
   - Measurement: CV ratio (upstream variance / downstream variance)
   - Mitigation strategies implemented: smoothing, staggering, derived demand

3. **Supply Chain Physics Laws**
   - Mass Balance: Input = Output + Scrap (conservation of mass)
   - Little's Law: Inventory = Throughput × Flow Time
   - Kinematic Consistency: Travel Time = Distance / Speed
   - Capacity Constraints: Cannot exceed Rate × Time

4. **Network Topology**
   - Echelon structure: Plants → RDCs → Customer DCs → Stores
   - Node types and their roles
   - Link types and capacity constraints

5. **Demand Modeling**
   - POS demand generation (base + seasonality + noise)
   - Promo lift and hangover effects
   - Category profiles and Zipf distribution

6. **Allocation & Fulfillment**
   - Fair Share allocation under scarcity
   - Fill-or-Kill vs backorder policies
   - FTL vs LTL shipping modes

7. **Manufacturing Physics**
   - OEE calculation (Availability × Performance × Quality)
   - Changeover penalties and batch sizing
   - Capacity planning and constraint management

---

## Key Concepts Quick Reference

### MRP Derived Demand Theory
```
Independent Demand: Consumer purchases at stores (POS) - must forecast
Derived Demand: Calculated from downstream orders - deterministic

Plants → RDCs → Customer DCs → Stores → Consumers
                                         ↑
                                    Independent (POS)
         ←←←←←←←←←←←←←←←←←←←←←←←←←
              Derived Demand (orders received)
```

### Network Node Types
- `STORE-*` - Retail stores (POS demand, order from customer DCs)
- `RET-DC-*`, `DIST-DC-*`, `ECOM-*` - Customer DCs (derived demand, order from RDCs)
- `RDC-*` - Manufacturer RDCs (inventory position for MRP)
- `PLANT-*` - Manufacturing plants

---

## Commands
```bash
# ALWAYS use 365 days for validation
poetry run python run_simulation.py --days 365 --streaming --output-dir data/results/validation

# Quick smoke test only (DO NOT use for validation)
poetry run python run_simulation.py --days 30 --no-logging

# Run tests (4 pre-existing failures unrelated to our changes)
poetry run pytest
```

## Pre-existing Test Failures (Not from our changes)
- `test_promo_lift` - Missing `add_promo` method
- `test_generates_production_order_when_low_inventory` - Outdated API
- `test_no_order_when_sufficient_inventory` - Outdated API
- `test_world_builder_initialization` - Missing product
