# Forward Tasks - Phase C Continuation

## Session Summary (Dec 31, 2024 - v0.15.5) - COMPLETED

### Fixes Completed

#### 1. Customer DC Bullwhip Cascade - FIXED (v0.15.4)
**Root Cause:** Customer DCs (RET-DC, DIST-DC, ECOM-FC) used POS demand signal (=0, floored to 0.1) instead of derived demand. When stores ordered on Day 1, customer DCs saw inventory drop below their tiny ROP and mass-ordered on Day 2 (272M orders).

**Fix (MRP Derived Demand Theory):**
- Customer DCs now track allocation outflow (orders fulfilled to downstream stores) as their demand signal
- Warm start from `POSEngine.get_average_demand_estimate()` × downstream store count
- Order staggering: Customer DCs order on 5-day cycle (stores use 3-day)
- Inventory priming: Customer DC initial inventory scaled by downstream store count

**Files:** `replenishment.py`, `orchestrator.py`

#### 2. MRP Ingredient Ordering Explosion - FIXED (v0.15.4)
**Root Cause:** ACTIVE_CHEM policy had 30-day ROP / 45-day target, causing plants to order 272M cases when inventory < 30M.

**Fix:**
- Reduced ACTIVE_CHEM policy to 7/14 days (was 30/45)
- Capped daily production estimate at 2x expected demand
- Capped ingredient order quantities

**Files:** `mrp.py`, `simulation_config.json`

#### 3. Fragmented Store Orders - FIXED (v0.15.5)
**Root Cause:** Stores ordered 20-40 cases but FTL required 300-1200 cases (5-20 pallets). Orders were held in `held_orders` for FTL consolidation, causing service level issues and low truck fill when they eventually shipped.

**Fix (LTL for Store Deliveries):**
- Differentiated shipping modes: FTL for DC-to-DC, LTL for DC-to-Store
- LTL shipments ship immediately without pallet minimum (min 10 cases)
- FTL shipments maintain pallet minimums for consolidation
- Added `store_delivery_mode: "LTL"` and `ltl_min_cases: 10` config options
- Added `default_ftl_min_pallets: 10` for routes without channel rules

**Files:** `logistics.py`, `simulation_config.json`

**Note on Truck Fill Metric:** The truck fill rate dropped because:
1. Most shipments are now LTL to stores (intentionally small)
2. FMCG products "cube out" (fill by volume) before "weighting out" (fill by weight)
3. Weight-based truck fill isn't appropriate for light, bulky products
4. The 85% target was designed for FTL-only networks; with LTL, service level is the better metric

### Results (v0.15.5)

| Metric | v0.15.4 | v0.15.5 | Target |
|--------|---------|---------|--------|
| Day 2 Orders | 100K | 100K | <1M ✓ |
| Service Level (90-day) | 83% | **92.8%** | 98.5% |
| Truck Fill Rate | 15% | 4.2% | 85% (see note) |
| Manufacturing OEE | 81% | **83%** | 75-85% ✓ |

---

## Remaining Issues (Priority Order)

### 1. Service Level (92.8% vs 98.5% target)
**Status:** Improved significantly, but still below target

**Problem:** Service level improved by ~10 percentage points but still short of 98.5% target.

**Potential Fixes:**
1. Higher initial inventory levels (increase `store_days_supply`, `rdc_days_supply`)
2. Safety stock adjustments (increase ROP/target gap)
3. Production capacity tuning (additional shifts or plants)
4. Demand forecast smoothing improvements

### 2. Truck Fill Rate (4.2% vs 85% target)
**Status:** Metric needs re-evaluation

**Analysis:** The low truck fill rate is expected behavior with LTL mode:
- LTL shipments are intentionally small (shipped by parcel/LTL carriers)
- FMCG products (toiletries) "cube out" before "weighting out"
- Weight-based fill rate isn't meaningful for volume-constrained products

**Recommendations:**
1. Track FTL and LTL fill rates separately
2. Use volume-based fill rate for FMCG products
3. Consider adjusting target to 30-50% for realistic FMCG operations

---

## Key Concepts

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
poetry run python run_simulation.py --days 30 --no-logging  # Quick validation
poetry run python run_simulation.py --days 90 --no-logging  # Full test
poetry run pytest  # Run tests (4 pre-existing failures unrelated to our changes)
```

## Pre-existing Test Failures (Not from our changes)
- `test_promo_lift` - Missing `add_promo` method
- `test_generates_production_order_when_low_inventory` - Outdated API
- `test_no_order_when_sufficient_inventory` - Outdated API
- `test_world_builder_initialization` - Missing product
