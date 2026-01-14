# Prism Sim: LLM Context & Developer One-Pager

> **System Prompt Context:** This document contains the critical architectural, functional, and physical constraints of the Prism Sim project. Use this as primary context when reasoning about code changes, bug fixes, or feature expansions.

**Version:** 0.32.0 | **Last Updated:** 2026-01-13

---

## 1. Project Identity

**Name:** Prism Sim
**Type:** Discrete-Event Simulation (DES) Engine for Supply Chain Digital Twins
**Scale:** 6,100+ nodes (4 Plants, 4 RDCs, 6,000+ Retail Stores, 50 Suppliers), 500 SKUs
**Core Philosophy:** **"Supply Chain Physics"** - The simulation adheres to fundamental physical laws (Little's Law, Mass Balance, Capacity Constraints) rather than statistical approximation.
**Goal:** Generate realistic, high-fidelity supply chain datasets exhibiting emergent behaviors (bullwhip effect, bottlenecks) for benchmarking optimization algorithms.

---

## Physics Laws (Non-Negotiable)

The simulation enforces these constraints - violations indicate bugs:
1. **Mass Balance:** Input (kg) = Output (kg) + Scrap. (Verified by `PhysicsAuditor`)
2. **Kinematic Consistency:** Travel time = Distance / Speed. Teleportation is banned.
   - **v0.19.12+:** Distances are real Haversine calculations between lat/lon coordinates.
3. **Little's Law:** Inventory = Throughput × Flow Time.
4. **Capacity Constraints:** Cannot produce more than Rate × Time. 
   - **v0.32.0+**: Capacity is modeled as **Discrete Parallel Lines** (e.g., 44 lines/plant) with per-line changeover penalties. The `production_rate_multiplier` hack is deprecated.
5. **Inventory Positivity:** Cannot ship what you don't have.
6. **Geospatial Coherence:**
   - Topology is distance-based (Nearest Neighbor).
   - Production routing is Demand-Proportional (Supply follows Demand).

## Engineering Standards

### Core Simulation Loop
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Daily orchestration** | `simulation/orchestrator.py` | `Orchestrator.run()`, `_run_day()` |
| **State tensors** | `simulation/state.py` | `StateManager` |
| **World construction** | `simulation/builder.py` | `WorldBuilder` |

### Demand & Orders
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **POS demand generation** | `simulation/demand.py` | `POSEngine.generate_daily_demand()` |
| **Replenishment orders** | `agents/replenishment.py` | `MinMaxReplenisher` (Full Physics SS) |
| **Order allocation** | `agents/allocation.py` | `AllocationAgent.allocate()` |

### Manufacturing
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Production planning** | `simulation/mrp.py` | `MRPEngine.generate_production_orders()` |
| **Recipe matrix (BOM)** | `network/recipe_matrix.py` | `RecipeMatrixBuilder` |
| **Production execution** | `simulation/transform.py` | `TransformEngine.execute_production()` |

### Logistics & Validation
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Shipment creation** | `simulation/logistics.py` | `LogisticsEngine.create_shipments()` |
| **Physics validation** | `simulation/monitor.py` | `PhysicsAuditor`, `RealismMonitor` |
| **Behavioral quirks** | `simulation/quirks.py` | `QuirkManager` |
| **Risk events** | `simulation/risk_events.py` | `RiskEventManager` |

### Data Models
| Concept | File | Key Classes |
|---------|------|-------------|
| **Network primitives** | `network/core.py` | `Node`, `Link`, `Order`, `Shipment`, `Batch` |
| **Channel enums** | `network/core.py` | `CustomerChannel`, `StoreFormat`, `OrderType` |
| **Product definitions** | `product/core.py` | `Product`, `Recipe`, `ProductCategory` |
| **Packaging enums** | `product/core.py` | `PackagingType`, `ContainerType`, `ValueSegment` |
| **Promo calendar** | `simulation/demand.py` | `PromoCalendar`, `PromoEffect` |

### Generators (Static World Creation)
| Concept | File | Key Classes |
|---------|------|-------------|
| **Product/SKU generation** | `generators/hierarchy.py` | `ProductGenerator` |
| **Network topology** | `generators/network.py` | `NetworkGenerator` |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/generate_static_world.py` | Generate static world data (products, recipes, nodes, links) |
| `scripts/calibrate_config.py` | Derive optimal simulation parameters from world definition using physics |

---

## 4. State Manager: The Single Source of Truth

`StateManager` holds all simulation state as NumPy tensors for O(1) access:

```python
# Tensor Shapes
inventory: np.ndarray        # [n_nodes, n_products] - Current stock (actual)
perceived_inventory: np.ndarray  # [n_nodes, n_products] - What system "sees" (may differ due to phantom inventory)
wip: np.ndarray              # [n_nodes, n_products] - Work in process at plants

# Index Mappings (str -> int for O(1) lookup)
node_id_to_idx: dict[str, int]
product_id_to_idx: dict[str, int]
```

**Critical:** Always use `state.update_inventory(node_id, product_id, delta)` - never modify tensors directly.

---

## 5. Daily Simulation Loop (Execution Order)

Every `tick()` (1 day) in `Orchestrator._run_day()`:

```
1. RISK EVENTS   → Trigger disruptions (RiskEventManager)
2. PRE-QUIRKS    → Apply Phantom Inventory shrinkage (QuirkManager)
3. ARRIVALS      → Process in-transit shipments that arrived today (Records realized lead time)
4. DEMAND        → POSEngine generates retail sales (consumes store inventory)
5. REPLENISHMENT → MinMaxReplenisher creates orders (Uses Physics-based SS + ABC Segmentation)
6. ALLOCATION    → AllocationAgent allocates inventory to orders (Fair Share)
7. LOGISTICS     → LogisticsEngine creates shipments (FTL rules, Emissions, Earliest Order Day)
8. MRP           → MRPEngine plans production (uses RDC→Store shipment signal)
9. PRODUCTION    → TransformEngine executes manufacturing
10. POST-QUIRKS  → Apply logistics delays/congestion (QuirkManager)
11. MONITORING   → PhysicsAuditor validates mass balance, records KPIs
```

---

## 5.1 Death Spiral Safeguards (v0.20.0)

The simulation has multiple safeguards to prevent feedback loops that collapse production:

### MRP Production Floors
ABC-based minimum production floors ensure demand is met regardless of inventory signals:
- **A-Items:** 90% of expected demand (absolute minimum)
- **B-Items:** 80% of expected demand
- **C-Items:** 70% of expected demand

**Critical:** SLOB throttling is applied BEFORE floors, so floors always win.

### Timeout Mechanisms
To prevent unbounded accumulation:
- **Production Orders:** 14-day timeout (stale orders dropped, MRP regenerates)
- **Held Logistics Orders:** 14-day timeout (FTL consolidation doesn't block indefinitely)
- **Pending Replenishment Orders:** 14-day timeout (allows retry after failed fulfillment)
- **Completed Batches:** 30-day retention (memory cleanup)

### Demand Signal Priority
MRP uses order-based demand (not shipment-based) as primary signal, with expected demand as floor:
```python
# mrp.py: record_order_demand()
daily_vol = np.maximum(daily_vol, self.expected_daily_demand)  # Floor
daily_vol = np.minimum(daily_vol, self.expected_daily_demand * 4.0)  # Cap
```

### Initial Inventory Buffers
Plants are seeded with 50M units per ingredient type (~90 days buffer) to allow production stability while ingredient ordering catches up.

---

## 6. Recipe Matrix: Vectorized BOM

The `RecipeMatrixBuilder` creates a dense matrix $\mathbf{R}$ for instant ingredient calculations:

```
Shape: [n_finished_goods, n_ingredients]
Value R[i,j] = quantity of ingredient j needed to make 1 unit of product i

Usage: ingredient_requirements = demand_vector @ recipe_matrix
```

This enables O(1) MRP calculations for thousands of SKUs instead of iterating BOMs.

---

## 7. Inventory Policies (Physics-Based)

As of **v0.17.0**, replenishment is driven by realized performance rather than static heuristics:

### The Full Safety Stock Formula
$SS = z \sqrt{\bar{L}\sigma_D^2 + \bar{D}^2\sigma_L^2}$
- **Demand Risk ($\sigma_D$):** Protected by rolling 28-day demand history.
- **Supply Risk ($\sigma_L$):** Protected by tracking realized lead times (order to arrival).

### Dynamic ABC Segmentation
Products are dynamically classified every 7 days based on cumulative sales volume:
- **A-Items (Top 80%):** Target 99% SL ($z=2.33$)
- **B-Items (Next 15%):** Target 95% SL ($z=1.65$)
- **C-Items (Bottom 5%):** Target 90% SL ($z=1.28$)

---

## 8. Customer Channels & Store Formats (v0.13.0)

The simulation models realistic B2B customer segmentation with channel-specific logistics rules:

### Customer Channels (`CustomerChannel` enum)
| Channel | Description | Logistics Mode | Min Order (Pallets) |
|---------|-------------|----------------|---------------------|
| `B2M_LARGE` | Big retailers (Walmart DC, Target DC) | FTL | 20 |
| `B2M_CLUB` | Club stores (Costco, Sam's Club) | FTL | 15 |
| `B2M_DISTRIBUTOR` | 3P Distributors (consolidate for small retailers) | FTL | 10 |
| `ECOMMERCE` | Amazon, pure-play digital | FTL | 5 |
| `DTC` | Direct to consumer | Parcel | N/A |

### Store Formats (`StoreFormat` enum)
`RETAILER_DC`, `HYPERMARKET`, `SUPERMARKET`, `CLUB`, `CONVENIENCE`, `PHARMACY`, `DISTRIBUTOR_DC`, `ECOM_FC`

### Channel Economics
Configured in `world_definition.json` under `channel_economics`:
```json
{
  "B2M_LARGE": {"volume_pct": 40, "margin_pct": 18, "payment_days": 45},
  "B2M_CLUB": {"volume_pct": 15, "margin_pct": 16, "payment_days": 30},
  "B2M_DISTRIBUTOR": {"volume_pct": 25, "margin_pct": 22, "payment_days": 30},
  "ECOMMERCE": {"volume_pct": 15, "margin_pct": 25, "payment_days": 15},
  "DTC": {"volume_pct": 5, "margin_pct": 35, "payment_days": 0}
}
```

**Key Insight:** CPG manufacturers ship FTL to Retailer DCs, Club stores, and Distributors—NOT to individual stores. Small retailers are served by 3P distributors.

---

## 9. Order Types (v0.13.0)

Orders are classified by type with different priority handling in allocation:

| Order Type | Distribution | Priority | Behavior |
|------------|--------------|----------|----------|
| `STANDARD` | 70% | 3 | Normal (s,S) replenishment |
| `RUSH` | 10% | 1 | Expedited, reduced lead time |
| `PROMOTIONAL` | 10% | 2 | Linked to promo calendar, larger batches |
| `BACKORDER` | 10% | 4 | Created when allocation fails, persists until filled |

**Order Dataclass Attributes:**
- `order_type`: `OrderType` enum
- `promo_id`: Link to promotion (for PROMOTIONAL orders)
- `priority`: 1=highest, 10=lowest
- `requested_date`: Target delivery day

---

## 10. Packaging Hierarchy & SKU Variants (v0.13.0)

Products have realistic packaging attributes for logistics and demand modeling:

### Container Types (`ContainerType` enum)
`TUBE` (toothpaste), `BOTTLE` (dish soap), `PUMP` (premium body wash), `POUCH` (refills), `GLASS` (premium)

### Value Segments (`ValueSegment` enum)
| Segment | Description | Channel Affinity |
|---------|-------------|------------------|
| `PREMIUM` | Glass, large pump bottles | DTC, Ecommerce |
| `MAINSTREAM` | Standard sizes | B2M_LARGE, B2M_DISTRIBUTOR |
| `VALUE` | Large refills, bulk | B2M_CLUB |
| `TRIAL` | Sachets, travel sizes | Ecommerce, DTC |

### PackagingType Dataclass
```python
@dataclass
class PackagingType:
    code: str               # PKG-TUBE-100
    name: str               # "100ml Tube"
    container: ContainerType
    size_ml: int
    material: str           # "plastic", "glass"
    recyclable: bool
    units_per_case: int     # Critical for logistics
    segment: ValueSegment
```

**Product Extended Attributes:**
- `brand`: PrismWhite, ClearWave, AquaPure
- `packaging_type_id`: Link to PackagingType
- `value_segment`: Premium/Mainstream/Value/Trial
- `recyclable`, `material`: Sustainability tracking

---

## 11. Promo Calendar (v0.13.0)

The `PromoCalendar` class manages promotional lift and hangover effects with O(1) vectorized lookup:

```python
# Core index structure: week → channel/account → sku → PromoEffect
_index: dict[int, dict[str, dict[str, PromoEffect]]]

@dataclass(frozen=True)
class PromoEffect:
    promo_id: str
    lift_multiplier: float      # e.g., 2.5x during promo
    hangover_multiplier: float  # e.g., 0.7x after promo ends
    discount_percent: float
    is_hangover: bool = False
```

**Promo Config Example** (`world_definition.json`):
```json
{
  "code": "PROMO-BF-2024",
  "name": "Black Friday 2024",
  "start_week": 48, "end_week": 48,
  "lift_multiplier": 3.0,
  "hangover_weeks": 1, "hangover_multiplier": 0.6,
  "discount_percent": 25.0,
  "affected_channels": ["B2M_LARGE", "ECOMMERCE"]
}
```

---

## 12. Risk Events (v0.13.0)

`RiskEventManager` triggers deterministic disruptions. All 5 events have individual enable/disable toggles:

| Code | Type | Trigger Day | Effect |
|------|------|-------------|--------|
| `RSK-BIO-001` | Contamination | 150 | Batches with target ingredient → `REJECTED` |
| `RSK-LOG-002` | Port Strike | 120 | 4x logistics delays for 14 days (Gamma dist) |
| `RSK-SUP-003` | Supplier Opacity | 100 | SPOF supplier OTD drops: 92% → 40% |
| `RSK-CYB-004` | Cyber Outage | 200 | Target DC WMS down for 72 hours |
| `RSK-ENV-005` | Carbon Tax | 180 | 3x CO2 cost multiplier |

Configured in `simulation_config.json` under `risk_events`.

---

## 13. Behavioral Quirks (v0.13.0)

`QuirkManager` injects realistic supply chain pathologies. All 6 quirks have individual toggles:

| Quirk | Effect | Key Parameters |
|-------|--------|----------------|
| `bullwhip_whip_crack` | Order batching during promos amplifies bullwhip | `batching_factor: 3.0` |
| `phantom_inventory` | Shrinkage creates actual vs perceived divergence | `shrinkage_pct: 0.02`, `detection_lag_days: 14` |
| `port_congestion_flicker` | AR(1) correlated delays create clustered late arrivals | `ar_coef: 0.70`, `cluster_size: 3` |
| `single_source_fragility` | SPOF ingredient delays cascade through BOM | `delay_multiplier: 2.5` |
| `human_optimism_bias` | Over-forecast for new products | `bias_pct: 0.15`, `affected_age_months: 6` |
| `data_decay` | Older batches have higher rejection rates | `base_rate: 0.02`, `elevated_rate: 0.05` |

Configured in `simulation_config.json` under `quirks`.

---

## 14. The Supply Chain Triangle

Every decision impacts the balance between:

```
        SERVICE (Fill Rate, OTIF)
              /\
             /  \
            /    \
           /      \
    COST ←────────→ CASH
(Freight, Prod)   (Inventory, WC)
```

**Key Metrics Tracked (Updated v0.31.0 with industry benchmarks):**
| Metric | Formula / Description | Target | Industry Reference |
|--------|----------------------|--------|-------------------|
| **Store Service Level (OSA)** | `Actual_Sales / Consumer_Demand` | 95-98% | P&G: >99% |
| **Perfect Order Rate** | OTIF + Damage Free + Doc Accuracy | >90% | - |
| **Cash-to-Cash Cycle** | DSO + DIO - DPO | 40-70 days | - |
| **Scope 3 Emissions** | kg CO2 per case shipped (`Shipment.emissions_kg`) | Track | - |
| **Inventory Turns** | `COGS / Avg_Inventory` | 5-7x | Colgate: 4.1x, P&G: 5.5x |
| **MAPE** | Forecast accuracy | 20-50% | - |
| **Shrinkage Rate** | Phantom inventory % | 1-4% | - |
| **SLOB %** | Slow/Obsolete inventory | <15% | World-class: 10% |
| **Truck Fill Rate** | `Actual_Load / Capacity` | >85% | - |
| **OEE** | Overall Equipment Effectiveness | 55-70% | Industry avg: 55-60% |

---

## 15. Known Issues & Current State

### Industry Calibration & Trade-Off Analysis (v0.31.0 - CURRENT)
**Status:** OPTIMAL CONFIGURATION WITHIN STRUCTURAL CONSTRAINTS ✅

**Current State (v0.31.0, 365-day):**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Service Level | **89.4%** | 95-98% | Improved +15pp from v0.30 |
| A-Items Service | **89.6%** | - | Good ABC differentiation |
| Inventory Turns | **5.00x** ✅ | 5-7x | On target (industry standard) |
| **SLOB** | **4.2%** ✅ | <15% | Excellent |
| OEE | **35.8%** | 55-70% | Expected with high inventory |

**Trade-Off Frontier Discovery:**
Through iterative tuning, discovered the Pareto frontier:
- **Max Service (91.3%)** requires Turns <4.5x and SLOB >13%
- **Balanced (89.4%)** achieves Turns=5.0x and SLOB=4.2%
- Each 1% service gain above 88% costs ~0.2x turns and ~2% SLOB

**Root Cause of 5% Service Gap (89% vs 95%):**
1. **Lead time variability**: 3-day lead times plus demand noise create stockout windows
2. **Seasonality peaks**: ±12% amplitude creates demand spikes buffers can't fully cover
3. **Multi-echelon delays**: Network topology compounds positioning delays

**Recommendations for reaching 95%+ service:**
- Reduce lead times (3 days → 1-2 days)
- Implement ABC-differentiated safety stock in replenishment engine
- Add expedited shipping for A-items during shortages

### SKU Expansion & Phase 2 Alignment (v0.19.15 - COMPLETE)
**Status:** ALL TARGETS ACHIEVED ✅

**Key Implementations:**
1. **Category-Level ABC:** Ensures valid A/B/C mix within every category.
2. **Differentiated Logic:** A-items get 50% production floor, C-items get 10%.
3. **SLOB Throttling:** Hard cap (50% demand) when DOS > 60 days.
4. **Fast-Path Logistics:** Bypasses bin-packing for single-truck LTL orders (critical for 500-SKU scale).
5. **Order Consolidation:** 3-day cycle + 50-case min qty reduces object overhead by 60%.

### Distribution Bottleneck (v0.19.11 - RESOLVED)
**Status:** RESOLVED via Phase 2 Logic & Consolidation

The service level degradation seen in v0.19.11 was resolved by:
1. **Inventory Priming:** Using ABC-specific targets (A=21d) ensures fast movers start with adequate buffers.
2. **Order Consolidation:** Larger, less frequent orders (3-day cycle) prevent the "tiny order" death spiral that was choking logistics.

### The Bullwhip Crisis (v0.12.1 - FIXED in v0.12.3)
**Status:** RESOLVED

**Original Symptom:** 365-day simulation showed orders exploding from ~460k to 66M cases/day.

**Root Cause:** MRP was using smoothed POS demand proxies, which dampened the natural demand signal variance. Combined with "Fill or Kill" allocation, this created an "Inverse Bullwhip" where upstream variance was *lower* than downstream.

**Fix (v0.12.3):** `MRPEngine` now uses a 7-day moving average of **actual RDC shipments** (lumpy signal) instead of smoothed POS demand. This restores realistic demand amplification upstream.

### OEE Tuning (v0.12.3 - FIXED)
**Original Symptom:** OEE at 28% (massive over-capacity).

**Fix:** Reduced `production_hours_per_day` from 24 to 8 (single shift), increased batch sizes. OEE now at ~99%.

### SPOF Isolation (v0.12.1 - IMPLEMENTED)
`ACT-CHEM-001` now isolated to ~20% of portfolio (Premium Oral Care only).
Supplier `SUP-001` constrained to 500k units/day.

### System Death Spiral (v0.14.0 - FIXED in v0.15.0)
**Status:** RESOLVED

**Original Symptom:** 365-day baseline simulation collapsed around day 22-27. Service Level dropped to 8.8%, Production/Shipments went to zero.

**Root Cause:** MRP used RDC→Store shipments as demand signal. When shipments stopped (due to inventory drain), MRP got zero signal → no production orders → no replenishment → death spiral.

**Fixes (v0.15.0):**
1. **C.1: MRP Demand Fallback** (`mrp.py`): Added expected demand vector as floor. When shipment signal drops below 10% of expected, MRP uses expected demand to prevent spiral.
2. **C.2: Supplier-Plant Routing** (`mrp.py`): Fixed `_find_supplier_for_ingredient()` to verify link exists before routing SPOF ingredient to a supplier.
3. **C.3: Production Capacity** (`simulation_config.json`): Increased `production_hours_per_day` from 20 to 24 (3-shift 24/7 operation).
4. **C.4: Realistic Inventory** (`simulation_config.json`): Reduced initial inventory to realistic levels (Store: 7d, RDC: 14d).
5. **C.5: Production Smoothing** (`mrp.py`): Added 7-day rolling average cap (1.5x) on production order volatility.

**Result:** System survives 365-day simulation without collapse. Service Level: 51.6% (vs 8.8% pre-fix).

### MRP Inventory Position Bug (v0.15.0 - FIXED in v0.15.1)
**Status:** RESOLVED

**Original Symptom:** 94 zero-production days in 365-day simulation. Manufacturing OEE at 44.9%.

**Root Cause:** MRP's `_cache_node_info()` included ALL `NodeType.DC` nodes in inventory position calculation, including customer DCs (RET-DC, DIST-DC, ECOM-FC) with ~4.5M units total. This inflated Days of Supply to 11.5 days > ROP 7 days, preventing production orders.

**Fix:** Only include manufacturer RDCs (nodes starting with `RDC-*`) in inventory position:
```python
if node_id.startswith("RDC-"):
    self._rdc_ids.append(node_id)
```

**Also Fixed:** C.5 smoothing history bug - was recording pre-scaled quantities instead of post-scaled actuals.

**Result:** Service Level: 60.19% (vs 51.6%), Manufacturing OEE: 88.2% (vs 44.9%), Zero-Production Days: 0 (vs 94).

### Ingredient Replenishment Mismatch (v0.15.1 - FIXED in v0.15.2)
**Status:** RESOLVED

**Original Symptom:** 365-day simulation collapsed on days 362-365 with production dropping to 0 due to ingredient exhaustion.

**Root Cause:** MRP's `generate_purchase_orders()` used POS demand signal (~400k/day) for ingredient replenishment, but actual ingredient consumption was driven by production orders (amplified by bullwhip to 5-6M/day). This caused a net burn rate of ~1,380 units/day shortfall over 362 days.

**Fix:** Changed `generate_purchase_orders()` to use production-based signal instead of POS demand:
```python
def generate_purchase_orders(
    self,
    current_day: int,
    active_production_orders: list[ProductionOrder],  # Changed from daily_demand
) -> list[Order]:
    # Calculate production signal from active production orders
    production_by_product = np.zeros(self.state.n_products, dtype=np.float64)
    for po in active_production_orders:
        p_idx = self.state.product_id_to_idx.get(po.product_id)
        if p_idx is not None:
            production_by_product[p_idx] += po.quantity_cases
```

**Result:** System survives full 365-day simulation. Production continues through day 365 (259,560 cases).

### Mass Balance FTL Timing Mismatch (v0.15.2 - FIXED in v0.15.3)
**Status:** RESOLVED

**Original Symptom:** Mass balance violations at customer DCs (DIST-DC-001, etc.) showing `Expected < 0, Actual = 0`.

**Root Cause:** FTL consolidation timing mismatch:
1. Allocation decrements inventory immediately when orders are created
2. Logistics can HOLD orders if they don't meet FTL minimum pallet thresholds
3. `shipments_out` was recorded when shipments are actually created (may be days later)
4. Mass balance equation didn't account for "allocated but not yet shipped" inventory

**Fix:** Replaced `shipments_out` tracking with `allocation_out`:
- `AllocationAgent` now returns `AllocationResult` with `allocation_matrix` tracking decrements
- `PhysicsAuditor.record_allocation_out()` records inventory decrements at allocation time
- Added minimum absolute difference threshold (1.0 case) to filter floating-point noise

**Result:** No false mass balance violations at customer DCs. Audit now correctly tracks inventory changes.

### Production Death Spiral (v0.23.0 - RESOLVED)
**Status:** FIXED via Campaign Batching

**Original Symptom:** 365-day simulation showed production declining from 8.8M to 4.3M cases/day (51% drop).

**Root Causes:**
1. **Changeover Time Accumulation:** 500 SKUs × 0.05h changeover = 25h/day (exceeded 20h capacity)
2. **POS-Demand Feedback Loop:** Stockouts → POS drops → production drops → more stockouts

**Solution (v0.23.0): Campaign Batching**

Instead of producing all 500 SKUs daily, use campaign-style production matching real FMCG plants:

1. **Trigger-Based Production:** Only produce when DOS < threshold
   - A-items: DOS < 14 days
   - B-items: DOS < 10 days
   - C-items: DOS < 7 days

2. **Batch Sizing:** Produce `production_horizon_days` worth (default 10 days) per SKU

3. **SKU Limit:** Max 60 SKUs/plant/day to cap changeover overhead

4. **Priority Sorting:** Lowest DOS first (most critical items)

5. **Expected Demand for DOS:** Use expected_daily_demand (not POS) to break feedback loop

**Configuration:** `simulation_config.json` → `manufacturing.mrp_thresholds.campaign_batching`

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| Prod Day 365 | 4.3M (51% drop) | 8.3M (stable) |
| Service Level | 66% | 80.5% |

**Key Files:**
- `simulation/mrp.py`: `_generate_rate_based_orders()` (campaign batching logic)

---

## 16. Configuration Files

| File | Purpose |
|------|---------|
| `config/simulation_config.json` | Runtime parameters (MRP, logistics, quirks, initialization) |
| `config/world_definition.json` | Static world (products, network topology, recipe logic) |
| `config/benchmark_manifest.json` | Risk scenarios, validation targets |

**Key Config Sections:**
```json
{
  "manufacturing": {
    "target_days_supply": 28.0,
    "reorder_point_days": 14.0,
    "inventory_policies": { ... },  // Tiered ROP/Target
    "spof": { "ingredient_id": "ACT-CHEM-001" }
  },
  "simulation": {
    "inventory": {
      "initialization": {
        "store_days_supply": 28.0,
        "rdc_days_supply": 35.0
      }
    }
  }
}
```

---

## 17. Key Commands

```bash
# Run Simulation (default 90 days)
poetry run python run_simulation.py
poetry run python run_simulation.py --days 365 --streaming

# Run Tests
poetry run pytest
poetry run pytest tests/test_milestone_4.py -v

# Type Check & Lint
poetry run mypy src/
poetry run ruff check src/

# Generate Static World (4,500 nodes)
poetry run python scripts/generate_static_world.py

# Analyze Results
poetry run python scripts/analyze_bullwhip.py data/results/run_001
poetry run python scripts/compare_scenarios.py data/results/baseline data/results/risk
```

---

## 18. Debugging Checklist

When simulation behaves unexpectedly:

1. **Service Level = 0%?**
   - Check `initialization.store_days_supply` - cold start starvation?
   - Check `PhysicsAuditor` for mass balance drift
   - Look for SPOF ingredient exhaustion

2. **Orders exploding?**
   - Bullwhip feedback loop - check `MinMaxReplenisher` order volumes
   - Fill-or-Kill creating infinite retry loop

3. **Production stalled?**
   - Check ingredient inventory at plants (`state.inventory[plant_idx, :]`)
   - Check `MRPEngine` purchase orders for raw materials
   - Verify `TransformEngine` capacity vs demand

4. **OEE too low?**
   - Check changeover times in recipes
   - Check `min_production_qty` (MOQ) - too small = changeover penalty

5. **Mass Balance violations?**
   - Check `PhysicsAuditor.audit()` output
   - Look for inventory updates outside `StateManager`

6. **Negative inventory detected?** (Should not happen after v0.12.2)
   - Check if new code bypasses `StateManager.update_inventory()` methods
   - Verify `actual_inventory` is used (not `perceived_inventory`) for allocation/consumption decisions
   - Look for direct tensor manipulation without floor guards
   - Run: `inv[inv['actual_inventory'] < 0]` on inventory.csv to find violating cells

7. **Metrics healthy at 30 days but degrade at 365 days?** (v0.24.0 finding)
   - This is a **drift problem**, not a structural issue
   - **SLOB drift:** C-items accumulating relative to A-items (slow movers don't sell)
   - **Service drift:** SLOB throttling reducing production → inventory draws down unevenly
   - **Truck fill drift:** Lower overall volume as inventory depletes
   - **Key diagnostic:** Compare ABC class inventory distribution at day 30 vs day 365
   - See `docs/planning/v025_forward_path.md` for root cause analysis

---

## 19. Architecture Diagrams

### System Layers
```
┌─────────────────────────────────────────────────────────┐
│                   SIMULATION LAYER                       │
│  Orchestrator → POSEngine → MRPEngine → TransformEngine │
│                → LogisticsEngine → QuirkManager          │
├─────────────────────────────────────────────────────────┤
│                     AGENT LAYER                          │
│         MinMaxReplenisher    AllocationAgent             │
├─────────────────────────────────────────────────────────┤
│                     CORE LAYER                           │
│              StateManager    WorldBuilder                │
├─────────────────────────────────────────────────────────┤
│                   DOMAIN MODELS                          │
│     Node, Link, Order, Shipment │ Product, Recipe        │
└─────────────────────────────────────────────────────────┘
```

### Daily Loop Sequence
```
Orchestrator
    │
    ├──→ POSEngine.generate_daily_demand()
    │         └──→ state.update_inventory() [subtract sales]
    │
    ├──→ MinMaxReplenisher.generate_orders()
    │         └──→ creates Order objects
    │
    ├──→ AllocationAgent.allocate()
    │         ├──→ Fair Share if constrained
    │         └──→ Fill-or-Kill (close unfilled)
    │
    ├──→ MRPEngine.generate_production_orders()
    │         ├──→ recipe_matrix @ demand_vector
    │         └──→ creates ProductionOrder objects
    │
    ├──→ TransformEngine.execute_production()
    │         ├──→ select line (sticky vs capacity)
    │         ├──→ check materials (vectorized)
    │         ├──→ consume ingredients
    │         └──→ produce finished goods (parallel lines)
    │
    ├──→ LogisticsEngine.create_shipments()
    │         ├──→ bin-packing (weight/cube)
    │         └──→ create Shipment objects
    │
    └──→ PhysicsAuditor.audit()
              └──→ validate mass balance
```

---

## 20. Version History (Key Milestones)

| Version | Key Changes |
|---------|-------------|
| 0.32.0 | **Multi-Line Manufacturing Physics** - Replaced `production_rate_multiplier` hack with explicit parallel production lines; Implemented discrete `LineState` tracking and sticky scheduling; Refined OEE calculation ($A \times P \times Q$); Scaled to 44 lines/plant. |
| 0.31.0 | **Industry-Calibrated Inventory Tuning** - Revised targets to industry benchmarks (Turns 5-7x, SLOB <15%); Discovered service-turns-SLOB Pareto frontier; **SL 89.4%, Turns 5.0x, SLOB 4.2%**; Documented structural 91% service ceiling |
| 0.30.0 | **Seasonal Capacity Calibration** - Physics-based `capacity_amplitude` derivation; Asymmetric flex to maintain trough buffer |
| 0.29.0 | **Flexible Production Capacity** - Seasonal capacity adjustment mirrors real FMCG practices; SLOB improved 83.8% → 26.2% |
| 0.28.0 | **Seasonally-Adjusted Priming** - Applied seasonal factor to Day 1 inventory; ABC-differentiated production (A: 1.1x buffer, C: 0.6x) |
| 0.27.0 | **Hardcode Audit & Physics Calibration** - Complete semgrep audit; Physics-based priming/trigger derivation |
| 0.25.0 | **Production Physics Fix** - ABC-differentiated network priming; Balanced production/consumption |
| 0.24.0 | **FTL/LTL Infrastructure** - Added truck fill and mode metrics |
| 0.23.0 | **Campaign Batching** - Trigger-based production (DOS thresholds); Max 60 SKUs/plant/day; Production stabilized |
| 0.19.15 | **Phase 2 Complete & Optimization** - Category-level ABC, SLOB throttling, ABC priming; "Fast-Path" logistics + Order consolidation (3-day) solved performance crash; **SL 94.7%, Turns 7.65x** |
| 0.19.14 | **SKU Generation Overhaul** - Expanded portfolio to 500+ SKUs with realistic packaging hierarchy and Zipfian distribution; Prepared ground for ABC/Long-tail analysis |
| 0.19.13 | **GIS Lead Time Fix** - Fixed critical bug where `dist/speed` was treated as days instead of hours; Service Level recovered 39% → 82.7%; OEE 30% → 86.8% |
| 0.19.12 | **Geospatial Physics (GIS)** - Replaced random topology with US Cities + Haversine distance; Fixed "Ghost RDC" black hole; Network now physically grounded |
| 0.19.11 | **POS-Driven Production (Closed-Loop)** - Uses actual consumer demand (POS) as primary signal, not expected demand; ABC class tracking for differentiated response dynamics; **SLOB target achieved: 28.6% (<30%)**; SL at 64.5% proves distribution is the bottleneck, not production; Next: distribution diagnosis (see `docs/planning/distribution_diagnosis.md`) |
| 0.19.10 | **ABC-Differentiated Production** - Experimental version superseded by v0.19.11; SL 71.7% but SLOB worsened to 76% |
| 0.19.9 | **Rate-Based Production (Option C)** - MRP produces at expected demand rate regardless of DOS signals; Prevents low-equilibrium trap; MRPDiagnostics for signal tracing; 365-day SL improved 50%→67.5%; Next: product mix optimization (see `docs/planning/mix_opt.md`) |
| 0.19.8 | **Starvation Loop Fixes** - Decoupled ingredient ordering from historical production; Warm-started history buffers; Smoothing cap uses max(history, expected) |
| 0.19.7 | **ABC Alignment Tuning** - Tuned ROP multipliers and allocation priorities |
| 0.19.6 | **Hardcode Refactoring** - Introduced `OrderPriority` and `ABCClass` enums; Moved logic parameters (`min_history_days`, `min_batch_size_absolute`, `default_store_count`) to config; Strict type safety enforcement |
| 0.19.3 | **ABC Prioritization (Phase 1 & 2)** - Implemented Pareto-based logic for Allocation (A-items first) and MRP (ROP multipliers 1.2x/0.8x) to fix product mix issues; Configurable thresholds/multipliers |
| 0.19.2 | **Signal Flow Optimization** - Daily DC ordering, Push Allocation, Demand-Proportional MRP batches; 90-day Service Level >90% achieved |
| 0.19.1 | **MEIO Bug Fixes** - Fixed Customer DC IP calculation (use Local IP not Echelon IP); MRP uses POS demand as floor; Added diagnostic scripts (`diagnose_service_level.py`, `diagnose_slob.py`); Root cause identified: negative feedback spiral |
| 0.19.0 | **Echelon Inventory Logic (MEIO)** - Customer DCs use aggregated downstream demand + inventory position; Config for store_batch_size_cases and lead_time_history_len |
| 0.18.2 | **Order Signal Stabilization** - Fixed order collapse by using 7-day inflow average for all nodes; Masked ingredients in Replenisher to prevent phantom orders; Reverted v0.18.0 band-aids |
| 0.18.0 | **Service Level Fix Attempt (Partially Reverted)** - Plant shipment routing fix; SLOB calculation fix; (Reverted: throughput floors) |
| 0.17.0 | **Physics Overhaul** - Full Safety Stock formula ($SS = z \sqrt{\bar{L}\sigma_D^2 + \bar{D}^2\sigma_L^2}$); Dynamic ABC Segmentation; Zero mypy errors; config-driven order cycles and scale factors |
| 0.16.0 | **Variance-Aware Safety Stock** - Implemented dynamic ROP ($ROP = \mu_L + z\sigma_L$) replacing static days-of-supply; (s,S) policies now use Inventory Position (IP) to prevent double-ordering; System stable with Zipfian demand (76% SL) |
| 0.15.9 | **Service Level Phase 2 (Demand Signal Fix)** - Inflow-based demand for customer DCs (orders received vs shipped); Daily DC ordering (5d→1d); Higher DC buffers (21/14 days); MRP order signal; Awaiting 365-day validation |
| 0.15.8 | **Service Level Phase 1** - ECOM FC demand signal fix; Increased replenishment targets (14/10 days); Daily store ordering; Higher initial inventory priming; Service level 75% → 80.5% |
| 0.15.7 | **Inventory Turns Fix** - Exclude raw materials from turns calculation; FG mask for metrics |
| 0.15.6 | **MRP Signal Stabilization** - Prevent 365-day production collapse with velocity tracking and production floor |
| 0.15.5 | **LTL Shipping** - Store deliveries use LTL mode; Service level 83% → 92.8% |
| 0.15.4 | **Bullwhip Dampening** - Customer DCs use derived demand (outflow) instead of POS; Warm start; Order staggering |
| 0.15.3 | **Mass Balance Audit Fix** - Replace shipments_out with allocation_out tracking to fix FTL timing mismatch; Add floating-point noise filtering |
| 0.15.2 | **Ingredient Replenishment Fix** - Use production-based signal for ingredient ordering instead of POS demand; System survives full 365-day simulation without collapse |
| 0.15.1 | **MRP Inventory Position Fix** - Only count manufacturer RDCs in inventory position (not customer DCs); Fix C.5 smoothing history bug; Document mass balance FTL timing issue |
| 0.15.0 | **Phase C Fixes** - MRP demand fallback, supplier-plant routing fix, production smoothing, realistic inventory levels |
| 0.14.0 | **Option C Architecture** - Multi-tier DC structure, capacity rebalancing |
| 0.13.0 | **Realism Overhaul** - CustomerChannel/StoreFormat/OrderType enums, PackagingType hierarchy, PromoCalendar with lift/hangover, 5 Risk Events, 6 Behavioral Quirks, Channel Economics, Scope 3 Emissions, Expanded KPIs (Perfect Order, Cash-to-Cash, MAPE, Shrinkage, SLOB) |
| 0.12.3 | **Inverse Bullwhip Fix** - MRP uses lumpy RDC shipment signals; Store Service Level metric added |
| 0.12.2 | **Negative inventory fix** - Inventory Positivity law enforced across all deduction paths |
| 0.12.1 | Tiered inventory policies, LIFR tracking, Bullwhip Crisis identified |
| 0.12.0 | Fill-or-Kill allocation, MRP look-ahead horizon |
| 0.11.0 | Streaming writers for 365-day runs |
| 0.10.0 | **Vectorized MRP** (Recipe Matrix), Mass Balance Audit, Procedural ingredients |
| 0.9.x | Deep NAM integration, inventory collapse fixes, OEE tracking |
| 0.8.0 | Static world generators (4,500 nodes) |
| 0.5.0 | Validation framework, quirks, risk events |
| 0.4.0 | MRP + Transform engines, SPOF simulation |
| 0.3.0 | Allocation agent, logistics (Tetris), transit physics |
| 0.2.0 | Orchestrator, POS demand, replenishment |
| 0.1.0 | Core primitives, state manager, world builder |
