# Prism Sim: LLM Context & Developer One-Pager

> **System Prompt Context:** This document contains the critical architectural, functional, and physical constraints of the Prism Sim project. Use this as primary context when reasoning about code changes, bug fixes, or feature expansions.

**Version:** 0.15.2 | **Last Updated:** 2025-12-30

---

## 1. Project Identity

**Name:** Prism Sim
**Type:** Discrete-Event Simulation (DES) Engine for Supply Chain Digital Twins
**Scale:** 4,500+ nodes (3 Plants, 8 RDCs, 4,400+ Retail Stores, 10 Suppliers)
**Core Philosophy:** **"Supply Chain Physics"** - The simulation adheres to fundamental physical laws (Little's Law, Mass Balance, Capacity Constraints) rather than statistical approximation.
**Goal:** Generate realistic, high-fidelity supply chain datasets exhibiting emergent behaviors (bullwhip effect, bottlenecks) for benchmarking optimization algorithms.

---

## 2. The "Physics" Laws (Non-Negotiable Invariants)

Any code change must preserve these. Violations indicate bugs:

| Law | Formula | Enforcement Location |
|-----|---------|---------------------|
| **Mass Balance** | $I_t = I_{t-1} + Receipts - Shipments - Consumed + Produced$ | `monitor.py:PhysicsAuditor` |
| **Little's Law** | $WIP = Throughput \times CycleTime$ | Implicit in logistics/transform |
| **Capacity Constraints** | $Production \leq Rate \times Time \times OEE$ | `transform.py` |
| **Inventory Positivity** | $Inventory \geq 0$ (cannot ship/consume what you don't have) | `state.py`, `allocation.py`, `transform.py`, `orchestrator.py` |
| **Kinematic Consistency** | $TransitTime = Distance / Speed$ (no teleporting) | `logistics.py` |

**Inventory Positivity Enforcement (v0.12.2):**
- `orchestrator.py`: Demand consumption constrained to `min(demand, available_actual)`
- `allocation.py`: Fill ratios calculated from `actual_inventory` (not perceived)
- `transform.py`: Material consumption constrained to `min(required, available_actual)`
- `state.py`: Floor guards `np.maximum(0, ...)` on all inventory updates

---

## 3. File-to-Concept Map (Where to Find Things)

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
| **Replenishment orders** | `agents/replenishment.py` | `MinMaxReplenisher.generate_orders()` |
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
3. ARRIVALS      → Process in-transit shipments that arrived today
4. DEMAND        → POSEngine generates retail sales (consumes store inventory)
5. REPLENISHMENT → MinMaxReplenisher creates orders (Store→RDC, RDC→Plant)
6. ALLOCATION    → AllocationAgent allocates inventory to orders (Fair Share)
7. LOGISTICS     → LogisticsEngine creates shipments (FTL rules, Emissions)
8. MRP           → MRPEngine plans production (uses RDC→Store shipment signal)
9. PRODUCTION    → TransformEngine executes manufacturing
10. POST-QUIRKS  → Apply logistics delays/congestion (QuirkManager)
11. MONITORING   → PhysicsAuditor validates mass balance, records KPIs
```

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

## 7. Inventory Policies (Tiered)

`MRPEngine` uses vectorized policy vectors based on product type:

| Category | Reorder Point (days) | Target (days) | Example Products |
|----------|---------------------|---------------|------------------|
| DEFAULT | 14 | 28 | Finished goods |
| INGREDIENT | 7 | 14 | Bulk base chemicals |
| PACKAGING | 7 | 14 | Bottles, caps, boxes |
| ACTIVE_CHEM | 30 | 45 | Specialty actives |
| SPOF | 30 | 45 | `ACT-CHEM-001` (constrained) |

Configured in `simulation_config.json` under `manufacturing.inventory_policies`.

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

**Key Metrics Tracked (Expanded in v0.13.0):**
| Metric | Formula / Description | Target |
|--------|----------------------|--------|
| **Store Service Level (OSA)** | `Actual_Sales / Consumer_Demand` | >93% |
| **Perfect Order Rate** | OTIF + Damage Free + Doc Accuracy | >90% |
| **Cash-to-Cash Cycle** | DSO + DIO - DPO | 40-50 days |
| **Scope 3 Emissions** | kg CO2 per case shipped (`Shipment.emissions_kg`) | Track |
| **Inventory Turns** | `COGS / Avg_Inventory` | 8-15x |
| **MAPE** | Forecast accuracy | 20-50% |
| **Shrinkage Rate** | Phantom inventory % | 1-4% |
| **SLOB %** | Slow/Obsolete inventory | <30% |
| **Truck Fill Rate** | `Actual_Load / Capacity` | >85% |
| **OEE** | Overall Equipment Effectiveness | 65-85% |

---

## 15. Known Issues & Current State

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

### Mass Balance Violations (Known Issue)
**Status:** OPEN - Expected behavior for FTL consolidation

**Symptom:** Mass balance violations at customer DCs (DIST-DC-001, etc.) showing `Expected < 0, Actual = 0`.

**Root Cause:** FTL consolidation timing mismatch:
1. Allocation decrements inventory immediately when orders are created
2. Logistics can HOLD orders if they don't meet FTL minimum pallet thresholds
3. `shipments_out` is recorded when shipments are actually created (may be days later)
4. Mass balance equation doesn't account for "allocated but not yet shipped" inventory

**Impact:** This is an **accounting/auditing issue**, not an actual physics violation. The inventory is correctly managed; the mass balance tracker can't account for temporal mismatches from FTL consolidation.

**Relevant Config:** `simulation_config.json` → `logistics.channel_rules.*.min_order_pallets`

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
    │         ├──→ check materials (vectorized)
    │         ├──→ consume ingredients
    │         └──→ produce finished goods
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
