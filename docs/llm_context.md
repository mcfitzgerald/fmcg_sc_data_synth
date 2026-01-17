# Prism Sim: LLM Context & Developer One-Pager

> **System Prompt Context:** This document contains the critical architectural, functional, and physical constraints of the Prism Sim project. Use this as primary context when reasoning about code changes, bug fixes, or feature expansions.

---

## 1. Project Identity

**Name:** Prism Sim
**Type:** Discrete-Event Simulation (DES) Engine for Supply Chain Digital Twins
**Core Philosophy:** **"Supply Chain Physics"** - The simulation adheres to fundamental physical laws (Little's Law, Mass Balance, Capacity Constraints) rather than statistical approximation.
**Goal:** Generate realistic, high-fidelity supply chain datasets exhibiting emergent behaviors (bullwhip effect, bottlenecks) for benchmarking optimization algorithms.

---

## 2. Physics Laws (Non-Negotiable)

The simulation enforces these constraints - violations indicate bugs:

1. **Mass Balance:** Input (kg) = Output (kg) + Scrap. (Verified by `PhysicsAuditor`)
2. **Kinematic Consistency:** Travel time = Distance / Speed. Teleportation is banned.
   - Distances are real Haversine calculations between lat/lon coordinates.
3. **Little's Law:** Inventory = Throughput × Flow Time.
4. **Capacity Constraints:** Cannot produce more than Rate × Time.
   - Capacity is modeled as **Discrete Parallel Lines** with per-line changeover penalties.
5. **Inventory Positivity:** Cannot ship what you don't have.
6. **Geospatial Coherence:**
   - Topology is distance-based (Nearest Neighbor).
   - Production routing is Demand-Proportional (Supply follows Demand).

---

## 3. File-to-Concept Map

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
| **Replenishment orders** | `agents/replenishment.py` | `MinMaxReplenisher` |
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
| `scripts/calibrate_config.py` | Derive optimal simulation parameters from world definition |
| `scripts/generate_warm_start.py` | Generate warm-start snapshot manually |

---

## 4. Order of Operations (CRITICAL)

When making configuration changes, follow this sequence **exactly**:

```bash
# 1. UPDATE CONFIG - Edit simulation_config.json
#    Located at: src/prism_sim/config/simulation_config.json
#    Change: num_lines, demand parameters, inventory targets, etc.

# 2. REGENERATE WORLD (only if topology changed)
poetry run python scripts/generate_static_world.py
#    Required when: products, recipes, node counts, or link distances change

# 3. CALIBRATE - Derives safety stock, triggers, priming FOR your config
poetry run python scripts/calibrate_config.py --apply
#    This step is MANDATORY after changing capacity parameters!
#    The calibration script derives planning parameters based on configured capacity.
#    Skipping this causes parameter mismatch and service level collapse.

# 4. CLEAR CHECKPOINT (config changes invalidate old checkpoints)
rm -f data/checkpoints/steady_state_*.json.gz

# 5. RUN SIMULATION
poetry run python run_simulation.py --days 365
```

**When to regenerate world:**
- Changed `world_definition.json` (products, recipes, node counts)
- Changed network topology or link distances
- First-time setup

**When to just recalibrate:**
- Changed `num_lines` or manufacturing parameters
- Changed inventory targets or service levels
- Tuning simulation behavior

**Simulation Run Lengths:**
- **Full diagnostic (365 days):** Required for accurate KPIs. Includes 90-day burn-in + 365 data days.
- **Sanity checks (30 days with `--no-checkpoint`):** Fast verification only. Metrics will NOT reflect true system performance.

---

## 5. Automatic Checkpointing

The simulation uses automatic checkpointing to eliminate cold-start artifacts:

```bash
poetry run python run_simulation.py --days 365
# First run: 90-day burn-in → saves checkpoint → 365 data days
# Subsequent runs: loads checkpoint → 365 data days (skips burn-in)
```

**Key Concepts:**
- `--days N` specifies N days of **steady-state data** (post burn-in)
- Checkpoints are named by config hash: `steady_state_{hash}.json.gz`
- Config changes invalidate old checkpoints (new burn-in runs automatically)
- `_metrics_start_day` excludes burn-in from Triangle Report metrics

**CLI Flags:**
- `--no-checkpoint` - Disable auto-checkpointing (always cold-start)
- `--warm-start PATH` - Use specific snapshot file
- `--skip-hash-check` - Load snapshot even if config changed

---

## 6. State Manager: The Single Source of Truth

`StateManager` holds all simulation state as NumPy tensors for O(1) access:

```python
# Tensor Shapes
inventory: np.ndarray        # [n_nodes, n_products] - Current stock (actual)
perceived_inventory: np.ndarray  # [n_nodes, n_products] - What system "sees"
wip: np.ndarray              # [n_nodes, n_products] - Work in process at plants

# Index Mappings (str -> int for O(1) lookup)
node_id_to_idx: dict[str, int]
product_id_to_idx: dict[str, int]
```

**Critical:** Always use `state.update_inventory(node_id, product_id, delta)` - never modify tensors directly.

---

## 7. Daily Simulation Loop (Execution Order)

Every `tick()` (1 day) in `Orchestrator._run_day()`:

```
1. RISK EVENTS   → Trigger disruptions (RiskEventManager)
2. PRE-QUIRKS    → Apply Phantom Inventory shrinkage (QuirkManager)
3. ARRIVALS      → Process in-transit shipments that arrived today
4. DEMAND        → POSEngine generates retail sales (consumes store inventory)
5. REPLENISHMENT → MinMaxReplenisher creates orders (Physics-based SS + ABC)
6. ALLOCATION    → AllocationAgent allocates inventory to orders (Fair Share)
7. LOGISTICS     → LogisticsEngine creates shipments (FTL rules, Emissions)
8. MRP           → MRPEngine plans production (uses demand signal)
9. PRODUCTION    → TransformEngine executes manufacturing
10. POST-QUIRKS  → Apply logistics delays/congestion (QuirkManager)
11. MONITORING   → PhysicsAuditor validates mass balance, records KPIs
```

---

## 8. Death Spiral Safeguards

The simulation has multiple safeguards to prevent feedback loops that collapse production:

### MRP Production Floors
ABC-based minimum production floors ensure demand is met regardless of inventory signals:
- **A-Items:** 90% of expected demand (absolute minimum)
- **B-Items:** 80% of expected demand
- **C-Items:** 70% of expected demand

### Timeout Mechanisms
- **Production Orders:** 14-day timeout (stale orders dropped)
- **Held Logistics Orders:** 14-day timeout (FTL consolidation doesn't block indefinitely)
- **Pending Replenishment Orders:** 14-day timeout (allows retry)
- **Completed Batches:** 30-day retention (memory cleanup)

### Demand Signal Priority
MRP uses order-based demand with expected demand as floor:
```python
daily_vol = np.maximum(daily_vol, self.expected_daily_demand)  # Floor
daily_vol = np.minimum(daily_vol, self.expected_daily_demand * 4.0)  # Cap
```

---

## 9. Recipe Matrix: Vectorized BOM

The `RecipeMatrixBuilder` creates a dense matrix for instant ingredient calculations:

```
Shape: [n_finished_goods, n_ingredients]
Value R[i,j] = quantity of ingredient j needed to make 1 unit of product i

Usage: ingredient_requirements = demand_vector @ recipe_matrix
```

This enables O(1) MRP calculations for thousands of SKUs.

---

## 10. Inventory Policies (Physics-Based)

### The Full Safety Stock Formula
$SS = z \sqrt{\bar{L}\sigma_D^2 + \bar{D}^2\sigma_L^2}$
- **Demand Risk ($\sigma_D$):** Protected by rolling 28-day demand history.
- **Supply Risk ($\sigma_L$):** Protected by tracking realized lead times.

### Dynamic ABC Segmentation
Products are dynamically classified every 7 days based on cumulative sales volume:
- **A-Items (Top 80%):** High service level target ($z=2.33$)
- **B-Items (Next 15%):** Medium service level target ($z=1.65$)
- **C-Items (Bottom 5%):** Lower service level target ($z=1.28$)

---

## 11. Campaign Batching

Instead of producing all SKUs daily, use campaign-style production:

1. **Trigger-Based Production:** Only produce when DOS < threshold
   - Configurable per ABC class (e.g., A=14, B=10, C=5 days)

2. **Batch Sizing:** Produce `production_horizon_days` worth per SKU

3. **SKU Limit:** Max SKUs/plant/day to cap changeover overhead

4. **Priority Sorting:** Lowest DOS first (most critical items)

**Configuration:** `simulation_config.json` → `manufacturing.mrp_thresholds.campaign_batching`

---

## 12. Customer Channels & Store Formats

### Customer Channels (`CustomerChannel` enum)
| Channel | Description | Logistics Mode |
|---------|-------------|----------------|
| `B2M_LARGE` | Big retailers (Walmart DC, Target DC) | FTL |
| `B2M_CLUB` | Club stores (Costco, Sam's Club) | FTL |
| `B2M_DISTRIBUTOR` | 3P Distributors | FTL |
| `ECOMMERCE` | Amazon, pure-play digital | FTL |
| `DTC` | Direct to consumer | Parcel |

### Store Formats (`StoreFormat` enum)
`RETAILER_DC`, `HYPERMARKET`, `SUPERMARKET`, `CLUB`, `CONVENIENCE`, `PHARMACY`, `DISTRIBUTOR_DC`, `ECOM_FC`

---

## 13. Order Types

| Order Type | Priority | Behavior |
|------------|----------|----------|
| `STANDARD` | 3 | Normal (s,S) replenishment |
| `RUSH` | 1 | Expedited, reduced lead time |
| `PROMOTIONAL` | 2 | Linked to promo calendar |
| `BACKORDER` | 4 | Created when allocation fails |

---

## 14. Risk Events

`RiskEventManager` triggers deterministic disruptions (all toggleable):

| Type | Effect |
|------|--------|
| Contamination | Batches with target ingredient → `REJECTED` |
| Port Strike | Logistics delays multiplied |
| Supplier Opacity | SPOF supplier OTD drops |
| Cyber Outage | Target DC WMS down |
| Carbon Tax | CO2 cost multiplier |

Configured in `simulation_config.json` under `risk_events`.

---

## 15. Behavioral Quirks

`QuirkManager` injects realistic supply chain pathologies (all toggleable):

| Quirk | Effect |
|-------|--------|
| `bullwhip_whip_crack` | Order batching amplifies bullwhip |
| `phantom_inventory` | Shrinkage creates actual vs perceived divergence |
| `port_congestion_flicker` | AR(1) correlated delays |
| `single_source_fragility` | SPOF ingredient delays cascade |
| `human_optimism_bias` | Over-forecast for new products |
| `data_decay` | Older batches have higher rejection rates |

Configured in `simulation_config.json` under `quirks`.

---

## 16. The Supply Chain Triangle

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

**Key Metrics Tracked:**
| Metric | Formula / Description |
|--------|----------------------|
| **Store Service Level (OSA)** | `Actual_Sales / Consumer_Demand` |
| **Perfect Order Rate** | OTIF + Damage Free + Doc Accuracy |
| **Cash-to-Cash Cycle** | DSO + DIO - DPO |
| **Scope 3 Emissions** | kg CO2 per case shipped |
| **Inventory Turns** | `COGS / Avg_Inventory` |
| **MAPE** | Forecast accuracy |
| **Shrinkage Rate** | Phantom inventory % |
| **SLOB %** | Slow/Obsolete inventory |
| **Truck Fill Rate** | `Actual_Load / Capacity` |
| **OEE** | Overall Equipment Effectiveness |

---

## 17. Configuration Files

| File | Purpose |
|------|---------|
| `config/simulation_config.json` | Runtime parameters (MRP, logistics, quirks, initialization) |
| `config/world_definition.json` | Static world (products, network topology, recipe logic) |
| `config/benchmark_manifest.json` | Risk scenarios, validation targets |

---

## 18. Key Commands

```bash
# Run Simulation
poetry run python run_simulation.py
poetry run python run_simulation.py --days 365

# Run Tests
poetry run pytest
poetry run pytest tests/test_milestone_4.py -v

# Type Check & Lint
poetry run mypy src/
poetry run ruff check src/

# Generate Static World
poetry run python scripts/generate_static_world.py

# Calibrate Config
poetry run python scripts/calibrate_config.py --apply
```

---

## 19. Debugging Checklist

When simulation behaves unexpectedly:

1. **Service Level = 0%?**
   - Check `initialization.store_days_supply` - cold start starvation?
   - Check `PhysicsAuditor` for mass balance drift
   - Look for SPOF ingredient exhaustion

2. **Orders exploding?**
   - Bullwhip feedback loop - check `MinMaxReplenisher` order volumes
   - Fill-or-Kill creating infinite retry loop

3. **Production stalled?**
   - Check ingredient inventory at plants
   - Check `MRPEngine` purchase orders for raw materials
   - Verify `TransformEngine` capacity vs demand

4. **OEE too low?**
   - Check changeover times in recipes
   - Check `min_production_qty` (MOQ)

5. **Mass Balance violations?**
   - Check `PhysicsAuditor.audit()` output
   - Look for inventory updates outside `StateManager`

6. **Negative inventory detected?**
   - Check if new code bypasses `StateManager.update_inventory()`
   - Verify `actual_inventory` is used for allocation decisions
   - Look for direct tensor manipulation without floor guards

7. **Metrics healthy at 30 days but degrade at 365 days?**
   - This is a **drift problem**, not a structural issue
   - **SLOB drift:** C-items accumulating (slow movers don't sell)
   - **Service drift:** SLOB throttling reducing production
   - **Key diagnostic:** Compare ABC class inventory at day 30 vs day 365

---

## 20. Architecture Diagrams

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
