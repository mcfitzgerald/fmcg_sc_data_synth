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
| **Returns (Reverse Logistics)** | `simulation/logistics.py` | `LogisticsEngine.generate_returns_from_arrivals()` |

### Manufacturing & Procurement
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Production planning** | `simulation/mrp.py` | `MRPEngine.generate_production_orders()` |
| **Ingredient procurement** | `simulation/mrp.py` | `MRPEngine.generate_purchase_orders()` |
| **Recipe matrix (BOM)** | `network/recipe_matrix.py` | `RecipeMatrixBuilder` |
| **Production execution** | `simulation/transform.py` | `TransformEngine.execute_production()` |

### Logistics, Validation & Export
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Shipment creation** | `simulation/logistics.py` | `LogisticsEngine.create_shipments()` |
| **Returns processing** | `simulation/logistics.py` | `LogisticsEngine.process_returns()` |
| **Physics validation** | `simulation/monitor.py` | `PhysicsAuditor`, `RealismMonitor` |
| **Behavioral quirks** | `simulation/quirks.py` | `QuirkManager` |
| **Risk events** | `simulation/risk_events.py` | `RiskEventManager` |
| **Data export** | `simulation/writer.py` | `SimulationWriter`, `ThreadedParquetWriter` |

### Data Models
| Concept | File | Key Classes |
|---------|------|-------------|
| **Network primitives** | `network/core.py` | `Node`, `Link`, `Order`, `Shipment`, `Batch` |
| **Work Orders** | `network/core.py` | `ProductionOrder` (Status: Planned/Released/Complete) |
| **Returns** | `network/core.py` | `Return`, `ReturnLine` (Status: Requested/Processed) |
| **Channel enums** | `network/core.py` | `CustomerChannel`, `StoreFormat`, `OrderType` |
| **Product definitions** | `product/core.py` | `Product`, `Recipe`, `ProductCategory` |
| **Packaging enums** | `product/core.py` | `PackagingType`, `ContainerType`, `ValueSegment` |
| **Promo calendar** | `simulation/demand.py` | `PromoCalendar`, `PromoEffect` |

### Generators (Static World Creation)
| Concept | File | Key Classes |
|---------|------|-------------|
| **Product/SKU generation** | `generators/hierarchy.py` | `ProductGenerator` |
| **Network topology** | `generators/network.py` | `NetworkGenerator` |

### Utility Scripts
| Script | Purpose |
|--------|---------|
| `scripts/generate_static_world.py` | Generate static world data (products, recipes, nodes, links) |
| `scripts/generate_warm_start.py` | Generate warm-start snapshot manually |
| `scripts/export_erp_format.py` | **ETL:** Transform sim output to normalized ERP tables (SQL-ready) |

### Diagnostic Analysis Scripts (Parquet-based, `data/output` default)
| Script | Purpose | Key technique |
|--------|---------|---------------|
| `scripts/analysis/diagnose_a_item_fill.py` | 4-layer A-item fill rate root cause analysis (measurement, stockout location, root cause, ranking) | PyArrow row-group streaming for inventory.parquet |
| `scripts/analysis/diagnose_service_level.py` | Service level trend, echelon breakdown, worst performers, degradation phases | PyArrow row-group streaming for inventory.parquet |
| `scripts/analysis/diagnose_slob.py` | SLOB inventory: echelon distribution, DOS, velocity, imbalance, production vs demand | PyArrow row-group streaming for inventory.parquet |
| `scripts/analysis/analyze_bullwhip.py` | Bullwhip effect: echelon variance amplification, production oscillation, order batching | Lightweight (no inventory load) |
| `scripts/analysis/check_plant_balance.py` | Plant production load balance (batches per plant, zero-production days) | Lightweight |
| `scripts/analysis/analyze_results.py` | General results overview (orders, shipments, batches, inventory) | **Legacy CSV — needs migration** |

**Note:** `find_missing_skus.py`, `find_trapped_inventory.py`, `analyze_batches.py`, `analyze_production_mix.py`, `quick_check.py`, and `debug_head.py` are legacy CSV scripts. Use the Parquet-based diagnostic scripts above for current analysis.

### Data Export (`simulation/writer.py`)
| Concept | Key Classes |
|---------|-------------|
| **Buffered mode** | `SimulationWriter` — accumulates dicts, writes at end (short runs) |
| **Streaming CSV** | `StreamingCSVWriter` — incremental row-by-row CSV output |
| **Streaming Parquet** | `StreamingParquetWriter` — batched row-group Parquet output |
| **Threaded inventory Parquet** | `ThreadedParquetWriter` — background-thread writer with `DictionaryArray` columns (v0.39.8) |

### Simulation Output Files
All files are `.csv` by default or `.parquet` with `--format parquet`. Parquet uses dictionary-encoded string columns and float32 for inventory.

| File | Contents |
|------|----------|
| `orders` | Replenishment orders (header + lines flattened) |
| `shipments` | Logistics shipments with `emissions_kg` for Scope 3 tracking |
| `batches` | Production batches (Work Order execution) |
| `batch_ingredients` | Ingredient consumption per batch (BOM traceability) |
| `production_orders` | Work orders (Plan-to-Produce lifecycle) |
| `forecasts` | S&OP demand forecasts (14-day horizon) |
| `returns` | Reverse logistics (RMAs) |
| `inventory` | Periodic inventory snapshots (frequency controlled by `--inventory-sample-rate`) |

---

## 4. Setup & Workflow (Order of Operations)

### Running the Simulation

```bash
# Full diagnostic (365 days, streaming Parquet — canonical run)
poetry run python run_simulation.py --days 365 --streaming --format parquet --inventory-sample-rate 1

# Quick sanity check (50 days, no data export)
poetry run python run_simulation.py --days 50 --no-logging

# Regenerate static world (after changing world_definition.json)
poetry run python scripts/generate_static_world.py
```

**Key Concepts:**
- **Auto-Checkpointing:** The Orchestrator automatically detects config changes (hash check). If changed, it runs a 90-day burn-in and saves a snapshot. If unchanged, it loads the snapshot and runs immediately.
- **Demand Sensing:** Agents (MRP/Replenishment) now use proactive demand forecasts from `POSEngine` to build stock ahead of promotions and seasonality.
- **P&G Scale:** The simulation is calibrated to ~4M cases/day (realistic North American FMCG volume) with ~33 production lines network-wide (15 default + 4 OH + 2 TX + 3 CA + 6 GA + 3 other).

**Simulation Run Lengths:**
- **Full diagnostic (365 days):** Required for accurate KPIs. Includes 90-day burn-in + 365 data days. Use `--streaming --format parquet` for logged runs.
- **Sanity checks (30-50 days with `--no-logging`):** Fast verification only.

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
inventory: np.ndarray           # [n_nodes, n_products] - Current stock (actual)
perceived_inventory: np.ndarray # [n_nodes, n_products] - What system "sees"
inventory_age: np.ndarray       # [n_nodes, n_products] - Weighted avg age (days)
wip: np.ndarray                 # [n_nodes, n_products] - Work in process at plants

# Index Mappings (str -> int for O(1) lookup)
node_id_to_idx: dict[str, int]
product_id_to_idx: dict[str, int]
```

**Critical:** Always use `state.update_inventory(node_id, product_id, delta)` - never modify tensors directly.

### Inventory Age Tracking (v0.39.2)
Used for industry-standard SLOB calculation (age-based, not DOS-based):
- `age_inventory(days)` - Age all positive inventory by N days (called daily)
- `receive_inventory_batch(delta)` - Receive fresh inventory with weighted average age blending
- `get_weighted_age_by_product()` - Get inventory-weighted average age per SKU

---

## 7. Daily Simulation Loop (Execution Order)

Every `tick()` (1 day) in `Orchestrator._run_day()`:

```
0. MASS BALANCE  → Start day tracking (PhysicsAuditor)
0a. AGE INVENTORY → Age all inventory by 1 day (for SLOB tracking)
1. RISK EVENTS   → Trigger disruptions (RiskEventManager)
2. PRE-QUIRKS    → Apply Phantom Inventory shrinkage (QuirkManager)
3. DEMAND        → POSEngine generates retail sales (consumes store inventory)
3a. CONSUMPTION  → Record actual sales to MRP (for demand calibration)
4. REPLENISHMENT → MinMaxReplenisher creates orders (Physics-based SS + ABC)
5. ALLOCATION    → AllocationAgent allocates inventory to orders (Fair Share)
6. LOGISTICS     → LogisticsEngine creates shipments (FTL rules, Emissions)
7. ARRIVALS      → Process in-transit shipments (age-aware receipt)
7a. RETURNS      → LogisticsEngine generates returns from arrivals (Damage/Recall)
8. MRP           → MRPEngine plans production (uses POS demand signal)
9. PRODUCTION    → TransformEngine executes manufacturing (Work Orders → Batches)
10. POST-QUIRKS  → Apply logistics delays/congestion (QuirkManager)
11. MONITORING   → PhysicsAuditor validates mass balance, records KPIs (inc. age-based SLOB)
```

---

## 8. Death Spiral Safeguards

The simulation has multiple safeguards to prevent feedback loops that collapse production:

### Demand Signal Floor (v0.39.3)
MRP uses expected demand as floor to prevent death spiral from low service:
```python
# v0.39.3: Weighted blend with expected as floor
blended = actual * (1 - demand_floor_weight) + expected * demand_floor_weight
demand_for_dos = max(expected, blended)  # Never go below expected
```
- **Config:** `demand_floor_weight` (default 0.65 = 65% expected, 35% actual)
- **Rationale:** Reduced from 0.8 in v0.42.0 to improve promo/peak responsiveness while retaining floor

### ABC Production Buffers (v0.39.3, updated v0.42.0)
- **A-Items:** `a_production_buffer` = 1.22x (raised from 1.15 in v0.42.0 — safe with ABC-aware Phase 4 clipping)
- **B-Items:** `b_production_buffer` = 1.1x (modest buffer, applied to batch qty)
- **C-Items:** No penalty factor - use longer horizons (21 days) instead

### Emergency Replenishment (v0.39.3, updated v0.42.0)
Bypass order staggering when any product DOS < `emergency_dos_threshold` (default 3.0):
- Raised from 2.0 to 3.0 in v0.42.0 for 1-day earlier stockout prevention
- Prevents stores with empty shelves from waiting for scheduled order day
- Critical stockouts trigger immediate action regardless of schedule

### Timeout Mechanisms
- **Production Orders:** 14-day timeout (stale orders dropped)
- **Held Logistics Orders:** 14-day timeout (FTL consolidation doesn't block indefinitely)
- **Pending Replenishment Orders:** 14-day timeout (allows retry)
- **Completed Batches:** 30-day retention (memory cleanup)

### Stockout Demand Tracking (v0.39.3)
Unmet demand from stockouts (`daily_demand - actual_sales`) is recorded and flows upstream to MRP:
- Prevents "demand signal collapse" where stockouts hide true demand
- C-items get proper production priority when shelves are empty

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

## 11. Production Scheduling (ABC-Branched)

Production uses ABC-branched scheduling (v0.40.0): A-items use net-requirement (MPS-style), B/C items use campaign triggers.

### A-Items: Net-Requirement Scheduling (v0.40.0)

Instead of trigger-based feast/famine, A-items compute the gap between target inventory and current position:

```python
target_inventory = demand_rate × horizon × buffer   # 14d × 1.15 = 16.1 DOS target
net_requirement  = target_inventory - inventory_position
batch_qty        = max(net_requirement, 0)           # Skip if at/above target
```

- **Self-regulating:** Items at target produce nothing; depleted items get proportionally larger batches
- **Demand-matched:** Total daily A-item production ≈ total A-item daily demand (natural equilibrium)
- **Smooth loading:** 310 A-items across 240 A-slots (100 SKUs × 0.60 × 4 plants) → each item produced every ~1.3 days
- **No trigger gate:** Eliminates idle capacity between trigger firings

### B/C Items: Campaign Trigger Scheduling

B/C items retain the original trigger-based approach:

1. **Trigger-Based Production:** Only produce when DOS < threshold
   - Configurable per ABC class: `trigger_dos_b`=5, `trigger_dos_c`=4 (v0.43.0)

2. **Batch Sizing:** Produce `production_horizon_days` worth per SKU

### Common Phases (All ABC Classes)

3. **Priority Sorting:** Critical Ratio (`DOS/Trigger`) with shuffle tie-breaker
4. **ABC Slot Reservation:** 60/25/15 split (A/B/C, config-driven) with overflow redistribution
5. **Capacity Cap:** 98% with ABC-aware clipping — A-items protected up to 65% of capacity, B/C absorb clipping first (v0.42.0)
6. **SKU Limit:** 100 SKUs/plant/day to cap changeover overhead (v0.43.0, raised from 80)

**Configuration:** `simulation_config.json` → `manufacturing.mrp_thresholds.campaign_batching`

### Capacity Planning (v0.36.3)

The `--derive-lines` calibration uses physics-based efficiency decomposition:

1. **DOS Cycling Factor:** For B/C items, lines sit idle when DOS > trigger. A-items now produce continuously via net-requirement, improving utilization.
   - Formula (B/C): `dos_coverage = horizon / (horizon + avg_trigger) × stagger_benefit`

2. **Variability Buffer (~1.25x):** Reserve capacity for demand peaks (seasonality + noise).
   - Formula: `buffer = 1 / (1 - safety_z × combined_cv)`

**Parameters:** `simulation_config.json` → `calibration.capacity_planning`:
- `variability_safety_z`: Z-score for capacity buffer (1.28 = 90%)

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
| **SLOB %** | Slow/Obsolete inventory (age-based, FIFO approximation) |
| **Truck Fill Rate** | `Actual_Load / Capacity` |
| **OEE** | Overall Equipment Effectiveness (planned time denominator) |
| **TEEP** | Total Effective Equipment Performance (`OEE × Utilization`, calendar time denominator) |

---

## 17. Configuration Files

| File | Purpose |
|------|---------|
| `config/simulation_config.json` | Runtime parameters (MRP, logistics, quirks, initialization) |
| `config/world_definition.json` | Static world (products, network topology, recipe logic) |
| `config/benchmark_manifest.json` | Risk scenarios, validation targets |

---

## 18. Key Commands

### Simulation
```bash
# Full diagnostic run (365 days, streaming Parquet — canonical)
poetry run python run_simulation.py --days 365 --streaming --format parquet --inventory-sample-rate 1

# Quick sanity check (50 days, metrics only, no data export)
poetry run python run_simulation.py --days 50 --no-logging
```

**CLI Flags:**
| Flag | Purpose |
|------|---------|
| `--days N` | N days of steady-state data (after automatic 90-day burn-in) |
| `--streaming` | Write data incrementally (required for 365-day logged runs) |
| `--format parquet` | Parquet output (columnar, dictionary-encoded strings) |
| `--inventory-sample-rate N` | Inventory snapshots every N days (1=daily, 7=weekly) |
| `--no-logging` | Skip data export (fastest, Triangle Report metrics only) |
| `--no-checkpoint` | Disable auto-checkpointing (always cold-start) |

### Post-Run Diagnostics
```bash
# A-item fill rate root cause (4-layer analysis)
poetry run python scripts/analysis/diagnose_a_item_fill.py

# Service level trend + degradation analysis
poetry run python scripts/analysis/diagnose_service_level.py

# SLOB inventory + echelon imbalance
poetry run python scripts/analysis/diagnose_slob.py

# Bullwhip effect detection
poetry run python scripts/analysis/analyze_bullwhip.py

# Plant production load balance
poetry run python scripts/analysis/check_plant_balance.py
```

All diagnostic scripts default to `data/output` and accept a positional path argument or `--data-dir`.

### Other
```bash
poetry run ruff check src/               # Lint
poetry run mypy src/                      # Type check
poetry run python scripts/generate_static_world.py  # Regenerate static world
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

## 20. First-Principles Debugging & Tuning

**STOP:** Do not guess configuration values (e.g., `num_lines`, `safety_stock`).
**LOOK:** The simulation is a closed system governed by physics. Use Mass Balance to isolate the bottleneck.

### Phase 1: The Hierarchy of Constraints (Check in Order)

1.  **Global Mass Balance (Supply vs. Demand)**
    *   *Equation:* `Total_Capacity (Cases/Day)` vs. `Total_Demand (Cases/Day)`
    *   *Check:* Is `Demand > Capacity`?
    *   *Fix:* Increase `num_lines` or `production_hours`. No amount of logic tuning can fix a physics deficit.

2.  **Utilization Balance (Plant Load)**
    *   *Symptom:* Global capacity looks fine, but service is low.
    *   *Check:* Compare `batches.parquet` counts by `plant_id` (use `check_plant_balance.py`).
    *   *Red Flag:* One plant has 20k batches, another has 2k.
    *   *Fix:* Adjust `supported_categories` in `simulation_config.json` to offload work to the idle plant.

3.  **Flow & Location (Trapped Inventory)**
    *   *Symptom:* High Global Inventory (Days of Supply > 60) but Low Service Level.
    *   *Check:* Where is the inventory? (Plant vs. RDC vs. Store).
    *   *Diagnosis:*
        *   Stuck at Plant? -> Logistics/Deployment failure.
        *   Stuck at RDC? -> Allocation failure.
        *   Stuck at Customer DC? -> Replenishment signal failure.
        *   **Empty Stores?** -> **Distribution failure.**

4.  **Starvation & Exclusion (The "Long Tail")**
    *   *Symptom:* Service Level is 50-60%. Top items are 98%, Bottom items are 0%.
    *   *Check:* Count unique `product_id` in `batches.parquet` vs. Total SKUs.
    *   *Red Flag:* 500 Total SKUs, but only 370 ever produced.
    *   *Fix:* The sorting logic is starving C-items. Implement **Fairness Sorting** (Critical Ratio) or **Randomized Tie-Breaking** to force rotation.

### Phase 2: The Tuning Cycle

Once physics are valid (Capacity > Demand) and flow is moving:

1.  **OEE Too Low (<50%)?**
    *   *Cause:* Too many changeovers.
    *   *Fix:* Increase `production_horizon_days` (7 -> 14) or reduce `max_skus_per_plant`.
    *   *Trade-off:* Higher Inventory (Cash) for Higher OEE (Service).

2.  **Inventory Too High (>6 Turns)?**
    *   *Cause:* Batches are too big.
    *   *Fix:* Reduce `production_horizon_days` (14 -> 7).
    *   *Risk:* OEE will drop. Ensure you have spare capacity (Lines) to handle the friction.

### Key Diagnostic Scripts

*   `diagnose_a_item_fill.py` — Full root cause analysis: measurement validation, stockout location, allocation/production/lead-time/replenishment breakdown, ranked causes
*   `diagnose_service_level.py` — Service level trend over time, by echelon, by product category, worst performers, degradation phases
*   `diagnose_slob.py` — Inventory distribution by echelon, DOS analysis, velocity/turns, SLOB identification, production vs demand ratio
*   `check_plant_balance.py` — Plant load balance (batches/day, zero-production days)
*   `analyze_bullwhip.py` — Echelon variance amplification (CV ratios), production oscillation, order batching effects

---

## 21. Architecture Diagrams

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

## 22. Known Hardcodes (Deferred Config Migration)

The following values are hardcoded but working correctly. They should be migrated to config files in a future hygiene pass:

| File | Line(s) | Value(s) | What it controls |
|------|---------|----------|-----------------|
| `simulation/logistics.py` | ~493 | `return_prob=0.05` | Return/RMA probability per shipment |
| `simulation/logistics.py` | ~511 | `min_return_qty=1.0` | Minimum return quantity (cases) |
| `simulation/logistics.py` | ~515 | `restock_probability=0.8` | Restock vs scrap disposition |
| `simulation/logistics.py` | ~510 | `uniform(0.1, 0.5)` | Return qty range (% of line qty) |
| `simulation/mrp.py` | ~1095 | `60.0/45.0`, `0.5/0.7` | B-item DOS throttling thresholds and scale factors |
| `agents/replenishment.py` | ~906 | `forecast_horizon=14` | Proactive demand sensing lookahead (days) |
| `simulation/orchestrator.py` | ~1192 | `min_fg_inventory_threshold=100.0` | Inventory turns divide-by-zero guard |

**Priority:** The `logistics.py` returns subsystem has the most values (4) with zero config coverage. The `mrp.py` B-item throttling thresholds are the most impactful but B-fill is 99%+ so there's no urgency.
