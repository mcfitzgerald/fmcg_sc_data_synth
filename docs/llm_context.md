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
| **Production planning** | `simulation/mrp.py` | `MRPEngine.generate_production_orders()` — SKU + dependent bulk intermediate orders |
| **DRP-Lite (B/C production)** | `simulation/drp.py` | `DRPPlanner.plan_requirements()` — forward-netting daily targets for B/C items |
| **Ingredient procurement** | `simulation/mrp.py` | `MRPEngine.generate_purchase_orders()` — two-step BOM explosion for 3-level BOM |
| **Recipe matrix (BOM)** | `network/recipe_matrix.py` | `RecipeMatrixBuilder` — dense [n_products, n_products] matrix |
| **Production execution** | `simulation/transform.py` | `TransformEngine` — two-pass: bulk intermediates first, then SKUs |

### Logistics, Validation & Export
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Shipment creation** | `simulation/logistics.py` | `LogisticsEngine.create_shipments()` |
| **Returns processing** | `simulation/logistics.py` | `LogisticsEngine.process_returns()` |
| **Physics validation** | `simulation/monitor.py` | `PhysicsAuditor`, `RealismMonitor` |
| **Behavioral quirks** | `simulation/quirks.py` | `QuirkManager` |
| **Risk events** | `simulation/risk_events.py` | `RiskEventManager` |
| **Data export** | `simulation/writer.py` | `SimulationWriter`, `ThreadedParquetWriter` |
| **Initialization & priming** | `simulation/orchestrator.py` | `_initialize_inventory()`, `_prime_synthetic_steady_state()` |
| **Warm-start loader** | `simulation/warm_start.py` | `load_warm_start_state()`, `WarmStartState` |

### Data Models
| Concept | File | Key Classes |
|---------|------|-------------|
| **Network primitives** | `network/core.py` | `Node`, `Link`, `Order`, `Shipment`, `Batch` |
| **Work Orders** | `network/core.py` | `ProductionOrder` (Status: Planned/Released/Complete) |
| **Returns** | `network/core.py` | `Return`, `ReturnLine` (Status: Requested/Processed) |
| **Channel enums** | `network/core.py` | `CustomerChannel`, `StoreFormat`, `OrderType` |
| **Product definitions** | `product/core.py` | `Product` (bom_level: 0=SKU,1=bulk,2=RM), `Recipe`, `ProductCategory` |
| **Packaging enums** | `product/core.py` | `PackagingType`, `ContainerType`, `ValueSegment` |
| **Promo calendar** | `simulation/demand.py` | `PromoCalendar`, `PromoEffect` |

#### Cached Integer Indices

`OrderLine`, `Order`, and `Shipment` carry cached integer indices to avoid repeated `dict.get()` lookups in inner loops:

- `OrderLine.product_idx: int = -1` — maps to `state.product_id_to_idx[line.product_id]`
- `Order.source_idx: int = -1` / `Order.target_idx: int = -1` — maps to `state.node_id_to_idx`
- `Shipment.source_idx: int = -1` / `Shipment.target_idx: int = -1` — maps to `state.node_id_to_idx`

Sentinel `-1` means "not populated" — all consumer sites use `x.product_idx if x.product_idx >= 0 else dict.get(...)` for backwards-compatible fallback. `OrderLine` uses `@dataclass(slots=True)` for reduced allocation overhead (155K created/day).

### Generators (Static World Creation)
| Concept | File | Key Classes |
|---------|------|-------------|
| **Product/SKU generation** | `generators/hierarchy.py` | `ProductGenerator` |
| **Network topology** | `generators/network.py` | `NetworkGenerator` |

### Utility Scripts
| Script | Purpose |
|--------|---------|
| `scripts/generate_static_world.py` | Generate static world data (products, recipes, nodes, links) |
| `scripts/calibrate_config.py` | Physics-based config calibration (capacity, safety stock, DOS targets) |
| `scripts/run_standard_sim.py` | Standard workflow runner (priming + stabilization + data run) |
| `scripts/erp/` | **Enterprise Data Generator:** DuckDB-based ETL, sim parquet → 36 normalized CSV tables (368.5M rows) |
| `scripts/erp_schema.sql` | ERP relational schema (PostgreSQL DDL, 36 tables + 14 indexes) |

### Validation & Planning Documents
| Document | Purpose |
|----------|---------|
| `docs/planning/spec.md` | Project status, remaining work, document map |
| `docs/planning/physics.md` | Supply chain physics theory (Little's Law, VUT, Mass Balance) — timeless reference |
| `docs/planning/triangle.md` | Desmet's SC Triangle (Service/Cost/Cash) — timeless reference |
| `docs/planning/archive/` | Historical: `intent.md`, `roadmap.md`, investigation docs (Dec 2024 – Jan 2025) |

### Diagnostic Suite (`scripts/analysis/diagnostics/`)

Shared modular backend used by diagnostics:

| Module | Key Functions | Layer |
|--------|---------------|-------|
| `diagnostics/loader.py` | `load_all_data()`, `classify_node()`, `is_demand_endpoint()`, `DOSTargets`, `load_dos_targets()`, `SeasonalityConfig`, `load_seasonality_config()`, `_pre_aggregate_orders()` | Data loading, enrichment, config-derived targets + seasonality. Also loads cost/price maps, channel map, batch_ingredients, cost_master.json, channel economics. v0.75.0: loads `requested_date` from orders (with graceful fallback). v0.74.0: int16 day columns, smart order pre-aggregation with `line_count` column, all groupby calls use `observed=True`. |
| `diagnostics/first_principles.py` | `analyze_mass_balance()`, `analyze_flow_conservation()`, `analyze_littles_law()` | Layer 1: Physics validation. v0.75.0: Little's Law uses outbound LT (source_echelon), includes Store/Club via inbound demand throughput. Customer DC verdict uses adjusted delta. |
| `diagnostics/operational.py` | `analyze_inventory_positioning()`, `analyze_service_levels()`, `analyze_production_alignment()`, `analyze_slob()` | Layer 2: Operational health. v0.75.0: DOS uses per-echelon throughput (outflow for Plant/RDC/DC, inflow for Store/Club) instead of total POS demand. SLOB reports warm-start baseline. |
| `diagnostics/flow_analysis.py` | `analyze_throughput_map()`, `analyze_deployment_effectiveness()`, `analyze_lead_times()`, `analyze_bullwhip()`, `analyze_control_stability()` | Layer 3: Flow & stability. v0.75.0: Plant→Store in retention rate, RDC→Store in ASCII diagram. |
| `diagnostics/cost_analysis.py` | `compute_per_sku_cogs()`, `compute_logistics_by_route()`, `stream_carrying_cost()`, `compute_cash_to_cash()`, `compute_otif()` | Cost & working capital. v0.75.0: `total_cogs` uses demand-endpoint shipments only (landed COGS). `avg_cases` divides by n_days. |
| `diagnostics/commercial.py` | `compute_channel_pnl()`, `compute_cost_to_serve()`, `compute_margin_by_abc()`, `compute_fill_by_abc_channel()`, `compute_concentration_risk()`, `compute_tail_sku_drag()` | Commercial & channel analysis. v0.75.0: cost-to-serve uses `is_demand_endpoint` (consistent with channel P&L). |
| `diagnostics/manufacturing.py` | `compute_bom_cost_rollup()`, `compute_changeover_analysis()`, `compute_upstream_availability()`, `compute_stockout_waterfall()`, `compute_forward_cover()` | Manufacturing, BOM, upstream. v0.75.0: waterfall uses grouped order events, BOM cost only for ingredients with valid weight_kg, `current_woc` field name. |

**DEMAND_PREFIXES** = `("STORE-", "ECOM-FC-", "DTC-FC-")` — NOT `"CLUB-"` (CLUB-DC nodes are intermediate warehouses, not demand endpoints).

### Diagnostic Playbook (Tiered)

**Standard workflow after a 365d streaming sim:**
1. Run `diagnose_supply_chain.py` → unified 35-question diagnostic (Sections 1-7, ~60s)
2. For inventory deep-dive → run with `--full` flag (adds Section 8, streams inventory.parquet, ~8min)
3. For specialized investigation → use Tier 2 scripts as needed

#### Tier 1: Unified Diagnostic (run after every 365d sim)
| Script | When | What |
|--------|------|------|
| `diagnose_supply_chain.py` | **Always** — comprehensive consultant's checklist | 35 questions across 8 sections: physics, scorecard, service, inventory, flow, manufacturing, financial, inventory deep-dive (--full) |

**Deprecated Tier 1 scripts** (superseded by `diagnose_supply_chain.py` in v0.72.0):
- `diagnose_365day.py` — executive scorecard (now Sections 1-2)
- `diagnose_flow_deep.py` — 20-question forensic deep-dive (now Sections 3-6)
- `diagnose_cost.py` — cost analytics (now Section 7)

#### ERP Database Diagnostic (run after PostgreSQL load)
| Script | When | What |
|--------|------|------|
| `diagnose_erp_database.py` | **After loading ERP CSVs into PostgreSQL** | 52 questions across 10 sections: data landscape, GL reconciliation, SCOR Source/Make/Deliver/Return, Desmet's Triangle, temporal integrity, friction audit, digital thread traceability. psycopg2 against PostgreSQL. |

#### Tier 2: Specialized Diagnostics (run when investigating specific issues)
| Script | When | What |
|--------|------|------|
| `diagnose_a_item_fill.py` | Fill rate drops below target | 4-layer root cause: measurement → stockout location → root cause → ranking |
| `diagnose_service_level.py` | Service degradation patterns | Daily/rolling trends, echelon breakdowns, worst performers |
| `diagnose_slob.py` | SLOB percentage rises | Echelon distribution, velocity, stagnant inventory |
| `analyze_bullwhip.py` | Order variance amplification | Echelon CV ratios, oscillation patterns |
| `check_plant_balance.py` | OEE/production imbalance | Batches per plant, zero-production days |

#### Tier 3: Utilities
| Script | Purpose |
|--------|---------|
| `slice_data.py` | Create small data subset (first N days) for fast diagnostic iteration |

**Shared infrastructure:** `diagnostics/loader.py` is the canonical source for `classify_node()`, `classify_abc()`, `DataBundle`, `load_all_data()`, `SeasonalityConfig`. Used by all diagnostic scripts. Standalone scripts (Tier 2) have local copies of classification functions — acceptable for isolation, but `loader.py` is the source of truth. `SeasonalityConfig.factor(day)` is used by stability and backpressure analyses for seasonal detrending.

**Bug fix pass (v0.75.0):** 18 bugs fixed across 8 files (3 HIGH, 6 MEDIUM, 9 LOW). Key fixes: COGS uses demand-endpoint only (was 3× inflated), warehouse cost uses n_days denominator (was ~5000× underestimated), DOS uses per-echelon throughput (was using total POS for all), stockout waterfall uses grouped order events (was mixing line counts with tuple counts), `requested_date` now loaded for OTIF.

**Memory optimization (v0.74.0):** Diagnostic modules avoid `.copy()` on large DataFrames, use category-level merge instead of row-level `.astype(str)` for echelon/route lookups, and use `observed=True` on all categorical groupby calls. Day columns use int16 (max 365 fits in [-32768, 32767]). Orders include a `line_count` column (int8, always 1 for unique orders; summed count if pre-aggregated).

**Archived scripts:** `scripts/analysis/archive/` contains 9 legacy scripts (CSV-based, shell subprocess, or version-specific) superseded by the diagnostic suite above.

### Enterprise Data Generator (`scripts/erp/`)

DuckDB-based ETL that transforms sim parquet output into 39 normalized ERP CSV tables. Loadable into PostgreSQL and Neo4j.

**Entry point:** `poetry run python -m scripts.erp --input-dir data/output --output-dir data/output/erp`

| Module | Purpose |
|--------|---------|
| `config.py` | ErpConfig + FrictionConfig: loads cost_master.json (incl. friction section), world_def, sim_config; 14-account Chart of Accounts |
| `id_mapper.py` | Bidirectional sim string ID ↔ integer PK mapping (JSON serializable) |
| `sequence.py` | Deterministic `transaction_sequence_id`: `day × 10M + category × 1M + counter`. Categories 0-7 (core), 8 (friction), 9 (payment) |
| `master_tables.py` | DuckDB SQL: 14 master CSVs from static world files + config |
| `transactional.py` | DuckDB-native large tables (orders 60.9M, shipments 69.9M, inventory 99.2M); Python for small tables |
| `gl_journal.py` | Pure DuckDB SQL: 7 reference_types (goods_receipt, production, shipment, freight, sale, return + friction variance/payment) × DR/CR pairs → ~47M per-shipment/batch GL entries with `reference_id` traceability. v0.77.0: production cost uses `batch_cost` CTE. v0.78.1: freight split from shipment (`reference_type='freight'`), freight+variance entries get `node_id` (source plant / receiving plant via AP lookup). node_id coverage: ~98% for physical events, 0% for treasury events (payment/receipt/bad_debt) — matches SAP/Oracle behavior. v0.79.1: inbound freight (`supplier_to_plant`) now capitalizes to DR 1100 RM Inventory (was DR 5300 Freight Expense); outbound freight unchanged (DR 5300). |
| `invoices.py` | DuckDB-native AP invoices (5.4M) and AR invoices (1.5M headers, 57.7M lines). v0.79.1: supplier_id resolved via `erp_shipments.source_sim_id` → `loc_map` (was broken pq_shipments re-join). |
| `friction.py` | v0.78.0: Phase 3.5 — controlled data quality friction. 4 tiers: entity resolution (dup suppliers, SKU aliases), 3-way match failures (price/qty variance → `invoice_variances`), data quality (null FKs, dup invoices w/ line items, status flips), payment timing (`ap_payments`, `ar_receipts`, early discounts, bad debt). All DuckDB SQL, all GL entries balanced. Config: `cost_master.json` → `friction.enabled`. v0.78.1: variance GL entries get plant `node_id` via `ap_node_lookup` (AP invoice → goods_receipt → shipment → GL chain); dup invoices now include `ap_invoice_lines`. Payment lag computed via CTE (single random per row) for seq/date consistency. GL re-sorted by `transaction_sequence_id`; AP invoices exported with ordered COPY. |
| `verify.py` | Post-gen: GL balance (DuckDB), COGS/Revenue ratio, reference_id coverage, FK integrity, sequence monotonicity, friction table stats |
| `neo4j_headers.py` | Neo4j-admin import header files (incl. friction tables) |

**Schema:** `scripts/erp_schema.sql` — 39 tables (9 domains incl. friction), 19 indexes, PostgreSQL DDL.
**Load scripts:** `data/output/erp/load_postgres.sh`, `data/output/erp/load_neo4j.cypher`

**Friction layer (v0.78.0):** Toggled by `cost_master.json` → `friction.enabled`. When enabled, Phase 3.5 injects controlled noise: duplicate suppliers (variant names, ~10%), SKU old-code aliases (~5%, `supersedes_sku_id`), AP invoice price/qty variances (8%/5%), null FKs, duplicate invoices, AR status flips, AP payments, AR receipts, early payment discounts (2%), bad debt writeoffs (0.5%). New GL accounts: 4200 Discount Income, 5500 Bad Debt Expense.

**Performance:** ~7.5 min end-to-end for 394M rows with friction enabled (Phase 1: 0.1s, Phase 2: 50s, Phase 3: 50s, Phase 3.5: 100s, Phase 4: 115s incl. verify). Without friction: ~2.5 min for 230M+ rows.

### Data Export (`simulation/writer.py`)
| Concept | Key Classes |
|---------|-------------|
| **Buffered mode** | `SimulationWriter` — accumulates dicts, writes at end (short runs) |
| **Streaming CSV** | `StreamingCSVWriter` — incremental row-by-row CSV output |
| **Streaming Parquet** | `StreamingParquetWriter` — batched row-group Parquet output |
| **Threaded inventory Parquet** | `ThreadedParquetWriter` — background-thread writer with `DictionaryArray` columns |

### Simulation Output Files
All files are `.csv` by default or `.parquet` with `--format parquet`. Parquet uses dictionary-encoded string columns and float32 for inventory.

| File | Contents |
|------|----------|
| `orders` | Replenishment orders (header + lines flattened). Includes `requested_date` for OTIF measurement. |
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
- **Two Init Paths:** Cold-start (formula priming → synthetic steady-state → 10-day stabilization) or warm-start (`--warm-start <dir>` → load converged state from prior run's parquet → 3-day stabilization).
- **Demand Sensing:** Agents (MRP/Replenishment) use proactive demand forecasts from `POSEngine` to build stock ahead of promotions and seasonality.
- **P&G Scale:** The simulation is calibrated to ~4M cases/day (realistic North American FMCG volume) with ~33 production lines network-wide (15 default + 4 OH + 2 TX + 3 CA + 6 GA + 3 other).

**Simulation Run Lengths:**
- **Full diagnostic (365 days):** Required for accurate KPIs. Includes 10-day stabilization + 365 data days. Use `--streaming --format parquet` for logged runs.
- **Sanity checks (30-50 days with `--no-logging`):** Fast verification only.

---

## 5. Initialization & Stabilization

Two initialization paths:

### Cold Start (default)
1. **Demand-proportional priming** (`_initialize_inventory()`): Seeds on-hand inventory at every node proportional to expected demand and ABC class. Priming targets match operational targets: stores use `store_days_supply=6.0` × ABC factors; DCs use `dc_buffer_days × 1.5/2.0/2.5` (A/B/C); RDCs use `rdc_days_supply=9` × ABC factors with pipeline adjustment. Both RDCs and Customer DCs subtract upstream lead time to prevent double-stocking with pipeline inventory.
2. **Synthetic steady-state priming** (`_prime_synthetic_steady_state()`): Adds pipeline shipments, production WIP (plant FG at `plant_fg_prime_days`=3.5 per plant → 14 DOS total, matching MRP A-item target ~17 DOS with WIP+transit), history buffers, and inventory age
3. **Stabilization** (default 10 days): Normal simulation steps excluded from metrics

### Warm Start (`--warm-start <dir>`)
Loads converged state from a prior run's parquet output, eliminating the synthetic→converged transient:
1. **Inventory**: Reads final-day snapshot from `inventory.parquet` (perceived + actual tensors)
2. **Pipeline**: Restores in-transit shipments from `shipments.parquet` (`arrival_day > checkpoint_day`)
3. **WIP**: Restores active production orders from `production_orders.parquet`
4. **Agent History** (v0.76.0): Restores agent memory from `agent_state/` subdirectory (if present):
   - MRP demand/consumption/production history (14-day buffers) + circular buffer pointers
   - Replenisher demand_history_buffer (28-day), outflow/inflow history (5-day), smoothed demand
   - Lead time history dicts (~4K links × 20 samples)
   - Inventory age matrix (FIFO aging by node×product)
   - **Seamless continuation**: No transient, no 50-day prod/demand ratio dip
5. **Stabilization**: Reduced to 3 days (`warm_start_stabilization_days` config) — or 0 with `--no-stabilization` when using agent state

**Backward Compatibility:** Old snapshots without `agent_state/` fall back to synthetic history priming (~14-28 day settling transient).

Module: `simulation/warm_start.py` — `load_warm_start_state()` returns `WarmStartState` dataclass. `_load_agent_state()` handles optional agent history.

### Snapshot Mode (`--snapshot`)
Writes final-day state to `{output_dir}/snapshot/` as 3 parquet files + agent history:
- **Parquet files**: `inventory.parquet`, `shipments.parquet`, `production_orders.parquet`
- **Agent state** (v0.76.0): `agent_state/` subdirectory with 4 NPZ files + metadata:
  - `mrp_history.npz` (~100KB)
  - `replenisher_history.npz` (~50-100MB compressed)
  - `lt_history.npz` (~1MB)
  - `inventory_age.npy` (~17MB)
  - `metadata.json` (checkpoint_day, cycle_days, seasonal_phase)

**Seasonal Phase Alignment:** Warns if `checkpoint_day % cycle_days != 0`. For seamless warm-start, use `--days N` where `N + stabilization_days` is a multiple of `cycle_days` (365):
```bash
# Phase-aligned snapshots (day 365, 730, 1095, ...)
poetry run python run_simulation.py --days 355 --no-logging --snapshot  # 355+10=365
poetry run python run_simulation.py --days 720 --no-logging --snapshot  # 720+10=730
# Warm-start from aligned snapshot (no seasonal discontinuity)
poetry run python run_simulation.py --days 365 --streaming --format parquet \
  --warm-start data/output/snapshot --no-stabilization
```

Method: `Orchestrator.save_snapshot(output_dir)` + `_save_agent_state()` — uses PyArrow + numpy, no SimulationWriter dependency.

`--days N` specifies N days of **steady-state data** (after stabilization).
Total simulated days = stabilization_days + N.

Config: `simulation_parameters.calibration.initialization.stabilization_days` (default: 10)

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

**Critical:** Always use `state.update_inventory(node_id, product_id, delta)` for simple changes. For inventory receipt (production output, arrivals), use `state.receive_inventory(node_idx, product_idx, qty)` which handles age blending. When deducting inventory, apply FIFO age reduction: `age *= (old_qty - shipped) / old_qty`.

**Read-only views:** `state.get_in_transit_by_target()` and `demand.get_base_demand_matrix()` return read-only NumPy views (not copies). Do NOT mutate the returned arrays — use `.copy()` first if mutation is needed.

### Inventory Age Tracking
Used for industry-standard SLOB calculation (age-based, not DOS-based):
- `age_inventory(days)` - Age all positive inventory by N days (called daily)
- `receive_inventory(node_idx, product_idx, qty)` - Receive fresh inventory with weighted average age blending
- `receive_inventory_batch(delta)` - Batch receive with age blending
- `get_weighted_age_by_product()` - Get inventory-weighted average age per SKU

**FIFO Age Pattern:** Every inventory deduction must reduce age proportionally:
```python
old_qty = max(0.0, float(state.actual_inventory[node_idx, prod_idx]))
if old_qty > 0:
    fraction_remaining = max(0.0, (old_qty - ship_qty) / old_qty)
    state.inventory_age[node_idx, prod_idx] *= fraction_remaining
```
This is applied in: POS consumption, allocation, plant deployment, RDC push, ingredient consumption.

---

## 7. Daily Simulation Loop (Execution Order)

Every `tick()` (1 day) in `Orchestrator._run_day()`:

```
0.  MASS BALANCE  → Start day tracking (PhysicsAuditor)
0a. AGE INVENTORY → Age all inventory by 1 day (for SLOB tracking)
0b. RISK & QUIRKS → Trigger disruptions + Phantom Inventory shrinkage
1.  DEMAND        → POSEngine generates retail sales
2.  CONSUMPTION   → Constrain sales to available inventory, record to MRP
3.  REPLENISHMENT → MinMaxReplenisher creates orders (Physics-based SS + ABC)
4.  ALLOCATION    → AllocationAgent allocates inventory to orders (Fair Share)
5.  LOGISTICS     → LogisticsEngine creates shipments (FTL rules, Emissions)
6.  TRANSIT       → Advance in-transit shipments
7.  ARRIVALS      → Process arrivals (age-aware receipt)
7a. RETURNS       → LogisticsEngine generates returns from arrivals (Damage/Recall)
8.  MRP           → MRPEngine plans production (uses POS demand signal + DRP for B/C)
9.  PRODUCTION    → TransformEngine executes manufacturing (Work Orders → Batches)
10. DEPLOYMENT    → _create_plant_shipments() need-based deployment (Plant→RDC/DC)
10a.PUSH EXCESS   → _push_excess_rdc_inventory() RDC→DC overflow valve
11. POST-QUIRKS   → Apply logistics delays/congestion (QuirkManager)
12. MONITORING    → PhysicsAuditor validates mass balance, records KPIs (inc. age-based SLOB)
13. DATA LOGGING  → SimulationWriter records daily metrics
```

---

## 8. Death Spiral Safeguards

The simulation has multiple safeguards to prevent feedback loops that collapse production:

### Demand Signal (Direct POS)
MRP uses raw POS demand directly — no blending or floor:
```python
demand_for_dos = pos_demand  # demand_floor_weight=0.0, mrp_floor_gating_enabled=false
```
- **Config:** `demand_floor_weight=0.0`, `mrp_floor_gating_enabled=false`
- Demand floor code exists but is disabled.

### Demand Smoothing
DOS calculation uses smoothed demand to reduce noise-driven overproduction:
```python
demand_for_dos = demand_smoothing_weight * expected + (1.0 - demand_smoothing_weight) * pos_today
# 70% expected + 30% POS
```
- **Config:** `demand_smoothing_weight=0.70`
- Stabilizes DOS calculation without interfering with the raw POS production signal

### DOS Cap Guard
Skip production when a product already has sufficient inventory:
```python
if dos_position > cap_dos:  # A=22d, B=25d, C=25d
    continue
```
- **Config:** `inventory_cap_dos_a=22`, `inventory_cap_dos_b=25`, `inventory_cap_dos_c=25`
- These are emergency brakes (~5-8d headroom above operating point), not continuous feedback
- Self-regulating negative feedback — production stops when inventory is sufficient, resumes when consumed

### SLOB Production Dampening — **DISABLED**
```python
slob_dampening_floor = 1.0  # No production reduction for aged inventory
```
- **Config:** `slob_dampening_floor=1.0`, `slob_dampening_ramp_multiplier=1.0`
- **Rationale:** Production should follow demand. Aged stock is a logistics/disposition concern, not a production throttling trigger. Real FMCG handles SLOB via markdown/donation, not by cutting supply.
- Graduated ramp code exists but is inactive (floor=1.0 means factor always 1.0).

### ABC Production Buffers
- **A-Items:** `a_production_buffer` = 1.22x (safe with ABC-aware Phase 4 clipping)
- **B-Items:** `b_production_buffer` = 1.1x (modest buffer, applied to batch qty)
- **C-Items:** `c_production_factor` = 1.05x — applied as buffer multiplier on C-item DRP batches in `mrp.py`

### Emergency Replenishment
Bypass order staggering when any product DOS < `emergency_dos_threshold` (default 3.0):
- Prevents stores with empty shelves from waiting for scheduled order day
- Critical stockouts trigger immediate action regardless of schedule

### Timeout Mechanisms
- **Production Orders:** 14-day timeout (stale orders dropped)
- **Held Logistics Orders:** 14-day timeout (FTL consolidation doesn't block indefinitely)
- **Pending Replenishment Orders:** 14-day timeout (allows retry)
- **Completed Batches:** 30-day retention (memory cleanup)

### Stockout Demand Tracking
Unmet demand from stockouts (`daily_demand - actual_sales`) is recorded and flows upstream to MRP:
- Prevents "demand signal collapse" where stockouts hide true demand
- C-items get proper production priority when shelves are empty

---

## 8a. Need-Based Deployment

Finished goods flow from plants to downstream nodes via need-based deployment:

### `_create_plant_shipments()` — Core Deployment
```python
need = max(0, target_dos × expected_demand × seasonal_factor - current_position)
# current_position = on_hand + in_transit_to_target
# seasonal_factor = MRPEngine._get_seasonal_factor(day)
```
1. `_compute_deployment_needs()` calculates per-target, per-product need vectors (seasonally adjusted)
2. Available FG per product summed across all plants
3. Fair-share allocation when constrained: `fill_ratio = min(available / total_need, 1.0)`
4. Share ceiling headroom: 1.5× prevents any single target from consuming all supply
5. `_select_sourcing_plant_for_product()`: Dynamic — picks plant with most FG for each product

### ABC-Differentiated Target DOS
| Echelon | A-Items | B-Items | C-Items | Config |
|---------|---------|---------|---------|--------|
| **DCs** | `dc_buffer_days × 1.5` (≈10.5d) | `dc_buffer_days × 2.0` (14d) | `dc_buffer_days × 2.5` (17.5d) | `dc_buffer_days=7.0` |
| **RDCs** | 9.0d | 9.0d | 9.0d | Flat `_rdc_target_dos` (flow-through cross-dock, actual DOS ≈8.4) |

### `_push_excess_rdc_inventory()` — RDC→DC Overflow
Active as secondary overflow valve: pushes excess RDC inventory to customer DCs when RDC DOS exceeds threshold (`push_threshold_dos=12.0`, ~1.33× the 9 DOS target). DOS calculations use seasonally-adjusted demand (same factor as deployment). Push receive cap is ABC-differentiated — `dc_buffer_days × ABC_mult × push_receive_headroom(1.15)` → A≈12.1, B≈16.1, C≈20.1 DOS.

### Key Design: Plant FG as Natural Backpressure
Unneeded FG stays at the plant and enters MRP's inventory position calculation (`_calculate_inventory_position()` includes plant FG in pipeline IP). This creates a natural negative feedback loop: high plant FG → high IP → MRP reduces production → equilibrium.

Production backpressure is handled entirely by MRP DOS caps + plant FG in IP — not by physical storage constraints. `Node.storage_capacity` does not exist (all nodes have infinite capacity). This matches real FMCG operations where MRP prevents overproduction at the planning level; plants don't physically block production lines due to warehouse capacity.

**Status:** MRP backpressure and B/C underproduction are resolved. Diagnostic Q12 detrended correlation is +0.14 (near-zero after removing seasonal confound); DOS caps are emergency brakes that correctly don't fire during normal operation (~17 DOS vs 22d cap).

---

## 9. Recipe Matrix: 3-Level BOM

The `RecipeMatrixBuilder` creates a dense matrix for BOM calculations:

```
Shape: [n_products, n_products]   (products = RM + bulk intermediates + SKUs)
Value R[i,j] = quantity of product j needed to make 1 unit of product i

3-Level BOM Structure:
  Level 2 (Raw Materials): BLK-*, ACT-*, PKG-*  — purchased leaf nodes
  Level 1 (Bulk Intermediates): BULK-*  — compounded in-house
  Level 0 (Finished SKUs): SKU-*  — packed shippable cases

Matrix entries:
  R[SKU-X, BULK-Y]     = 1.0    (SKU needs 1 bulk intermediate)
  R[SKU-X, PKG-TUBE-1] = 1.0    (SKU needs packaging)
  R[BULK-Y, BLK-WATER] = 0.005  (bulk needs raw material)
  R[BULK-Y, ACT-SLS]   = 3.57   (bulk needs active chemical)

MRP uses two-step explosion:
  Step 1: sku_production @ R → bulk + packaging needs
  Step 2: bulk_needs @ R → raw material needs
```

This enables O(1) MRP calculations with dependent demand explosion.

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
- **C-Items (Bottom 5%):** Medium service level target ($z=1.65$)

### Throughput-Based DC Ordering
Customer DCs use throughput-based ordering instead of echelon-demand:
```python
dc_order_rate = outflow + correction
# outflow = rolling 5-day average of shipments to stores
# correction = (local_target - local_ip) / dc_correction_days
# correction capped at ±(outflow × dc_correction_cap_pct)
```
- **Config:** `dc_buffer_days=7.0`, `dc_correction_days=7.0`, `dc_correction_cap_pct=0.5`, `throughput_floor_pct=0.7`
- **Physics-derived DC DOS caps:** `dc_buffer_days × mult` → A≈10.5, B=14, C=17.5
- **Effect:** DC ordering tracks actual outflow rather than upstream demand signal, reducing bullwhip

### Anti-Windup Floor Gating
All demand floors are conditional on inventory state to prevent accumulation:
```python
floor_weight = clip((target_dos - local_dos) / target_dos, 0, 1)
# At DOS=0: floor_weight=1.0 (full floor — protect against death spiral)
# At DOS=target: floor_weight=0.0 (floor disengages — prevent accumulation)
```
- **Config:** `floor_gating_enabled=true`
- **Effect:** Floors only activate when inventory is below target, preventing the over-ordering that floors would otherwise cause at steady state

---

## 11. Production Scheduling (ABC-Branched)

Production uses ABC-branched scheduling: A-items use net-requirement (MPS-style), B/C items use DRP-Lite with campaign triggers. All ABC classes use a 14-day production horizon.

### A-Items: Net-Requirement Scheduling

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

### B/C Items: DRP-Lite + Campaign Triggers

B/C items use DRP for forward-netting production targets, with campaign triggers as fallback:

1. **DRP-Driven Production (Primary):** `DRPPlanner.plan_requirements()` projects inventory forward, nets against in-transit/in-production, and level-loads daily production targets
   - DRP daily target is scaled by ABC horizon for batch sizing
   - Campaign triggers still gate when production fires, but DRP modulates batch size

2. **Campaign Trigger Fallback:** When DRP planner is absent, pure trigger-based:
   - `trigger_dos_b`=7, `trigger_dos_c`=6
   - Batch sizing: `production_horizon_days` worth per SKU

### Common Phases (All ABC Classes)

3. **Priority Sorting:** Critical Ratio (`DOS/Trigger`) with shuffle tie-breaker
4. **ABC Slot Reservation:** 60/25/15 split (A/B/C, config-driven) with overflow redistribution
5. **Capacity Cap:** 98% with ABC-aware clipping — A-items protected up to 65% of capacity, B/C absorb clipping first
6. **SKU Limit:** 100 SKUs/plant/day to cap changeover overhead

**Configuration:** `simulation_config.json` → `manufacturing.mrp_thresholds.campaign_batching`

### Capacity Planning

The `--derive-lines` calibration uses physics-based efficiency decomposition:

1. **DOS Cycling Factor:** For B/C items, lines sit idle when DOS > trigger. A-items produce continuously via net-requirement, improving utilization. DRP level-loads B/C production, further improving utilization.
   - Formula (B/C): `dos_coverage = horizon / (horizon + avg_trigger) × stagger_benefit`

2. **Variability Buffer (~1.25x):** Reserve capacity for demand peaks (seasonality + noise).
   - Formula: `buffer = 1 / (1 - safety_z × combined_cv)`

**Parameters:** `simulation_config.json` → `calibration.capacity_planning`:
- `variability_safety_z`: Z-score for capacity buffer (1.28 = 90%)

---

## 12. Customer Channels & Store Formats

### Customer Channels (`CustomerChannel` enum)
| Channel | Description | Logistics Mode | Upstream Routing |
|---------|-------------|----------------|------------------|
| `MASS_RETAIL` | Big retailers (Walmart DC, Target DC) | FTL | **Plant-direct** |
| `GROCERY` | Traditional grocery (Kroger, Albertsons) | FTL | RDC |
| `CLUB` | Club stores (Costco, Sam's Club) | FTL | **Plant-direct** |
| `PHARMACY` | Pharmacy chains (CVS, Walgreens) | FTL | RDC |
| `DISTRIBUTOR` | 3P Distributors | FTL | RDC |
| `ECOMMERCE` | Amazon, pure-play digital | FTL | RDC |
| `DTC` | Direct to consumer | Parcel | RDC |

### Store Formats (`StoreFormat` enum)
`RETAILER_DC`, `HYPERMARKET`, `SUPERMARKET`, `CLUB`, `CONVENIENCE`, `PHARMACY`, `DISTRIBUTOR_DC`, `ECOM_FC`, `DTC_FC`

### Network Topology
| Echelon | Count | Routing |
|---------|-------|---------|
| Plants | 4 | — |
| RDCs | 6 | Plant→RDC (full mesh) |
| Mass Retail DCs | 15 | Plant→DC (direct) |
| Grocery DCs | 10 | RDC→DC |
| Club Depot DCs | 3 | Plant→DC (direct) |
| Pharmacy DCs | 4 | RDC→DC |
| Distributor DCs | 3 | RDC→DC |
| Ecom FCs | 18 | RDC→FC |
| DTC FCs | 3 | RDC→FC |
| Stores | ~3,822 | DC→Store (hierarchical) |
| **Total** | **~4,238** | |

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
| `config/cost_master.json` | Post-sim cost parameters: per-route logistics (FTL/LTL), echelon warehouse rates, manufacturing cost structure (labor/overhead % of material by category), channel DSO, penalty costs, product_costs (deprecated fallback). Not used by simulation engine. |

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
| `--days N` | N days of steady-state data (after 10-day stabilization) |
| `--streaming` | Write data incrementally (required for 365-day logged runs) |
| `--format parquet` | Parquet output (columnar, dictionary-encoded strings) |
| `--inventory-sample-rate N` | Inventory snapshots every N days (1=daily, 7=weekly) |
| `--no-logging` | Skip data export (fastest, Triangle Report metrics only) |
| `--snapshot` | Write final-day state to `{output_dir}/snapshot/` for warm-start |
| `--warm-start DIR` | Load converged state from prior run's parquet output |
| `--no-stabilization` | Skip stabilization burn-in (0 days). Use with warm-start from fully converged state |

### Post-Run Diagnostics
```bash
# Tier 1: Unified diagnostic — 35 questions, 8 sections (~60s)
poetry run python scripts/analysis/diagnose_supply_chain.py --data-dir data/output

# Tier 1: With inventory deep-dive (streams inventory.parquet, ~8min)
poetry run python scripts/analysis/diagnose_supply_chain.py --data-dir data/output --full

# Tier 1: Single section for dev/debug
poetry run python scripts/analysis/diagnose_supply_chain.py --section 6

# Tier 2: Specialized diagnostics (as needed)
poetry run python scripts/analysis/diagnose_a_item_fill.py
poetry run python scripts/analysis/diagnose_service_level.py
poetry run python scripts/analysis/diagnose_slob.py
poetry run python scripts/analysis/analyze_bullwhip.py
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
   - **Key diagnostic:** Compare ABC class inventory at day 30 vs day 365
   - Run `diagnose_365day.py` for comprehensive 3-layer diagnostic
   - Note: Stability analysis uses seasonal detrending — ensure drift isn't just seasonal oscillation

8. **Inventory turns low but production/demand ratio near 1.0?**
   - This is a **distribution-level problem**, not a production problem
   - Check echelon-level inventory growth: Customer DCs and Club DCs may be accumulating
   - **Key diagnostic:** Run `diagnose_365day.py` → Layer 2 (operational) + Layer 3 (flow analysis)
   - Use `diagnose_slob.py` for echelon breakdown

9. **CLUB-DC double-counting trap**
   - CLUB-DC nodes are **intermediate warehouses**, not demand endpoints
   - `DEMAND_PREFIXES = ("STORE-", "ECOM-FC-", "DTC-FC-")` — do NOT include `"CLUB-"`
   - Including "CLUB-" causes ~11% demand double-counting (~179M/yr)
   - Actual club stores are `STORE-CLUB-*` (matched by `"STORE-"` prefix)

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
    ├──→ LogisticsEngine.create_shipments()
    │         ├──→ bin-packing (weight/cube)
    │         └──→ create Shipment objects
    │
    ├──→ MRPEngine.generate_production_orders()
    │         ├──→ DRPPlanner.plan_requirements() [B/C items]
    │         ├──→ recipe_matrix @ demand_vector
    │         └──→ creates ProductionOrder objects
    │
    ├──→ TransformEngine.execute_production()
    │         ├──→ select line (sticky vs capacity)
    │         ├──→ check materials (vectorized)
    │         ├──→ consume ingredients
    │         └──→ produce finished goods (parallel lines)
    │
    ├──→ _create_plant_shipments() [DEPLOYMENT]
    │         ├──→ _compute_deployment_needs() per target
    │         ├──→ _select_sourcing_plant_for_product()
    │         └──→ fair-share allocation when constrained
    │
    ├──→ _push_excess_rdc_inventory() [OVERFLOW]
    │         └──→ RDC→DC when RDC DOS > threshold
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
| `simulation/logistics.py` | ~510 | `uniform(0.1, 0.5)` | Return qty range (% of line qty) |
| `simulation/logistics.py` | ~511 | `min_return_qty=1.0` | Minimum return quantity (cases) |
| `simulation/logistics.py` | ~515 | `restock_probability=0.8` | Restock vs scrap disposition |
| `simulation/mrp.py` | ~1199 | `60.0/45.0`, `0.5/0.7` | B-item DOS throttling thresholds and scale factors (fallback, only when DRP absent) |
| `agents/replenishment.py` | ~956 | `forecast_horizon=14` | Proactive demand sensing lookahead (days) |
| `simulation/orchestrator.py` | ~1627 | `min_fg_inventory_threshold=100.0` | Inventory turns divide-by-zero guard |

**Note:** `changeover_time_multiplier=0.1` is intentional (SMED-optimized lines), not a hardcode.

**Priority:** The `logistics.py` returns subsystem has the most values (4) with zero config coverage. The `mrp.py` B-item throttling thresholds are in a fallback code path (only active when DRP planner is absent).
