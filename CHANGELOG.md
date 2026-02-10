# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.62.0] - 2026-02-10

### Fix Plant FG Priming to Match MRP Steady State

Plant FG grew +61% over 365 days because per-plant priming was 2.0 DOS (total 8 DOS across 4 plants) while MRP's A-item MPS target was ~17 DOS. MRP saw a 9 DOS deficit on day 1 and overproduced at +25% above demand for 30 days, building 22M cases that persisted all year.

#### Fix: Increase plant FG priming to match MRP IP target (orchestrator.py, simulation_config.json)
- New config: `calibration.initialization.plant_fg_prime_days` = 3.5 (was hardcoded 2.0)
- Per-plant FG: 3.5 DOS × 4 plants = 14 DOS total
- With ~2 DOS in-production WIP + ~1 DOS in-transit = ~17 DOS total plant IP
- Matches MRP A-item target: `horizon(14) × a_buffer(1.22)` = 17.08 DOS
- MRP starts in equilibrium — minimal catch-up production needed
- Expected plant FG growth: ~14% (was +61%)

## [0.61.0] - 2026-02-10

### Seasonal-Aware Deployment + Tighter MRP Caps

v0.60.0 diagnostic (365-day) revealed deployment uses static `base_demand_matrix` while actual POS swings ±12% seasonally (34% peak-to-trough). During peaks, deployment under-targets causing DC drain; during troughs, deployment over-targets causing plant FG accumulation. MRP DOS caps (A=30/B=35/C=35) provided no backpressure — 13–18d of unused headroom above operating points.

#### Fix 1: Seasonal-Aware Deployment (orchestrator.py)
- Scale `expected_demand` by `MRPEngine._get_seasonal_factor(day)` in `_compute_deployment_needs()` — deployment targets now track actual POS seasonality
- Scale demand in `_push_excess_rdc_inventory()` — RDC DOS and DC push suppression DOS use seasonal demand
- During peaks: higher targets → more deployment → plant FG drains (seasonal buffer)
- During troughs: lower targets → less deployment → plant FG accumulates with MRP cap ceiling

#### Fix 2: Tighten MRP DOS Caps (simulation_config.json)
- `inventory_cap_dos_a`: 30 → 22 (5d above ~17d A-item IP)
- `inventory_cap_dos_b`: 35 → 25 (8d above ~17d post-batch IP peak)
- `inventory_cap_dos_c`: 35 → 25 (same as B)
- With seasonal deployment reducing FG variance, caps can be tighter
- Caps only bite deep into troughs (~42 days of accumulation before cap triggers for A-items)

## [0.60.0] - 2026-02-09

### Fix Priming Mismatches & RDC Accumulation

Diagnostic investigation of v0.59.0 revealed monotonic inventory trends at every echelon — Customer DCs draining (-53K/day), Stores draining (-117K/day), RDCs growing (+26K/day). Root cause: 3 priming bugs + 1 push threshold issue.

#### Bug 1: Customer DC priming used RDC targets (71% over-prime for A-items)
- Code used `rdc_abc_target_dos` (18/15/12 DOS for A/B/C) instead of DC deployment targets (`dc_buffer_days × mult` = 10.5/14/17.5 DOS)
- A-items were over-primed by 71% — spent the entire year draining toward steady state
- **Fix:** Build `dc_abc_target_dos` from `dc_buffer_days × 1.5/2.0/2.5` to match operational deployment

#### Bug 2: Store priming DOS didn't match channel profiles (40-100% over-prime)
- `store_days_supply=10.0` × ABC factors gave 12/10/8 DOS, but channel profiles target 5-8 DOS (most at 6.0)
- **Fix:** `store_days_supply` 10.0 → 6.0 (ABC factors give A=7.2, B=6.0, C=4.8)

#### Bug 3: Customer DCs lacked pipeline adjustment
- RDC priming subtracted upstream lead time to prevent double-stocking with pipeline inventory
- Customer DCs had no such adjustment — got full on-hand + pipeline arriving
- **Fix:** Apply same `_get_avg_upstream_lead_time()` adjustment as RDCs (floor at 2.0 days)

#### Issue 4: RDC push threshold too high (dead zone)
- `push_threshold_dos=40.0` only activated when RDC DOS exceeded 40 — 2.7× the 15 DOS target
- Created 25-DOS dead zone where inventory accumulated with no escape
- **Fix:** `push_threshold_dos` 40.0 → 20.0 (~1.3× target, real-world push behavior)

#### Config Changes
| Parameter | Old | New | Rationale |
|---|---|---|---|
| `store_days_supply` | 10.0 | 6.0 | Match channel profiles |
| `push_threshold_dos` | 40.0 | 20.0 | ~1.3× RDC target (15 DOS) |
| `customer_dc_days_supply` | 10.0 | DEPRECATED | Replaced by `dc_buffer_days × ABC mult` |

#### Files Modified
| File | Changes |
|---|---|
| `src/prism_sim/simulation/orchestrator.py` | DC priming uses `dc_abc_target_dos` (dc_buffer_days × ABC mult); DC pipeline adjustment added; dead `customer_dc_days` variable removed |
| `src/prism_sim/config/simulation_config.json` | `store_days_supply` 10→6, `push_threshold_dos` 40→20, `customer_dc_days_supply` deprecated |

## [0.59.0] - 2026-02-09

### Remove Artificial Config Knobs & Fix Age Tracking Bugs

End-to-end flow tracing revealed two independent root causes of poor KPIs: (1) artificial configuration parameters with no real-world supply chain analog were inflating store ordering, and (2) inventory age tracking had bugs in 4 of 5 deduction paths — only POS sales properly reduced age via FIFO.

#### Config: Remove Artificial Controls
- **`default_min_qty`** 100 → 10: Was forcing C-items into 33-day orders at stores (100 units / 3 units/day demand). Reduced to 10 (= batch_size, representing real case-pack minimums)
- **`min_order_qty`** 100 → 25: DC minimum order had no supply chain basis at 100
- **`order_cycle_days`** 1 → 3: Real stores order 2-3x/week, not daily
- **`min_safety_stock_days`** 3.0 → 0.5: Artificial floor was overriding the variance-based safety stock formula
- **`drp_replenish_multiplier`** 1.75 → 1.0: Fudge factor with no DRP analog — real DRP uses net requirements

#### Code: Fix Store Priority
- Store order priority changed from `LOW` → `STANDARD` in replenishment agent — stores were being deprioritized vs DCs in allocation with no real-world basis

#### Code: Fix FIFO Age Tracking (4 bugs)
Age tracking used a weighted-average FIFO approximation (`new_age = old_age × fraction_remaining`), but only POS consumption applied it. Four other inventory deduction paths bypassed age reduction entirely, causing plant FG age to climb monotonically (~1 day/day):

1. **Production output** (`transform.py`): Used `update_inventory()` (no age blending) instead of `receive_inventory()` — fresh FG wasn't blending with existing stock age
2. **Ingredient consumption** (`transform.py`): No FIFO age reduction when ingredients consumed for production
3. **Plant deployment** (`orchestrator.py`): No FIFO age reduction when FG shipped from plants to DCs/RDCs
4. **RDC push** (`orchestrator.py`): No FIFO age reduction when RDC overflow pushed to DCs
5. **Allocation** (`allocation.py`): No FIFO age reduction when DCs/RDCs fill replenishment orders

#### Validated Results (365-day)
| Metric | v0.58.0 | v0.59.0 | Change |
|---|---|---|---|
| Fill Rate | 99.54% | 98.5% | -1% (GREEN) |
| Inventory Turns | 6.90x | 10.31x | **+49%** |
| SLOB | 37.3% | 0.0% | **Fixed** |
| Store DOS | 24.1 | 6.1 | **On target** |
| Cash-to-Cash | 38.1d | 20.6d | **-46%** |
| Prod/Demand | 1.01 | 0.98 | GREEN |
| Perfect Order | 97.5% | 97.5% | Stable |
| Bullwhip | 0.67x | 0.46x | Better |
| OEE | 58.5% | 54.3% | YELLOW |

#### Remaining Observations
- Customer DC accumulation (+25.1% inflow vs outflow)
- RDC inventory diverging (+26K/day)
- A-item production -2.5% below demand
- OEE slightly below target (54.3% vs 55%)

#### Files Modified
| File | Changes |
|---|---|
| `src/prism_sim/config/simulation_config.json` | 5 config parameter changes (min_qty, order_cycle, safety_stock, drp_multiplier) |
| `src/prism_sim/agents/replenishment.py` | Store priority LOW → STANDARD |
| `src/prism_sim/simulation/transform.py` | Production output uses `receive_inventory()` for age blending; ingredient consumption gets FIFO age reduction |
| `src/prism_sim/simulation/orchestrator.py` | Plant deployment and RDC push get FIFO age reduction |
| `src/prism_sim/agents/allocation.py` | Order allocation gets FIFO age reduction |

## [0.58.0] - 2026-02-09

### Removed
- Checkpoint/snapshot system (`snapshot.py`, `generate_warm_start.py`)
- CLI flags: `--warm-start`, `--no-checkpoint`, `--skip-hash-check`
- ~90MB of stale checkpoint files

### Changed
- Single initialization path: demand-proportional priming → synthetic steady-state → stabilization
- Config: `calibration.warm_start` → `calibration.initialization` (simplified)
- Synthetic priming always runs (no config gate)

## [0.57.0] - 2026-02-09

### Replace intent.md + roadmap.md with spec.md

The original planning docs (`intent.md`, `roadmap.md`) were written at project inception (Dec 2024) to define "what should we build." The core engine is now complete through v0.56.2 — all milestones achieved. This replaces the build-phase docs with a single spec that captures where we are and what remains.

#### New
- `docs/planning/spec.md` — project status, validated scorecard (v0.56.1), iterative validation process, likely code change areas, ERP export status, document map

#### Archived → `docs/planning/archive/`
- `intent.md` — original project vision (all milestones complete)
- `roadmap.md` — original task roadmap (all milestones complete)
- `365_day_drift_diagnosis.md`, `regression_investigation.md`, `calibration_diagnostic_report.md` — historical investigations
- `notes/fresh_start_comparison.md`, `notes/schema_gap_analysis.md` — historical analyses

#### Updated
- `CLAUDE.md` — Prime Directive #1 now references `spec.md`; Spec-Driven Development section updated
- `docs/llm_context.md` — Validation & Planning Documents table updated with `spec.md`, `physics.md`, `triangle.md`, and `archive/`
- `pyproject.toml` — version bump 0.56.2 → 0.57.0

## [0.56.2] - 2026-02-08

### Documentation Update — Sync `llm_context.md` and `CLAUDE.md` to v0.56.1

Both docs were last substantively updated at v0.45.0. This brings them current with 11 versions of changes (v0.46.0–v0.56.1).

#### `docs/llm_context.md`
- **Section 3 (File Map):** Added `drp.py`, `snapshot.py`, diagnostic suite package, utility scripts (`calibrate_config.py`, `run_standard_sim.py`, `erp_schema.sql`)
- **Section 4-5 (Setup/Checkpointing):** Updated burn-in from 90→10 days, added synthetic priming explanation
- **Section 7 (Daily Loop):** Added steps 10 (DEPLOYMENT), 10a (PUSH EXCESS), 13 (DATA LOGGING); restructured to match actual `_step()` code
- **Section 8 (Death Spiral):** Rewrote demand signal (direct POS), added demand smoothing, updated DOS caps (A=30/B=35/C=35), marked SLOB dampening DISABLED
- **NEW Section 8a:** Need-based deployment (v0.55.0) — `_create_plant_shipments()`, ABC target DOS table, plant FG backpressure
- **Section 10 (Inventory):** Added throughput-based DC ordering (v0.52.0) and anti-windup floor gating (v0.51.0)
- **Section 11 (Production):** Updated B/C items from campaign triggers to DRP-Lite + triggers; updated trigger/horizon values
- **Section 19 (Debugging):** Updated items 7-8 with v0.56.1 results; added item 9 (CLUB-DC double-counting trap)
- **Section 21 (Diagrams):** Updated daily loop sequence with deployment and DRP steps
- **Section 22 (Hardcodes):** Corrected line numbers (mrp.py ~1199, replenishment.py ~956, orchestrator.py ~1627)

#### `CLAUDE.md`
- Added `drp.py` and `snapshot.py` to directory tree
- Updated data flow with deployment steps (7-8) and DRP note
- Updated node count 4,500+ → ~4,200+
- Removed stale "default 90 days" comment
- Added Prime Directive #6: keep `docs/llm_context.md` current after code changes

## [0.56.1] - 2026-02-08

### Fix Diagnostic Double-Counting & Disable SLOB Dampening

The v0.56.0 diagnostic showing prod/demand ratio = 0.89 was a **measurement error**, not a production issue. Production actually matches demand at ratio 1.01.

#### Root Cause: CLUB-DC Double-Counting
- `DEMAND_PREFIXES` included `"CLUB-"` which matched CLUB-DC warehouse nodes
- CLUB-DC nodes are intermediate DCs — they receive from RDCs (~179M/yr) and ship to STORE-CLUB stores (~181M/yr)
- The same inventory was counted as demand TWICE: once at CLUB-DC inbound, once at STORE-CLUB inbound
- Actual club stores are `STORE-CLUB-*` (already matched by `"STORE-"` prefix)
- **Fix**: Remove `"CLUB-"` from `DEMAND_PREFIXES` in `loader.py`

#### Other Fixes
- **Diagnostic order status**: Changed `"FULFILLED"` → `"CLOSED"` in fill rate calculation — orders use OPEN→IN_TRANSIT→CLOSED lifecycle (no "FULFILLED" status exists)
- **SLOB dampening disabled**: Set `slob_dampening_floor = 1.0` — production should follow demand, not inventory age. Real FMCG handles aged stock via disposition, not production throttling

#### Validated Results (365-day)
| Metric | v0.56.0 (broken diagnostic) | v0.56.1 (corrected) |
|---|---|---|
| Prod/Demand Ratio | 0.89 (RED) | **1.01 (GREEN)** |
| Fill Rate | 99.7% | 99.7% |
| Turns | 6.93x | 6.93x |
| OEE | 58.5% | 58.5% |
| SLOB | 38.3% | 38.3% |
| Bullwhip | 0.50x | 0.50x |

#### Files Modified
- `scripts/analysis/diagnostics/loader.py` — Remove "CLUB-" from DEMAND_PREFIXES
- `scripts/analysis/diagnostics/operational.py` — Fix order status check
- `src/prism_sim/config/simulation_config.json` — Disable SLOB dampening (floor=1.0)

## [0.56.0] - 2026-02-07

### Fix Production-Demand Divergence

The v0.55.1 diagnostic suite revealed the system was **diverging**: production/demand ratio = 0.88 (declining 0.91→0.84 over 365 days) while ALL echelons accumulated inventory. Root cause: four compounding MRP throttles (SLOB dampening, DOS caps, demand noise, min batch filter) created a death spiral where aged products were silently dropped from production.

#### Fix 1: Graduated SLOB Dampening (highest impact)
- **Replaced** binary 0.25× cut with linear ramp from 1.0 → 0.50 floor
- Products just crossing the age threshold now get ~95% production (was 25%)
- Only products at 2× threshold age hit the 50% floor
- **Breaks the dropout cascade**: floor at 0.50 keeps batches above `absolute_min`, preventing silent production drops
- New `_apply_slob_dampening()` helper replaces 3 identical inline blocks

#### Fix 2: Smoothed Demand Signal for DOS Calculation
- **Replaced** raw single-day POS demand with 70/30 blend (expected + POS)
- Single-day demand spikes/drops no longer cause false DOS cap hits
- Batch sizing is more stable (less volatile → fewer min-batch dropouts)

#### Fix 3: Raised C-item DOS Cap (25 → 35)
- C-items had only 1.8× headroom above 14-day production horizon
- A single campaign batch could push DOS to 20+, leaving only 5 DOS before binary cutoff
- Now matches B-items at 2.5× horizon, giving adequate buffer for campaign cycling

#### Config Changes
| Parameter | Old | New |
|---|---|---|
| `slob_dampening_factor` | 0.25 (binary) | **Removed** |
| `slob_dampening_floor` | — | 0.50 |
| `slob_dampening_ramp_multiplier` | — | 1.0 |
| `demand_smoothing_weight` | — | 0.7 |
| `inventory_cap_dos_c` | 25 | 35 |

#### Files Modified
| File | Changes |
|---|---|
| `src/prism_sim/simulation/mrp.py` | Added `_apply_slob_dampening()` method; replaced 3 inline dampening blocks; smoothed `demand_for_dos`; loaded new config params |
| `src/prism_sim/config/simulation_config.json` | Removed `slob_dampening_factor`; added graduated dampening + smoothing params; raised C-item DOS cap |

## [0.55.1] - 2026-02-07

### Comprehensive Supply Chain Diagnostic Suite

Replaced the monolithic 982-line `diagnose_365day.py` (written for v0.52.0 MRP bug) with a modular three-layer diagnostic suite. The old script had hardcoded root-cause conclusions pointing to a specific MRP demand floor bug that was fixed in v0.53.0.

#### Three-Layer Analysis Pyramid

- **Layer 1: First Principles** — Mass balance, echelon flow conservation waterfall, Little's Law validation
- **Layer 2: Operational Health** — Inventory positioning (DOS by echelon x ABC), service level decomposition, production/demand alignment, SLOB decomposition
- **Layer 3: Flow & Stability** — E2E throughput map with ASCII flow diagram, deployment effectiveness, lead time analysis, bullwhip measurement, control system stability assessment

#### Key Features

- Executive scorecard with traffic-light status for 8 KPIs
- Automated data-driven issue detection (no hardcoded conclusions)
- Modular file structure: 6 files, each under 400 lines
- Memory-safe inventory streaming (100M+ rows via PyArrow row groups)
- All findings ranked by severity with supporting evidence

#### Files

| File | Action | Lines |
|------|--------|-------|
| `scripts/analysis/diagnose_365day.py` | Rewrite — entry point + exec summary | ~460 |
| `scripts/analysis/diagnostics/__init__.py` | New — package exports | ~47 |
| `scripts/analysis/diagnostics/loader.py` | New — data loading, ABC, helpers | ~230 |
| `scripts/analysis/diagnostics/first_principles.py` | New — mass balance, flow, Little's Law | ~400 |
| `scripts/analysis/diagnostics/operational.py` | New — inventory, service, production, SLOB | ~630 |
| `scripts/analysis/diagnostics/flow_analysis.py` | New — E2E flow, deployment, bullwhip, stability | ~630 |

## [0.55.0] - 2026-02-07

### Need-Based Deployment — Replace Push with Physics

v0.54.3 decoupled MRP from deployed inventory (pipeline-only IP) and added RDC receive DOS cap for backpressure. Results: Fill 96.8%, Turns 6.88x, but Prod/Demand ratio still declined to 0.807, SLOB 38.6%. Plant FG grew +156% because the receive cap blocked RDC shipments, FG piled up at plant, pipeline IP approached the MRP DOS cap, and MRP throttled production.

**Root cause:** `_create_plant_shipments()` pushed ALL production to targets by fixed demand shares, then artificial caps tried to prevent overflow. The caps created bottlenecks that chain-reacted into production suppression.

**Fix:** Replace fixed-share push with need-based deployment. Each target receives what it needs to reach target DOS. Unneeded FG stays at plant as a natural buffer. Plant FG in MRP pipeline IP provides natural backpressure — over-stocked products get production suppressed, freeing capacity for under-stocked products.

#### 365-Day Validation Results

| Metric | v0.54.3 | v0.55.0 | Target | Status |
|--------|---------|---------|--------|--------|
| Fill Rate | 96.8% | 99.68% | >=97% | PASS |
| A-Items | - | 99.9% | - | PASS |
| B-Items | - | 99.7% | - | PASS |
| C-Items | - | 96.4% | - | PASS |
| Turns | 6.88x | 6.97x | >=6x | PASS |
| SLOB | 38.6% | 39.1% | <15% | FAIL (pre-existing) |
| OEE | - | 58.1% | 55-70% | PASS |

#### Changes

- **Need-based deployment** (`orchestrator.py`): Rewrote `_create_plant_shipments()`. New algorithm:
  1. `_compute_deployment_needs()`: Per-target, per-product need = max(0, target_dos x demand - current_position), where position = on_hand + in_transit
  2. ABC-differentiated DC target DOS: dc_buffer_days x 1.5/2.0/2.5 for A/B/C (matching v0.54.1 physics-derived caps)
  3. Fair-share allocation when total need > available FG
  4. Deployment share ceiling (share x headroom 1.5x) prevents monopolization
  5. `_select_sourcing_plant_for_product()`: Dynamic plant selection — picks plant with most FG for each product (all targets, not just RDCs)
- **Removed from `_create_plant_shipments()`**: Fixed-share per-batch iteration, DC DOS cap gating, RDC receive DOS cap, DC-to-RDC redirect logic, per-batch `_make_shipment` closure
- **Init precomputation** (`orchestrator.py`): `_precompute_deployment_targets()` builds per-target expected demand, source plant mapping, plant-direct DC IDs
- **Pipeline IP with plant FG** (`mrp.py`): `_calculate_inventory_position()` includes plant FG + in-transit (all plant-sourced) + in-production. Plant FG provides natural MRP backpressure: over-stocked products suppress production, freeing capacity for starved items.
- **DRP with plant FG** (`drp.py`): `plan_requirements()` current_inv includes plant FG. Scheduled arrivals include all plant-sourced in-transit (not just RDC-bound). Gives DRP accurate pipeline picture.
- **New diagnostic counters**: `_deployment_total_need`, `_deployment_total_deployed`, `_deployment_retained_at_plant` replace redirect/skip counters
- **Config** (`simulation_config.json`):
  - Added: `rdc_target_dos` (15.0), `share_ceiling_headroom` (1.5)
  - Removed: `dc_deploy_target_dos` (replaced by physics-derived ABC values), `rdc_receive_dos_cap`, `rdc_push_floor_dos`, `rdc_push_cap_dos`, `plant_push_floor_dos`, `plant_push_cap_dos`
  - Kept: `push_allocation_enabled`, `push_threshold_dos` (40.0), `push_receive_dos_cap` (12.0) — RDC->DC overflow safety valve unchanged

#### Key Design Decisions

1. **Dynamic plant sourcing**: All targets source from whichever plant has FG for each product (not hard-mapped to linked plant). Plant-direct DC network links determine lead time, not sourcing exclusivity. This prevents 114 products from being unavailable at DCs whose linked plant doesn't manufacture them.
2. **Plant FG in MRP IP**: Plant FG is the deployment buffer. Including it in IP creates a natural feedback loop: when targets don't need inventory, FG accumulates -> MRP suppresses production -> capacity reallocated to starved products. Excluding plant FG caused bimodal distribution: some products had 50+ DOS at plant (wasted capacity) while 114 products had zero FG at all locations.
3. **ABC-differentiated DC deployment targets**: DC target DOS matches the physics-derived caps from v0.54.1 (dc_buffer_days x multiplier). B/C items get higher buffers (14/17.5d) than A-items (10.5d) to account for their burstier replenishment patterns.

## [0.54.3] - 2026-02-06

### Pipeline-Only IP — Decouple MRP from Deployed Inventory

v0.54.2 redirected over-cap DC production to RDCs → RDC inventory grew +106% (28M → 58M). This inflated MRP's Inventory Position, causing the DOS cap guard and A-item net requirement to suppress production. Production/demand ratio declined from 0.92 to 0.83 over the year; plant FG hit 0 by day 252.

**Root cause:** MRP included RDC on-hand in IP. RDC inventory is *deployed buffer* — already committed to the distribution system and being consumed by downstream DC/store orders. Including it in IP is like a factory manager throttling production because a distant warehouse has stock.

#### Changes

- **Pipeline-only MRP IP** (`mrp.py`): `_calculate_inventory_position()` now returns Plant FG + in-transit + in-production only. RDC on-hand removed. Steady-state IP drops from ~25-35d to ~9-18d — MRP DOS caps (30/35/25) transform from daily regulators to genuine emergency brakes.
- **Pipeline-only DRP IP** (`drp.py`): `plan_requirements()` current_inv now sums only Plant FG on-hand. Same rationale — RDC is deployed buffer, not production pipeline.
- **RDC receive DOS cap** (`orchestrator.py`): `_create_plant_shipments()` gates RDC shipments at `rdc_receive_dos_cap` (25d). When RDC is over cap, FG stays at plant → pipeline IP rises → MRP throttles. Natural backpressure path replaces IP-based coupling.
  - Value chain: init target=15d, receive cap=25d, push threshold=40d (15d gap prevents oscillation)
  - Diagnostic counters: `_rdc_skip_count`, `_rdc_skip_qty`
- **Config** (`simulation_config.json`): Added `rdc_receive_dos_cap: 25`. MRP caps unchanged (30/35/25).

## [0.54.2] - 2026-02-06

### Fix Plant-Direct DC Push — Root Cause of +583% DC Drift

v0.54.1's physics-derived DOS caps only controlled DC **orders** (pull). But 55% of plant production was **pushed** directly to plant-direct DCs (RET-DC, CLUB-DC) via `_create_plant_shipments()` with no DOS gating. Result: plant-direct DCs grew +977%/+500% while RDC-sourced DCs were stable (-2% to -34%).

#### Changes

- **DOS gating for plant-direct DC shipments** (`orchestrator.py`): `_create_plant_shipments()` now checks target DC DOS before shipping. DCs over their ABC-classified cap (10.5/14/17.5) have their share redirected to RDCs instead.
  - RDCs receive freely (distribution hubs, designed to buffer)
  - Redirected inventory not lost — redistributed to RDCs for normal pull flow
  - Same physics-derived caps as replenishment system (dc_buffer_days x ABC mult)
- **Diagnostic counters** added: `_plant_dc_redirect_count`, `_plant_dc_redirect_qty`

## [0.54.1] - 2026-02-05

### Derive DC Caps from Physics — Fix Customer DC +633% Drift

v0.54.0 fixed production/demand ratio (~1.0) and eliminated plant inventory trapping, but Customer DC inventory grew +633% (40M → 293M), accounting for 75.3% of system inventory. Root cause: DC DOS caps (25/30/35) were 3-5× the steady-state target (~4-7d), turning safety valves into de facto accumulation targets.

#### Physics Derivation

Little's Law: DC on-hand ≈ outflow × (dc_buffer_days − lead_time) ≈ 4d. Max after batch spike ≈ 7d (= dc_buffer_days). DOS cap = dc_buffer_days × ABC multiplier.

#### Changes

- **Physics-derived DC DOS caps** (`replenishment.py`): Replace hardcoded `dc_dos_cap_a/b/c` with `dc_buffer_days × multiplier`. With buffer=7: A≈10.5, B=14, C=17.5. Self-adjusts if buffer changes.
- **Config: multipliers replace absolutes** (`simulation_config.json`): `dc_dos_cap_mult_a/b/c` = 1.5/2.0/2.5 replace `dc_dos_cap_a/b/c` = 25/30/35
- **Reverted over-widened parameters:**
  - `dc_correction_cap_pct`: 0.75 → 0.5 (v0.52.0 original)
  - `order_cap_base_demand_multiplier`: 5.0 → 3.0 (v0.50.0 original)
  - `push_receive_dos_cap`: 25 → 12 (≈ dc_buffer_days × 1.7)

## [0.54.0] - 2026-02-05

### Natural Flow Restoration — Strip Bandaids, Trust Physics

v0.53.1 showed production/demand ratio ~0.85 (underproducing), yet plant FG grew +121% and RDCs drained -58%. Root cause: push gating trapped inventory at plants, making plant FG and RDC inventory non-fungible while MRP treated them as fungible. 27+ accumulated control mechanisms (floors, caps, gates, tapers, blends, dampeners) created feedback loop confusion.

**Design philosophy:** Remove artificial flow barriers. Let production match demand. Let inventory flow freely. Use DOS caps only as emergency safety valves.

#### Part A: Remove Flow Barriers (`orchestrator.py`)

- **Stripped `_create_plant_shipments()` push gating** (~90 lines removed)
  - Removed RDC push gating (v0.53.1 soft taper, DOS checks)
  - Removed plant-direct DC push gating (v0.52.0 soft taper)
  - Removed excess redirect to RDCs block
  - Method now ~40 lines: iterate batches, distribute by share, create shipments
- **Removed helper methods:** `_precompute_rdc_downstream_demand()`, `_precompute_plant_direct_dc_demand()`
- **Removed init attributes:** `_plant_direct_dc_ids`, `_plant_direct_dc_demand`, `_rdc_downstream_demand`
- **Raised `_push_excess_rdc_inventory()` thresholds:** `push_threshold_dos` 30→40, `push_receive_dos_cap` 12→25

#### Part B: Fix Demand Signal (`mrp.py`)

- **Replaced demand signal blend+floor (~60 lines) with direct POS demand (~3 lines)**
  - MRP receives unconstrained POS demand (not constrained sales), so death spiral concern is invalid
  - Seasonal troughs: produce less (correct). Peaks: produce more (correct)
  - Daily noise smoothed by 14-day campaign horizon
- **Disabled floor gating:** `mrp_floor_gating_enabled` → false, `demand_floor_weight` → 0.0

#### Part C: Widen Safety Valves (`simulation_config.json`)

- **MRP DOS caps (emergency brakes, not daily regulators):**
  - A: 22→30 (~2× horizon), B: 25→35 (~2.5× horizon), C: 20→25 (~1.8× horizon)
- **DC ordering caps (allow free pull from RDCs):**
  - A: 10→25, B: 14→30, C: 18→35
- **Other caps:** `dc_correction_cap_pct` 0.5→0.75, `order_cap_base_demand_multiplier` 3.0→5.0

## [0.53.1] - 2026-02-05

### Fix v0.53.0 Overcorrection — Holistic Tuning (6 Layers)

v0.53.0 overcorrected — production dropped to ~0.90 of demand (was 1.088), causing MFG_RDC growth of +143% (worse than v0.52.0's +92%), RETAILER_DC growth of +79%, and plant inventory drain of -51%. Fill rate held at 97.24%.

#### Layer 1: Widen DOS Caps (`simulation_config.json`)
- A: 18 → 22 (4.9d headroom over 17.08d target, one batch cycle)
- B: 20 → 25 (10d buffer for DRP cycles)
- C: 20 → 20 (kept — SLOB risk)

#### Layer 2: Fix Floor Gating Reference Target (`mrp.py`)
- A-item floor gating now uses buffered target: `expected * horizon * a_production_buffer` (×1.22)
- Was using `expected * horizon` (14d) — floor disengaged prematurely at IP=14d instead of ~17d
- B/C items unchanged (no buffer applied)

#### Layer 3: Plant→RDC Push Gating with Soft Taper (`orchestrator.py`)
- **Highest-impact change** — closes the structural missing feedback loop
- New: DOS-based soft taper for plant→RDC push (mirrors plant-direct DC taper)
  - Below `rdc_push_floor_dos` (15): Full push
  - Between 15-35: Linear taper
  - Above `rdc_push_cap_dos` (35): Zero push — excess stays at plant FG
- **Key: Excess stays at plant** (not redirected). MRP sees high plant IP → suppresses production
- New helper: `_precompute_rdc_downstream_demand()` caches per-RDC downstream POS demand
- Also gates redirected plant-direct DC excess through same RDC DOS taper
- New config: `rdc_push_floor_dos: 15.0`, `rdc_push_cap_dos: 35.0`

#### Layer 4: DRP Replenish Multiplier 1.5 → 1.75 (`simulation_config.json`)
- Midpoint between aggressive 1.5 and original 2.0
- Restores B/C production signal without overshoot

#### Layer 5: Priming 14 → 15 (`simulation_config.json`)
- `rdc_days_supply`: 14 → 15 (1d buffer for service level during first week)

#### Layer 6: Keep demand_floor_weight at 0.50 (no change)
- Analysis confirmed 50/50 blend is correct (blended ≈ 0.93×expected in steady state)

## [0.53.0] - 2026-02-05

### Fix MFG_RDC Drift (+92.4%) — Inventory-Conditional MRP Floor

Production exceeded demand by 8.8% systemically over 365 days, causing MFG_RDC inventory to grow +92.4%. The production/demand ratio was 1.13 in Q1, converging only to 1.01 in Q4 — a structural overproduction problem, not just a priming artifact.

**Root cause:** MRP's seasonal demand floor at 85% of expected (`seasonal_floor_min_pct: 0.85`) was unconditional — it fired every day, every product, regardless of inventory state. This prevented production from suppressing when the system was overstocked.

#### Inventory-Conditional MRP Demand Floor (`mrp.py`)

- Gated the seasonal floor on inventory position relative to target (anti-windup pattern)
- `floor_weight = clip(1.0 - IP/target, 0, 1)` — linear ramp, not binary
- IP=0 → full floor (death spiral protection); IP=target → no floor (production can suppress)
- New config: `mrp_floor_gating_enabled: true`

#### Tightened MRP DOS Caps (`simulation_config.json`)

- A: 25 → 18 (1.3× campaign horizon)
- B: 35 → 20 (1.4× horizon)
- C: 45 → 20 (1.4× horizon, was 3.2×)

#### Configurable DRP Replenish Multiplier (`drp.py`)

- `replenish_target = safety_stock * drp_replenish_multiplier` (was hardcoded `* 2.0`)
- New config: `drp_replenish_multiplier: 1.5` — reduces persistent positive production signal

#### Pipeline-Adjusted RDC Priming (`orchestrator.py`)

- Subtract average upstream lead time from RDC on-hand seed to prevent double-stocking
- `pipeline_adjusted_days = max(rdc_days_vec - avg_upstream_lt, 2.0)`
- Reduced `rdc_days_supply` from 21 → 14 (with pipeline priming, 21 was excessive)
- New helper: `_get_avg_upstream_lead_time(node_id)`

## [0.52.0] - 2026-02-04

### Fix DC Inventory Drift (+467%) — Throughput-Based DC Ordering

Customer DC inventory grew +467% over 365 days due to a structural ordering-vs-shipping asymmetry: DCs ordered based on echelon demand (DC + ~100 downstream stores), but stores manage their own inventory independently. The DC held whatever stores didn't pull, causing persistent accumulation.

**Root cause fix:** Replace echelon-demand-based ordering with outflow-based (throughput) ordering. DC order rate = what it ships to stores + local buffer correction. Matches real FMCG practice (retail DCs order ≈ throughput ± buffer).

#### Throughput-Based Echelon Logic (`replenishment.py`)

- Replaced echelon demand/target/correction block in `_apply_echelon_logic()` with:
  ```
  dc_order_rate = dc_outflow + (dc_local_target - dc_local_ip) / correction_days
  ```
- `dc_outflow` = `get_outflow_demand()` (rolling 5-day shipment average to stores)
- `dc_local_target` = `dc_outflow * dc_buffer_days` (local flow-based target, not echelon)
- `dc_local_ip` = DC on-hand + DC in-transit (LOCAL, not echelon aggregate)
- Correction capped at ±50% of outflow to prevent overshoot
- Throughput floor at 70% of expected for demand ramp protection
- DOS cap safety valve retained (uses outflow for consistency)

#### Outflow-Based DC Demand Signal (`replenishment.py`)

- Changed `_calculate_average_demand()` DC branch from inflow (orders received) to outflow (shipments sent)
- Consistent with throughput-based ordering — both use the same signal
- Removed `inflow_cap_multiplier` (no longer needed; replaced by `throughput_floor_pct`)

#### Plant Push Gating with RDC Redirect (`orchestrator.py`)

**Second root cause:** `_create_plant_shipments()` distributed ALL production output to deployment targets (RDCs + plant-direct DCs) unconditionally. Plant-direct DCs received 573M via push vs only 12M via replenishment orders, completely bypassing ordering/DOS cap systems.

**Fix:** Soft taper for plant-direct DC push based on Days of Supply (DOS):
- Full push below `plant_push_floor_dos` (default 20)
- Linear taper to zero at `plant_push_cap_dos` (default 60)
- **Critical: Gated excess redirected to RDCs** (not left at plants)
  - Without redirect: plant inventory backs up → MRP cuts production → system starves
  - With redirect: plants ship everything, RDCs absorb excess, distribute via `_push_excess_rdc_inventory()` with DOS caps

New helper: `_precompute_plant_direct_dc_demand()` identifies plant-direct DCs and computes per-product demand for DOS calculation.

#### Config Changes (`simulation_config.json`)

- Added: `dc_buffer_days: 7.0` — local buffer target in days of outflow (tuned from 10 to 7)
- Added: `dc_correction_days: 7.0` — smoothing period for inventory correction
- Added: `dc_correction_cap_pct: 0.5` — max correction as fraction of outflow
- Added: `throughput_floor_pct: 0.7` — outflow floor for demand ramp protection
- Added: `plant_push_floor_dos: 20.0` — below this DOS, plant pushes full qty to plant-direct DCs
- Added: `plant_push_cap_dos: 60.0` — above this DOS, plant push fully gated (redirected to RDCs)
- Lowered: `push_receive_dos_cap`: 15 → 12 (aligns with dc_buffer_days + margin)
- Removed: `echelon_safety_multiplier`, `echelon_correction_cap_pct`, `inflow_cap_multiplier` (obsolete)

#### 365-Day Validation Results

| Metric | v0.51.0 | v0.52.0 | Target | Status |
|--------|---------|---------|--------|--------|
| Fill Rate | 91.4% | 97.61% | >90% | PASS |
| A-Item Fill | 91.4% | 97.5% | >90% | PASS |
| SLOB | 18.6% | 4.2% | <20% | PASS |
| Turns | 5.67x | 6.25x | 6-8x | PASS |
| Bullwhip | 1.55x | 1.12x | 1.0-2.5x | PASS |
| Total System Growth | — | +10.0% | <+50% | PASS |
| RETAILER_DC Growth | +876% | +0.7% | <+50% | PASS |
| CLUB_DC Growth | +368% | -19.4% | <+50% | PASS |
| MFG_RDC Growth | -33% | +92.4% | Stable | DRIFT |

**Known issue:** MFG_RDC growth (+92.4%) reflects system-wide overproduction accumulating at the buffer echelon. RDCs absorb redirected excess from plant push gating by design. Root cause is MRP-level production planning (out of scope for v0.52.0).

## [0.51.0] - 2026-02-03

### Fix DC Drift, Inverse Bullwhip, and A-Item Shortfall — Anti-Windup Controls

Three symptoms (DC inventory growth, inverse bullwhip 0.30x, A-item fill rate 91.8%) traced to a single root cause: four unconditional demand floors stacked at every layer preventing the system from ever reducing throughput. This is integrator windup — floors only prevent signal collapse but never suppress signal when inventory is excessive.

**Fix: Make all demand floors conditional on inventory state (anti-windup).**

#### Phase 1: Inventory-Conditional Demand Signal

##### 1A. Replenishment floor gating (`replenishment.py`)
- Replaced unconditional `base_signal = max(capped_inflow, expected)` with inventory-gated floor
- Floor weight = `clip((target_dos - local_dos) / target_dos, 0, 1)` — smooth ramp
- At DOS=0: full floor protection (prevents death spiral). At DOS=target: floor disengages (allows drawdown)
- **Config:** `floor_gating_enabled: true`

##### 1B. Tightened DC DOS caps (`simulation_config.json`)
- `dc_dos_cap_a`: 15 → 10, `dc_dos_cap_b`: 20 → 14, `dc_dos_cap_c`: 25 → 18
- Suppression now kicks in at ~1x target DOS (was ~1.5x)

##### 1C. Reduced smoothing window (`replenishment.py`)
- Outflow/inflow history buffer: 7 days → 5 days (config-driven via `outflow_history_days`)
- Shorter window passes weekly demand variance, enabling bullwhip emergence
- Buffer size now configurable instead of hardcoded

#### Phase 2: Production Rebalancing

##### 2A. Equalized C-item production horizon (`simulation_config.json`)
- `production_horizon_days_c`: 21 → 14 (matches A/B items)
- With DRP (v0.48.0) generating right-sized batches, the 21-day horizon was inflating C batches

##### 2B. Inventory-conditional DRP floor (`drp.py`)
- Replaced unconditional `daily_target = max(daily_target, demand_floor)` with:
  `daily_target = where(projected_inv < safety_stock, max(daily_target, floor), daily_target)`
- Floor only activates when inventory is below safety stock (death-spiral zone)
- When inventory is adequate, DRP's net-requirement logic drives production naturally

##### 2C. Reduced MRP demand floor weight (`simulation_config.json`)
- `demand_floor_weight`: 0.65 → 0.50 (actual demand gets equal weight vs expected)
- Previous 65% expected bias inflated B/C production where actual << expected

#### 50-Day Sanity Check Results

| Metric | v0.50.0 | v0.51.0 | Target |
|--------|---------|---------|--------|
| A-item fill rate | 91.8% | 97.1% | >90% |
| B-item fill rate | 98.9% | 100.0% | >95% |
| Inventory turns | 4.8x | 7.27x | 5-8x |
| SLOB % | 20.8% | 0.0% | <20% |
| Overall service | — | 97.66% | >95% |

## [0.50.0] - 2026-02-02

### Fix Runaway Order Behavior — Break Three Feedback Loops

Orders were exploding from 1M/day to 77M/day over 50 days while POS demand stayed flat at 3.3M/day. Diagnostic confirmed this is structural (not caused by synthetic priming). Three compounding feedback loops in the replenishment logic caused unbounded order amplification.

#### Fix 1: Anchor Order Cap to Base Demand (breaks inflow cascade)

- **Modified:** `replenishment.py` — `_finalize_quantities()`
- Previously, order cap used `get_inflow_demand()` (endogenous, inflates with orders) as reference
- Now uses exogenous POS `base_demand_matrix` from POSEngine as absolute cap anchor
- For customer DCs, uses `_expected_throughput` (aggregate downstream POS demand)
- **Config:** `order_cap_base_demand_multiplier: 3.0` (default)
- **Effect:** Cap can no longer inflate with the signal it's supposed to limit

#### Fix 2: Non-Additive Unmet Demand Signal (breaks unmet ratchet)

- **Modified:** `replenishment.py` — `_calculate_average_demand()`
- Previously: `base_signal = base_signal + unmet_signal` (additive, unbounded)
- Now: `base_signal = np.maximum(base_signal, unmet_signal)` (capped at whichever is larger)
- **Modified:** `state.py` — `record_unmet_demand_batch()`
- Previously: `self._unmet_demand += unmet_matrix` (double-counts store stockouts + allocation failures)
- Now: `np.maximum(self._unmet_demand, unmet_matrix)` (takes worst-case, no double-counting)
- **Effect:** Signal can never exceed max(demand, weighted_unmet) instead of their sum

#### Fix 3: Cap Echelon Correction (breaks unbounded correction)

- **Modified:** `replenishment.py` — `_apply_echelon_logic()`
- Previously: `daily_correction = inv_error / 7.0` with no cap — depleted echelon yields 3-10x demand
- Now: correction clipped to ±50% of echelon demand via `np.clip()`
- **Config:** `echelon_correction_cap_pct: 0.5` (default)
- **Effect:** DC order rate bounded to 1.5x echelon demand regardless of inventory gap

#### Fix 4: Cap DC Inflow Signal (defense-in-depth)

- **Modified:** `replenishment.py` — `_calculate_average_demand()`
- Caps inflow signal at multiple of expected throughput before using as demand reference
- **Config:** `inflow_cap_multiplier: 2.0` (default)
- **Effect:** Prevents bullwhip-inflated store orders from propagating as DC demand

#### Results

- Orders stabilize at ~30-35M/day (down from 77M unbounded growth)
- Store fill rate: 97.9% (up from 83.5%)
- A-item fill rate: 97.6% (up from 79.7%)
- Inventory turns: 6.56x (healthy range)
- Inventory mean stabilizes (no longer monotonically declining)

## [0.49.0] - 2026-02-02

### Synthetic Steady-State Initialization — Eliminate 90-Day Burn-In

Instead of simulating 90 days to reach steady state, reverse-engineers what steady state looks like and injects it at day 0. Four synthetic injections make day 1 look like day 91, reducing burn-in from 90 to 10 days.

#### Hack 1: Pipeline Priming (structural)

- **New method:** `orchestrator._prime_pipeline()`
- Creates synthetic in-transit `Shipment` objects on every link, sized to match expected daily flow
- For each link, one shipment per day of lead time arrives on days 1 through `ceil(lead_time)`
- Flow estimation respects echelon topology: store demand for DC→Store, aggregated demand for RDC→DC, deployment shares for Plant→RDC
- **Effect:** Replenisher sees realistic Inventory Position (on-hand + in-transit) from day 1

#### Hack 2: WIP Priming (structural)

- **New method:** `orchestrator._prime_production_wip()`
- Creates synthetic `ProductionOrder` objects at 67% completion, due day 1
- Seeds 2 days of finished goods buffer at all plants
- **Effect:** TransformEngine finds WIP on day 1, FG available for shipment immediately

#### Hack 3: History Buffer Forgery (signal quality)

- **New method:** `orchestrator._prime_history_buffers()`
- Pre-fills `demand_history_buffer` with 28 days of noisy synthetic demand
- Sets `smoothed_demand` to clean seasonal baseline
- Populates `_lt_history` deques for every link with realistic lead-time samples
- Marks `history_idx` as full so variance calculations activate immediately
- **Effect:** Safety stock non-zero from day 1, eliminates 7-14 days of soft instability

#### Hack 4: Inventory Age Seeding (SLOB continuity)

- **New method:** `orchestrator._prime_inventory_age()`
- Sets realistic FIFO ages based on ABC class (A=3d, B=7d, C=15d)
- **Effect:** Prevents SLOB discontinuity at day 60 when thresholds first apply

#### Hack 5: Structural Config Hashing (checkpoint resilience)

- **Modified:** `snapshot.compute_config_hash()`
- Now hashes only physics-relevant keys: manufacturing, demand, logistics, agents, inventory, calibration
- Includes sorted product and node IDs from manifest
- **Effect:** Checkpoints survive changes to writer, validation, quirks, risk_events, and key reordering
- Also updated `orchestrator._compute_config_hash()` to delegate to shared function

#### Hack 6: Reduced Default Burn-In (config)

- `default_burn_in_days` 90 → 10
- New `synthetic_steady_state: true` flag gates all priming logic
- **Effect:** ~80% reduction in cold-start simulation time

#### Files Modified

- `src/prism_sim/simulation/orchestrator.py` — New `_prime_synthetic_steady_state()` + 4 sub-methods + 2 helpers
- `src/prism_sim/simulation/snapshot.py` — Structural config hashing
- `src/prism_sim/config/simulation_config.json` — Burn-in reduction + synthetic flag
- `CHANGELOG.md` — This entry
- `pyproject.toml` — Version bump to 0.49.0

## [0.48.0] - 2026-02-02

### Add DRP-Lite Planning Layer for Fill Rate Improvement

Structural fix targeting the ~85% store fill rate ceiling. The simulation faithfully modeled a purely reactive supply chain; real FMCG companies achieve 97%+ through planning-based systems (DRP, MPS, MEIO). This release adds a simplified DRP layer and rate-based ordering to coordinate production and distribution.

#### Root Cause: Five Structural Issues

1. **Campaign batching feast-famine** — B/C items produce only when DOS < trigger, creating 14-21 day oscillation cycles
2. **MRP demand signal mix attenuation** — expected demand floor produces correct total volume but wrong product mix
3. **Multi-echelon independence** — no coordination between echelons (stores, DCs, RDCs, plants)
4. **Allocation failure ratchet** — shortage → partial fill → re-order → signal decay before supply arrives
5. **Batch rounding amplification** — DC batch sizes (200-500) systematically over-order

#### Fix 1: DRP-Lite Planner (New — structural)

- **New file:** `src/prism_sim/simulation/drp.py` — Simplified Distribution Requirements Planning
- **Mechanism:** Projects inventory forward at RDC level, nets against in-transit and in-production, generates time-phased daily production targets
- **Integration:** B/C items use DRP daily targets instead of binary trigger-based campaigns. A-items keep net-requirement scheduling (already near-continuous).
- **Effect:** Replaces feast-famine oscillation with level-loaded production matched to net requirements

#### Fix 2: Rate-Based DC Ordering (Structural — replaces binary trigger)

- **Where:** `replenishment.py:_apply_echelon_logic()`
- **Old:** `needs_order = echelon_ip < echelon_rop` (binary trigger)
- **New:** `daily_order = echelon_demand + (inv_error / correction_days)` with `correction_days=7`
- **Effect:** Smooth, continuous DC ordering stream with negative feedback (order less when over-stocked, more when under-stocked)

#### Fix 3: Unmet Demand Decay Matched to Lead Time

- **Config:** `unmet_demand_decay` 0.85 → 0.93
- **Old half-life:** 4.25 days (signal dies before supply chain responds in 10 days)
- **New half-life:** 9.6 days (matches 10-day supply chain response time)

#### Fix 4: Reduce Batch Amplification

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| MASS_RETAIL `batch_size` | 500 | 50 | DCs order exact quantities from own RDCs |
| GROCERY `batch_size` | 400 | 50 | Same rationale |
| CLUB `batch_size` | 200 | 50 | Same rationale |
| `store_batch_size_cases` | 20 | 10 | Reduce store-level batch amplification |
| `trigger_dos_b` | 5 | 7 | Higher trigger = more frequent, smaller batches |
| `trigger_dos_c` | 4 | 6 | Same rationale |

#### Files Modified

- `src/prism_sim/simulation/drp.py` — **NEW** DRP planner module
- `src/prism_sim/simulation/mrp.py` — DRP integration for B/C item production
- `src/prism_sim/agents/replenishment.py` — Rate-based DC ordering
- `src/prism_sim/simulation/orchestrator.py` — DRP planner initialization
- `src/prism_sim/config/simulation_config.json` — Parameter tuning
- `CHANGELOG.md` — This entry
- `pyproject.toml` — Version bump to 0.48.0

## [0.47.0] - 2026-02-02

### Fix Distribution-Level Inventory Accumulation

Five fixes targeting the DC inventory buildup problem: Retailer DCs (+462%) and Club DCs (+341%) accumulated inventory because replenishment had no negative feedback loop equivalent to MRP's DOS caps. Production was already controlled (1.02-1.08x demand) from v0.46.0.

#### Fix 1: DC-Level DOS Cap in Replenishment (Primary — structural)

- **Root cause:** DCs order whenever echelon IP < ROP, regardless of local inventory level. No ceiling on DC inventory.
- **Fix:** Add ABC-differentiated DOS guard to `_apply_echelon_logic()`. Suppresses ordering when DC local DOS > cap.
- **Config:** `dc_dos_cap_a=15`, `dc_dos_cap_b=20`, `dc_dos_cap_c=25`
- **Effect:** Prevents DCs from over-ordering when already well-stocked

#### Fix 2: Dampen Unmet Demand Signal (Signal — breaks ratchet)

- **Root cause:** Unmet demand weight=1.0 and decay=0.95 (5%/day) meant every 1-day shortage persisted ~60 days as elevated signal → permanent surplus
- **Fix:** Config-driven weight (1.0→0.5) and decay (0.95→0.85, 15%/day). Shortage now persists ~15 days instead of ~60.
- **Config:** `unmet_demand_weight=0.5`, `unmet_demand_decay=0.85`
- **Why not 0.0?** C-items need recovery signal; 0.5 weight × 0.85 decay gives ~7 days of boost per shortage

#### Fix 3: Guard RDC Push Against Over-Stocked DCs (Flow — prevents push override)

- **Root cause:** Push mechanism was blind to DC inventory. RDCs pushed excess to DCs without checking if DCs already had sufficient stock, undermining DOS cap (Fix 1).
- **Fix:** Before pushing, check target DC DOS per product. Skip push for products where DC DOS >= cap.
- **Config:** `push_receive_dos_cap=15.0`

#### Fix 4: Config Parameter Tuning (Calibration)

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| CLUB/MASS_RETAIL `target_days` | 8.0 | 6.0 | Industry: 3-6 days for high-velocity FTL |
| CLUB/MASS_RETAIL `reorder_point_days` | 4.0 | 3.0 | Proportional to target reduction |
| `push_threshold_dos` | 21.0 | 30.0 | Less aggressive push; let pull system work |
| `customer_dc_days_supply` (init) | 14.0 | 10.0 | Reduce initialization overshoot |
| `a_rop_multiplier` | 1.5 | 1.25 | Reduce A-item demand amplification |
| `slob_dampening_factor` | 0.5 | 0.25 | More aggressive production cut for aged product |

#### Fix 5: Diagnostic Instrumentation (Observability)

- Per-echelon DOS tracking by channel (MASS_RETAIL, CLUB, GROCERY, etc.)
- `dc_order_suppression_count`: Orders suppressed by DC DOS cap (Fix 1)
- `push_suppression_count`: Push allocations blocked by DC DOS check (Fix 3)
- `unmet_demand_magnitude`: Daily sum of unmet demand signal to track ratchet effect

#### Files Modified

- `src/prism_sim/agents/replenishment.py` — Fix 1 (DC DOS cap), Fix 2 (unmet demand dampening)
- `src/prism_sim/simulation/orchestrator.py` — Fix 3 (push guard), Fix 5 (diagnostics)
- `src/prism_sim/config/simulation_config.json` — Fix 4 (parameter tuning), new config keys for Fixes 1-3
- `CHANGELOG.md` — This entry
- `pyproject.toml` — Version bump to 0.47.0

## [0.46.0] - 2026-02-02

### Fix 365-Day Inventory Drift

Four fixes to break the production-SLOB oscillation by adding negative feedback from inventory levels back to production decisions.

#### Fix 1: ABC-Differentiated DOS Cap Guard (Primary)

- **Root cause:** `inventory_cap_dos=30.0` loaded in MRP but never used — no ceiling on production when inventory is already sufficient
- **Fix:** Before adding a candidate to the production list, check if the product's current DOS exceeds its ABC-class cap. If so, skip production for that product.
- **Config:** `inventory_cap_dos_a=25`, `inventory_cap_dos_b=35`, `inventory_cap_dos_c=45` (replaces single `inventory_cap_dos=30`)
- **Effect:** Self-regulating negative feedback loop — production stops when inventory is sufficient, resumes when consumed

#### Fix 2: Seasonal-Aware Demand Floor (Secondary)

- **Root cause:** `demand_for_dos = max(expected, blended)` always floors at annual average. During 6-month seasonal troughs (demand = 0.88x avg), this prevents production from adjusting downward → ~12% systematic overproduction
- **Fix:** Replace with `max(seasonal_floor, blended)` where `seasonal_floor = expected * max(seasonal_factor, 0.85)`. Floor now tracks seasonality but never drops below 85% of expected (death spiral prevention preserved)
- **Config:** `seasonal_floor_min_pct=0.85`

#### Fix 3: SLOB Production Dampening (Tertiary)

- **Root cause:** When inventory is flagged as SLOB-aged (age > ABC threshold), production isn't notified — keeps producing at full rate for products with excess aged stock
- **Fix:** Check weighted inventory age per product before appending candidates. If age exceeds SLOB threshold for its ABC class, reduce batch size by 50%
- **Config:** `slob_dampening_factor=0.5` (uses existing `slob_abc_thresholds`: A=60d, B=90d, C=120d)

#### Fix 4: Diagnostic Demand Proxy (Validation)

- **Root cause:** `diagnose_365day.py` only counted `STORE-*` shipments as demand proxy, missing ~20-30% of demand from CLUB, ECOM-FC, DTC-FC channels (added in v0.44.0)
- **Fix:** Expand demand filter to all demand-generating endpoints: `STORE-`, `CLUB-`, `ECOM-FC-`, `DTC-FC-`
- Also added `PHARM-DC-` and `CLUB-DC-` to `classify_node` for correct echelon classification

#### 365-Day Validation Results

| Metric | v0.45.0 | v0.46.0 | Target | Status |
|--------|---------|---------|--------|--------|
| A-fill | 84.1% | 85.1% | >=85% | PASS |
| B-fill | — | 95.7% | >=90% | PASS |
| C-fill | — | 91.5% | >=85% | PASS |
| Turns | 4.3x | 4.29x | 6-14x | FAIL |
| SLOB | 38.6% | 31.2% | <30% | MARGINAL |
| OEE | — | 62.3% | 55-85% | PASS |

- **Production/demand ratio:** 1.02-1.08x (was 1.27-1.47x) — production-side drift resolved
- **Cumulative excess monotonicity:** 67% (was 100%) — self-correcting
- **Remaining issue:** Customer DC (+357%) and Club DC (+341%) inventory growth — distribution-level problem, not production
- Full analysis: `V046_VALIDATION_STATE.md`

#### Files Modified

- `src/prism_sim/simulation/mrp.py` — Fix 1 (DOS cap guard), Fix 2 (seasonal floor), Fix 3 (SLOB dampening)
- `src/prism_sim/config/simulation_config.json` — New params: `inventory_cap_dos_a/b/c`, `seasonal_floor_min_pct`, `slob_dampening_factor`
- `scripts/analysis/diagnose_365day.py` — Fix 4 (expand demand proxy to all channels)
- `V046_VALIDATION_STATE.md` — Validation results, open questions, next steps
- `CHANGELOG.md` — This entry
- `pyproject.toml` — Version bump to 0.46.0

## [0.45.0] - 2026-02-01

### Network Restructure + Echelon/Batch Fixes

Three work streams in one release: network restructure (6 RDCs, pharmacy DCs, club depots, plant-direct links), echelon inventory position bug fix, and format-scaled store batch sizing.

#### Network Restructure

- **6 RDCs (from 4):** Added RDC-SE (Jacksonville, FL) and RDC-SW (Phoenix, AZ) to fill Southeast and Southwest geographic gaps
- **Pharmacy DCs (4 new):** 375 pharmacy stores now route through 4 regional pharmacy DCs (PHARM-DC-001 through PHARM-DC-004) instead of directly to RDCs
- **Club Depot DCs (3 new):** 47 club stores now route through 3 regional club depots (CLUB-DC-001 through CLUB-DC-003) instead of directly to RDCs
- **Plant-direct links:** Mass retail DCs (15) and club depot DCs (3) now source from nearest plant instead of RDCs, matching real-world high-volume account logistics
- **RDC-routed channels:** Grocery, pharmacy, distributor, ecommerce, DTC continue routing through RDCs

| Echelon | v0.44.0 | v0.45.0 |
|---------|---------|---------|
| RDCs | 4 | 6 (+2) |
| Club DCs | 0 | 3 (new) |
| Pharmacy DCs | 0 | 4 (new) |
| Mass Retail DCs | →RDC | →Plant (direct) |
| Club DCs | — | →Plant (direct) |

#### Echelon IP Bug Fix

- **Root cause:** `_apply_echelon_logic` used DC-local inventory position instead of true echelon IP (DC + downstream stores + in-transit). With 100 stores per DC, local IP ~1,000 vs echelon ROP ~4,000 → DCs always ordered → +290-360% inventory buildup.
- **Fix:** Replace `local_ip` with `echelon_matrix @ (actual_inventory + in_transit)` for correct echelon-wide inventory visibility.

#### Plant-Direct DC Deployment Fix

- **Root cause:** `_create_plant_shipments` routed 100% of production to RDCs only. Plant-direct DCs (15 mass retail + 3 club = 44% of volume) never received production output. Additionally, MRP signal filtered to RDC-sourced orders only, excluding plant-direct demand.
- **Fix A:** Renamed `rdc_demand_shares` → `deployment_shares`, expanded `_calculate_deployment_shares` to include DCs sourced from plants as deployment targets alongside RDCs. Production now flows proportionally to all deployment targets.
- **Fix B:** Widened MRP order filter to include `PLANT-`-sourced orders, giving MRP visibility into plant-direct demand signal.
- **Result:** Orders/demand ratio reduced, InvMean stable (not climbing), inventory turns 6.45x, A-item fill rate ~85%.

#### Format-Scaled Store Batch Size

- **Root cause:** Flat `store_batch_size=20` for all stores. Convenience stores (0.5x scale, ~1.6 cases/day) get 20-case batches → 12.5 days supply per order.
- **Fix:** Scale batch and min_qty by `format_scale_factor` (capped at 1.0, floor of 5 cases). Pharmacy→16, convenience→10, others unchanged.

#### Files Modified

- `src/prism_sim/generators/network.py` — Pharmacy DC gen (replaced sec D), Club DC gen (replaced sec C), plant-direct link routing, removed RDC→direct-store links
- `src/prism_sim/config/world_definition.json` — Added 2 RDCs to fixed_nodes, added `pharmacy_dcs: 4`, `club_dcs: 3`, updated `rdcs: 6`
- `src/prism_sim/agents/replenishment.py` — Loaded format_scale_factors, scaled store batch/min, fixed echelon IP calculation
- `src/prism_sim/simulation/orchestrator.py` — Renamed `rdc_demand_shares` → `deployment_shares`, expanded deployment target calculation to include plant-direct DCs, widened MRP signal filter
- `docs/llm_context.md` — Updated network topology and channel routing docs
- `CHANGELOG.md` — This entry
- `pyproject.toml` — 0.44.0 → 0.45.0

## [0.44.0] - 2026-02-01

### Channel Restructure — 7-Channel Model + B2M Rename

Replaces the 5-channel model (3 B2M + ECOMMERCE + DTC) with an industry-standard 7-channel model. Fixes volume distribution to match HPC benchmarks and eliminates dead code (DTC nodes never generated, volume_pct mismatch).

#### Channel Rename (B2M removal)

- `B2M_LARGE` → `MASS_RETAIL` — Walmart, Target (hypermarket format)
- `B2M_CLUB` → `CLUB` — Costco, Sam's Club
- `B2M_DISTRIBUTOR` → `DISTRIBUTOR` — Independent grocers, convenience stores

#### New Channels

- **GROCERY** — Kroger, Albertsons (10 DCs × 100 stores, supermarket format, ~20% volume)
- **PHARMACY** — CVS, Walgreens (375 stores direct to RDC, pharmacy format, ~6% volume)

#### DTC Implementation

- DTC nodes now actually generated (3 DTC_FC fulfillment centers)
- Added `DTC_FC` store format enum
- Added `DTC_FC` format scale factor (50.0, matching ECOM_FC)

#### Network Rebalance

| Channel | Old Nodes | New Nodes | Old Volume | New Volume | Industry HPC |
|---------|-----------|-----------|------------|------------|-------------|
| MASS_RETAIL | 2,020 (20 DCs × 100) | 1,515 (15 DCs × 100) | ~40% | ~30% | 30-35% |
| GROCERY | — | 1,010 (10 DCs × 100) | — | ~20% | 15-20% |
| CLUB | 30 | 47 | ~9% | ~14% | 12-16% |
| PHARMACY | — | 375 | — | ~6% | 5-8% |
| DISTRIBUTOR | 4,008 (8 DCs × 500) | 903 (3 DCs × 300) | ~40% | ~9% | 8-15% |
| ECOMMERCE | 10 | 18 | ~10% | ~18% | 17-19% |
| DTC | 0 (dead code) | 3 | 0% | ~3% | 2-3% |

- **Total weighted demand:** ~5,005 (vs ~4,950 — demand neutral)
- **Total nodes:** ~3,933 (vs ~4,130 — slightly fewer, faster sim)
- **Distributor stores:** 900 (vs 4,000) — higher per-SKU demand → better (s,S) behavior

#### Mass retail format fix

- Mass retail stores now use `HYPERMARKET` format (was incorrectly `SUPERMARKET`)
- Grocery stores correctly use `SUPERMARKET` format

#### Files Modified

- `src/prism_sim/network/core.py` — Renamed 3 enum values, added GROCERY + PHARMACY channels, added DTC_FC format
- `src/prism_sim/generators/network.py` — Renamed refs, added GROCERY/PHARMACY/DTC generation blocks, pharmacy→RDC linking
- `src/prism_sim/config/world_definition.json` — Renamed all B2M keys, rebalanced node counts, added GROCERY + PHARMACY + DTC configs
- `src/prism_sim/config/simulation_config.json` — Renamed all B2M keys, added DTC_FC scale factor, added GROCERY + PHARMACY channel configs
- `src/prism_sim/agents/replenishment.py` — Renamed 3 keys in DEFAULT_CHANNEL_POLICIES, added GROCERY + PHARMACY + DTC entries
- `src/prism_sim/simulation/demand.py` — Renamed 3 keys + 2 fallback strings, added GROCERY + PHARMACY entries
- `docs/llm_context.md` — Updated channel tables

## [0.43.0] - 2026-02-01

### Store Replenishment Fix, Production Throughput, Analysis Script Migration

Addresses the two root causes of A-item fill rate miss (92.4% vs 95-97% target): store order gap of 27.1 days (vs config order_cycle=3) and production/demand ratio of 72%.

#### Store Replenishment Frequency (Phase 1)

1. **`order_cycle_days`: 3 → 1** — Eliminates `hash(n_id) % cycle` stagger that blocked 2/3 of stores from evaluating (s,S) policy each day. Stores still only order when inventory < ROP.

2. **Channel profile target/ROP reduction** — All channels reduced to tighter (s,S) bands:
   - B2M_LARGE: 16/12 → 8/4, B2M_CLUB: 14/10 → 8/4, B2M_DISTRIBUTOR: 14/10 → 8/4
   - ECOMMERCE: 8/6 → 6/3, DTC: 7/5 → 5/2, default: 12/8 → 6/3

3. **Base replenishment params** — `target_days_supply` 12→6, `reorder_point_days` 8→3

4. **Store initialization** — `store_days_supply` 14→8 (faster initial cycle)

#### Production Throughput (Phase 2)

5. **`max_skus_per_plant_per_day`: 80 → 100** — With 60/25/15 ABC slot split and 4 plants: 240 A-slots/day → 310 A-items rotate every ~1.3 days (was 1.6d)

6. **DOS trigger thresholds lowered** — `trigger_dos_a` 10→7, `trigger_dos_b` 8→5, `trigger_dos_c` 6→4. More frequent, smaller batches reduce feast/famine oscillation.

#### Analysis Script Migration (Phase 3)

7. **4 scripts migrated from CSV to Parquet** with proper defaults and argparse:
   - `diagnose_service_level.py` — Parquet + PyArrow streaming for inventory + `--data-dir`
   - `check_plant_balance.py` — Full rewrite with Parquet, argparse, pathlib
   - `diagnose_slob.py` — Parquet + PyArrow streaming for inventory + `--data-dir`
   - `analyze_bullwhip.py` — Parquet + default path + removed unused inventory load

#### 50-Day Sanity Check Results
| Metric | v0.42.0 | v0.43.0 (50d) | Target |
|--------|---------|---------------|--------|
| A-item fill | 92.4% | 93.2% | 95-97% |
| B-item fill | 97.6% | 98.9% | >96% |
| C-item fill | 97.1% | 97.1% | >95% |
| OEE | 54.5% | 56.4% | >50% |
| Inventory Turns | 4.81x | 6.17x | 5.0-6.0x |
| Performance | 1.34 s/day | 1.16 s/day | <2.5 s/day |

#### Files Modified
- `src/prism_sim/config/simulation_config.json` — All config parameter changes
- `scripts/analysis/diagnose_service_level.py` — CSV→Parquet + streaming + default path
- `scripts/analysis/check_plant_balance.py` — CSV→Parquet + argparse + default path
- `scripts/analysis/diagnose_slob.py` — CSV→Parquet + streaming + default path
- `scripts/analysis/analyze_bullwhip.py` — CSV→Parquet + default path
- `CHANGELOG.md` — v0.43.0 entry
- `pyproject.toml` — Version bump

---

## [0.42.0] - 2026-02-01

### Fill Rate Optimization — A-Item Fill 92.3% → 95-97% Target

Addresses structural bottlenecks in MRP slot allocation, Phase 4 capacity clipping, and replenishment tuning identified by v0.41.0 diagnostics.

#### MRP Structural Fixes (Phase 1)

1. **Config-driven ABC slot percentages** (`a_slot_pct` 0.50→0.60, `b_slot_pct` 0.30→0.25, `c_slot_pct` 0.20→0.15)
   - 310 A-items / (80 SKUs × 0.60 × 4 plants) = 1.61d rotation (was 2.2d)
   - Each A-item produces ~37% more often

2. **ABC-aware Phase 4 capacity clipping** (replaces uniform clipping)
   - A-items (demand-matched, small batches) protected up to 65% of capacity
   - B/C campaign batches absorb clipping first — they have 97%+ fill headroom
   - New config: `a_capacity_share` = 0.65 in `mrp_thresholds`

3. **Raise A-item production buffer** (`a_production_buffer` 1.15→1.22)
   - Target = 14d × 1.22 = 17.1 DOS — enough headroom for 1.61d rotation
   - Safe with ABC-aware clipping preventing uniform Phase 4 cuts

4. **Increase SKU throughput** (`max_skus_per_plant_per_day` 70→80)
   - 48 A-slots/plant × 4 = 192 total → 1.61d rotation for all classes
   - OEE at 54% has headroom for more changeovers

#### Replenishment Tuning (Phase 2)

5. **Raise emergency DOS threshold** (2.0→3.0)
   - Gives stores 1-day earlier stockout warning for A-items with 1-day lead times

6. **Reduce demand floor weight** (0.80→0.65)
   - Lets 35% actual demand signal through (was 20%) for promo/peak responsiveness
   - Expected demand still provides floor to prevent death spiral

#### Bug Fixes (Phase 4)

7. **Fix DTC channel profile silently dropped** in `replenishment.py`
   - Config-only channels not in `DEFAULT_CHANNEL_POLICIES` were silently ignored
   - Now loads any channel present in config but not in defaults

8. **Clean up deprecated commands** in `docs/llm_context.md`
   - Removed references to `scripts/calibrate_config.py` and `scripts/run_standard_sim.py`
   - Updated canonical run command

#### Hardcode Fixes (semgrep audit)

9. **`replenishment.py`**: Emergency DOS demand floor used hardcoded `0.01` instead of config-driven `min_demand_floor` (0.1) — now consistent
10. **`replenishment.py`**: Default lead time `_lt_default_mu` was hardcoded `3.0` — now reads `lead_time_days` from config
11. **`quirks.py`**: `QuirkManager.seed` was hardcoded `42` — now reads `random_seed` from config

#### Expected Outcomes
| Metric | v0.41.0 | v0.42.0 Target |
|--------|---------|----------------|
| A-item fill | 92.3% | 95-97% |
| B/C fill rates | 97%+ | Stable |
| OEE | 54% | 55-60% |
| Performance | ~1.34s/day | ~1.3s/day |

#### Files Modified
- `src/prism_sim/simulation/mrp.py` — Config-driven slot %, ABC-aware Phase 4 clipping, `_get_abc_class` helper
- `src/prism_sim/config/simulation_config.json` — slot pcts, a_capacity_share, a_production_buffer, emergency_dos, demand_floor_weight, max_skus
- `src/prism_sim/agents/replenishment.py` — Fix DTC channel loading bug, demand floor consistency, lead time from config
- `src/prism_sim/simulation/quirks.py` — Read seed from config instead of hardcoding
- `docs/llm_context.md` — Updated parameters, removed deprecated commands, added known hardcodes section
- `CHANGELOG.md` — v0.42.0 entry
- `pyproject.toml` — Version bump

---

## [0.41.0] - 2026-02-01

### Close Remaining A-Item Fill Rate Gaps

Address two gaps identified by v0.40.0 diagnostics: replenishment too infrequent (mean order gap 27.8d vs config 5d) and production capacity shortfall (prod/demand ratio 0.85).

#### Changes

1. **Reduce store order cycle** (`order_cycle_days` 5 → 3)
   - With cycle=5, stores only evaluate replenishment every 5th day via `hash(n_id) % order_cycle_days` stagger
   - Cycle=3 means ~33% of stores order each day (vs 20%), catching low-velocity A-items sooner
   - Expected mean order gap reduction: 27.8d → 15-18d

2. **Add production line to PLANT-GA** (`num_lines` 5 → 6)
   - PLANT-GA is the most flexible plant (supports all 3 categories)
   - Adds ~315K cases/day (~7% capacity increase): `24h × 0.93 × 0.80 × 17,667 avg_rate`
   - Network capacity: ~4.55M → ~4.87M cases/day

3. **Reduce A-item production buffer** (`a_production_buffer` 1.3 → 1.15)
   - With net-requirement scheduling (v0.40.0), buffer sets target inventory
   - 1.3 → 18.2 DOS target → net-requirements routinely exceed capacity → Phase 4 clips uniformly
   - 1.15 → 16.1 DOS target → net-requirements closer to capacity → less clipping, more B/C headroom

4. **Raise Phase 4 capacity cap** (0.95 → 0.98)
   - TransformEngine already enforces physical line-level capacity with changeover penalties
   - Excess orders carry to next day as IN_PROGRESS — no physics violation
   - Lets ~3% more production through while retaining 2% safety margin

#### Expected Outcomes
| Metric | v0.40.0 | v0.41.0 Target |
|--------|---------|----------------|
| A-item fill | 92.8% | 95-97% |
| Prod/demand ratio | 0.85 | 0.93-0.97 |
| B/C fill rates | 97-98% | Stable |
| Mean A-item order gap | 27.8d | 15-18d |
| OEE | 59.4% | 55-65% |

#### Files Modified
- `src/prism_sim/config/simulation_config.json` — `order_cycle_days`, `a_production_buffer`, PLANT-GA `num_lines`
- `src/prism_sim/simulation/mrp.py` — Phase 4 capacity cap

---

## [0.40.0] - 2026-02-01

### Net-Requirement Production Scheduling for A-Items (MPS-Style)

Replace reactive trigger-based production for A-items with schedule-based net-requirement planning. B/C items keep their existing campaign triggers unchanged.

#### Problem
Trigger-based scheduling creates feast/famine oscillation: items produce nothing until DOS < threshold, then produce a huge batch. Plants alternate between idle and overloaded. The v0.40.0 attempt of raising `trigger_dos_a` flooded the candidate pool and shrank all batches via the capacity cap, dropping A-item fill from 90.6% to 72.3%.

#### Solution
A-items now use net-requirement (MPS-style) batch sizing:
```python
target_inventory = demand_rate * horizon * buffer  # 14d × 1.3 = 18.2 DOS target
net_requirement  = target_inventory - inventory_position
batch_qty        = max(net_requirement, 0)
```
- Each A-item gets a small, demand-matched batch every ~2.2 days (310 items / 140 A-slots)
- Items at/above target produce nothing (self-regulating)
- No trigger gate = no idle capacity = no feast/famine
- Phase 2-4 (sorting, slot reservation, capacity cap) unchanged

#### Results (365-day simulation)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| A-item fill | 90.6% | 92.8% | +2.2pp |
| OEE | 50.7% | 59.4% | +8.7pp |
| Prod/demand ratio | 0.71 | 0.85 | +0.14 |
| B-item fill | ~98.7% | 97.8% | Stable |
| C-item fill | ~98.5% | 97.0% | Stable |

#### Files Modified
- `src/prism_sim/simulation/mrp.py` — Phase 1 branched: A-items use net-requirement, B/C keep triggers

---

## [0.39.8] - 2026-01-31

### Inventory Serialization Optimization (Parquet + DictionaryArray + Background Thread)

Streaming inventory export was the dominant bottleneck for logged runs (~220s overhead on a 365-day run). This release eliminates that overhead with a threaded Parquet writer and removes the hardcoded weekly inventory gate.

#### 1. Remove Orchestrator `day % 7` Hardcode (`orchestrator.py`)
- **Before:** Inventory logging was gated by `if day % 7 == 0:` regardless of user config
- **After:** Inventory logging frequency is controlled solely by `--inventory-sample-rate` (default 1 = daily)
- **Impact:** Users get true daily snapshots (~150M+ rows for 50 days) when requested

#### 2. Parquet Inventory Schema Optimization (`writer.py`)
- `node_id` and `product_id` columns now use `pa.dictionary(pa.int32(), pa.string())` instead of plain `pa.string()`
- `perceived_inventory` and `actual_inventory` narrowed from `float64` to `float32` (matches state tensor dtype)
- **Impact:** ~50% smaller Parquet files, faster encoding

#### 3. `ThreadedParquetWriter` Class (`writer.py`)
- New class that offloads Parquet encoding/compression/IO to a background thread
- **Main thread:** numpy ops (mask, nonzero, fancy-index) → puts raw arrays on bounded `queue.Queue(maxsize=4)`
- **Background thread:** builds `pa.DictionaryArray` columns from pre-cached dictionaries → writes row groups (GIL released during C++ Parquet IO)
- Pre-built `pa.Array` dictionaries for node/product IDs (built once on first call, ~4,500 + ~500 entries)
- Error propagation: thread exceptions surfaced via `submit()` and `close()`
- Only activates for `--streaming --format parquet`; CSV and buffered paths unchanged

#### 4. Added `pyarrow` as Project Dependency
- `pyarrow >= 23.0.0` added to `pyproject.toml` (was previously optional)

#### Results (50-day simulation, `--streaming --format parquet --inventory-sample-rate 1`)
| Metric | Value |
|--------|-------|
| Inventory rows written | 150,960,760 |
| Runtime (50-day) | 47s |
| Schema | dictionary-encoded strings, float32 values |

#### Files Modified
- `src/prism_sim/simulation/writer.py` — Schema fix, DictionaryArray caching, `ThreadedParquetWriter`, `log_inventory` rewrite
- `src/prism_sim/simulation/orchestrator.py` — Remove `if day % 7 == 0:` gate
- `pyproject.toml` — Add `pyarrow` dependency, bump version

---

## [0.39.7] - 2026-01-31

### OEE/TEEP Overhaul & Lint Compliance

#### 1. OEE Fix: Eliminate Double-Counted Efficiency (`transform.py`)
- **Bug:** `efficiency_factor` was baked into line capacity hours AND applied again as the OEE Performance component, double-counting speed losses
- **Fix:** Capacity hours now use `hours_per_day * (1 - downtime)` only; efficiency is captured solely via the OEE Performance component
- **Impact:** OEE numbers now reflect the standard SMRP/Vorne decomposition

#### 2. OEE Denominator: Planned Production Time (`transform.py`)
- **Before:** OEE denominator was raw calendar time (24h × all lines), penalizing plants with no demand
- **After:** OEE denominator is planned production time (active lines only); idle lines with no demand are a Schedule Loss excluded from OEE per industry standards
- **Impact:** OEE reflects true manufacturing effectiveness, not utilization

#### 3. TEEP Metric (`transform.py`, `orchestrator.py`, `monitor.py`)
- Added Total Effective Equipment Performance (TEEP = OEE × Utilization)
- TEEP uses raw calendar time (24h × all lines) as denominator, revealing total hidden capacity
- Tracked via `RealismMonitor.teep_tracker`, reported in Triangle Report

#### 4. Config Rebalancing (`simulation_config.json`)
- Differentiated category demand profiles: ORAL_CARE 8.0, PERSONAL_WASH 5.5, HOME_CARE 2.0 (was 3.2 uniform)
- Rebalanced plant line counts to match category throughput needs (OH:4, TX:2, CA:3, GA:5)
- Added ORAL_CARE to PLANT-GA supported categories for overflow capacity

#### 5. Full Lint Compliance
- **Ruff:** Fixed all 169 errors (E501, E701, RUF002/3, PLR2004, B023, RUF012); adjusted pylint limits for DES simulation complexity
- **Mypy:** Fixed all 5 errors (no-any-return, no-redef, unused-ignore, var-annotated)
- **Semgrep:** Audited all 237 hardcoded-values findings — confirmed zero config violations
- Removed 6 stale `noqa` directives; added `PLC0415`, `PLW0603` to ignore list
- All 33 source files now pass `ruff check`, `mypy --strict`, and `semgrep`

## [0.39.6] - 2026-01-25

### Performance Optimization Release

Major performance improvements reducing 30-day simulation runtime from 388s to 43s (~89% faster, 9x speedup) and memory usage from 6.3 GB to ~3 GB.

#### 1. Batch Inventory Updates (`allocation.py`)
- **Before:** 500K+ individual `update_inventory()` calls per day
- **After:** Single vectorized tensor operation per source node
- **Impact:** Eliminated dict lookups and per-product function call overhead

#### 2. Pre-computed Lookup Caches (`replenishment.py`)
- **Before:** 49M+ `dict.get()` calls per day in `_create_order_objects()`
- **After:** Direct numpy array indexing with pre-computed caches
- **Impact:** O(1) array access replaces O(1) amortized dict lookups (lower constant)
- Added `_build_lookup_caches()` method for product_id, category, and node_id arrays

#### 3. Sparse Lead Time Storage (`replenishment.py`)
- **Before:** Dense [6126 x 6126 x 20] tensor = 2.86 GB
- **After:** Sparse dict with deque circular buffers = ~2 MB
- **Impact:** ~3.5 GB memory savings (99.9% reduction in lead time tracking)
- Only ~6000 links exist in network; 99.99% of dense tensor was zeros
- Updated `orchestrator.py` warm-start and `snapshot.py` to use sparse API

#### 4. Scatter-Add for Inflow Recording (`replenishment.py`)
- **Before:** Nested loops with individual element accumulation
- **After:** `np.add.at()` scatter operation for sparse accumulation
- **Impact:** Replaces N individual updates with single vectorized operation

#### Results (30-day simulation, --no-logging)
| Metric | v0.39.5 | v0.39.6 | Change |
|--------|---------|---------|--------|
| Runtime | 388s | 43s | -89% |
| Peak Memory | 6.3 GB | ~3 GB | -52% |
| Per-day Step | ~13s | ~1.4s | -89% |

All optimizations are behavior-preserving refactors - Triangle Report metrics unchanged.

## [0.39.5] - 2026-01-25

### SLOB & C-Item Service Root Cause Fix

Fixed critical SLOB calculation bug and C-item starvation parameters. SLOB dropped from 42.1% to 18.7% (-23.4 pts).

#### Problem Statement
After v0.39.4 config fixes, 365-day simulation showed:
- **SLOB: 42.1%** (target <15%) - exploded over time
- **Service Level: 75.2%** (target 97%)
- **C-Item Service: 45.5%** - catastrophically low

Investigation found SLOB calculation bug where age accumulated regardless of inventory turnover.

#### 1. FIFO Age Approximation (`state.py`)
- **Bug:** Age incremented daily regardless of consumption
  - Day 1: Receive 100 cases (age = 0)
  - Day 60: Sell down to 20 cases, but age = 60 days (FALSE!)
  - SLOB flagged despite inventory turning normally
- **Fix:** Reduce age proportionally when inventory is consumed
  - Approximates FIFO: selling removes "oldest" units along with their age
  - Formula: `new_age = old_age × fraction_remaining`
  - Example: 100 units at age 50, sell 50 → age becomes 25
- **Impact:** SLOB dropped from 42.1% to 18.7%

#### 2. C-Item Z-Score Boost (`simulation_config.json`)
- **Bug:** C-item z-score 1.0 (84% service) caused systematic starvation
- **Fix:** Increased to 1.65 (95% service target, match B-items)
- **Impact:** C-items now get adequate safety stock

#### 3. Unmet Demand Decay Slowdown (`replenishment.py`)
- **Bug:** 15%/day decay collapsed C-item signal before replenishment cycle
- **Fix:** Reduced decay from 0.85 to 0.95 (5%/day)
- **Impact:** Signal persists ~60 days vs ~7 days before

#### 4. Production Trigger Tuning (`simulation_config.json`)
- **Fix:** Lower DOS triggers (A: 15→10, B: 12→8, C: 10→6)
- **Fix:** Increase SKU slots per plant (50→70)
- **Impact:** More proactive production, better C-item coverage

#### Results (365-day simulation)
| Metric | v0.39.4 | v0.39.5 | Change |
|--------|---------|---------|--------|
| **SLOB** | 42.1% | **18.7%** | -23.4 pts |
| Service Level | 75.2% | 74.5% | -0.7 pts |
| C-Item Service | 45.5% | 56.5% | +11.0 pts |
| Inventory Turns | - | 8.61x | Good |

#### Files Modified
- `src/prism_sim/simulation/state.py` - FIFO age approximation
- `src/prism_sim/agents/replenishment.py` - Unmet demand decay
- `src/prism_sim/config/simulation_config.json` - Z-scores, triggers, SKU slots

---

## [0.39.4] - 2026-01-24

### Demand Signal & Inventory Initialization Fix

Fixed critical configuration mismatches causing low service levels despite v0.39.3 demand floor fixes.

#### Problem Statement
365-day simulation showed 84% service level (target: 97%) with production at only 15% of demand. Investigation revealed initialization and Zipf alpha misconfigurations.

#### 1. Inventory Initialization Aligned to FMCG Benchmarks (`simulation_config.json`)
- **Bug:** `store_days_supply: 5.0` caused immediate stockouts, triggering death spiral
- **Fix:** Aligned with `baseline_reference` values:
  - `store_days_supply`: 5.0 → 14.0 (P&G targets 66 DOS, Colgate 89 DOS)
  - `rdc_days_supply`: 14.0 → 21.0
  - `customer_dc_days_supply`: 7.0 → 14.0
- **Impact:** System starts with adequate safety stock to prevent cold-start death spiral

#### 2. Zipf Alpha Aligned to Benchmark (`simulation_config.json`)
- **Bug:** `sku_popularity_alpha: 0.5` created nearly uniform demand across SKUs
- **Fix:** Aligned with `benchmark_manifest.json`: 0.5 → 1.05
- **Impact:** Proper Pareto 80/20 distribution - top 20% SKUs now drive 80% of volume
- **Industry Reality:** FMCG Zipf alpha typically 0.8-1.2

#### 3. Inventory Turns Guard (`orchestrator.py`)
- **Bug:** Division by near-zero inventory produced 59 million x turns (physically impossible)
- **Fix:** Added minimum threshold (100 cases) and cap at 50x
- **Impact:** Inventory turns now report realistic values (target: 5-6x)

#### 4. MRP Diagnostic Logging (`mrp.py`)
- Added debug logging to verify demand floor execution
- Logs expected vs actual vs blended demand signals for first 5 SKUs

#### 5. Memory Usage Fix for Long Runs (`simulation_config.json`, `orchestrator.py`)
- **Bug:** 365-day runs with buffered logging used 20GB+ RAM
- **Root Cause:** `inventory_sample_rate: 1` (daily) created 8GB+ of inventory history
- **Fix:** Changed default to 7 (weekly) - reduces memory by ~7x
- **Added:** Warning message when running >100 days with buffered logging
- **Recommendation:** Use `--streaming` or `--no-logging` for long runs

#### Results (50-day simulation after 90-day burn-in)
| Metric | v0.39.3 | v0.39.4 | Target |
|--------|---------|---------|--------|
| SLOB | 0.1% | **4.8%** | 5-10% |
| Inventory Turns | 59M (bug) | **6.72x** | 5-6x |
| Service Level | 84% | **80.9%** | 95%+ |
| OEE | - | **33.1%** | 55%+ |

*Note: Service level expected to improve with 365-day run as system stabilizes.*

#### Files Modified
- `src/prism_sim/config/simulation_config.json` - Inventory init, Zipf alpha
- `src/prism_sim/simulation/orchestrator.py` - Inventory turns guard
- `src/prism_sim/simulation/mrp.py` - Diagnostic logging

---

## [0.39.3] - 2026-01-24

### Production Under-Run & Demand Signal Fix

Fixed the over-correction from v0.39.2 that caused production to under-run demand by ~28%. The v0.39.2 SLOB fix swung from over-production (129%) to under-production (~72%).

#### Problem Statement
After v0.39.2, SLOB improved (71% → 28.3%), but we had the opposite problem:
- Production/Demand ratio dropped to ~72% (target: 100-105%)
- Service remained at 89% (target: 98%)
- Death spiral: actual sales → lower MRP signal → less production → lower sales

#### 1. Demand Signal Floor (`mrp.py`)
- **Bug:** Using `pos_demand_vec` (actual sales) directly caused death spiral when service < 100%
- **Fix:** Use expected demand as FLOOR: `demand_for_dos = max(expected, weighted_blend)`
- **New Config:** `demand_floor_weight` (default 0.8 = 80% expected, 20% actual)
- **Industry Reality:** P&G, Colgate, Unilever use forecast as floor, actual sales only for upward adjustments

#### 2. Remove Cascading C-Item Penalties (`mrp.py`)
- **Bug:** `c_production_factor` (0.5) × DOS throttle (0.3) = 0.15 (85% cut!)
- **Fix:** C-items use longer horizons (21 days), no penalty factor, no DOS throttling
- **New Config:** `production_horizon_days_c` increased 5 → 21 (3-week campaigns)
- **Industry Reality:** C-items run monthly campaigns vs A-items weekly

#### 3. Emergency Replenishment Bypass (`replenishment.py`)
- **Bug:** Stores with zero inventory still waited for scheduled order day
- **Fix:** Bypass stagger when any product DOS < `emergency_dos_threshold` (default 2.0)
- **Industry Reality:** Walmart, Target use emergency orders for critical stockouts

#### 4. Track Stockout-Based Unmet Demand (`orchestrator.py`)
- **Bug:** Unmet demand only tracked from allocation failures, not store stockouts
- **Fix:** Record `daily_demand - actual_sales` to state for MRP signal calibration
- **Impact:** Lost sales at shelf now flow upstream to drive production

#### 5. Dead Code Removal (`mrp.py`)
- Removed unused `c_demand_factor` (was defined but never used in calculations)

#### 6. Test Suite Archived
- Moved `tests/` to `archive/tests/` - use full simulations for integration testing
- Updated `CLAUDE.md` and `llm_context.md` to reflect new testing approach

#### Results (50-day simulation after 90-day burn-in)
| Metric | v0.39.2 | v0.39.3 | Target |
|--------|---------|---------|--------|
| SLOB | 28.3% | **0.1%** | 5-10% |
| Inventory Turns | 3.71x | **4.95x** | 5-6x |
| Service Level | 89% | **90.9%** | 98% |
| A-Item Service | - | **92.8%** | 95%+ |
| B-Item Service | - | **83.9%** | 90%+ |
| C-Item Service | - | **82.6%** | 85%+ |

#### Files Modified
- `src/prism_sim/simulation/mrp.py` - Demand floor, remove cascading penalties
- `src/prism_sim/agents/replenishment.py` - Emergency bypass
- `src/prism_sim/simulation/orchestrator.py` - Stockout unmet demand
- `src/prism_sim/config/simulation_config.json` - Updated parameters
- `CLAUDE.md` - Updated commands, removed test references
- `docs/llm_context.md` - Updated for v0.39.3 changes

---

## [0.39.2] - 2026-01-23

### SLOB & Over-Production Fix

Major fix to reduce SLOB (Slow-moving and Obsolete) inventory from 71% to 28% by addressing over-production and implementing industry-standard age-based SLOB calculation.

#### Problem Statement
After v0.39.1, SLOB was at 71% (industry target: 5-10%). Root cause: production exceeded consumption by 29%, causing inventory accumulation. Daily production was 4.1M cases vs 3.2M actual consumption.

#### 1. Production-Consumption Balance Fix (`mrp.py`)
- **Batch Sizing Fix:** Changed batch sizing to use actual POS demand (`pos_demand_vec`) instead of recycled 14-day forecast totals. Previous bug: `network_forecast` already contained 14-day totals, then was re-multiplied by horizon.
- **Consumption Tracking:** Added `record_consumption()` and `get_actual_daily_demand()` methods to track actual sales vs expected demand. MRP now has feedback on real consumption.
- **Reduced Buffers:** `a_production_buffer` 1.5→1.2, `c_production_factor` 0.4→0.5.

#### 2. Age-Based SLOB Calculation (`state.py`, `orchestrator.py`)
- **Industry Standard:** SLOB now uses inventory AGE (how long sitting) instead of DOS (how long it COULD last). A fresh batch with 90 days supply is NOT obsolete—but inventory sitting for 90 days IS.
- **New State Tracking:** Added `inventory_age` tensor and methods: `age_inventory()`, `receive_inventory()`, `receive_inventory_batch()`, `get_weighted_age_by_product()`.
- **Weighted Average Age:** When fresh inventory arrives, it blends with existing inventory using weighted average age calculation.
- **Age Thresholds:** Config changed from DOS to age thresholds: A=60d, B=90d, C=120d.

#### 3. Daily Loop Changes (`orchestrator.py`)
- Added `state.age_inventory(1)` at start of each day.
- Added `mrp_engine.record_consumption(actual_sales)` after inventory consumption.
- Updated `_process_arrivals()` to use age-aware `receive_inventory_batch()`.

#### Results (365-day simulation)
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| SLOB | 71% | **28.3%** | 15-25% |
| Service Level | ~89% | **89.01%** | 87-90% |
| Inventory Turns | 3.3x | **3.71x** | 5-6x |

#### Files Modified
- `src/prism_sim/simulation/mrp.py` - Batch sizing fix, consumption tracking
- `src/prism_sim/simulation/state.py` - Inventory age tracking
- `src/prism_sim/simulation/orchestrator.py` - Age methods, SLOB calculation
- `src/prism_sim/config/simulation_config.json` - Buffer and threshold adjustments

---

## [0.39.0] - 2026-01-19

### ETL Layer Enhancement - Expanded ERP Schema

Extended the ETL pipeline to export additional tables, closing the gap with enterprise ERP data models.

#### 1. Simulation Writer Enhancements
- **Emissions Tracking:** Added `emissions_kg` column to `shipments.csv` output (was calculated but not logged).
- **Batch Genealogy:** New `batch_ingredients.csv` logs ingredient consumption per production batch (enables full BOM traceability).

#### 2. Bug Fixes in Export Script
- **Line Number Fix:** `order_lines` and `shipment_lines` now have incrementing `line_number` per parent (was hardcoded to 1, violating PK constraint).
- **Unit Conversion Fix:** Uses product-specific `weight_kg` from `products.csv` instead of hardcoded 10.0 kg/case.

#### 3. New ERP Tables (SOURCE Domain)
- `purchase_orders` / `purchase_order_lines` - Ingredient procurement orders (filtered from `PO-ING-*` orders).
- `goods_receipts` / `goods_receipt_lines` - Inbound deliveries to plants (derived from plant-bound shipments).

#### 4. New ERP Tables (TRANSFORM Domain)
- `batch_ingredients` - Ingredient consumption per batch (enables yield analysis and traceability).

#### 5. New ERP Tables (ESG Domain)
- `shipment_emissions` - CO2 emissions per shipment (Scope 3 transport emissions).

#### Files Modified
- `src/prism_sim/simulation/writer.py` - Added `emissions_kg` to shipments, added `log_batch_ingredients()` method.
- `src/prism_sim/simulation/orchestrator.py` - Calls `log_batch_ingredients()` after transform step.
- `scripts/erp_schema.sql` - Added 7 new table definitions.
- `scripts/export_erp_format.py` - Fixed bugs, added 4 new `process_*()` functions.

---

## [0.38.0] - 2026-01-19

### ERP Data Model & Returns Logistics

Implemented the final "Tier 3" capabilities to bridge the gap between simulation physics and enterprise ERP data models.

#### 1. Returns Simulation (Reverse Logistics)
- **Physics:** `LogisticsEngine` now generates returns (RMAs) based on a 5% random probability from arriving shipments, simulating damage or rejection.
- **Data Model:** Added `Return` and `ReturnLine` primitives to `core.py`.
- **Export:** Returns are logged to `returns.csv` and transformed into `returns` and `return_lines` tables.

#### 2. S&OP Export (Consensus Forecast)
- **Orchestrator:** Now generates a daily 14-day deterministic forecast using `POSEngine`'s demand sensing logic.
- **Export:** Logged to `forecasts.csv` and transformed into `demand_forecasts` table (Versioned by generation date).
- **Goal:** Allows calculation of "Forecast Accuracy" and "Bias" metrics in external BI tools.

#### 3. ERP Data Transformation (ETL)
- **New Script:** `scripts/export_erp_format.py` transforms flat simulation CSVs into a normalized 3rd Normal Form (3NF) PostgreSQL-ready schema.
- **Features:**
  - Splitting Orders/Shipments into Header/Line tables
  - Integer ID mapping for all entities
  - Generating static reference data (Carriers, Certifications, Costs)
  - Extracting "hidden" config data (Production Lines, Channels, Promotions)

---

## [0.37.0] - 2026-01-19

### Procurement & Work Order Lifecycle

Upgraded the manufacturing and sourcing engines to support full "Plan-to-Produce" and "Source-to-Pay" data lifecycles.

#### 1. Realistic Procurement
- **Removed Cheat:** Reduced `plant_ingredient_buffer` from 5M (infinite) to 200k (tight), forcing the system to rely on actual replenishment.
- **Logic:** `Orchestrator` now initializes ingredient inventory based on demand velocity, ensuring a stable start without infinite safety stock.
- **Result:** `MRPEngine` successfully generates `PO-ING-` orders, which flow through allocation, logistics, and receipt logic.

#### 2. Work Order Tracking
- **Lifecycle:** `ProductionOrder` objects are now logged at creation (PLANNED status) and linked to execution `Batch` objects.
- **Data Model:** Export script generates `work_orders` table linking `wo_id` to `batches`.
- **Analytics:** Enables "Schedule Adherence" (Planned vs Actual Date) and "Yield" (Planned vs Actual Qty) analysis.

---

## [0.36.3] - 2026-01-19

### Capacity Planning - Physics-Based Efficiency Decomposition

**Problem:** The `--derive-lines` calibration derived **11 lines** but empirically **22 lines** were needed to achieve 87% service level. The 2x gap stemmed from overly optimistic assumptions about campaign batching efficiency.

**Root Cause:** The single `campaign_batch_efficiency: 0.66` factor assumed 66% of theoretical capacity is utilized. In reality, DOS cycling idle time and demand variability reduce actual utilization to ~40-45%.

**Solution:** Replaced single efficiency factor with physics-based decomposition:

1. **DOS Cycling Factor (~50%)** - Lines sit idle when DOS > trigger. The production cycle is:
   - Produce when DOS < trigger (e.g., 31 days)
   - After producing 14-day horizon, DOS jumps to ~45
   - Wait ~31 days for DOS to drop below trigger again
   - Formula: `dos_coverage = production_horizon / (production_horizon + weighted_avg_trigger) × stagger_benefit`

2. **Variability Buffer (~1.25x)** - Point estimate of demand doesn't account for ±2σ swings:
   - Formula: `buffer = 1 / (1 - safety_z × combined_cv)`
   - With seasonality 12% + noise 10%, CV≈0.16, z=1.28: buffer = 1.25

**Combined Effect:** Old: 66% efficiency → 11 lines. New: 45% efficiency × 1.25x buffer → ~20-24 lines.

**Files Modified:**
- `scripts/calibrate_config.py`:
  - Added `calculate_campaign_efficiency()` - DOS cycling efficiency calculation
  - Added `calculate_variability_buffer()` - demand variability reserve capacity
  - Updated `derive_num_lines_from_oee()` to use physics-based decomposition
  - Added efficiency decomposition to output report
- `src/prism_sim/config/simulation_config.json`:
  - Added `variability_safety_z: 1.28` parameter
  - Removed static `campaign_batch_efficiency` (now calculated dynamically)

---

## [0.36.2] - 2026-01-18

### System Stabilization & Load Balancing

Resolved the critical production starvation and plant load imbalance issues, achieving **87.7% Service Level** (up from 56%).

#### 1. Starvation Fix: Shuffle Tie-Breaker
- **Problem:** "Campaign Batching" logic sorted production candidates by `(ABC, DOS)`. When capacity was tight, C-items with `DOS=0` were always at the bottom of the list and never produced.
- **Fix:** Implemented **Critical Ratio Sorting** (`DOS/Trigger`) and added a **Random Shuffle** tie-breaker. This ensures that starving items rotate into the production schedule fairly.

#### 2. Load Balancing: Ohio Plant Activation
- **Problem:** `PLANT-OH` was idle (4% load) while `PLANT-GA` and `PLANT-CA` were capped (33% load). `PLANT-OH` was restricted to `HOME_CARE` only.
- **Fix:** Updated `simulation_config.json` to allow `PLANT-OH` to produce `PERSONAL_WASH`. This offloaded ~15% of network volume to Ohio, balancing utilization.

#### 3. Capacity Optimization
- **Lines:** Increased to **19 Lines** (derived from 40% OEE target) to handle changeover friction.
- **Horizon:** Set `production_horizon_days` to **14** to prioritize stability over inventory turns.
- **Result:** System successfully meets 4M case/day demand with stable 46% OEE.

---

## [0.36.1] - 2026-01-18

### Critical Ratio Sorting (Phase 1)

Initial attempt to fix C-item starvation by replacing ABC hierarchy with Critical Ratio (`DOS/Trigger`) sorting. This improved fairness but revealed the secondary plant imbalance issue.

---

## [0.36.0] - 2026-01-18

### P&G Scale Recalibration & Demand Sensing

This release implements a structural pivot to realistic North American FMCG scale and introduces proactive demand sensing to fix 365-day inventory drift.

#### 1. P&G Scale Alignment (Downsizing)
- **Volume:** Reduced network demand from ~21M cases/day to **~4M cases/day**. This aligns with P&G North America (~$42B revenue/year).
- **Capacity:** Reduced production lines from 176 (44/plant) to **32 (8/plant)**.
- **Inventory:** Tightened inventory buffers (Stores 14d, RDCs 21d) to reflect the new stable flow.
- **Rationale:** The previous 21M scale was a "Hyper-Enterprise" that amplified computational friction. Downsizing makes physics traceable and realistic.

#### 2. Proactive Demand Sensing
- **MRP & Replenishment:** Agents now query `POSEngine` for deterministic future demand (forecasts) rather than relying solely on reactive history.
- **Drift Fix:** This allows the system to build inventory *ahead* of promotional peaks and seasonal shifts, preventing the "slow bleed" of A-item inventory observed in previous versions.
- **Implementation:** Added `get_deterministic_forecast()` to POSEngine and integrated it into `MRPEngine` batch sizing and `MinMaxReplenisher` demand smoothing.

#### 3. Codified Workflow
- **New Runner:** Added `scripts/run_standard_sim.py` which enforces the standard pipeline:
  - `python scripts/run_standard_sim.py --rebuild --recalibrate --days 365`
- **Automation:** Handles config validation, world generation, calibration, and checkpointing in a single command.

#### 4. Physics Fixes
- **Ingredient Scaling:** Moved `plant_ingredient_buffer` to config to scale with network demand (prevents 8B unit memory usage).
- **Unbound Local Fix:** Resolved variable scope issue in MRP campaign batching logic.

---

## [0.35.4] - 2026-01-16

### Capacity Planning - Campaign Batching OEE Calibration

**Problem:** Calibration targeted 60% OEE but simulation consistently achieved only 46% OEE regardless of line count or trigger settings.

**Root Cause:** Campaign batching creates natural idle time - products only produce when DOS drops below trigger, creating cycling patterns. MRP scales production to capacity, so reducing lines doesn't increase utilization - production scales down proportionally.

**Analysis:**
- With campaign batching (7-day horizon, DOS triggers 31/27/22), actual production is ~66% of capacity
- This results in ~57% availability, yielding OEE ≈ 57% × 82% × 98.5% = 46%
- The 46-50% OEE is a fundamental characteristic of campaign batching, not a bug

**Fixes:**
1. **Explicit changeover calculation** - Replaced simple overhead factor with batch-count-based formula:
   - `changeover_hours = (num_products / production_horizon_days) × avg_changeover_time`
2. **Capacity-constrained production** - Formula now uses `min(demand, capacity)` when capacity-bound
3. **Campaign batch efficiency** - Added config parameter (0.66) to account for DOS cycling idle time
4. **Warning system** - Calibration now warns when target OEE exceeds campaign batching limit (~50%)

**Files Modified:**
- `scripts/calibrate_config.py` — Major rewrite of `derive_num_lines_from_oee()`:
  - Explicit changeover calculation based on batches per day
  - Capacity-constrained demand handling
  - Campaign batch efficiency factor
  - Warning for unrealistic OEE targets
- `src/prism_sim/config/simulation_config.json`:
  - Added `campaign_batch_efficiency: 0.66` parameter
  - Updated capacity planning comments

**Calibration Output Example:**
```
Campaign Batching Parameters:
  Finished products:    500
  Production horizon:   7 days
  Batches per day:      71.4
  Campaign batch eff:   66%
  Effective production: 7,540,911 cases/day

Line Count:
  Derived total lines:  33
  Estimated OEE:        49.9%
  Realistic OEE:        49.9% (campaign batching limit)

*** WARNING: Target OEE 60% exceeds campaign batching limit ***
```

**Usage:**
```bash
# Use realistic 50% target for campaign batching
poetry run python scripts/calibrate_config.py --target-oee 0.50 --derive-lines --apply
```

**Key Insight:** To achieve >50% OEE, need to either:
1. Reduce `production_horizon_days` (smaller, more frequent batches)
2. Increase DOS triggers (more products always need production)
3. Switch from campaign batching to continuous production

---

## [0.35.1] - 2026-01-16

### Streaming Mode Bug Fix (Memory Explosion)

**Problem:** 365-day simulations with logging enabled caused memory explosion (30GB+), crashing the simulation.

**Root Cause:** `run_simulation.py` passed `args.streaming` (defaults to `False` with `store_true`) instead of `None` to the Orchestrator. This bypassed the config fallback (`writer.streaming: true`), forcing buffered mode which accumulates all orders/shipments/inventory in memory.

```python
# Before (broken):
streaming=args.streaming if enable_logging else False,  # False bypasses config

# After (fixed):
streaming=args.streaming if args.streaming else None,  # None uses config default
```

**Impact:**
- With fix: 365-day simulation completes in ~9 minutes, ~4GB memory
- Without fix: Memory grows to 30GB+ and crashes

**Files Modified:**
- `run_simulation.py` — Fixed streaming parameter passthrough

**Documentation:**
- `docs/llm_context.md` — Added "Setup & Workflow (Order of Operations)" section

---

## [0.35.0] - 2026-01-15

### Truck Fill Rate & OEE Metrics Fixes

Major metrics accuracy improvements addressing three issues discovered during deep analysis:

#### Issue 1: Plant Shipment Weight Bug (Fixed)

**Problem:** Plant→RDC shipments had `total_weight_kg = 0` because `_create_plant_shipments()` didn't calculate weight/volume.

**Fix:** Added weight/volume calculation using product attributes (`orchestrator.py:1407-1411`).

**Impact:** Plant shipments now correctly contribute to fill rate metrics and CSV exports.

#### Issue 2: FTL Metric Miscategorization (Fixed)

**Problem:** Supplier→Plant inbound shipments (57% of "FTL" volume, 8-20% fill) were mixed with outbound finished goods (97% fill), dragging the combined FTL metric down to 26%.

**Fix:** Separated inbound vs outbound FTL metrics:
- Added `inbound_fill_tracker` and `outbound_ftl_tracker` to `RealismMonitor`
- Updated `_record_daily_metrics()` to classify shipments by source node type
- Triangle Report now shows **Outbound FTL** as the primary truck fill metric

**New Triangle Report format:**
```
3. COST (Truck Fill Rate):      97.2%    <-- Outbound FTL (finished goods)
   - Inbound Fill (raw mat):    4.6%     <-- Supplier→Plant (visible separately)
```

#### Issue 3: OEE Formula Flaws (Fixed)

**Problem:** OEE showed 94.7% (unrealistic) due to:
1. Availability = `run/(run+changeover)` ≈ 98% — ignored idle capacity
2. Performance = 1.0 (hardcoded) — ignored `efficiency_factor` (0.78-0.88)
3. Idle lines excluded from calculation

**Fix:** Implemented standard OEE formula (`transform.py:379-414`):
- Availability = `(run + changeover) / total_scheduled` (includes utilization)
- Performance = plant's `efficiency_factor` from config
- Stores efficiency factors during `_initialize_plant_states()`

**Expected OEE:** 55-70% (realistic) vs 94.7% (inflated)

#### Additional Fix: Metrics Start Day

**Problem:** `--no-checkpoint` runs set `_metrics_start_day = 91`, causing no metrics to be recorded for short runs.

**Fix:** When `auto_checkpoint=False` and no warm-start, set `_metrics_start_day = 1`.

#### Files Modified
- `src/prism_sim/simulation/orchestrator.py` — Plant weight, FTL categorization, metrics start day
- `src/prism_sim/simulation/transform.py` — OEE formula
- `src/prism_sim/simulation/monitor.py` — Inbound/outbound trackers
- `tests/test_manufacturing_lines.py` — Updated OEE test expectations

## [0.34.0] - 2026-01-14

### Replenishment Logic Refactor & Hardcode Fixes

Refactored the core replenishment agent to improve maintainability, fix logic gaps, and enforce strict configuration compliance.

#### Key Changes

- **Refactored `MinMaxReplenisher`:**
  - Broke down the monolithic `generate_orders` function (154 statements) into 6 focused helper methods:
    - `_identify_target_nodes`: Handles staggering logic
    - `_update_demand_smoothing`: Maintains POS averages
    - `_calculate_average_demand`: Unifies Inflow/POS signal logic
    - `_calculate_base_order_logic`: Computes standard (s,S) targets
    - `_apply_echelon_logic`: Overrides targets for Multi-Echelon (MEIO) nodes
    - `_finalize_quantities`: Applies masking and batching
    - `_create_order_objects`: Generates Order entities
  - Resolved complexity violations and improved readability.

- **Config Enforcement (Physics Hardcodes Removed):**
  - **Rush Threshold:** Replaced hardcoded `2.0` days threshold for rush orders with `rush_threshold_days` (default 2.0) from configuration.
  - **Burn-In Days:** Removed hardcoded 90-day fallback in `Orchestrator`. Now strictly respects `simulation_config.json` value.
  - **Earth Radius:** Moved hardcoded `6371.0` km radius to a module constant in `generators/network.py`.

- **Test Infrastructure:**
  - Added `tests/test_replenishment_refactor.py` covering Store Replenishment and Customer DC Echelon logic to ensure regression safety during refactoring.

#### Validation
- Verified regression safety with full test suite pass.
- Verified logic correctness for both standard store ordering and multi-echelon DC replenishment.

## [0.33.0] - 2026-01-14

### Automatic Steady-State Checkpointing

Major UX improvement that eliminates the need for manual warm-start management. The simulation now transparently handles burn-in and checkpointing.

#### Key User Experience Change

**Before (manual 2-step process):**
```bash
poetry run python scripts/generate_warm_start.py --burn-in-days 90
poetry run python run_simulation.py --days 275 --warm-start data/snapshots/warm_start_90d.json.gz
```

**After (automatic):**
```bash
poetry run python run_simulation.py --days 365
# First run: auto-creates checkpoint after 90-day burn-in, then runs 365 data days
# Subsequent runs: loads checkpoint, runs 365 data days immediately
```

#### How It Works

1. **First Run (no checkpoint):** Runs 90-day burn-in phase (configurable via `warm_start.default_burn_in_days`), saves checkpoint to `data/checkpoints/steady_state_{config_hash}.json.gz`, then runs requested data days.

2. **Subsequent Runs:** Loads existing checkpoint (if config hash matches), skips burn-in, runs data days immediately.

3. **Config Change Detection:** Checkpoints are named by config hash. When `simulation_config.json` or `world_definition.json` changes, the old checkpoint is ignored and a new burn-in runs.

#### New CLI Flags

| Flag | Description |
|------|-------------|
| `--no-checkpoint` | Disable auto-checkpointing (always cold-start) |
| `--warm-start PATH` | Use specific snapshot file (legacy behavior) |
| `--skip-hash-check` | Load snapshot even if config changed (not recommended) |

#### Architecture Changes

- **`simulation/orchestrator.py`:**
  - Added `auto_checkpoint` parameter to `__init__`
  - Added `_needs_burn_in` flag for deferred burn-in execution
  - Added `_metrics_start_day` to exclude burn-in from Triangle Report metrics
  - Implemented `_save_checkpoint()` method using shared snapshot utilities
  - Modified `run()` to handle burn-in + data phases transparently

- **`simulation/snapshot.py`** (NEW):
  - Extracted snapshot utilities from `generate_warm_start.py` for code sharing
  - `compute_config_hash()` - SHA256 hash of config + manifest
  - `capture_minimal_state()` - Inventory tensors, active shipments, production orders
  - `derive_initialization_params()` - History buffer priming values
  - `save_snapshot()` - Gzipped JSON checkpoint writer

- **`run_simulation.py`:**
  - Updated docstring with auto-checkpoint behavior
  - Added `--no-checkpoint`, `--warm-start`, `--skip-hash-check` flags

#### Configuration Additions (`simulation_config.json`)

```json
"calibration": {
  "warm_start": {
    "default_burn_in_days": 90,
    "validate_config_hash": true,
    "history_buffer_init": {
      "demand_noise_cv": 0.1,
      "lead_time_noise_cv": 0.1,
      "use_derived_abc": true
    }
  },
  "multi_echelon_lead_times": {
    "store_from_dc": 1.0,
    "customer_dc_from_rdc": 3.0,
    "rdc_from_plant": 5.0,
    "plant_production": 3.0,
    "ftl_consolidation_buffer": 2.0
  }
}
```

#### Testing

New test file `tests/test_calibration.py` with 8 regression tests:
- Checkpoint creation/loading verification
- `--no-checkpoint` flag behavior
- Metrics count verification (days flag meaning)
- Cold-start burn-in exclusion
- Service level consistency across checkpoint loads
- Inventory turns consistency
- Config hash validation

#### Day Continuation Fix

Critical fix: Warm-start simulations now continue from `burn_in_days + 1` (e.g., day 91) instead of resetting to day 1. This preserves:
- Seasonality patterns (`sin(2*pi*(day-phase)/365)`)
- Random seed consistency
- Weekly promo cycles
- ABC reclassification timing

## [0.32.1] - 2026-01-13

### Fix: Restore v0.31.0 Inventory Calibration

The v0.32.0 release inadvertently overwrote the v0.31.0 empirically-tuned inventory parameters when the calibration script was re-run for multi-line physics. This caused severe metric regression (Service: 82%, SLOB: 85%, Turns: 3.5x).

#### Root Cause

The `calibrate_config.py` script re-derived all parameters from physics formulas, overwriting the manually-tuned v0.31.0 values that achieved the optimal service/turns/SLOB balance.

#### Restored Parameters (`simulation_config.json`)

| Parameter | v0.32.0 (broken) | v0.32.1 (restored) |
|-----------|------------------|-------------------|
| `store_days_supply` | 14.0 | **27.0** |
| `rdc_days_supply` | 21.3 | **41.0** |
| `customer_dc_days_supply` | 14.0 | **27.0** |
| `abc_velocity_factors.A` | 1.2 | **1.5** |
| `abc_velocity_factors.C` | 0.85 | **0.5** |
| `slob_abc_thresholds.A` | 64 | **120** |
| `slob_abc_thresholds.B` | 91 | **170** |
| `slob_abc_thresholds.C` | 137 | **250** |
| `a_production_buffer` | 1.1 | **1.2** |
| `c_production_factor` | 0.6 | **0.35** |
| `c_demand_factor` | 0.8 | **0.6** |
| `trigger_dos_a` | 21 | **14** |
| `trigger_dos_b` | 17 | **10** |
| `trigger_dos_c` | 12 | **5** |

#### Known Issue

The calibration script (`calibrate_config.py`) produces suboptimal ABC velocity factors and SLOB thresholds. Future work needed to:
1. Add cold-start buffer component to priming formulas
2. Derive ABC factors from demand distribution, not arbitrary values
3. Validate calibrated values against simulation results before applying

## [0.32.0] - 2026-01-13

### Multi-Line Manufacturing Physics

Significant architectural upgrade to the production engine, replacing scalar capacity hacks with explicit discrete production line modeling.

#### Architectural Changes

- **TransformEngine (`transform.py`)**: 
  - Implemented `LineState` to track capacity, last product, and OEE metrics per discrete production line.
  - Replaced aggregate plant capacity pools with parallel line scheduling.
  - Implemented sticky line selection logic to minimize changeover overhead.
  - Refined OEE calculation to use $Availability \times Performance \times Quality$ per line.
- **MRPEngine (`mrp.py`)**: 
  - Updated network capacity calculations to be line-aware, ensuring accurate planning signals.

#### Configuration & Calibration

- **`simulation_config.json`**:
  - Deprecated `production_rate_multiplier` (set to 1.0) in favor of explicit `num_lines`.
  - Scaled network to **44 lines per plant** (176 total) to support the 21M case/day industry-scale demand.
- **`calibrate_config.py`**:
  - Updated capacity derivation physics to use line counts.
  - Re-calibrated MRP triggers and inventory priming for the new physics model.

#### Results

- **OEE Realism**: OEE now accurately reflects changeover efficiency (~74% in baseline) rather than aggregate utilization (~36%).
- **Physics Compliance**: Eliminated the "magic multiplier" hack, making manufacturing throughput fully traceable to line run rates and discrete changeover events.

## [0.31.0] - 2026-01-13

### Industry-Calibrated Inventory Tuning

Comprehensive calibration to align simulation metrics with real FMCG industry benchmarks. Discovered and documented the fundamental trade-off frontier between Service, Turns, and SLOB.

#### Research & Target Revision

Based on industry analysis (P&G, Colgate, Unilever annual reports):
- **Turns target**: Revised from 12-16x to **5-7x** (Colgate: 4.1x, P&G: 5.5x, Unilever: 6.2x)
- **Service target**: Confirmed at **95-98%** (P&G achieves >99%)
- **SLOB target**: Confirmed at **<15%** (world-class: 10%)
- **OEE target**: Revised from 65-85% to **55-70%** (industry average: 55-60%)

#### Trade-Off Frontier Discovery

Through iterative tuning, mapped the service-turns-SLOB Pareto frontier:

| Configuration | Service | Turns | SLOB | Notes |
|---------------|---------|-------|------|-------|
| v0.30.0 baseline | 74.3% | 10.32x | 25.1% | Too lean, high SLOB |
| High priming (35d) | 91.3% | 4.57x | 13.6% | Maximum achievable service |
| **Balanced (final)** | **89.4%** | **5.00x** | **4.2%** | Optimal within constraints |

**Key Finding**: The simulation architecture has a structural service ceiling around **91%**. Pushing beyond requires violating turns (<4.5x) or SLOB (>14%) constraints.

#### Configuration Changes (`simulation_config.json`)

**Inventory Initialization** (physics-based priming):
- `store_days_supply`: 14.0 → **27.0** days
- `rdc_days_supply`: 21.0 → **41.0** days
- `customer_dc_days_supply`: 14.0 → **27.0** days
- `abc_velocity_factors`: A=1.3/B=1.0/C=0.6 → **A=1.5/B=1.0/C=0.5**

**Replenishment Policy**:
- `target_days_supply`: 21.0 → **27.0** days
- `reorder_point_days`: 14.0 → **20.0** days

**Calibration Section** (new industry benchmarks):
- Added `industry_benchmarks.target_turns: 6.0`
- Added `industry_benchmarks.reference_companies` (Colgate, P&G, Unilever)
- Added `echelon_proportions` for DOS breakdown (store: 23%, DC: 23%, RDC: 35%, plant: 12%, pipeline: 7%)

#### Metrics Improvement (365-day)

| Metric | v0.30.0 | v0.31.0 | Target | Status |
|--------|---------|---------|--------|--------|
| Service | 74.3% | **89.4%** | 95-98% | +15.1pp |
| A-Items | - | **89.6%** | - | Good differentiation |
| Turns | 10.32x | **5.00x** | 5-7x | ✓ On target |
| SLOB | 25.1% | **4.2%** | <15% | ✓ Excellent |
| OEE | 41.6% | **35.8%** | 55-70% | Expected with higher inventory |

#### Root Cause of Service Gap (89% vs 95%)

The ~5% gap to the 95% target stems from structural factors:
1. **Lead time variability**: 3-day lead times plus demand variability create stockout windows
2. **Demand spikes**: Seasonality (±12%) creates peaks that buffers can't fully cover
3. **Network topology**: Multi-echelon delays compound through the network

**Recommendations for future improvements**:
- Reduce lead times (3 days → 1-2 days)
- Implement safety stock differentiation by ABC class in replenishment engine
- Add expedited shipping capability for A-items during shortages

## [0.30.0] - 2026-01-10

### Seasonal Capacity Calibration Enhancement

Enhanced the calibration script to validate and optimize seasonal capacity parameters, addressing the service degradation introduced in v0.29.0.

#### Root Cause Analysis

v0.29.0 set `capacity_amplitude = 0.12` to match `demand_amplitude = 0.12`, creating symmetric flex. During trough periods, when both demand AND capacity drop by 12%, there's no margin for demand variability, causing stockouts.

#### Changes

**Calibration Script (`scripts/calibrate_config.py`)**
- Added `validate_seasonal_balance()` function to check capacity meets demand across all seasons
- Added `derive_seasonal_capacity_params()` function to derive optimal capacity_amplitude based on physics
- Integrated seasonal validation into the main calibration flow
- Added seasonal analysis section to calibration report output
- Updated `apply_recommendations()` to set `capacity_amplitude`

**Configuration (`simulation_config.json`)**
- Changed `capacity_amplitude` from 0.12 to 0.108 (physics-derived: `demand_amp × (1 - trough_buffer)`)
- Added seasonal validation thresholds to `validation` section:
  - `seasonal_min_peak_margin`: 0.05 (5% minimum margin at peak)
  - `seasonal_min_trough_margin`: 0.10 (10% minimum margin at trough)
  - `seasonal_max_peak_oee`: 0.95 (flag if OEE > 95% at peak)
  - `seasonal_min_trough_oee`: 0.40 (flag if OEE < 40% at trough)
  - `seasonal_target_trough_buffer`: 0.10 (target 10% buffer during trough)

#### Physics Rationale

The key insight is that `capacity_amplitude` should be LESS than `demand_amplitude` to maintain safety buffer during troughs:

```
demand_amplitude = 0.12 (±12%)
capacity_amplitude = 0.108 (±10.8%)

At TROUGH (day ~58):
  Demand:   base × 0.88 = 88%
  Capacity: base × 0.892 = 89.2%
  Margin:   ~1.4% buffer (was 0% with symmetric 0.12)
```

This small asymmetry maintains service during trough while preserving most of the SLOB improvement from v0.29.0.

#### Actual Metrics Impact (365-day)

| Metric | v0.29.0 | v0.30.0 | Target |
|--------|---------|---------|--------|
| SLOB | 26.2% | **25.1%** | <30% |
| Service | 73.7% | **74.3%** | >85% |
| Turns | 10.43x | **10.32x** | 12-16x |
| OEE | 41.4% | **41.6%** | 65-85% |

**Analysis**: Modest service improvement (+0.6%) with slight SLOB improvement. The small change (0.12 → 0.108) provides minimal buffer. Further reduction to 0.10 or 0.08 may be needed if service target remains priority.

## [0.29.0] - 2026-01-08

### Flexible Production Capacity (Seasonal Capacity Flex)

Implemented seasonal capacity adjustment that mirrors real FMCG manufacturing practices:
- **Peak season**: Overtime, extra shifts → capacity INCREASES
- **Trough season**: Reduced shifts, maintenance → capacity DECREASES

This allows production to track seasonal demand without massive inventory buffers.

#### Changes

**TransformEngine (`transform.py`)**
- Added `_get_seasonal_capacity_factor(day)` method calculating capacity multiplier
- Daily capacity reset now applies seasonal factor: `max_capacity × seasonal_factor`
- Fixed OEE calculation to use effective daily capacity (prevents >100% OEE at peaks)

**MRPEngine (`mrp.py`)**
- Added `_get_daily_capacity(day)` method for day-aware capacity planning
- Final capacity check (Phase 4) now uses seasonally-adjusted capacity threshold

**Configuration (`simulation_config.json`)**
- Added `capacity_amplitude` to `demand.seasonality` section
- Set to 0.12 (matches demand amplitude for 1:1 tracking)

#### Metrics Impact (365-day run)

| Metric | Before (v0.27) | After (v0.29) | Target |
|--------|----------------|---------------|--------|
| SLOB | 83.8% | **26.2%** | <30% |
| Turns | 7.87x | **10.43x** | 12-16x |
| Service | 78.2% | 73.7% | >85% |
| OEE | 57.8% | 41.4% | 65-85% |

**Analysis**: SLOB improved dramatically (−69%), Turns improved. Service slightly degraded during trough periods when capacity is reduced. OEE appears lower because denominator is now seasonally-adjusted (correct calculation).

#### Tests Added

- `tests/test_seasonal_capacity.py` - 9 tests covering:
  - Seasonal factor at peak/trough
  - Zero amplitude returns 1.0
  - Sinusoidal pattern verification
  - MRP/TransformEngine factor alignment
  - OEE never exceeds 100%

## [0.28.0] - 2026-01-08

### Seasonally-Adjusted Inventory Priming & Mix Optimization

Fixed systematic inventory drift where SLOB inventory accumulated from 27% to 84% over 365 days due to cold-start misalignment with seasonal demand.

#### Key Changes

1.  **Seasonally-Adjusted Priming** (`orchestrator.py`)
    *   Applied seasonal factor (~0.94x for Day 1 trough) to initial inventory priming.
    *   Ensures Day 1 inventory matches actual demand rather than annual average.
    *   Prevents immediate overstocking that creates a permanent inventory wedge.

2.  **ABC-Differentiated Production** (`mrp.py`)
    *   **A-items**: Added 1.1x production buffer to prevent stockouts.
    *   **C-items**: Applied 0.6x production factor to prevent accumulation.
    *   Activated previously dead code to enforce mix discipline.

3.  **SLOB Metric Recalibration** (`simulation_config.json`)
    *   Updated SLOB thresholds to reflect realistic multi-echelon lead times.
    *   A-Items: 60 days (was 23)
    *   B-Items: 75 days (was 33)
    *   C-Items: 120 days (was 50)
    *   Reduces false positives for naturally slow-moving inventory.

#### Results

*   **Inventory Drift**: Significantly reduced.
*   **Product Mix**: Better alignment between production and consumption by class.

## [0.27.0] - 2026-01-07

### Hardcode Audit & Physics-Based Calibration

Completed comprehensive semgrep audit and moved all hardcoded values to config. Enhanced calibration script with physics-based derivations.

#### Hardcodes Fixed (moved to config)

| Location | Hardcode | New Config Key |
|----------|----------|----------------|
| `orchestrator.py:248` | ABC velocity factors `{0:1.2, 1:1.0, 2:0.8}` | `inventory.initialization.abc_velocity_factors` |
| `orchestrator.py:519` | Production order timeout `14` | `manufacturing.production_order_timeout_days` |
| `orchestrator.py:528` | Batch retention days `30` | `manufacturing.batch_retention_days` |
| `logistics.py:104` | Stale order threshold `14` | `logistics.stale_order_threshold_days` |
| `calibrate_config.py:263` | SLOB velocity factors | `validation.slob_abc_velocity_factors` |
| `calibrate_config.py:267` | SLOB margin `1.5` | `validation.slob_margin` |

#### Calibration Script Enhancements (`calibrate_config.py`)

1. **Physics-Based Priming Derivation**
   - Store DOS derived from: cycle_stock + safety_stock + lead_time
   - Safety stock uses z-scores: A=2.33 (99%), B=1.65 (95%), C=1.28 (90%)
   - CV by echelon: store=0.4, DC=0.25, RDC=0.15 (aggregation effect)

2. **Network-Level Trigger Derivation**
   - trigger = production_time + transit_time + safety_buffer
   - Derived: A=14d, B=12d, C=10d

3. **Config Consistency Validation**
   - Multi-echelon priming vs trigger check
   - SLOB threshold vs expected DOS check
   - Capacity utilization bounds (50%-95%)
   - Expected turns sanity check

#### Root Cause Discovery: MRP Uses Static Demand

Investigation revealed MRP batch sizing uses static `expected_daily_demand` (no seasonality) while actual demand varies ±12% with seasonality. This causes:
- Trough overproduction → SLOB accumulation
- Peak underproduction → stockouts

**Fix planned for v0.28.0:** Use actual demand signal for batch sizing.

## [0.25.0] - 2026-01-07

### Production Physics Fix: Balanced Production/Consumption

Fixed critical production overshoot that caused inventory accumulation and metric drift over 365 days.

#### Root Cause Identified

Initial inventory priming (6.7 days RDC DOS) was below MRP trigger thresholds (A:14, B:10, C:7 days),
causing 100% of SKUs to trigger production on Day 1. This created a 2x production spike that
accumulated as SLOB inventory.

#### Key Changes

1. **ABC-Differentiated Network Priming** (`orchestrator.py`)
   - Applied ABC-based initial inventory to ALL echelons (stores, customer DCs, RDCs)
   - A-items: 21 days, B-items: 16 days, C-items: 12 days
   - Initial RDC DOS now 17.4 days (was 6.7 days)
   - Prevents 100% Day-1 production trigger

2. **Campaign Batching Calibration** (`simulation_config.json`)
   - Reduced `production_horizon_days` from 10 to 6
   - Math: With ~90 SKUs triggering daily, horizon=6 gives 90/500 × 6 = 1.08x ratio
   - Previous horizon=10 gave 1.8x ratio (production >> consumption)

#### Results: 365-Day Production/Consumption Ratio

| Day | Before Fix | After Fix |
|-----|------------|-----------|
| 30  | 1.90x      | 1.54x     |
| 90  | 1.50x      | 1.20x     |
| 180 | ~1.3x      | 1.04x     |
| 365 | ~1.2x      | **0.99x** |

Production is now balanced (0.99x) over 365 days. The system self-corrects by under-producing
after the initial spike to draw down excess inventory.

#### Remaining Issues (Not Physics)

1. **SLOB Metric**: Still high (81.9%) - uses volatile daily demand, not expected demand
2. **Service Level Drift**: 87% → 83% - product mix mismatch (A-items short, C-items excess)
3. **FTL Fill**: Shows 0.0% - metric measurement issue to investigate

These are metric definition issues, not simulation physics issues.

## [0.24.0] - 2026-01-07

### Metric Infrastructure & Forward Path Analysis

Added FTL/LTL metric tracking infrastructure and comprehensive analysis of 30-day vs 365-day metric degradation.

#### Key Changes

1. **FTL/LTL Metric Infrastructure** (`monitor.py`)
   - Added `ftl_fill_tracker` for Full Truckload fill rate measurement
   - Added `ltl_shipment_count` for Less Than Truckload tracking
   - Added `record_ftl_fill()` and `record_ltl_shipment()` methods
   - Added FTL metrics to `get_summary_report()` output
   - Infrastructure ready for orchestrator integration in v0.25.0

2. **Forward Path Documentation** (`docs/planning/v025_forward_path.md`)
   - Comprehensive root cause analysis of metric gaps
   - Key finding: 30-day metrics healthy, 365-day metrics degrade (drift problem)
   - Physics-first diagnostic approach defined
   - Prioritized fix list with expected impacts

#### Key Finding: Metric Drift Problem

| Metric | 30-Day | 365-Day | Target |
|--------|--------|---------|--------|
| Service | 86.5% ✅ | 80.2% ❌ | >85% |
| Truck Fill | 70.0% | 31.0% ❌ | >85% |
| SLOB | 29.0% ✅ | 81.0% ❌ | <30% |

Metrics are healthy at startup but degrade over time, suggesting feedback loops or drift rather than structural issues.

#### Research References

Industry research on FMCG consolidation practices:
- [Walmart Freight Consolidation](https://fstlogistics.com/walmart-freight-consolidation/)
- [Target Consolidation Program](https://www.hubgroup.com/)
- [APQC DOS Benchmarks](https://www.apqc.org/)

#### Next Steps (v0.25.0)

See `docs/planning/v025_forward_path.md` for implementation plan:
1. Diagnose root cause of 365-day metric drift
2. Implement Plant→RDC truck consolidation
3. Refine SLOB metric calculation

## [0.23.0] - 2026-01-06

### Death Spiral Fix: Campaign Batching Production (RESOLVED)

The production death spiral (8.8M → 4.3M cases/day over 365 days) has been **FIXED** with campaign-style production batching that matches how real FMCG plants operate.

#### Root Causes Identified

1. **Changeover Time Accumulation**: With 500 SKUs and daily production orders for ALL of them, changeover time exceeded daily capacity (500 SKUs × 0.05h = 25h/day vs 20h available).

2. **POS-Demand Feedback Loop**: When stores stocked out, POS dropped → production dropped → more stockouts → POS dropped further.

#### Solution: Campaign Batching

Instead of producing all 500 SKUs daily (causing 25h+ changeovers), produce larger batches for fewer SKUs per day. This matches how real FMCG plants operate with "campaign runs".

**Key changes to `_generate_rate_based_orders()` (mrp.py):**

1. **Trigger-Based Production**: Only produce when DOS drops below threshold (not daily)
   - A-items: Trigger at DOS < 14 days
   - B-items: Trigger at DOS < 10 days
   - C-items: Trigger at DOS < 7 days

2. **Batch Sizing**: Produce `production_horizon_days` worth per SKU (default 10 days)

3. **SKU Limit per Plant**: Max 60 SKUs/plant/day to cap changeover overhead

4. **Priority Sorting**: Lowest DOS first (most critical items get produced)

5. **Expected Demand for DOS**: Use expected_daily_demand (not POS) to prevent feedback loop

#### Configuration Parameters

New `campaign_batching` section in `simulation_config.json`:
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

#### Results (365-day Simulation)

| Metric | Before (Death Spiral) | After (Campaign Batching) |
|--------|----------------------|---------------------------|
| Production Day 365 | 4.3M (51% drop) | **8.3M (stable)** |
| Service Level | 66% | **80.5%** |
| Production Stability | Declining | **Stable** |

The death spiral is eliminated - production now matches demand (~8M/day) throughout the 365-day run.

---

## [0.22.0] - 2026-01-06

### Memory Explosion Fix: Real-World Replenishment Model

This release fixes the memory explosion issue that caused 365-day runs to crash with 33GB+ memory usage. The fix aligns the replenishment model with how real retail systems (Walmart, Target) actually work.

### Root Cause Analysis

The `pending_orders` dictionary in `MinMaxReplenisher` tracked every unfulfilled order by `(source_id, target_id, product_id)` tuple. With 6,000 stores and 500 SKUs, this could grow to **3M+ entries** (420MB+) during stockout scenarios.

**Key insight from research**: Real retail systems like Walmart's Retail Link don't track pending orders per-SKU. Instead, they:
1. Use **Inventory Position** (on-hand + in-transit) for reorder decisions
2. Recalculate requirements fresh each cycle
3. Generate consolidated orders at the warehouse level, not per-SKU

### Removed

- **`pending_orders` dict** (`replenishment.py`): Removed the dictionary that tracked unfulfilled orders per (source, target, product) combination
- **`record_fulfilled_orders()`**: Removed - no longer needed
- **`record_unfulfilled_orders()`**: Removed - no longer needed
- **`expire_stale_pending_orders()`**: Removed - no longer needed
- **Deduplication check in `generate_orders()`**: Removed the `pending_key in self.pending_orders` check

### Changed

- **`all_violations` list** (`monitor.py`): Now capped at 1,000 entries (configurable via `max_violations_tracked`) to prevent unbounded growth

### Technical Notes

The Inventory Position logic at `replenishment.py:727` already correctly prevents double-ordering:
```python
inventory_position = on_hand_inv + in_transit_inv  # Line 727
needs_order = inventory_position < reorder_point   # Line 849
```

If a shipment is in-transit, it's counted in `in_transit_inv`, preventing duplicate orders without needing a separate tracking dictionary.

### References

- [Walmart Retail Link](https://supplierwiki.supplypike.com/articles/retail-link-how-does-it-help)
- [Walmart Real-Time Replenishment with Kafka](https://www.confluent.io/blog/how-walmart-uses-kafka-for-real-time-omnichannel-replenishment/)
- [Walmart National PO Program](https://supplierwiki.supplypike.com/articles/what-is-walmarts-national-po-program)

---

## [0.21.0] - 2026-01-06

### SKU Scale Expansion & Config Calibration

This release expands the simulation to 500 SKUs (10x previous) and adds a physics-based calibration script to derive optimal configuration parameters.

### Added

- **Calibration Script** (`scripts/calibrate_config.py`): New tool that derives optimal simulation parameters from world definition using supply chain physics rather than guesswork.
  - Calculates expected daily demand from store count, SKU count, and category profiles
  - Calculates plant theoretical capacity from recipe run rates
  - Derives `production_rate_multiplier` to achieve target OEE (default 85%)
  - Derives inventory DOS parameters based on lead times and service level targets
  - Run with `--apply` flag to update simulation_config.json automatically

### Changed

- **SKU Count**: 50 → 500 SKUs (10x scale increase)
- **production_rate_multiplier**: 25.0 → 0.2 (calibrated for 85% OEE with 500 SKUs)
- **store_days_supply**: 21.0 → 4.5 days (reduced initial inventory)
- **rdc_days_supply**: 28.0 → 7.5 days
- **customer_dc_days_supply**: 21.0 → 7.5 days
- **rdc_store_multiplier**: 500.0 → 150.8 (scaled to store count)
- **ltl_min_cases**: 10 → 100 (improve truck fill rate)

### Technical Notes

The calibration script solves the fundamental equation:
- **OEE = Actual Production / Theoretical Capacity**
- For target OEE of 85%, capacity should be ~1.18x demand
- `production_rate_multiplier = (demand / target_oee) / theoretical_capacity`

With 500 SKUs across 6,030 stores, daily demand is ~21M cases. The base recipe run rates provide 141M cases/day theoretical capacity, requiring a 0.2 multiplier to achieve 85% OEE.

---

## [0.20.0] - 2026-01-05

### Death Spiral Fix: Root Cause Resolution

This release fixes the "death spiral" issue that caused production to collapse from 7M to 3.5M cases/day and service levels to drop below 85%. The simulation now runs stably for 90+ days with 500 SKUs.

### Root Cause Analysis

Three interrelated issues caused the death spiral:

1. **SLOB Throttling Override:** The SLOB (Slow/Obsolete) throttling logic in MRP was applied AFTER the ABC production floors, allowing it to override safety minimums and crash production when DOS appeared high.

2. **Raw Material Starvation:** Plants ran out of specific ingredients (ACT-CHEM-003, PKG-CAP-003) because initial inventory (5M per ingredient) was insufficient, and the ingredient supply chain wasn't replenishing fast enough.

3. **Production Order Backlog Explosion:** Unfulfilled production orders accumulated unboundedly (164M requested vs 6M produced by Day 90), hiding actual production capacity issues.

### Fixed

- **MRP Floor Priority** (`mrp.py:840-857`): Moved SLOB throttling BEFORE the ABC production floor so the floor acts as an absolute safety net. Production floors (A=90%, B=80%, C=70% of expected demand) are now always respected.

- **Plant Ingredient Inventory** (`orchestrator.py:324-336`): Increased initial plant ingredient inventory from 5M to 50M per ingredient type to provide 90+ day buffer while MRP ingredient ordering catches up.

- **Production Order Timeout** (`orchestrator.py:475-483`): Added 14-day timeout for stale production orders. Orders that can't be fulfilled within 14 days are dropped to prevent unbounded backlog accumulation. MRP regenerates if demand persists.

- **Order Demand Signal** (`mrp.py:920-940`): Made order-based demand the primary MRP signal (replacing shipment-based), with expected demand as floor to prevent signal collapse during constrained periods.

- **Held Order Timeout** (`logistics.py:101-110`): Added 14-day timeout for held logistics orders to prevent unbounded accumulation.

- **Pending Order Deduplication** (`replenishment.py:206-210, 1019-1075`): Added tracking to prevent stores from generating duplicate orders for SKUs already awaiting fulfillment.

- **Batch Memory Cleanup** (`orchestrator.py:485-491`): Added 30-day retention limit for completed batches to prevent unbounded memory growth.

### Results

| Metric | Before Fix | After Fix (30d) | After Fix (90d) |
|--------|------------|-----------------|-----------------|
| Service Level | 83% → crashed | **95.86%** | **90.50%** |
| Production | 7M → 3.5M/day | **7M stable** | **6M stable** |
| OEE | 68% | **91.3%** | **89.9%** |
| Truck Fill Rate | 47% | **76.0%** | **53.6%** |

### Technical Notes

- The ingredient supply chain needs long-term improvement in MRP ordering logic to maintain supply without relying on large initial buffers.
- Production order timeout prevents backlog explosion but means some demand may be temporarily unfulfilled during constrained periods.
- The 500 SKU scale is now stable for 90+ day runs, suitable for virt-graph stress testing.

---

## [0.19.16] - 2026-01-05

### Performance Optimizations & Death Spiral Diagnosis

This release adds performance optimizations and documents the root cause of the "death spiral" issue affecting 365-day runs.

### Added
- **Changeover Time Multiplier:** New config option `changeover_time_multiplier` (default 0.1) scales recipe changeover times. Addresses changeover starvation where 500 SKUs × 0.5h = 250h changeover needed per plant vs 18h available.
- **Production Order Grouping:** Transform engine now sorts orders by (plant, ABC priority, product_id, due_day) to minimize changeover events by processing same products consecutively.
- **Forward Fix Plan:** Created `docs/planning/forward_fix.md` documenting root causes and proper solutions for death spiral and order explosion issues.

### Performance
- **Lead Time Stats Caching:** Added dirty-link tracking to replenishment agent, reducing np.std() calls from 55k to only changed links.
- **Incremental In-Transit Tensor:** StateManager now tracks shipments incrementally instead of recomputing from scratch.
- **Batch Arrival Processing:** Orchestrator processes arrivals in batch with single numpy update instead of per-shipment loops.
- **Product Attribute Caching:** LogisticsEngine pre-computes product weight/volume/cases_per_pallet as numpy arrays, avoiding 6M+ dict lookups per day.
- **Volume Caching:** Product.volume_m3 is now cached on creation instead of computed per-access.

### Changed
- `production_rate_multiplier`: 15.0 → 25.0 (bandaid for capacity gap)
- `changeover_time_multiplier`: 0.1 (new, reduces 30min changeover to 3min)
- `min_order_qty`: 50 → 100 cases
- `order_cycle_days`: 3 → 5 days
- `store_batch_size_cases`: 50 → 100 cases

### Known Issues
- **Order Explosion:** 316M+ orders generated on day 1 due to structural demand-production mismatch. See `forward_fix.md` for proper solution (order aggregation).
- **Memory Blowup:** 365-day runs still exhaust memory. Root cause is order explosion, not simulation logic.

---

## [0.19.15] - 2026-01-05

### Phase 2: Demand-Production Alignment & Performance Optimization

This release implements the core logic for Phase 2 (closed-loop demand alignment) and solves a critical performance bottleneck caused by the 20x SKU expansion.

### Added
- **Category-Level ABC Classification (Phase 2.1):** `MRPEngine` now classifies products into A/B/C buckets *within each category* (Oral, Wash, Home) rather than globally. This ensures high-volume categories don't monopolize A-status.
- **Differentiated Production Floors (Phase 2.5):** MRP now applies variable production floors based on ABC class (A=50%, B=30%, C=10%) to protect availability for key items while reducing SLOB risk for tail items.
- **SLOB Throttling Override (Phase 2.2):** Implemented a hard override in MRP: if Days-of-Supply > 60, production is capped at 50% of demand regardless of other signals.
- **ABC-Based Inventory Priming (Phase 2.3):** `Orchestrator` now initializes store inventory using ABC-specific targets (A=21d, B=14d, C=7d) instead of a uniform 14d, creating a more realistic starting state.

### Performance (Critical Fix)
- **"Fast-Path" Logistics Optimization:** The 500-SKU expansion caused an object explosion (~132k orders/day), crashing the simulation. Implemented a "Fast-Path" in `LogisticsEngine` that pre-calculates weight/volume for LTL routes. If a route fits in a single truck (99% of cases), it bypasses the expensive bin-packing logic.
- **Order Consolidation:** Updated `simulation_config.json` to force consolidation:
  - `min_order_qty`: 10 → 50 cases
  - `store_batch_size_cases`: 20 → 50 cases
  - `order_cycle_days`: 1 → 3 days
  - **Result:** Reduced daily order count by ~60%, stabilizing memory usage and restoring simulation speed (~2s/day).

### Results (30-day Validation)
| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Service Level | >85% | **94.71%** | ✅ Exceeded |
| Truck Fill | 30-50% | **52.0%** | ✅ Optimized |
| Inventory Turns | 6-14x | **7.65x** | ✅ In Range |
| SLOB | <30% | **6.3%** | ✅ Low |

### Fixed
- **Linting:** Resolved multiple E501 line length and unused variable violations in `orchestrator.py` and `mrp.py`.

## [0.19.14] - 2026-01-04

### SKU Generation Overhaul (Phase 1)
Implemented a massive expansion of the product portfolio to support realistic long-tail demand analysis.

### Added
- **500+ SKU Scale:** Rewrote `ProductGenerator` to support target-based generation (`n_skus=500`).
- **Zipfian Packaging:** SKU generation now uses Zipfian distribution to select packaging types, simulating "Mainstream" vs "Niche" sizes.
- **Expanded Packaging Types:** Added 25+ new packaging definitions (Tubes, Bottles, Pumps, Pouches, Glass) with realistic dimensions and `units_per_case`.
- **Variant Palettes:** Added configuration for category-specific scent/flavor variants (e.g., "Mint", "Lavender", "Ocean Mist").
- **Configurable SKU Proportions:** Added `sku_proportion` to `simulation_config.json` to control SKU count per category (Oral Care 40%, Personal Wash 35%, Home Care 25%).

### Changed
- **Generator Logic:** `ProductGenerator.generate_products()` now iteratively samples unique (Brand, Pack, Variant) combinations until the target count is met, rather than exhaustive enumeration.
- **Generation Script:** `scripts/generate_static_world.py` now accepts `--skus` and `--stores` CLI arguments.

### Results
- Generated a static world with **544 Products** (500 Finished Goods + 44 Ingredients) and **6,126 Nodes**.
- This enables proper ABC classification testing (Phase 2) which was impossible with the previous 24-SKU dataset.

## [0.19.13] - 2026-01-04

### Fixed
- **Critical Lead Time Formula Bug:** Fixed `network.py:add_geo_link()` where `dist / speed` produced hours but was used as days. Formula now correctly converts: `(dist / speed / 24) + handling`.
  - Before: 9,411 km / 80 km/h = **117.6 hours treated as 117.6 days!**
  - After: 9,411 km / 80 km/h / 24 = **4.9 days + 1 day handling = 5.9 days**
- **Documented `us_cities.csv` dependency:** File must exist at `data/static/us_cities.csv`. Fallback to (0,0) coordinates causes 9,000+ km distances.

### Results (365-day Simulation)
| Metric | v0.19.12 (Broken) | v0.19.13 (Fixed) | Change |
|--------|-------------------|------------------|--------|
| **Service Level** | 39.28% | **82.70%** | **+43pp** |
| **Inventory Turns** | 31.69x | **8.71x** | Normalized |
| **OEE** | 30.8% | **86.8%** | **+56pp** |
| **SLOB** | 0.0% | 65.3% | Expected |

### Root Cause Analysis
The v0.19.12 GIS overhaul had two bugs:
1. **Missing `us_cities.csv`:** When deleted, fallback sets all non-fixed nodes to (0,0) coordinates (Atlantic Ocean), creating 9,000+ km distances.
2. **Lead time formula:** `dist / speed` gives hours, not days. This caused 110+ day lead times instead of 1-2 days.

Result: Shipments took 3+ months to arrive, starving the entire network.

## [0.19.12] - 2026-01-04

### Fixed
- **Structural Distribution Bottleneck ("The Ghost RDC"):** Identified and fixed a critical topology flaw where `RDC-WE` had 0 downstream customers due to preferential attachment logic, creating a "black hole" for 25% of production.
- **Geospatial Linking:** Replaced random preferential attachment with **Physics-Based Geospatial Linking**.
  - Implemented `us_cities.csv` lookup for 100+ real US cities with lat/lon coordinates.
  - Links are now created based on **Haversine Distance** (nearest neighbor logic).
  - Suppliers, Plants, RDCs, DCs, and Stores now have physical coordinates.
- **Demand-Proportional Routing:** Updated `Orchestrator` to route production to RDCs based on their **Actual Demand Share** (POS aggregate) rather than an even 25/25/25/25 split. This ensures `RDC-SO` (50% demand) gets 50% of supply.
- **Inventory Priming:** RDCs with zero demand (if any) now initialize with zero inventory, preventing dead stock accumulation.

### Added
- **"Super Lean" GIS Layer:**
  - `StaticDataPool`: Loads curated city data.
  - `Node`: Added `lat` and `lon` fields.
  - `NetworkGenerator`: Implemented "Metropolitan Jitter" to spread 4,500 stores realistically across 100 metro areas.

### Results (365-day Simulation)
| Metric | v0.19.11 | v0.19.12 (GIS Fix) | Change |
|--------|----------|-------------------|--------|
| **Service Level** | 64.5% | **70.69%** | +6.19pp |
| **SLOB** | 28.6% | **1.0%** | -27.6pp (Hyper-Lean) |
| **Turns** | 7.9x | **19.16x** | +11.3x (Too Fast) |
| **OEE** | ~85% | **65.0%** | -20pp (Starved) |

### Analysis
The structural fix worked: flow is now physically correct, and the "Black Hole" is gone. However, the system is now **Hyper-Lean**. Because Haversine lead times are much shorter (1-2 days vs 3-5 random), the `MinMaxReplenisher` automatically slashed safety stocks. The system now runs with near-zero buffers (19x turns!), causing immediate stockouts on demand spikes.

**Next Steps:** Re-calibrate inventory policies (increase `z-score` and `min_safety_days`) to account for the new, faster physics.

## [0.19.11] - 2026-01-03

### Added
- **POS-Driven Production (Physically Correct Approach):** Replaced open-loop expected-based production with closed-loop POS-driven production.
  - Uses actual consumer demand (POS) as the primary signal
  - Inventory feedback loop: production adjusts based on actual DOS
  - ABC differentiation in response dynamics, not baseline rates
- **ABC Class Tracking:** Added `abc_class` array to MRPEngine (0=A, 1=B, 2=C) for production decisions.

### Changed
- **`_generate_rate_based_orders()`:** Complete rewrite for closed-loop control:
  - A-items: Fast response (130% catch-up when low, 90% when high)
  - B-items: Balanced response (110% catch-up, 85% when high)
  - C-items: Slow response, aggressive reduction when overstocked (down to 30%)
  - Production tracks actual demand, not expected demand
- **`generate_production_orders()`:** Now passes POS demand vector to rate-based orders.

### Results
| Metric | v0.19.9 | v0.19.11 | Target |
|--------|---------|----------|--------|
| 30-day Service Level | 93% | 92.5% | >90% |
| 365-day Service Level | 67.5% | 64.5% | >85% |
| **SLOB** | 70% | **28.6%** | <30% ✓ |
| Inventory Turns | 4.8x | 7.9x | Higher ✓ |
| Cash-to-Cash | 68 days | 33 days | Lower ✓ |

### Analysis
**SLOB target achieved!** The POS-driven approach dramatically reduced SLOB from 70% to 28.6% (<30% target).

However, 365-day service level dropped from 67.5% to 64.5%. Extensive experimentation revealed:

1. **Production is NOT the bottleneck for service level.** Even with 150% A-item production, SL stays at ~66%. This proves the remaining SL gap is a **distribution problem**, not a production problem.

2. **The physics is correct.** POS-driven production creates a self-correcting system:
   - C-items: Production drops when DOS > threshold → SLOB decreases
   - A-items: Production increases when DOS < threshold → should improve SL
   - But goods aren't reaching stores (distribution bottleneck)

3. **Tradeoff confirmed:**
   - Aggressive C-item reduction → SLOB ↓↓↓, but SL slightly ↓
   - Aggressive A-item boost → SL barely changes (stuck at ~66%)
   - Proves distribution flow is the constraint

**Next Steps for 85%+ SL:**
The service level improvement requires fixing **distribution**, not production:
- RDC-to-store replenishment tuning
- Push allocation parameters
- Potentially reduce initial inventory priming at RDCs
- Check if goods are stuck at RDCs instead of flowing to stores

See `docs/planning/mix_opt.md` for the original analysis.

## [0.19.10] - 2026-01-03 (Superseded by v0.19.11)

Experimental version with ABC-differentiated production. Improved SL to 71.7% but SLOB worsened to 76%. Replaced by POS-driven approach in v0.19.11.

## [0.19.9] - 2026-01-03

### Added
- **Rate-Based Production (Option C):** Implemented anticipatory production mode that always produces at expected demand rate, preventing the low-equilibrium trap.
  - `rate_based_production`: Config toggle for new production mode (default: true)
  - `rate_based_min_batch`: Lower min batch (100 vs 1000) to capture C-items
  - `inventory_cap_dos`: Only throttle production if DOS > 45 days
  - Catch-up mode: If DOS < ROP, produce EXTRA to recover deficit
- **MRP Diagnostics:** Added `MRPDiagnostics` dataclass for debugging signal flow.
  - Tracks demand signals, production orders, inventory position, DOS
  - Logs to `prism_sim.mrp.diagnostics` logger at INFO level

### Fixed
- **Low-Equilibrium Trap:** Rate-based production prevents the system from stabilizing at low production levels when demand signals collapse.
- **C-Item Production:** Lower min batch (100) prevents filtering out slow-moving products entirely.

### Changed
- **MRPEngine:** Added `_generate_rate_based_orders()` method for Option C logic.
- **Config:** Added `rate_based_production`, `rate_based_min_batch`, `inventory_cap_dos`, `diagnostics_enabled` to `mrp_thresholds`.

### Results
| Metric | v0.19.8 | v0.19.9 | Target |
|--------|---------|---------|--------|
| 30-day Service Level | 93% | 93% | >90% |
| 365-day Service Level | 50-58% | **67.5%** | >85% |
| OEE | 60-75% | **88%** | 75-85% |
| SLOB | - | 70% | <30% |

### Analysis
Rate-based production improved 365-day SL by +17.5pp. The remaining gap to 85%+ is due to:
1. **Product Mix Mismatch:** Fixed expected mix doesn't match actual demand variations
2. **SLOB at 70%:** Wrong products being stocked (C-items accumulating)
3. **Distribution Flow:** Goods produced but not flowing to stores efficiently

See `docs/planning/debug_foxtrot.md` for detailed analysis.

## [0.19.8] - 2026-01-03

### Fixed
- **MRP Starvation Loop (Partial):** Decoupled ingredient ordering from historical production flow.
  - Added `_calculate_max_daily_capacity()` to compute network production capacity.
  - Changed `generate_purchase_orders()` to use expected demand (not historical avg) when backlog is low.
  - Changed production smoothing cap to use `max(avg_recent, expected)` as baseline, preventing cap degradation.
- **Warm Starts:** Warm-started `demand_history` and `production_order_history` buffers.
- **Bullwhip Clamping:** Clamped `record_order_demand` signal to 4x expected demand.

### Changed
- **MRPEngine:**
  - Ingredient ordering now uses expected demand as minimum baseline (was: coupled to backlog only).
  - Production smoothing cap now floors at expected production (was: could degrade with history).
- **Orchestrator:** Added `auditor.record_plant_shipments_out` for push shipments (Mass Balance fix).
- **Config:** Tuned `production_floor_pct` (0.3 → 0.5) and `min_production_cap_pct` (0.5 → 0.7).

## [0.19.7] - 2026-01-03

### Fixed
- **ABC Alignment (Phase 1):** Aligned `MRPEngine` ABC classification with `Replenisher` and `TransformEngine` by injecting `base_demand_matrix` (Zipf-aware) into MRP. This resolves the regression where popular A-items were misclassified as B/C in production planning.
- **Service Level Recovery:** 90-day simulation Service Level recovered to **86.50%** (from 71%), exceeding the >85% target.
- **Engine Bugs:**
  - Fixed `AttributeError` in `POSEngine` by initializing `channel_sku_weights`.
  - Fixed `AttributeError` in `TransformEngine` by correcting scope of `_get_abc_priority`.

### Changed
- **Parameter Tuning (Phase 2 & 3):**
  - Increased A-item ROP multiplier (1.2 → 1.5) to buffer against demand/supply variability.
  - Decreased C-item ROP multiplier (0.8 → 0.5) and Service Level Z-score (1.28 → 1.0) to reduce SLOB.
- **MRPEngine:** Now calculates `expected_daily_demand` by summing the injected `base_demand_matrix` instead of using static config profiles.
- **Orchestrator:** Passes `base_demand_matrix` to `MRPEngine` during initialization.

## [0.19.6] - 2026-01-03

### Refactoring
- **Config-Driven Logic:** Replaced "genuine logic hardcodes" with configuration parameters and enums to improve maintainability and flexibility.
  - **Enums:** Introduced `OrderPriority` (`RUSH`, `HIGH`, `STANDARD`, `LOW`) and `ABCClass` (`A`, `B`, `C`) in `core.py` to replace integer/string literals.
  - **Configuration:** Added `min_history_days` to `replenishment` config and `min_batch_size_absolute`, `default_store_count` to `manufacturing` config in `simulation_config.json`.
  - **Agents:** Updated `replenishment.py`, `mrp.py`, and `transform.py` to use the new enums and configuration values instead of hardcoded numbers.

## [0.19.5] - 2026-01-03

### Status
- **365-Day Validation:** Run completed. Service Level (71%) and SLOB (78%) indicate a regression from baseline.
- **Root Cause:** Identified ABC misalignment between Replenisher (Zipf-aware) and MRP (Zipf-blind config).
- **Plan:** Created `docs/planning/alignment_and_param_fix.md` to address the architecture gap before further tuning.

## [0.19.4] - 2026-01-03

### Added
- **ABC Prioritization (Phase 3 & 4):** Completed remaining phases of the ABC prioritization plan.
  - **ABC-Aware Replenishment (Phase 3):** `MinMaxReplenisher` now uses config-driven thresholds (80/95%) for dynamic ABC classification, ensuring consistency with MRP logic.
  - **Production Capacity Reservation (Phase 4):** `TransformEngine` now reserves capacity for A-items by classifying products based on expected demand and prioritizing A-item production orders before C-items.
  - **Configuration:** Updated `simulation_config.json` with `abc_prioritization` block enabled by default.

### Changed
- **Replenishment:** Updated `replenishment.py` to read ABC thresholds from config.
- **Transform:** Updated `transform.py` to sort production orders by ABC priority (A > B > C) then due date.

## [0.19.3] - 2026-01-03

### Added
- **ABC Prioritization (Phase 1 & 2):** Implemented Pareto-based prioritization for Allocation and MRP to resolve product mix imbalances.
  - **Allocation:** Scarcity logic now prioritizes A-items (high velocity) over C-items within the same order priority tier. This ensures fast movers get dibs on inventory.
  - **MRP:** Production planning uses ROP multipliers (A=1.2x, C=0.8x) to ensure higher availability for fast movers and reduce SLOB for slow movers.
  - **Configuration:** Added `abc_prioritization` section to `simulation_config.json` with configurable thresholds (80/95%) and multipliers.

### Changed
- **Orchestrator:** Now calculates and injects product velocity into `AllocationAgent` during initialization.
- **MRPEngine:** Now classifies products by velocity and applies dynamic ROP multipliers during production planning.

## [0.19.2] - 2026-01-03

### Service Level Improvement: Signal Flow Optimization

This release implements the fixes outlined in `docs/planning/new-fix.md` to break the negative feedback spiral causing service level degradation.

### Fixed
- **Daily Ordering for Customer DCs:** Removed the 3-day order cycle restriction for Customer DCs using echelon logic. DCs now order every day, ensuring demand signals flow continuously upstream without accumulation delays.
- **Increased Customer DC Targets:** Raised target_days from 21 to 35 and reorder_point_days from 14 to 21 for B2M_LARGE, B2M_CLUB, and B2M_DISTRIBUTOR channels. Higher targets ensure DCs order sufficient quantities to cover downstream demand.
- **Increased Store Targets:** Raised default target_days from 14 to 21 and reorder_point_days from 10 to 14 to provide more buffer at store level.
- **Echelon Safety Multiplier:** Added `echelon_safety_multiplier` (default 1.3) to echelon target/ROP calculations. This provides a buffer beyond raw echelon demand to account for variance at the echelon level.
- **Demand-Proportional MRP Batches:** Changed MRP minimum batch size from fixed 50,000 cases to demand-proportional (7 days of demand, minimum 1,000 cases). Prevents SLOB accumulation from massive batches of low-demand products.

### Added
- **Push-Based Allocation:** Implemented `_push_excess_rdc_inventory()` in Orchestrator to push excess RDC inventory to Customer DCs when Days of Supply exceeds threshold. Uses POS-based demand signal (stable) instead of outflow demand (which collapses during the spiral).
- **Production Prioritization Support:** Added `set_base_demand()` to TransformEngine for future demand-based production scheduling.
- **Configuration Parameters:**
  - `echelon_safety_multiplier`: Buffer multiplier for echelon targets (default: 1.3)
  - `push_allocation_enabled`: Toggle for push-based allocation (default: true)
  - `push_threshold_dos`: Days of supply threshold for push (default: 21)

### Results
| Metric | v0.19.1 Baseline | v0.19.2 (90-day) | v0.19.2 (365-day) | Target |
|--------|------------------|------------------|-------------------|--------|
| Service Level | 73% | **91.84%** ✅ | 76% | >90% |
| SLOB | 65% | 54% | 73% | <30% |
| Inventory Turns | 4.73x | 6.12x ✅ | 4.69x | 6-14x |

### Root Cause Analysis
The 90-day simulation achieves >90% service level, but 365-day degrades to ~76%. Investigation revealed:
- **Product Mix Issue:** Highly concentrated demand (top 10 SKUs = 60% of volume) with SLOB at 73-80% suggests wrong products are stocked
- **Zipfian Distribution:** A-items (16 SKUs, 80% of demand) may be stocking out while C-items (47 SKUs) accumulate
- **Slow Drift:** System starts well (initial inventory priming) but drifts to suboptimal equilibrium over time

### Next Steps for v0.19.3
1. **ABC-Prioritized Allocation:** When inventory is scarce, prioritize A-items over C-items
2. **ABC-Prioritized MRP:** Weight production planning toward high-velocity SKUs
3. **Inventory Distribution Monitoring:** Track echelon-level inventory vs demand alignment
4. **Store-Level Push:** Extend push allocation to Customer DC → Store link

## [0.19.1] - 2026-01-03

### Fixed
- **MEIO Inventory Position Bug:** Fixed echelon logic to use **Local IP** (DC inventory + in-transit) instead of **Echelon IP** (DC + all downstream stores). The original implementation caused Customer DCs to under-order because downstream store inventory inflated the IP calculation, making the system appear "well-stocked" even as stores depleted.
- **MRP Demand Signal Collapse:** Added POS demand as a floor for MRP demand signal. When order-based demand declines (due to downstream starvation), MRP now uses actual consumer demand (POS) to maintain production levels, preventing the death spiral where low orders → low production → more starvation.

### Analysis
Comprehensive diagnostic scripts created in `scripts/analysis/`:
- `diagnose_service_level.py` - Analyzes service degradation patterns by echelon, product, and time
- `diagnose_slob.py` - Analyzes inventory distribution, velocity, and SLOB (slow/obsolete) products

### Known Issues
- **365-Day Service Level: 73%** (target >90%) - Inventory accumulates at MFG RDCs (93% of total) while stores starve (4.5%). Customer DCs are ordering only 54% of demand despite MEIO fix.
- **Root Cause Identified:** The signal flow architecture creates a negative feedback loop where declining downstream inventory → declining orders → declining production. The MEIO/MRP fixes are applied but the 3-day ordering cycle and target_days parameters may need tuning.
- **Next Steps:** Consider (1) daily ordering for Customer DCs, (2) higher target_days, (3) push-based allocation from RDCs to supplement pull-based ordering.

## [0.19.0] - 2026-01-03

### Added
- **Echelon Inventory Logic (MEIO):** Implemented Multi-Echelon Inventory Optimization for Customer DCs. DCs now use aggregated downstream Echelon Inventory Position (DC + Stores) and Echelon Demand (POS) to trigger replenishment, resolving the "Signal Trap" where DCs stopped ordering when stores were empty.
- **Configuration:** Added `store_batch_size_cases` and `lead_time_history_len` to `simulation_config.json` to eliminate hardcoded values in `MinMaxReplenisher`.

### Changed
- **Replenishment Logic:** Customer DCs now bypass "Orders Received" signal and link directly to POS data via Echelon logic.
- **Refactoring:** Removed hardcoded overrides for store batch sizes and lead time history length in `replenishment.py` in favor of config-driven values.

## [0.18.2] - 2026-01-02

### Fixed
- **Order Signal Collapse:** Fixed demand signal attenuation at Customer DCs (RET-DC, etc.) by switching `Replenisher` to use a 7-day average of **Inflow Demand** (orders received) for all nodes, replacing the collapsing exponential smoothing logic.
- **Phantom Ingredient Replenishment:** Explicitly masked `ProductCategory.INGREDIENT` in `MinMaxReplenisher` to prevent Stores and DCs from ordering millions of units of raw materials (chemicals, packaging).
- **Sporadic Demand:** Increased store order cycle from 1 day to 3 days to consolidate demand signals and prevent "zero-demand" days from crashing safety stock calculations.

### Changed
- **Reverted v0.18.0 Band-aids:** Removed "Expected Throughput Floor" and "Flow-based Minimum Order" logic in favor of fixing the root cause (ingredient filtering + inflow demand).
- **Configuration:** Restored baseline inventory initialization days (21d Stores, 28d RDCs, 21d DCs).

## [0.18.0] - 2026-01-02

### Bug Fixes: Plant Shipment Routing & SLOB Calculation

This release fixes two critical bugs identified during 365-day simulation analysis.

### Fixed
- **Plant Shipment Routing Bug (Critical):** Plants were shipping production to ALL 44 DC nodes instead of just 4 manufacturer RDCs. Added `n_id.startswith("RDC-")` filter to `_ship_production_to_rdcs()` in `orchestrator.py:699-705`. This prevented 150M+ units of unauthorized PUSH shipments to customer DCs.
- **SLOB Calculation Bug:** SLOB (Slow/Obsolete) inventory was calculated as binary (0% or 100%) based on global days-of-supply. Fixed to per-SKU calculation: flags each SKU where `DOS > threshold`, then reports `sum(SLOB inventory) / total FG inventory`. Now correctly identifies which portion of inventory is slow-moving.

### Added (Experimental - May Revert)
- **Expected Throughput Floor for Customer DCs:** Customer DCs now use `max(inflow_demand, expected_throughput)` as their demand signal. `expected_throughput` is calculated by aggregating base demand from all downstream stores. This prevents cold-start under-ordering when stores haven't placed orders yet.
- **Flow-Based Minimum Order:** Customer DCs order at least `expected_throughput * lead_time` even when `IP > ROP`, to maintain inventory flow.

### Changed
- **Initialization Inventory Levels:** Reduced to prevent over-priming:
  - `customer_dc_days_supply`: 21 → 10 days
  - `store_days_supply`: 21 → 14 days
  - `rdc_days_supply`: 28 → 21 days

### Known Issues
- **365-Day Service Level Degradation:** Service level degrades from ~92% (30-day) to ~70% (365-day). The bullwhip effect is intentional realism, but the system should stabilize rather than degrade. Root cause under investigation - may be allocation/logistics bottleneck rather than replenishment policy.
- **Inventory Imbalance:** 93% of finished goods remain at RDCs, only 3% at stores. This suggests a flow bottleneck between RDCs and downstream nodes.

### Files Modified
- `src/prism_sim/simulation/orchestrator.py` - Plant shipment filter, SLOB calculation, pass base_demand_matrix
- `src/prism_sim/agents/replenishment.py` - Expected throughput floor, flow-based minimum
- `src/prism_sim/config/simulation_config.json` - Reduced initialization inventory levels

---

## [0.17.0] - 2026-01-02

### Physics Overhaul: First-Principles Supply Chain Physics

This major release overhauls the replenishment physics to use realized performance data and textbook inventory theory, moving away from heuristic safety stock.

### Added
- **Realized Lead Time Tracking (Phase 1):**
  - `Shipment` now tracks `original_order_day`.
  - `LogisticsEngine` captures the earliest order creation day during consolidation.
  - `Orchestrator` records realized lead time upon arrival.
  - `MinMaxReplenisher` maintains a rolling history of lead times per link.
- **Full Safety Stock Formula (Phase 2):**
  - Implemented the robust formula: $SS = z \sqrt{\bar{L}\sigma_D^2 + \bar{D}^2\sigma_L^2}$.
  - This formula protects against both **Demand Risk** (variability in sales) and **Supply Risk** (variability in logistics/fulfillment delays).
- **Dynamic ABC Segmentation (Phase 3):**
  - Products are dynamically classified every 7 days based on cumulative sales volume.
  - **A-Items (Top 80%):** Target 99% Service Level ($z=2.33$).
  - **B-Items (Next 15%):** Target 95% Service Level ($z=1.65$).
  - **C-Items (Bottom 5%):** Target 90% Service Level ($z=1.28$).
  - Optimizes inventory budget by prioritizing high-velocity items.

### Fixed
- **Zero Mypy Errors:** Resolved all type safety issues across the entire `src` directory.
- **Ruff Compliance:** Fixed over 100 style, complexity, and linting issues.
- **Hardcode Elimination:** Replaced multiple magic numbers with the config paradigm:
  - `order_cycle_days`: Configurable replenishment frequency.
  - `format_scale_factors`: Moved store format demand multipliers to `simulation_config.json`.
  - `stores_per_retailer_dc`: Moved to topology config in `world_definition.json`.
  - `mass_balance_min_threshold`: Moved to validation config.

### Technical Details
The system is now "self-healing." If a disruption occurs (e.g., port strike), the realized lead time variance ($\sigma_L$) will increase, causing the Replenisher to automatically raise safety stock buffers without manual intervention.

### Results (15-day simulation)
- Store Service Level: **97.9%** (Target >95% met)
- Inventory Turns: **7.9x** (Target ~8x met)
- System Stability: High (Physics laws strictly enforced with zero drift)

## [0.16.0] - 2026-01-02

### Service Level Fix: Physics-Based Multi-Phase Approach

This release implements a multi-phase fix for the 81% service level problem.

#### Phase 1: Inventory Position Fix for (s,S) Replenishment

This release implements the fundamental Inventory Position fix for (s,S) replenishment decisions. Per Zipkin's "Foundations of Inventory Management", (s,S) policies must use Inventory Position (On-Hand + In-Transit) rather than just On-Hand inventory to prevent double-ordering oscillation.

### Added
- **`get_in_transit_by_target()` (`state.py`):** New method calculates in-transit inventory per target node by aggregating quantities across all active shipments.
- **Variance-Aware Safety Stock (`replenishment.py`):** Implemented dynamic safety stock calculation based on demand variability ($ROP = \mu_L + z\sigma_L$).
  - Tracks rolling demand history per SKU via `record_demand()`.
  - Automatically adapts inventory buffers: popular/stable SKUs get lower relative safety stock, erratic/niche SKUs get higher buffers.
  - Replaces static "Days of Supply" heuristic which failed under Zipfian demand concentration.

### Changed
- **Replenishment (s,S) Decision (`replenishment.py`):**
  - Now uses Inventory Position (IP = On-Hand + In-Transit) for reorder point comparison
  - Order quantity calculated as Target Stock - IP (not Target - On-Hand)
  - This prevents double-ordering when shipments are already in transit
- **Manufacturing Targets (`simulation_config.json`):**
  - `target_days_supply`: 14 → 28 days
  - `reorder_point_days`: 7 → 21 days
  - Creates larger safety stock buffers at upstream nodes

### Removed
- **Legacy Tests:** Deleted low-value unit tests ("sham tests") that relied heavily on mocking without validating emergent behavior. The project now prioritizes full 365-day simulation runs to evaluate system stability and physics compliance.

### Technical Details

**The Double-Ordering Problem (Fixed):**
```
Before (v0.15.9):
  Store has 50 on-hand, 100 in-transit → compares 50 vs ROP → orders more
  Result: Double-ordering when shipments already cover the gap

After (v0.16.0):
  Store has 50 on-hand, 100 in-transit → IP = 150 → compares 150 vs ROP
  Result: No order if in-transit already covers replenishment need
```

### Files Modified
- `src/prism_sim/simulation/state.py`: Added `get_in_transit_by_target()` method
- `src/prism_sim/agents/replenishment.py`: Uses Inventory Position for (s,S) decisions, added variance tracking
- `src/prism_sim/simulation/orchestrator.py`: Passes daily demand to replenisher for history tracking
- `src/prism_sim/config/simulation_config.json`: Added safety stock parameters (`service_level_z`, `lead_time_days`)

#### Phase 2: Multi-Echelon Service Level Targets

Upstream nodes (Plants, RDCs) need higher inventory targets because end-to-end service level is the product of individual node service levels (0.95³ ≈ 85%).

### Changed
- **Manufacturing Targets (`simulation_config.json`):**
  - `target_days_supply`: 14 → 28 days
  - `reorder_point_days`: 7 → 21 days
  - Creates larger safety stock buffers at upstream nodes

### Validation
Service level stabilized at 76.20% in 365-day run with Zipfian demand enabled (System survives full year without collapse). Inventory turns at 4.73x. Further tuning of Z-scores required to reach >90% target.

### Future
- **Physics Overhaul (v0.17.0+):** Created `physics_overhaul.md` outlining a first-principles approach to fix the Service/Inventory paradox by instrumenting Effective Lead Time, implementing the full Safety Stock formula ($\sigma_L$), and adding dynamic ABC segmentation.

---

## [0.15.9] - 2026-01-01

### Service Level Improvement Phase 2: Demand Signal Fix

This release implements the core fix for demand signal attenuation identified in v0.15.8. Customer DCs now use **inflow-based demand** (orders received) instead of **outflow-based demand** (orders shipped), preventing the demand signal from being attenuated when DCs are short on inventory.

### Added
- **Inflow Tracking (`replenishment.py`):** New `inflow_history` array and methods to track orders received by each node:
  - `record_inflow(orders)`: Records orders received (pre-allocation)
  - `get_inflow_demand()`: Returns 7-day rolling average of inflow
  - Warm-start initialization to prevent cold-start issues
- **MRP Order Demand (`mrp.py`):** New `record_order_demand(orders)` method captures pre-allocation order quantities for production planning, preventing MRP from under-planning when supply chain is constrained.

### Changed
- **Customer DC Demand Signal (Critical Fix):**
  - Changed from outflow (what was shipped) to inflow (what was ordered)
  - This prevents demand attenuation cascade: when a DC is short on inventory, it now still sees the true demand from downstream stores
- **Customer DC Order Frequency:**
  - Reduced from 5-day cycle to 1-day cycle (daily ordering)
  - Creates smoother, more responsive demand signals upstream
- **Customer DC Replenishment Policy (Increased Buffers):**
  - B2M_LARGE: 14/10 → 21/14 days (target/ROP)
  - B2M_CLUB: 14/10 → 21/14 days
  - B2M_DISTRIBUTOR: 14/10 → 21/14 days
  - ECOMMERCE: 7/5 → 10/7 days
- **MRP Demand Signal:**
  - Now receives order quantities (pre-allocation) in addition to shipments
  - Uses `max(orders, shipments)` to capture true demand

### Technical Details

**The Demand Attenuation Problem (Fixed):**
```
Before (v0.15.8):
  Store requests 100 → DC ships 50 (low inv) → DC sees demand=50 → DC orders 50
  Result: 50% demand attenuation at each stage

After (v0.15.9):
  Store requests 100 → DC ships 50 (low inv) → DC sees demand=100 → DC orders 100
  Result: True demand propagates upstream
```

### Files Modified
- `src/prism_sim/agents/replenishment.py`: Inflow tracking, policy updates, daily DC ordering
- `src/prism_sim/simulation/orchestrator.py`: Record inflow, pass orders to MRP
- `src/prism_sim/simulation/mrp.py`: Order-based demand recording

### Validation
Requires 365-day simulation to validate service level improvement. Target: ≥95% (from 80.5%).

---

## [0.15.8] - 2026-01-01

### Service Level Improvement Phase 1 (75% → 80.5%)

This release improves store service level from 75.32% to 80.50% through policy tuning and demand signal fixes. Part of ongoing effort to reach 98.5% target.

### Fixed
- **ECOM FC Demand Signal (Critical):** ECOM FCs were classified as customer DCs, causing them to use outflow-based demand (which was 0 since they have no downstream stores). Now excluded from `customer_dc_indices` so they use POS demand correctly.

### Changed
- **Replenishment Policy (Increased Buffers):**
  - All channels: Target and ROP increased (~40-100% higher)
  - B2M_LARGE: 7/5 → 14/10 days (target/ROP)
  - B2M_CLUB: 10/7 → 14/10 days
  - ECOMMERCE: 5/3 → 7/5 days
  - Default: 10/7 → 14/10 days
- **Order Frequency:** Store order cycle reduced from 3 days to 1 day (daily ordering)
- **Initial Inventory Priming:**
  - Store days supply: 14 → 21 days
  - RDC days supply: 21 → 28 days
  - RDC-store multiplier: 50 → 500 (RDCs hold 500× store inventory)
  - Customer DC days supply: Added explicit config (21 days)

### Results (365-day simulation)
| Metric | v0.15.7 | v0.15.8 |
|--------|---------|---------|
| Store Service Level | 75.32% | **80.50%** |
| Inventory Turns | 6.18x | 5.11x |
| Manufacturing OEE | 82.0% | 81.9% |

### Known Issues
- **Service Level Gap:** Still 18pp below 98.5% target. Root cause identified as demand signal attenuation at customer DCs (orders based on outflow, not inflow).
- **SLOB Metric:** Shows 94.8% due to broken threshold logic (system-wide inventory > 60 days). Not a real problem.

### Next Steps
The plan to reach 98.5% service level involves changing customer DC demand signal from outflow-based (what was shipped) to inflow-based (what was ordered). This prevents demand attenuation as signals propagate upstream.

---

## [0.15.7] - 2026-01-01

### Fix Inventory Turns Calculation (Exclude Raw Materials)

This release fixes a critical metrics bug where inventory turns were calculated using ALL inventory (including raw materials at plants) instead of only finished goods.

### Fixed
- **Inventory Turns Calculation (Critical):** Inventory turns now correctly uses only finished goods (SKU-*) inventory in the denominator, excluding raw materials/ingredients at plants. Previously, 523M units of ingredients inflated the denominator, causing turns to show 0.23x instead of the actual ~6x.

### Added
- **Finished Goods Mask:** New `_build_finished_goods_mask()` method in Orchestrator creates a boolean mask to identify finished goods products (non-INGREDIENT category).

### Changed
- **SLOB Calculation:** Now uses finished goods inventory only (raw materials can't be "slow/obsolete" in the consumer sense).
- **Shrinkage Rate:** Now calculated against finished goods inventory for consistency.

### Results (365-day simulation)
| Metric | v0.15.6 | v0.15.7 |
|--------|---------|---------|
| Inventory Turns | 0.23x | **6.18x** ✓ |
| SLOB | 100% | 60.3% |
| Store Service Level | 73.34% | 75.32% |
| OEE | 78.6% | 82.0% ✓ |

### Technical Details
The inventory turns formula is: `Annual Sales / Average Inventory`

Previously, the denominator included all inventory across all nodes and products, including ~523M units of raw materials (chemicals, packaging) sitting at plants. This massively inflated the denominator, making turns appear artificially low (0.23x).

The fix creates a mask for finished goods products (24 SKUs out of 68 total products) and only sums inventory for those products when calculating turns.

---

## [0.15.6] - 2025-12-31

### MRP Demand Signal Stabilization & Production Floor

This release eliminates production collapse in 365-day simulations by improving MRP demand signal handling and adding a minimum production floor.

### Fixed
- **Production Collapse (Critical):** Production dropped to zero for days 252-279 due to demand signal dampening. When stores had sufficient inventory, order volumes dropped, which reduced the MRP shipment signal, causing MRP to calculate high Days-of-Supply and skip production orders. Eventually stores depleted, triggering massive bullwhip (35M orders on day 281).

### Added
- **Demand Velocity Tracking:** MRP now tracks week-over-week demand trends. If week 1 demand falls below 60% of week 2, the fallback signal is activated to prevent cascading decline.
- **Minimum Production Floor (30%):** When production orders fall below 30% of expected demand, MRP either scales up existing orders or creates new orders for top products, ensuring production never stops completely.

### Changed
- **MRP Signal Threshold:** Raised the collapse detection threshold from 10% to 40% of expected demand (`mrp.py:228`). The previous threshold was too low to catch gradual decline.
- **Smoothing Window:** Extended demand and production history from 7 to 14 days for smoother signals and better trend detection.

### Results (365-day simulation)
| Metric | v0.15.5 | v0.15.6 |
|--------|---------|---------|
| Store Service Level | 69.95% | **73.34%** |
| Manufacturing OEE | 62.0% | **78.6%** |
| Production Days 252-279 | **0** (collapsed) | 227K-484K |
| Total Inventory | 603M | 587M |

### Technical Details
The production collapse occurred because:
1. Stores had sufficient inventory after initial bullwhip → ordered less
2. Lower orders → lower shipments → lower MRP demand signal (40-50% of expected)
3. Previous 10% threshold didn't trigger fallback
4. MRP calculated high DOS (inventory / low_demand) → skipped production
5. Days 252-279: zero production → stores depleted → Day 281: 35M order explosion

The fix uses three mechanisms:
1. **Higher threshold (40%)** catches gradual decline earlier
2. **Velocity tracking** detects week-over-week declining trends
3. **Production floor (30%)** ensures minimum production regardless of signal

---

## [0.15.5] - 2025-12-31

### LTL Mode for Store Deliveries & Service Level Improvement

This release introduces LTL (Less Than Truckload) shipping mode for store deliveries, improving service level from 83% to 92.8%.

### Fixed
- **Fragmented Store Orders:** Stores ordered 20-40 cases but FTL required 300-1200 cases (5-20 pallets). Orders were held indefinitely for consolidation, hurting service level and causing low truck fill.

### Added
- **LTL Shipping Mode:** Differentiated shipping modes based on destination:
  - **FTL (Full Truckload):** DC-to-DC shipments maintain pallet minimums for consolidation
  - **LTL (Less Than Truckload):** DC-to-Store shipments ship immediately with minimum 10 cases
- **Config Options:** `store_delivery_mode`, `ltl_min_cases`, `default_ftl_min_pallets` in `simulation_config.json`
- **FTL/LTL Tracking:** `LogisticsEngine` now tracks `ftl_shipment_count` and `ltl_shipment_count`

### Changed
- **LogisticsEngine:** `create_shipments()` now checks if target is a Store and applies LTL mode (no pallet minimum)
- **Default FTL Minimum:** Routes without channel-specific rules now use `default_ftl_min_pallets` (10) instead of 0

### Results
| Metric | v0.15.4 | v0.15.5 |
|--------|---------|---------|
| Service Level (90-day) | 83% | **92.8%** |
| Truck Fill Rate | 15% | 4.2% |
| Manufacturing OEE | 81% | 83% |

### Note on Truck Fill Rate
The truck fill rate dropped because:
1. Most shipments are now LTL to stores (intentionally small)
2. FMCG products "cube out" (fill by volume) before "weighting out" (fill by weight)
3. Weight-based truck fill isn't appropriate for light, bulky products
4. With LTL, service level is the better metric for overall performance

---

## [0.15.4] - 2025-12-31

### Bullwhip Dampening & Service Level Stabilization

This release eliminates the Day 2 bullwhip cascade (272M → 100K orders) and stabilizes service level over 90-day simulations.

### Fixed
- **Customer DC Bullwhip Cascade (Critical):** Customer DCs now use derived demand (allocation outflow) instead of POS demand, which was floored to 0.1. Added warm start from POSEngine equilibrium estimate and order staggering (5-day cycle) for customer DCs.
- **MRP Ingredient Ordering Explosion:** Capped ingredient ordering at 2x expected demand to prevent bullwhip-driven explosions. Reduced ACTIVE_CHEM inventory policy from 30/45 days to 7/14 days ROP/target.
- **Customer DC Inventory Priming:** Customer DCs now initialize with inventory scaled by downstream store count, preventing Day 1 mass-ordering.

### Changed
- **Replenisher:** Added `warm_start_demand` parameter, `record_outflow()` method for derived demand tracking, and order staggering for customer DCs.
- **Orchestrator:** Passes warm start demand to replenisher and records allocation outflow.
- **Config:** Reduced ACTIVE_CHEM policy to 7/14 days (was 30/45).

### Results
| Metric | v0.15.3 | v0.15.4 |
|--------|---------|---------|
| Day 2 Orders | 272M | 100K |
| Service Level (90-day) | 80% | 83% |
| Truck Fill Rate | 8% | 15% |
| Manufacturing OEE | 92% | 81% |

### Technical Details
The bullwhip cascade occurred because:
1. Customer DCs (RET-DC, DIST-DC, ECOM-FC) don't generate POS demand
2. Their demand signal was floored to 0.1, creating tiny ROP/targets
3. When stores ordered on Day 1, DC inventory dropped below the tiny ROP
4. All DCs mass-ordered on Day 2 to refill

The fix uses MRP theory's "derived demand" concept: customer DCs track orders they fulfill (allocation outflow) as their demand signal, not consumer POS.

---

## [0.15.3] - 2025-12-31

### Mass Balance Audit Fix & Service Level Optimization

This release fixes the mass balance tracking issue and improves service level from 58% to 86%.

### Fixed
- **Mass Balance FTL Timing Mismatch (Option B1):** Replaced `shipments_out` tracking with `allocation_out` to fix false violations at customer DCs. Inventory decrements are now tracked at allocation time (when they actually occur) rather than at shipment creation time (which could be delayed by FTL consolidation).
- **Floating-Point Noise Filtering:** Added minimum absolute difference threshold (1.0 case) to filter negligible floating-point violations caused by floor guards.

### Changed
- **AllocationAgent:** Now returns `AllocationResult` dataclass containing both `allocated_orders` and `allocation_matrix` for mass balance tracking.
- **PhysicsAuditor:** New methods `record_allocation_out()` and `record_plant_shipments_out()` replace the single `record_shipments_out()` method.
- **Initial Inventory Levels:** Increased `store_days_supply` from 7 to 14 days, `rdc_days_supply` from 14 to 21 days to prevent synchronized reordering.
- **Replenishment Policy:** Tightened ROP-Target gap across all channels (e.g., default ROP 4→7 days) for smaller, more frequent orders.
- **Order Staggering:** Stores now order on different days based on hash(node_id) to spread orders across 3-day cycle, reducing bullwhip amplitude.

### Results
| Metric | v0.15.2 | v0.15.3 |
|--------|---------|---------|
| Service Level (30-day) | 58% | 86% |
| Manufacturing OEE | 62% | 88% |
| Simulation Time (30-day) | 10s | 2.4s |

### Known Issues
- Day 2 bullwhip (271M orders) still occurs from customer DC cascade
- Service level declines over longer runs (80% at 60 days)
- Truck fill rate remains low (8%) due to fragmented orders

### Technical Details
The FTL consolidation timing mismatch occurred because:
1. Allocation decrements inventory immediately when orders are created
2. Logistics can HOLD orders if they don't meet FTL minimum pallet thresholds
3. `shipments_out` was only recorded when shipments were actually created (potentially days later)
4. Mass balance equation couldn't account for "allocated but not yet shipped" inventory

The fix tracks inventory decrements at the source (allocation) rather than reconstructing from downstream flows.

---

## [0.15.2] - 2025-12-30

### Ingredient Replenishment Fix

This release fixes the critical ingredient replenishment mismatch that caused 365-day simulations to collapse on days 362-365 due to ingredient exhaustion.

### Fixed
- **Ingredient Replenishment Signal (Critical):** `generate_purchase_orders()` now uses production-based signal (active production orders) instead of POS demand for ingredient ordering. Previously, ingredient replenishment was based on POS demand (~400k/day), but actual consumption was driven by production orders amplified by bullwhip (~5-6M/day), causing a net burn rate of ~1,380 units/day shortfall.

### Changed
- **Orchestrator:** Updated to pass `active_production_orders` to `generate_purchase_orders()` instead of `daily_demand`.

### Results
| Metric | v0.15.1 (Collapse) | v0.15.2 |
|--------|-------------------|---------|
| Service Level | 52.54% | 58.16% |
| Manufacturing OEE | 55.1% | 61.8% |
| Production Day 365 | 0 (collapse) | 259,560 cases |
| System Survival | Collapsed day 362-365 | Full year |

---

## [0.15.1] - 2025-12-30

### MRP Inventory Position Fix

This release fixes the critical MRP inventory position bug that caused 94 zero-production days in 365-day simulations.

### Fixed
- **MRP Inventory Position (Critical):** `_cache_node_info()` now only includes manufacturer RDCs (`RDC-*` prefix) in inventory position calculation. Previously included ALL `NodeType.DC` nodes including customer DCs (RET-DC, DIST-DC, ECOM-FC) with ~4.5M units, inflating Days of Supply to 11.5 days > ROP 7 days, preventing production orders.
- **C.5 Smoothing History Bug:** Production order history now records post-scaled (actual) quantities instead of pre-scaled values, preventing rolling average inflation.

### Known Issues
- **Mass Balance Violations:** Customer DCs show mass balance violations due to FTL consolidation timing mismatch. Allocation decrements inventory immediately, but logistics can hold orders for FTL consolidation. This is an accounting/auditing issue, not an actual physics violation.

### Results
| Metric | v0.15.0 | v0.15.1 |
|--------|---------|---------|
| Service Level | 51.6% | 60.19% |
| Manufacturing OEE | 44.9% | 88.2% |
| Zero-Production Days | 94 | 0 |

---

## [0.15.0] - 2025-12-30

### Phase C: System Stabilization

This release fixes the critical "death spiral" bug that caused 365-day simulations to collapse around day 22-27. The system now survives full-year simulations without collapse.

### Fixed
- **Death Spiral Prevention (C.1):** Added expected demand fallback in `MRPEngine`. When RDC→Store shipment signals collapse below 10% of expected demand, MRP now uses expected demand as a floor to continue generating production orders.
- **Supplier-Plant Routing (C.2):** Fixed `_find_supplier_for_ingredient()` to verify link exists before routing SPOF ingredient. Previously returned SUP-001 for all plants even when no link existed.
- **Production Smoothing (C.5):** Added 7-day rolling average tracking for production orders with 1.5x cap on daily volatility to reduce wild swings.

### Changed
- **Production Capacity (C.3):** Increased `production_hours_per_day` from 20 to 24 hours (3-shift 24/7 operation) to ensure capacity exceeds demand.
- **Realistic Initial Inventory (C.4):** Reduced starting inventory to realistic levels:
  - Store days of supply: 14 → 7 days
  - RDC days of supply: 21 → 14 days
  - RDC-store multiplier: 100 → 50
  - Raw material inventory remains high (10M units) to isolate finished goods dynamics.

### Results
| Metric | Pre-Fix | Post-Fix | Target |
|--------|---------|----------|--------|
| System Collapse | Day 27 | Never | Never ✅ |
| Service Level | 8.8% | 51.6% | >90% |
| Production (day 365) | 0 | 148k+ | >0 ✅ |
| OEE | 1.8% | 44.9% | 75-85% |

---

## [0.14.0] - 2025-12-30

### Phase A: Capacity Rebalancing & Option C Architecture

This release implements Phase A of the brownfield digital twin stabilization plan, scaling demand/capacity to realistic FMCG industry benchmarks (400-500k cases/day) and implementing hierarchical DC + Store architecture.

### Added
- **Option C Network Architecture:**
  - Hierarchical structure: DCs (logistics layer) + Stores (POS layer)
  - 20 Retailer DCs with ~100 stores each (B2M_LARGE)
  - 8 Distributor DCs with 500 small retailers each (B2M_DISTRIBUTOR)
  - 30 Club stores direct to RDC (B2M_CLUB)
  - 10 Ecom FCs and 2 DTC FCs
  - Total: ~6,600 nodes (vs 155 aggregated DCs before)
- **Store Format Scale Factors:** SUPERMARKET=1.0, CONVENIENCE=0.5, CLUB=15.0, ECOM_FC=50.0
- **Phase A Implementation Notes** section in `fix_plan_v2.md` documenting all changes and learnings

### Changed
- **Capacity Scaling (2.5x increase):**
  - Run rates: ORAL 3000→7500, PERSONAL 3600→9000, HOME 2400→6000 cases/hour
  - Production hours: 8→20 hours/day (3-shift operation)
  - Plant efficiency factors leveled to 78-88% OEE range
- **Demand Calibration:**
  - Base daily demand: 1.0→7.0 cases/SKU/store
  - Target network demand: ~420k cases/day (aligned with multi-category CPG operations)
- **Seasonality & Promos:**
  - Seasonality amplitude: ±20%→±12% (staple category realism)
  - Black Friday lift multiplier: 3.0x→2.0x
- **Inventory Priming:**
  - Store days supply: 28→14 days
  - RDC-store multiplier: 1500→100 (fixed over-priming issue)

### Fixed
- **Demand Double-Counting:** DCs (RETAILER_DC, DISTRIBUTOR_DC) no longer generate POS demand; only stores do
- **Network Generator Bugs:** `sample_company()`→`sample_companies(1)[0]`, `sample_city()`→`sample_cities(1)[0]`
- **CSV Loader:** Added parsing for `channel`, `store_format`, `parent_account_id` enums in `builder.py`

### Known Issues (Phase B Required)
- Bullwhip effect still extreme (orders explode 60k→12.9M by day 20)
- Production stops ~day 20 due to bullwhip-induced collapse
- OEE at 39% (below target 75-85%), expected given order chaos

---

## [0.13.0] - 2025-12-29

### Realism & Architecture Overhaul (Phases 0-5)
This release implements a massive overhaul of the simulation physics and network structure to align with FMCG industry standards (P&G/Colgate benchmarks).

### Added
- **Customer Channel Structure (Fix 0A):**
  - Added `CustomerChannel` (B2M_LARGE, CLUB, DISTRIBUTOR, ECOM) and `StoreFormat` enums.
  - Implemented channel-specific logistics rules (FTL vs. LTL) and minimum order quantities.
- **Packaging Hierarchy (Fix 0B):**
  - Added `PackagingType` and `ContainerType` to support realistic SKU variants (Tubes, Bottles, Pumps).
  - Procedurally generated SKU variants based on packaging profiles.
- **Order Types (Fix 0C):**
  - Added `OrderType` (STANDARD, RUSH, PROMOTIONAL, BACKORDER) with priority handling in Allocation.
- **Promo Calendar (Fix 0D):**
  - Ported vectorized `PromoCalendar` from reference implementation for realistic demand lift and hangover effects.
- **Risk Events (Fix 5):**
  - Implemented full `RiskEventManager` with 5 scenarios: Contamination, Port Strike, Supplier Opacity, Cyber Outage, Carbon Tax.
- **Behavioral Quirks (Fix 6):**
  - Added 6 behavioral pathologies: `BullwhipWhipCrack`, `SingleSourceFragility`, `DataDecay`, `PortCongestion`, `OptimismBias`, `PhantomInventory`.
- **Sustainability Metrics (Fix 10):**
  - Added Scope 3 CO2 emissions tracking in `LogisticsEngine` (0.1 kg/ton-km).
  - Added `emissions_kg` tracking to Shipments.
- **Expanded KPIs (Fix 8):**
  - Added trackers for Perfect Order Rate, Cash-to-Cash Cycle, MAPE, Shrinkage Rate, and SLOB %.

### Changed
- **Orchestrator:**
  - Updated MRP signal to use **RDC-to-Store Shipments** (true pull signal) instead of allocated orders to fix inverse bullwhip.
  - Integrated `RiskEventManager` and `QuirkManager` into the daily loop.
  - Updated `Triangle Report` to include new KPIs.
- **Configuration:**
  - Overhauled `simulation_config.json` with comprehensive settings for all new engines.
  - Updated `benchmark_manifest.json` with strict validation targets (OSA 93-99%, Turns 8-15x).
  - Increased inventory initialization targets (Store: 28d, RDC: 35d) to prevent drain.

## [0.12.3] - 2025-12-29

### Added
- **Bullwhip Analysis Script:** Added `scripts/analyze_bullwhip.py` for analyzing variance amplification across supply chain echelons. Calculates CV ratios (Store → RDC → Plant), detects inverse bullwhip anomalies, and reports production oscillation patterns.
- **Scenario Comparison Script:** Added `scripts/compare_scenarios.py` for comparing two simulation runs (e.g., baseline vs risk events). Shows shipment volume differences around key disruption days with day-by-day breakdown.

### Fixed
- **Inverse Bullwhip Effect:** Resolved the "Inverse Bullwhip" anomaly (where upstream variance was lower than downstream) by updating `MRPEngine` to use a 7-day moving average of **actual RDC shipments** (lumpy signal) instead of smoothed POS demand proxies. This restores realistic demand amplification upstream.
- **Low OEE (28% -> 99%):** Fixed massive plant over-capacity by:
  - Reducing `production_hours_per_day` from 24 to 8 (single shift).
  - Increasing `batch_size_cases` in `simulation_config.json` to 100.
  - Increasing `min_production_qty` logic in `mrp.py` to 2x net requirement.
- **Service Level Metrics:** Introduced **Store Service Level (OSA)** tracking in `Orchestrator` and `RealismMonitor` to correctly measure On-Shelf Availability (Actual Sales / Demand). The legacy "Fill Rate" metric was skewed by massive ingredient orders hitting supplier capacity caps.

### Changed
- **Inventory Policy Tuning:** Increased `replenishment` target days (21d) and reorder points (10d) in `simulation_config.json` to support higher service levels.
- **Reporting:** Updated "The Triangle Report" in `Orchestrator` to feature Store Service Level as the primary Service metric.

## [0.12.2] - 2025-12-29

### Fixed
- **Negative Inventory Physics Violation:** Eliminated all sources of negative inventory that violated the Inventory Positivity law.
  - **Demand Consumption:** Updated `Orchestrator` to constrain sales to available actual inventory (lost sales model). Previously subtracted demand blindly.
  - **Allocation Agent:** Changed `AllocationAgent` to use `actual_inventory` instead of `perceived_inventory` for fill ratio calculations. When phantom inventory quirk creates divergence, allocation now respects physical reality.
  - **Material Consumption:** Updated `TransformEngine._consume_materials()` to check and consume from `actual_inventory`, constraining consumption to available amounts.
  - **Shrinkage Quirk:** Fixed `PhantomInventoryQuirk` to base shrinkage on actual inventory and constrain shrink amount to prevent negatives.
  - **State Guards:** Added `np.maximum(0, ...)` floor guards in `StateManager.update_inventory()` and `update_inventory_batch()` to catch floating point precision errors.

### Added
- **Analysis Script:** Added `scripts/analyze_results.py` for reusable simulation output analysis. Supports JSON output and handles large inventory files via chunked reading.

### Changed
- **Category Demand Balancing:** Equalized `base_daily_demand` to 1.0 for all product categories (ORAL_CARE, PERSONAL_WASH, HOME_CARE).
- **Plant Flexibility:** Added ORAL_CARE to PLANT-CA's `supported_categories` for better capacity balancing.

### Results
- **Before fix:** 3.4M cases backlog, 1,161 cells with negative inventory (min: -212.86)
- **After fix:** 0 cases backlog, 0 cells with negative inventory

## [0.12.1] - 2025-12-29

### Changed
- **Simulation Physics Tuning:**
  - **SPOF Isolation:** Updated `hierarchy.py` to restrict `ACT-CHEM-001` (SPOF ingredient) to only ~20% of the portfolio (Premium Oral Care), reducing systemic vulnerability.
  - **Tiered Inventory Policies:** Updated `mrp.py` and `simulation_config.json` to support granular ROP/Target levels. Ingredients now use "Commodity" (7d/14d) vs. "Specialty" (30d/45d) policies.
  - **Supplier Capacity:** Updated `NetworkGenerator` to set infinite capacity for bulk suppliers while constraining the SPOF supplier (`SUP-001`) to 500k units/day.
  - **Allocation Logic:** Updated `AllocationAgent` to respect finite supplier capacity limits using Fair Share logic.
  - **Initialization:** Increased Store/RDC initialization days (14d/28d) and implemented robust Plant ingredient seeding (5M units) to prevent cold-start starvation.
- **Service Level Tracking:**
  - Updated `RealismMonitor` and `Orchestrator` to track `Service Level (LIFR)` based on `Shipped / Ordered`, replacing the legacy backlog-based index.

### Fixed
- **MRP Initialization Bug:** Fixed `AttributeError` in `MRPEngine` caused by accessing uninitialized cache variables in `_build_policy_vectors`.

### Identified
- **The Bullwhip Crisis:** 365-day simulation revealed a catastrophic feedback loop where "Fill or Kill" logic combined with zero inventory triggers infinite reordering (460k -> 66M orders/day). Mitigation plan documented in `docs/planning/sim_tuning.md`.

## [0.12.0] - 2025-12-29

### Fixed
- **MRP Physics Fix:** Implemented a "Look-ahead Horizon" (reorder point window) for `in_production_qty` calculation. This prevents the MRP engine from overestimating available supply based on distant future production orders, which previously led to "Pipeline Silence" and systemic inventory collapse.
- **Fill-or-Kill Logic:** Refactored `AllocationAgent` to implement "Fill or Kill" (Cut) logic for retailer-to-DC orders. Unfulfilled quantities are now marked as `CLOSED` rather than remaining `OPEN`. This aligns with high-velocity FMCG industry standards and prevents exponential computational backlog growth during stockouts.

### Added
- **Simulation Tuning Plan:** Created `docs/planning/sim_tuning.md` to document the strategy for stabilizing the "Deep NAM" network, including Pareto-based SPOF isolation and tiered reorder points.

## [0.11.0] - 2025-12-29

### Added
- **Streaming Writers (Task 7.3):** Implemented incremental disk writes for 365-day runs without memory exhaustion.
  - **StreamingCSVWriter:** Writes rows directly to disk as they arrive, preventing memory accumulation.
  - **StreamingParquetWriter:** Batches rows and flushes periodically for optimal compression (requires `pyarrow`).
  - **CLI Flags:** Added `--streaming`, `--format`, and `--inventory-sample-rate` to `run_simulation.py`.
  - **Config-Driven:** Writer settings can be configured in `simulation_config.json` under `writer` section.
- **Inventory Sampling:** Added `inventory_sample_rate` parameter to reduce inventory data volume (e.g., log weekly instead of daily).

### Changed
- **SimulationWriter:** Refactored to support both buffered (legacy) and streaming modes with backward compatibility.
- **Orchestrator:** Now reads writer configuration from `simulation_config.json` with CLI override support.

### Performance
- **30-day streaming test:** 549K orders, 557K shipments, 1.6M inventory records written incrementally in 10.4s.
- **Memory efficiency:** Streaming mode eliminates in-memory accumulation for high-volume tables.

### Fixed
- **SPOF Config Alignment:** Updated SPOF ingredient from hardcoded `ING-SURF-SPEC` to procedural `ACT-CHEM-001` to match Task 8.1 world builder overhaul.
- **Test Maintenance:** Updated `test_world_builder.py` to validate procedural ingredient patterns instead of hardcoded IDs. Added new tests: `test_ingredients_generated`, `test_spof_ingredient_exists`.
- **Mass Balance Test:** Fixed `test_mass_balance_detects_leak` to inject proportionally larger leaks for reliable drift detection.

## [0.10.0] - 2025-12-28

### Architecture Overhaul
- **World Builder Overhaul (Task 8.1):** Transitioned from hardcoded "2-ingredient" logic to a fully procedural ingredient generation system.
  - **Procedural Ingredients:** `ProductGenerator` now creates Packaging (Bottles, Caps, Boxes) and Chemicals (Actives, Bulk Base) dynamically based on `world_definition.json` profiles.
  - **Semantic Recipes:** BOMs are now logic-driven (e.g., "Liquid" = Bottle + Cap + Label + Base + Active) rather than random.
- **Vectorized MRP (Task 8.2):** Implemented `RecipeMatrixBuilder` to convert object-based BOMs into a dense NumPy matrix ($\mathbf{R}$) for $O(1)$ dependency lookups.
  - **Matrix Algebra:** `MRPEngine` now calculates ingredient requirements via vector-matrix multiplication ($\mathbf{req} = \mathbf{d} \cdot \mathbf{R}$), enabling instant planning for thousands of SKUs.
  - **Vectorized Transform:** Refactored `TransformEngine` to use direct tensor operations for material feasibility checks and consumption updates.
- **Mass Balance Physics Audit (Task 6.6):** Implemented a rigorous validation gate to enforce the conservation of mass.
  - **Flow Tracking:** Added `DailyFlows` to track every inventory movement (Sales, Receipts, Shipments, Production, Consumed, Shrinkage).
  - **Audit Loop:** `PhysicsAuditor` now calculates expected inventory levels daily and flags "Drift" violations if actual state diverges from physics laws.

### Added
- **Recipe Matrix:** New core component `src/prism_sim/network/recipe_matrix.py` for handling dense BOM structures.
- **Configurable Profiles:** Added `ingredient_profiles` and `recipe_logic` to `world_definition.json` to control the procedural generation of supply chains.

### Changed
- **Performance:** Simulation speed validated at ~4.6s for 30 days of the full 4,500-node network despite 10x increase in BOM complexity.
- **Refactoring:** Removed hardcoded ingredients from `generate_static_world.py` and `hierarchy.py`.

## [0.9.8] - 2025-12-28

### Added
- **LLM Context Guide:** Added `docs/llm_context.md`, a consolidated "One-Pager" for developers and AI assistants, covering physics laws, architecture, and core components.
- **Dynamic Priming:** Implemented `get_average_demand_estimate` in `POSEngine` to calculate realistic initial inventory levels based on actual demand profiles, replacing the hardcoded `1.0` case/day proxy.
- **Strict Logistics Constraints:** Enhanced `LogisticsEngine` to raise `ValueError` if an item physically exceeds truck dimensions, preventing silent data errors.

### Fixed
- **Logistics Crash:** Resolved `ValueError` in `LogisticsEngine` caused by `fit_qty` calculating to zero for fractional remaining quantities. The engine now supports partial case packing for "Fair Share" allocation scenarios.
- **Linting & Types:** Resolved ~100 `ruff` errors (E501, F841, D200) and `mypy` strict type errors in `mrp.py` and `builder.py`.
- **Code Refactoring:**
  - Simplified `MRPEngine.generate_production_orders` by extracting inventory position logic.
  - Reduced argument complexity in `Orchestrator._print_daily_status` and `_record_daily_metrics`.
  - Refactored `TransformEngine._process_single_order` to reduce return statements.

### Documentation
- **API Docs:** Configured `mkdocs` to auto-generate API reference pages from source docstrings upon build.
- **Navigation:** Updated `mkdocs.yml` to include the new "LLM Context" guide.

## [0.9.7] - 2025-12-28

### Fixed
- **Service Index Reporting:** Fixed scaling issue where `backlog_penalty_divisor` was too small (1,000 vs 100,000) for the Deep NAM network, causing Service Index to floor at 0%.
- **Cash Metric Reporting:** Implemented `Inventory Turns` tracking in `RealismMonitor` and `Orchestrator`. Now correctly reports Annualized Inventory Turns (previously 0.00x).
- **Serialization Error:** Fixed `TypeError: float32 is not JSON serializable` in `RealismMonitor` by ensuring metric accumulators store native Python floats.

### Documentation
- **Quick Start:** Updated `README.md` with clear instructions on running the simulation via `run_simulation.py` and locating generated artifacts.

### Added
- **CLI Runner:** Added `run_simulation.py` with `argparse` support for `--days`, `--no-logging`, and `--output-dir`. Replaced the rigid `run_benchmark.py`.

## [0.9.6] - 2025-12-28

### Fixed
- **Systemic Inventory Collapse:** Resolved the core physics bottleneck that caused production to stall and inventory to drain.
  - **Inventory Sync:** Fixed `AllocationAgent` and `TransformEngine` to correctly update both `actual` and `perceived` inventory, eliminating "phantom stock" and reporting errors.
  - **Material Deadlock:** Modified `TransformEngine` to check material availability only for the quantity that can be produced in the current day, preventing large orders from blocking production due to partial ingredient shortages.
  - **Batching Latency:** Implemented daily batch creation for partial production in `TransformEngine`, reducing the lead time penalty of large MOQs.
  - **MRP Robustness:** Increased `target_days_supply` (28d) and `reorder_point_days` (14d) in `simulation_config.json` to provide a sufficient buffer against lead time and demand variability.
  - **MRP Visibility:** Updated `MRPEngine` to include Plant-level finished goods in the Inventory Position calculation.

### Changed
- **OEE Stabilization:** Achieved stable 73% OEE (Target: 65-85%) across the Deep NAM network.
- **Logistics Efficiency:** Doubled Truck Fill Rate to ~52% by increasing store replenishment targets.

## [0.9.5] - 2025-12-28

### Fixed
- **Structural Capacity Deficit:** Tuned manufacturing physics to resolve the ~160k production cap vs ~230k demand:
  - **MOQ Increase:** Raised `min_production_qty` from 5k to 25k to amortize changeover penalties.
  - **Changeover Reduction:** Reduced changeover times in `world_definition.json` (0.5-1.5h) to improve plant OEE.
  - **Efficiency Boost:** Increased plant `efficiency_factor` to 0.95.
- **Initialization Bias:** Updated `simulation_config.json` to use "Steady State" midpoints for inventory initialization (Stores: 5d, RDCs: 10.5d) instead of max capacity, eliminating the artificial "destocking" period.

### Changed
- **Config-Driven Hierarchy:** Refactored `generators/hierarchy.py` and `scripts/generate_static_world.py` to load product run rates and dimensions from `world_definition.json`, removing all hardcoded "magic numbers".
- **Benchmark Config:** Reduced simulation horizon to 90 days and enabled full CSV logging by default for deeper debugging.

## [0.9.4] - 2025-12-27

### Fixed
- **Structural Deficit Resolution:** Resolved system-wide inventory collapse where production could not meet demand.
  - **MRP Logic:** Updated `MRPEngine` to use Inventory Position (On Hand + Pipeline) instead of just On Hand, eliminating redundant orders and the Bullwhip effect.
  - **Ingredient Replenishment:** Implemented `generate_purchase_orders` in `MRPEngine` to replenish raw materials at plants, preventing production halts due to ingredient exhaustion.
  - **Partial Production:** Fixed `TransformEngine` to correctly handle partial production days and calculate remaining capacity, ensuring plants don't lose time on large orders.
- **Capacity Tuning:** Doubled theoretical run rates in `recipes.csv` and increased `production_hours_per_day` to 24.0 to ensure the network has sufficient capacity to meet the ~225k case/day demand.

### Added
- **Plant-Specific Configuration:** Implemented granular control over plant capabilities in `simulation_config.json`:
  - **Efficiency & Downtime:** Configurable `efficiency_factor` (e.g., 0.70-0.95) and `unplanned_downtime_pct` per plant.
  - **Product Restrictions:** Added `supported_categories` to restrict which plants can produce specific product categories (e.g., only PLANT-TX makes Oral Care).

## [0.9.3] - 2025-12-27

### Fixed
- **Zero Orders Reporting Bug:** Corrected `Orchestrator` logging to calculate "Ordered" quantity *before* the `AllocationAgent` modifies orders in-place. This reveals the true unconstrained demand signal (approx. 11M cases/day) instead of zero.
- **Production Starvation:** Increased `initial_plant_inventory` for raw materials (`ING-BASE-LIQ`, `ING-SURF-SPEC`) from 5k to 10M units to prevent immediate SPOF (Single Point of Failure) triggering. Plants now successfully produce ~100k cases/day to meet demand.
- **Cold Start Service Collapse:** Fixed RDC initialization logic in `Orchestrator`. RDCs now initialize with 1500x the base inventory (approx. 30k cases/sku) to cover network aggregate demand, preventing immediate stockouts.
- **System Priming (Configurable):** Added `initialization` block to `simulation_config.json` allowing precise control over Store and RDC start-up inventory (Days of Supply). This moves the simulation from a "Cold Start" to a "Warm Start" state.
- **OEE Tracking Implementation:** Updated `TransformEngine` and `Orchestrator` to calculate and record plant-level capacity utilization (OEE). OEE now correctly reports in the Supply Chain Triangle Report.
- **Reporting Cleanup:** Clamped Service Index reporting to 0-100% range to avoid confusing negative values during backlog recovery phases.

## [0.9.2] - 2025-12-27

### Changed
- **Performance Optimization (Replenisher):** Implemented O(1) Store-to-Supplier lookup caching in `MinMaxReplenisher` to resolve O(N*L) performance bottleneck (22M iterations/day).
- **Config Adjustment:** Reduced `base_daily_demand` to ~1.0 case/day/store and `initial_fg_level` to 5.0 to align simulation physics with realistic NAM benchmarks.
- **Debugging:** Added extensive debug instrumentation to `Replenishment` agent and `Orchestrator` to diagnose order silence.

## [0.9.1] - 2025-12-27

### Changed
- **Performance Optimization (Task 7.3):** Implemented `enable_logging` flag in `SimulationWriter` and `Orchestrator`.
  - Default behavior now skips expensive I/O and in-memory list appending when logging is disabled.
  - `run_benchmark.py` updated to run in "In-Memory Validation Mode" (logging=False) for accurate speed measurement.

## [0.9.0] - 2025-12-27

### Added
- **Deep NAM Integration (Task 7.3):** fully integrated the 4,500-node static world into the runtime simulation.
  - **CSV Loading:** `WorldBuilder` now automatically loads `products.csv`, `locations.csv`, and `links.csv` from `data/output/static_world` if present.
  - **Dynamic Demand:** Updated `POSEngine` to generate demand based on `ProductCategory` rather than hardcoded string matching, enabling support for generated SKUs (e.g., `SKU-ORAL-001`).
  - **Test Suite Updates:** Refactored `test_milestone_3`, `test_state_manager`, and `test_world_builder` to validate against the massive network topology.

## [0.8.0] - 2025-12-27

### Added
- **Deep NAM Static Generators (Task 2.1, 2.2):** Implemented high-performance generators for massive scale simulation:
  - **ProductGenerator:** Generates 50+ SKUs across 3 categories with Zipfian popularity and realistic physical dimensions.
  - **NetworkGenerator:** Generates a 4,500-node retail network using Barabási-Albert preferential attachment for hub-and-spoke realism.
  - **StaticDataPool:** Vectorized Faker sampling for O(1) attribute generation.
  - **Distributions:** Statistical helpers for Zipf and Power-Law network topology.
- **Static World Writer:** Implemented `StaticWriter` to export Levels 0-4 (Products, Recipes, Locations, Partners, Links) to CSV format.
- **World Generation Script:** Added `scripts/generate_static_world.py` to automate the creation of the 4,500-node Deep NAM environment.

## [0.7.0] - 2025-12-27

### Added
- **MkDocs Documentation:** Implemented comprehensive documentation system with:
  - **mkdocs-material** theme with dark/light mode, navigation tabs, and search
  - **mkdocstrings[python]** for automatic API reference generation from docstrings
  - **mkdocs-gen-files** for auto-generating reference pages from source code
  - **Architecture diagrams** using Mermaid (system overview, data flow, component interactions)
  - **Getting Started** guides (installation, quick start)
  - **Changelog integration** via pymdownx.snippets
- **World Definition Config (`world_definition.json`):** Separated static world data (products, network topology, recipes) from runtime simulation parameters.
- **Semgrep Rule (`.semgrep/detect_literals.yaml`):** Added custom rule to detect hardcoded literals in source code.

### Changed
- **Config-Driven Architecture Enforcement:** Eliminated all remaining hardcoded values from source code:
  - **Deleted `constants.py`:** Moved `EPSILON` and `WEEKS_PER_YEAR` to `simulation_config.json` under `global_constants`.
  - **Refactored `builder.py`:** Now reads products, recipes, and network topology from `world_definition.json` instead of hardcoding.
  - **Updated `allocation.py`:** Receives config and reads `epsilon` from `global_constants`.
  - **Updated `demand.py`:** Reads seasonality (amplitude, phase, cycle) and noise (gamma_shape, gamma_scale) from config.
  - **Updated `logistics.py`:** Reads `epsilon_weight_kg` and `epsilon_volume_m3` from config.
  - **Updated `orchestrator.py`:** Reads scoring weights from config for triangle report.
  - **Updated `quirks.py`:** Added configurable `cluster_delay_min/max_hours` and `shrinkage_factor_min/max`.

## [0.6.0] - 2025-12-27

### Added
- **Simulation Writer (Task 7.2):** Implemented `SimulationWriter` to export SCOR-DS compatible datasets (Orders, Shipments, Batches, Inventory) to CSV/JSON.
- **Triangle Report (Task 7.3):** Added automated generation of "The Triangle Report," summarizing Service (Fill Rate), Cost (Truck Fill), and Cash (Inventory Turns) performance.
- **Reporting Infrastructure:** Integrated data collection directly into the `Orchestrator` loop for seamless end-of-run reporting.

## [0.5.0] - 2025-12-26

### Added
- **Validation Framework (Task 6.1, 6.6):** Implemented comprehensive validation in `src/prism_sim/simulation/monitor.py`:
  - **WelfordAccumulator:** O(1) streaming mean/variance calculation for real-time statistics.
  - **MassBalanceChecker:** Physics conservation tracking (input = output + scrap).
  - **RealismMonitor:** Online validator for OEE (65-85%), Truck Fill (>85%), SLOB (<30%), Inventory Turns (6-14x), Cost-per-Case ($1-3).
  - **PhysicsAuditor:** Mass balance, inventory positivity, and kinematic consistency checks.
- **Resilience Metrics (Task 6.2):** Implemented `ResilienceTracker` for TTS (Time-to-Survive) and TTR (Time-to-Recover) measurement during disruptions per Simchi-Levi framework.
- **Behavioral Quirks (Task 6.3):** Implemented realistic supply chain pathologies in `src/prism_sim/simulation/quirks.py`:
  - **PortCongestionQuirk:** AR(1) auto-regressive delays creating clustered late arrivals (coefficient=0.70, clustering when delay >4h).
  - **OptimismBiasQuirk:** 15% over-forecast for new products (<6 months old).
  - **PhantomInventoryQuirk:** 2% shrinkage with 14-day detection lag (dual inventory model).
  - **QuirkManager:** Unified interface for all quirks.
- **Risk Scenarios (Task 6.4):** Implemented `RiskEventManager` in `src/prism_sim/simulation/risk_events.py` to trigger deterministic disruptions:
  - **Port Strike (RSK-LOG-002):** 4x logistics delay.
  - **Cyber Outage (RSK-CYB-004):** 10x logistics delay.
- **Legacy Validation (Task 6.5):** Ported validation checks from reference implementation:
  - Pareto distribution check (top 20% SKUs = 75-85% volume).
  - Hub concentration check (Chicago ~20-30%).
  - Named entities verification.
  - Bullwhip ratio check (Order CV / POS CV = 1.5-3.0x).
  - Referential integrity checks.
- **Metrics Dataclasses:** Added `ProductionMetrics` and `ShipmentMetrics` for clean parameter passing.

### Changed
- **Config Paradigm Enforcement:** Refactored `Orchestrator`, `RiskEventManager`, and `QuirkManager` to eliminate hardcoded simulation parameters, moving them to `simulation_config.json`.
- **Code Quality:** All ruff and mypy strict checks pass for new modules.
- **Inventory State:** Expanded `StateManager` to track `perceived_inventory` vs `actual_inventory` to support phantom inventory simulation.

## [0.4.1] - 2025-12-26

### Changed
- **Config-Driven Values:** Moved all magic numbers to `simulation_config.json` per CLAUDE.md directives:
  - `calendar.weeks_per_year` (52)
  - `manufacturing.recall_batch_trigger_day` (30)
  - `manufacturing.default_yield_percent` (98.5)
  - `manufacturing.spof.warning_threshold` (10.0)
  - `logistics.default_lead_time_days` (3.0)
- **Refactored Allocation Agent:** Extracted helper methods (`_group_orders_by_source`, `_calculate_demand_vector`, `_calculate_fill_ratios`, `_apply_ratios_to_orders`) to reduce branch complexity.
- **Refactored Demand Engine:** Extracted `_apply_multiplier_to_cells` helper and created `PromoConfig` dataclass to reduce branch/argument complexity.
- **Code Quality:** Fixed all ruff linting issues (D200 docstrings, E501 line length, PLR1714 comparisons, RUF059 unused vars, PLC0415 imports).

### Fixed
- All ruff checks now pass with zero warnings.
- All mypy strict type checks pass (20 source files).

## [0.4.0] - 2025-12-26

### Added
- **MRP Engine (Task 5.1):** Implemented `MRPEngine` in `src/prism_sim/simulation/mrp.py` to translate DRP (Distribution Requirements Planning) into Production Orders for Plants.
- **Transform Engine (Task 5.2):** Implemented `TransformEngine` in `src/prism_sim/simulation/transform.py` with full production physics:
  - **Finite Capacity:** Enforces `run_rate_cases_per_hour` from Recipe definitions.
  - **Changeover Penalty:** Implements Little's Law friction when switching products, consuming capacity based on `changeover_time_hours`.
  - **Batch Tracking:** Creates `Batch` records with ingredient genealogy for traceability.
  - **Deterministic Batch:** Schedules `B-2024-RECALL-001` contaminated batch as per roadmap.
- **SPOF Simulation (Task 5.3):** Implemented raw material constraints for `ING-SURF-SPEC` (Specialty Surfactant). Production fails when ingredients are unavailable.
- **Network Expansion:** Added Plant nodes (`PLANT-OH`, `PLANT-TX`), backup supplier (`SUP-SURF-BACKUP`), and complete supplier-plant-RDC link topology.
- **Manufacturing Primitives:** Added `ProductionOrder`, `ProductionOrderStatus`, `Batch`, and `BatchStatus` to `network/core.py`.
- **Testing:** Added comprehensive integration tests in `tests/test_milestone_5.py` covering MRP, finite capacity, changeover penalties, SPOF, and recall batch scheduling.

### Changed
- **Orchestrator:** Extended daily loop to include MRP planning and production execution steps.
- **WorldBuilder:** Added recipes for all finished goods (Toothpaste, Soap, Detergent).
- **Configuration:** Added `manufacturing` section to `simulation_config.json` with production parameters and SPOF settings.

## [0.3.0] - 2025-12-26

### Added
- **Allocation Agent:** Implemented `AllocationAgent` in `src/prism_sim/agents/allocation.py` to handle inventory scarcity using "Fair Share" logic.
- **Logistics Engine:** Created `LogisticsEngine` in `src/prism_sim/simulation/logistics.py` to simulate physical bin-packing ("Tetris") for trucks, enforcing Weight vs. Cube constraints.
- **Transit Physics:** Implemented `Shipment` tracking and transit delays in `Orchestrator`, replacing "Magic Fulfillment" with realistic lead times.
- **Network Primitives:** Added `Shipment` and `ShipmentStatus` to `src/prism_sim/network/core.py`.
- **Testing:** Added comprehensive integration tests in `tests/test_milestone_4.py` covering allocation, bin-packing, and transit delays.

## [0.2.0] - 2025-12-26

### Added
- **Orchestrator:** Implemented the daily time-stepper loop in `src/prism_sim/simulation/orchestrator.py`.
- **Demand Engine:** Created `POSEngine` and `PromoCalendar` in `src/prism_sim/simulation/demand.py`, porting the vectorized "Lift & Hangover" physics directly from the `fmcg_example_OLD` reference to ensure high-fidelity demand signal generation.
- **Replenishment Agent:** Implemented `MinMaxReplenisher` in `src/prism_sim/agents/replenishment.py` to simulate store-level ordering and trigger the Bullwhip effect.
- **Network Expansion:** Added `Order` and `OrderLine` primitives to `network/core.py` and instantiated Retail Stores in `WorldBuilder`.
- **Testing:** Added integration tests for POS demand, promo lifts, and replenishment logic in `tests/test_milestone_3.py`.

## [0.1.1] - 2025-12-26

### Maintenance
- Enforced code quality standards using `ruff`, `mypy`, and `semgrep`.
- Fixed linting errors, unused imports, and type mismatches across the codebase.
- Added `mypy` to project dependencies.
- Refactored hardcoded simulation parameters into `simulation_config.json` to enforce the configuration paradigm.

## [0.1.0] - 2025-12-26

### Added
- **Core Primitives:** Implemented `Node` (RDC, Supplier) and `Link` (Route) classes in `src/prism_sim/network/core.py`.
- **Product Physics:** Implemented `Product` class with Weight/Cube attributes and `Recipe` for BOMs in `src/prism_sim/product/core.py`.
- **World Builder:** Created `WorldBuilder` to deterministically generate the "Deep NAM" network and product portfolio (Soap, Toothpaste, Detergent).
- **State Management:** Implemented `StateManager` using `numpy` for vectorized, O(1) inventory tracking.
- **Configuration:** Ported legacy `benchmark_manifest.json` to `src/prism_sim/config/`.
- **Testing:** Added unit tests for core primitives, world builder, and state manager.
- `physics.md`: Core reference for Supply Chain Physics theory and validation rules.
- Initial project structure and documentation.
