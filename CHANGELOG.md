# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
