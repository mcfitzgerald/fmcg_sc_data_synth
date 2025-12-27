# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Legacy Validation (Task 6.5):** Ported validation checks from reference implementation:
  - Pareto distribution check (top 20% SKUs = 75-85% volume).
  - Hub concentration check (Chicago ~20-30%).
  - Named entities verification.
  - Bullwhip ratio check (Order CV / POS CV = 1.5-3.0x).
  - Referential integrity checks.
- **Metrics Dataclasses:** Added `ProductionMetrics` and `ShipmentMetrics` for clean parameter passing.

### Changed
- **Code Quality:** All ruff and mypy strict checks pass for new modules.

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
