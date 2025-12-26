# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
