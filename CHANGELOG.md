# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-25

### Added
- **Core Primitives:** Implemented `Node` (RDC, Supplier) and `Link` (Route) classes in `src/prism_sim/network/core.py`.
- **Product Physics:** Implemented `Product` class with Weight/Cube attributes and `Recipe` for BOMs in `src/prism_sim/product/core.py`.
- **World Builder:** Created `WorldBuilder` to deterministically generate the "Deep NAM" network and product portfolio (Soap, Toothpaste, Detergent).
- **State Management:** Implemented `StateManager` using `numpy` for vectorized, O(1) inventory tracking.
- **Configuration:** Ported legacy `benchmark_manifest.json` to `src/prism_sim/config/`.
- **Testing:** Added unit tests for core primitives, world builder, and state manager.
- `physics.md`: Core reference for Supply Chain Physics theory and validation rules.
- Initial project structure and documentation.
