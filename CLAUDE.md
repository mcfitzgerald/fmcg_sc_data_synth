# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prism Sim is a high-fidelity supply chain Digital Twin built on Discrete-Event Simulation (DES) and Supply Chain Physics. It simulates 4,500+ nodes across a North American FMCG network, enforcing mass balance, capacity constraints, and realistic behavioral quirks.

### Critical Documentation

**Read `docs/llm_context.md` first** - Contains comprehensive architecture guide, file-to-concept map, state tensor shapes, daily loop sequence, and debugging checklist.

## Prime Directives

1. **Always read `docs/planning/intent.md` and `docs/planning/roadmap.md`** before starting work
2. **Always use `poetry run` for all Python execution** in this project
3. **Never skip or ignore bugs** - if tests fail or something seems off, alert the user and find root cause together
4. **Commit and push** after completing new code, bug fixes, or features
5. **Don't reinvent the wheel** - search for robust libraries, prefer simple (but complete and correct) implementations over complex ones, don't over-engineer

## Commands

```bash
# Install dependencies
poetry install

# Run simulation (default 90 days)
poetry run python run_simulation.py
poetry run python run_simulation.py --days 365 --output-dir data/results/custom
poetry run python run_simulation.py --days 30 --no-logging  # Fast mode
poetry run python run_simulation.py --days 50 --no-logging  # Quick sanity check

# Streaming mode (required for 365-day logged runs)
poetry run python run_simulation.py --days 365 --streaming --format parquet
poetry run python run_simulation.py --days 365 --streaming --format parquet --inventory-sample-rate 1

# Linting and type checking
poetry run ruff check .
poetry run ruff check . --fix  # Auto-fix
poetry run mypy .
```

## Architecture

The simulation uses a hybrid DES architecture with a daily time-stepper orchestrator:

```
src/prism_sim/
├── simulation/
│   ├── orchestrator.py   # Main time-loop controller - coordinates all engines
│   ├── state.py          # StateManager - vectorized numpy tensors for inventory
│   ├── world.py          # World object - holds all nodes, links, products
│   ├── builder.py        # WorldBuilder - constructs World from config
│   ├── demand.py         # POSEngine - generates daily consumer sales (promo + seasonality)
│   ├── logistics.py      # LogisticsEngine - bin-packing, shipments, transit physics
│   ├── mrp.py            # MRPEngine - material requirements planning (vectorized)
│   ├── transform.py      # TransformEngine - manufacturing physics (capacity, changeover)
│   ├── quirks.py         # QuirkManager - behavioral realism (phantom inventory, bias)
│   ├── risk_events.py    # RiskEventManager - disruption scenarios (port strikes, cyber)
│   ├── monitor.py        # RealismMonitor, PhysicsAuditor, ResilienceTracker
│   └── writer.py         # SimulationWriter, ThreadedParquetWriter - streaming data export
├── agents/
│   ├── allocation.py     # AllocationAgent - Fair Share allocation, Fill-or-Kill
│   └── replenishment.py  # MinMaxReplenisher - (s,S) policy, creates Bullwhip behavior
├── network/
│   ├── core.py           # Node, Link, Order, Shipment, Batch primitives
│   └── recipe_matrix.py  # RecipeMatrixBuilder - vectorized BOM for O(1) MRP
├── product/core.py       # Product, Recipe, ProductCategory definitions
├── config/
│   ├── loader.py         # Config loading utilities
│   └── *.json            # simulation_config, world_definition, benchmark_manifest
├── generators/           # Static world generation (4,500 nodes)
│   ├── hierarchy.py      # ProductGenerator - SKUs, ingredients, recipes
│   ├── network.py        # NetworkGenerator - nodes, links, topology
│   ├── distributions.py  # Zipf, power-law distributions
│   └── static_pool.py    # Faker-based attribute sampling
└── writers/              # Static world generation export
    ├── base.py           # BaseWriter abstract class
    └── static_writer.py  # StaticWriter for world generation output
```

**Key Data Flow (per day):**
1. `POSEngine` generates demand → 2. `Replenisher` creates orders → 3. `AllocationAgent` allocates inventory → 4. `LogisticsEngine` builds shipments → 5. `MRPEngine` plans production → 6. `TransformEngine` executes batches → 7. `QuirkManager` injects realism

## Spec-Driven Development

Key reference documents in `docs/planning/`:
- `intent.md` - Technical source of truth (physics, ontology, validation targets)
- `roadmap.md` - Implementation sequence with Task IDs (initial build complete)

Reference Task IDs in commits when working on roadmap items. For bug fixes and enhancements, use conventional commit format (e.g., `fix(mrp):`, `feat(allocation):`).

## Configuration

All simulation parameters are config-driven (no hardcodes):
- `src/prism_sim/config/simulation_config.json` - Manufacturing, logistics, quirk parameters
- `src/prism_sim/config/world_definition.json` - Network topology, products, BOMs
- `src/prism_sim/config/benchmark_manifest.json` - Risk scenarios, validation targets

Use `semgrep` to detect hardcoded values.

## Physics Laws (Non-Negotiable)

The simulation enforces these constraints - violations indicate bugs:
1. **Mass Balance:** Input (kg) = Output (kg) + Scrap
2. **Kinematic Consistency:** Travel time = Distance / Speed (no teleporting)
3. **Little's Law:** Inventory = Throughput × Flow Time
4. **Capacity Constraints:** Cannot produce more than Rate × Time
5. **Inventory Positivity:** Cannot ship what you don't have

## Engineering Standards

- Use `ruff` and `mypy --strict` for all code
- Create and maintain robust documentation for all code
- Use full simulations (50-365 days) for integration testing - check Triangle Report metrics
- Split files at 700-1000 lines
- Use vectorized numpy operations for performance (loops banned for heavy lifting)
- No hardcoded values - use config files; run `semgrep` to detect violations
- Update `CHANGELOG.md`, `README.md`, `docs/`, and `pyproject.toml` on changes (semantic versioning)
- Use Context7 MCP tools for library documentation
- Unless noted otherwise, do not plan or code for backwards compatibility

## Reference Code

`reference/fmcg_example_OLD/` contains legacy stochastic implementation. Consult for logic patterns but scrutinize before reusing - it lacks proper physics enforcement.
