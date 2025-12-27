# Prism Sim

A high-fidelity supply chain simulation engine built on the principles of Supply Chain Physics.

## Environment Notes

Run all python through poetry

## Project Status
**Current Milestone:** Milestone 7 (Final Delivery) - *In Progress*
- **Completed (Milestone 6):** Validation Framework, Resilience Metrics, Behavioral Quirks, Risk Scenarios, Legacy Validation.
- **In Progress (Milestone 7):** Final 365-day run, SCOR-DS Data Export, "Triangle Report" generation.

## Core Principles

This project adheres strictly to the laws of Supply Chain Physics to ensure kinematic consistency and mass balance in all simulations. See [physics.md](docs/planning/physics.md) for the theoretical framework and validation rubric.

## Architecture

The simulation uses a hybrid **Discrete-Event Simulation (DES)** architecture with a modular component design:

- **`src/prism_sim/network`**: Defines the physical topology (Nodes, Links, Orders).
- **`src/prism_sim/product`**: Defines the physical goods (Products, Recipes) with Weight/Cube constraints.
- **`src/prism_sim/simulation`**: Contains the `Orchestrator`, `POSEngine`, and `StateManager` for vectorized physics.
- **`src/prism_sim/simulation/monitor.py`**: **Realism Framework** (Validation, Resilience, Physics Audit).
- **`src/prism_sim/simulation/quirks.py`**: **Behavioral Engine** (Port Congestion, Optimism Bias, Phantom Inventory).
- **`src/prism_sim/agents`**: Pluggable replenishment and planning logic (e.g., `MinMaxReplenisher`).
- **`src/prism_sim/config`**: Manages configuration and the "Gold Standard" benchmark manifest.

## Key Frameworks
- Little's Law
- VUT Equation (Kingman's Formula)
- Mass Balance
- Signal Resonance (Bullwhip Effect)

## Code Quality

All code passes strict quality checks:
- **ruff**: Full linting with pylint, pycodestyle, pyflakes, pydocstyle rules
- **mypy**: Strict type checking enabled
- **semgrep**: Security and hardcode detection
- **Config-driven**: No magic numbers - all parameters in `simulation_config.json`

## Documentation

Full documentation is available via MkDocs. To build and serve locally:

```bash
# Install docs dependencies
poetry install --with docs

# Serve documentation locally (http://127.0.0.1:8000)
poetry run mkdocs serve

# Build static site
poetry run mkdocs build
```

Documentation includes:
- **Getting Started**: Installation and quick start guides
- **Architecture**: System design with mermaid diagrams
- **API Reference**: Auto-generated from docstrings
- **Supply Chain Physics**: Theoretical foundations
