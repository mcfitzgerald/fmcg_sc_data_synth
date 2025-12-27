# Installation

This guide covers how to install and configure Prism Sim for local development.

## Prerequisites

- **Python 3.12+** - Required for type hints and performance features
- **Poetry** - Dependency management and virtual environment

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/prism-sim.git
cd prism-sim
```

### 2. Install Dependencies

Prism Sim uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
# Install all dependencies (including dev tools)
poetry install

# To include documentation dependencies
poetry install --with docs
```

### 3. Verify Installation

```bash
# Run the test suite to verify everything works
poetry run pytest
```

## Project Structure

```
prism-sim/
├── src/prism_sim/           # Main package
│   ├── agents/              # Decision-making agents
│   ├── config/              # Configuration loading
│   ├── network/             # Network topology (Nodes, Links)
│   ├── product/             # Product definitions & recipes
│   ├── simulation/          # Core simulation engines
│   └── constants.py         # Shared constants
├── tests/                   # Integration tests
├── data/                    # Simulation data files
├── docs/                    # Documentation
├── simulation_config.json   # Main configuration file
└── pyproject.toml           # Project configuration
```

## Configuration Files

Prism Sim uses a **config-driven paradigm** - all simulation parameters are externalized:

### `simulation_config.json`

The main configuration file containing:

- **Network topology** - RDCs, suppliers, retailers
- **Product definitions** - SKUs, weight/cube attributes
- **Policy parameters** - Reorder points, safety stock levels
- **Simulation settings** - Duration, random seed, risk events

Example structure:

```json
{
  "simulation": {
    "duration_days": 365,
    "seed": 42,
    "start_date": "2024-01-01"
  },
  "network": {
    "rdcs": [...],
    "suppliers": [...],
    "retailers": [...]
  },
  "products": [...],
  "policies": {
    "replenishment": {...},
    "allocation": {...}
  }
}
```

### `benchmark_manifest.json`

Defines the "Gold Standard" benchmark including:

- Named risk scenarios (port strikes, SPOF events)
- Validation thresholds
- Target KPIs

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRISM_CONFIG_PATH` | Path to config file | `./simulation_config.json` |
| `PRISM_DATA_DIR` | Output data directory | `./data/` |
| `PRISM_LOG_LEVEL` | Logging verbosity | `INFO` |

## Code Quality Tools

The project enforces strict quality standards:

```bash
# Run linting
poetry run ruff check src/

# Run type checking
poetry run mypy src/

# Format code
poetry run ruff format src/
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to run your first simulation.
