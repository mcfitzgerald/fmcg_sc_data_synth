# Prism Sim

**A high-fidelity supply chain simulation engine built on the principles of Supply Chain Physics.**

Prism Sim is a **Digital Twin** framework capable of simulating large-scale supply chain networks (4,500+ nodes) with kinematic consistency. It models the flow of goods, information, and cash from Raw Material Suppliers to Retail Stores, enforcing strict mass balance and capacity constraints at every step.

It allows you to:
- **Simulate Physics:** Mass balance, finite capacity, changeover penalties, and logistics bin-packing.
- **Inject Pathologies:** Model realistic behaviors like "Phantom Inventory", "Port Congestion", and "Optimism Bias".
- **Stress Test:** Trigger risk events (Port Strikes, Cyber Outages, SPOF failures) and measure Time-to-Recover (TTR).
- **Optimize:** Tune plant efficiency, downtime, and sourcing logic to find the "efficient frontier" of Service, Cost, and Cash.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Poetry** and **Python 3.12+** installed.

```bash
# Install dependencies
poetry install
```

### 2. Run a 365-Day Simulation
Execute the benchmark script to run a full year of simulation for the "Deep NAM" (North American Market) network. This will simulate ~82 million cases of demand across 4,500 stores, 4 RDCs, 4 Plants, and 50 Suppliers.

```bash
poetry run python run_benchmark.py
```

**What to expect:**
- **Runtime:** ~1-2 minutes (depending on hardware).
- **Output:** Daily status logs showing Demand, Orders, Shipments, Production, and Inventory.
- **Final Report:** A "Supply Chain Triangle" report summarizing Service (Fill Rate), Cost (Truck Fill), and Cash (Inventory Turns).

### 3. View Results
Artifacts are saved to `data/output/sim_<timestamp>/`:
- `triangle_report.txt`: The executive summary.
- `orders.csv`, `shipments.csv`, `batches.csv`: Transactional logs (SCOR-DS format).
- `inventory.csv`: Weekly inventory snapshots.

## ⚙️ Configuration

Prism Sim is entirely config-driven. You can tune the physics without touching code.

### Simulation Parameters
**File:** `src/prism_sim/config/simulation_config.json`

- **Manufacturing:**
  - `production_hours_per_day`: Set shift length (e.g., 24.0).
  - `plant_parameters`: Configure specific plants (e.g., `PLANT-OH` efficiency: 0.70).
  - `efficiency_factor` & `unplanned_downtime_pct`: Global defaults.
- **Logistics:**
  - `truck_max_weight_kg`: Bin-packing constraints.
- **Quirks:**
  - Enable/Disable `phantom_inventory`, `optimism_bias`, etc.

### World Definition
**File:** `src/prism_sim/config/world_definition.json` or `data/output/static_world/*.csv`

- Define the Network Topology (Nodes, Links).
- Define the Product Catalog and Recipes (BOMs).
- Define Demand Profiles.

## 📚 Documentation

Comprehensive documentation is available via MkDocs.

```bash
# Install docs dependencies
poetry install --with docs

# Serve documentation locally (http://127.0.0.1:8000)
poetry run mkdocs serve
```

### Key Topics
- **[Supply Chain Physics](docs/planning/physics.md):** The theoretical framework (Little's Law, Kingman's Formula).
- **Architecture:** System design and component interaction.
- **Resilience:** How risks and disruptions are modeled.

## 🏗️ Architecture

The simulation uses a hybrid **Discrete-Event Simulation (DES)** architecture:

- **`src/prism_sim/network`**: Physical topology (Nodes, Links).
- **`src/prism_sim/simulation`**: The `Orchestrator` time-stepper and physics engines (`TransformEngine`, `LogisticsEngine`).
- **`src/prism_sim/agents`**: Decision-making logic (`MRPEngine`, `AllocationAgent`).
- **`src/prism_sim/config`**: Configuration management.

## ✅ Code Quality

We adhere to strict engineering standards:
- **Linting:** `ruff` (pylint, pycodestyle, etc.)
- **Typing:** `mypy` (strict mode)
- **Security:** `semgrep` (hardcode detection)

Run checks locally:
```bash
poetry run pytest
poetry run ruff check .
poetry run mypy .
```