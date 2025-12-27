# Quick Start

This guide walks you through running your first Prism Sim simulation.

## Running a Simulation

### Basic Simulation Run

The simplest way to run a full 365-day simulation:

```bash
poetry run python run_benchmark.py
```

This will:

1. Initialize the simulation world (network, products, agents)
2. Run 365 daily time-steps
3. Generate SCOR-DS output files
4. Create a Triangle Report summary

### Expected Output

```
Initializing Prism Digital Twin...
Starting 365-Day 'Deep NAM' Simulation Run...

Simulation completed in 45.23 seconds.

Generating Artifacts...

================================================================================
                     PRISM DIGITAL TWIN - TRIANGLE REPORT
================================================================================
Service Level:     94.2%
Inventory Turns:   8.3x
Fill Rate:         91.5%
...

Triangle Report saved to data/simulation_output/triangle_report.txt
```

## Programmatic Usage

You can also run simulations programmatically:

```python
from prism_sim.simulation.orchestrator import Orchestrator

# Initialize the simulation
sim = Orchestrator()

# Run for a specific number of days
sim.run(days=90)  # Quarter simulation

# Access results
sim.save_results()  # Export to CSV/JSON
report = sim.generate_triangle_report()
print(report)
```

### Custom Configuration

Load a custom configuration file:

```python
from prism_sim.simulation.orchestrator import Orchestrator

# Use custom config path
sim = Orchestrator(config_path="my_custom_config.json")
sim.run(days=365)
```

## Understanding the Output

### Output Directory Structure

After running, find results in `data/simulation_output/`:

```
data/simulation_output/
├── orders.csv              # All orders generated
├── shipments.csv           # Shipment records
├── inventory_daily.csv     # Daily inventory snapshots
├── production_orders.csv   # Manufacturing orders
├── batches.csv             # Batch genealogy
└── triangle_report.txt     # Summary metrics
```

### Key Metrics in Triangle Report

The Triangle Report summarizes the three competing dimensions:

| Dimension | Key Metrics |
|-----------|-------------|
| **Service** | Fill Rate, On-Time Delivery, Stockout Days |
| **Cost** | Freight Cost, Production Cost, Cost-to-Serve |
| **Cash** | Inventory Turns, Days of Supply, Working Capital |

## Injecting Risk Scenarios

The simulation supports deterministic risk injection:

```python
from prism_sim.simulation.orchestrator import Orchestrator

sim = Orchestrator()

# Risk scenarios are defined in benchmark_manifest.json
# They trigger automatically based on configured day ranges
sim.run(days=365)
```

Common risk scenarios include:

- **Port Strike**: 14-day USLAX shutdown (affects West RDC)
- **SPOF Event**: Specialty surfactant supplier outage
- **Cyber Event**: WMS outage at DTC fulfillment center

## Validation

Verify simulation physics are working correctly:

```bash
# Run the test suite
poetry run pytest tests/ -v

# Run specific milestone tests
poetry run pytest tests/test_milestone_6.py -v
```

The tests validate:

- Little's Law compliance ($WIP = TH \times CT$)
- Mass balance integrity
- Bullwhip effect emergence
- Resilience metric calculation (TTS/TTR)

## Next Steps

- Read the [Architecture](../architecture/index.md) guide to understand the system design
- Explore the [API Reference](../reference/prism_sim/index.md) for detailed module documentation
- Review [Supply Chain Physics](../planning/physics.md) for theoretical background
