# Prism Sim

A high-fidelity supply chain simulation engine built on the principles of **Supply Chain Physics**.

Prism Sim is a Discrete-Event Simulation (DES) framework that generates realistic supply chain data by modeling the fundamental physical laws that govern inventory flow, production capacity, and logistics constraints.

## Key Features

<div class="grid cards" markdown>

-   :material-scale-balance:{ .lg .middle } **Physics-Based Simulation**

    ---

    Built on Little's Law, VUT Equation, and Mass Balance principles to ensure kinematically consistent data generation.

-   :material-truck-delivery:{ .lg .middle } **Logistics Realism**

    ---

    Models weight/cube constraints, truck fill optimization, and multi-echelon distribution networks.

-   :material-factory:{ .lg .middle } **Manufacturing Physics**

    ---

    Simulates production lines with OEE, changeover penalties, yield tracking, and batch genealogy.

-   :material-chart-timeline-variant:{ .lg .middle } **Behavioral Quirks**

    ---

    Includes realistic human factors: optimism bias, phantom inventory, and bullwhip amplification.

</div>

## The Supply Chain Triangle

Prism Sim forces the fundamental trade-off between:

- **Service**: Fill rate, on-time delivery
- **Cost**: Freight, production, warehousing
- **Cash**: Inventory investment, working capital

Every decision in the simulation affects this balance, creating realistic tension in the generated data.

## Core Frameworks

The simulation is validated against five non-negotiable physical laws:

| Framework | Description |
|-----------|-------------|
| **Little's Law** | $WIP = TH \times CT$ - The fundamental constraint |
| **VUT Equation** | Non-linear cycle time as utilization increases |
| **Mass Balance** | $I_t = I_{t-1} + Receipts - Shipments$ |
| **Signal Resonance** | Bullwhip effect amplification upstream |
| **Capacity Constraints** | Physical limits on production and logistics |

## Quick Links

- [Getting Started](getting-started/index.md) - Install and configure Prism Sim
- [Quick Start](getting-started/quickstart.md) - Run your first simulation
- [Architecture](architecture/index.md) - Understand the system design
- [API Reference](reference/prism_sim/index.md) - Module documentation

## Project Status

**Current Version:** 0.6.0

| Milestone | Status |
|-----------|--------|
| Core Network & Product Models | Complete |
| Simulation Orchestrator | Complete |
| Demand Generation & POS Engine | Complete |
| Manufacturing Physics (Transform) | Complete |
| Behavioral Quirks & Risk Scenarios | Complete |
| Validation Framework | Complete |
| Final Benchmarking & Reporting | In Progress |
