# Prism Sim

A high-fidelity supply chain simulation engine built on the principles of Supply Chain Physics.

## Core Principles

This project adheres strictly to the laws of Supply Chain Physics to ensure kinematic consistency and mass balance in all simulations. See [physics.md](docs/planning/physics.md) for the theoretical framework and validation rubric.

## Architecture

The simulation uses a hybrid **Discrete-Event Simulation (DES)** architecture with a modular component design:

- **`src/prism_sim/network`**: Defines the physical topology (Nodes, Links, Orders).
- **`src/prism_sim/product`**: Defines the physical goods (Products, Recipes) with Weight/Cube constraints.
- **`src/prism_sim/simulation`**: Contains the `Orchestrator`, `POSEngine`, and `StateManager` for vectorized physics.
- **`src/prism_sim/agents`**: Pluggable replenishment and planning logic (e.g., `MinMaxReplenisher`).
- **`src/prism_sim/config`**: Manages configuration and the "Gold Standard" benchmark manifest.

## Key Frameworks
- Little's Law
- VUT Equation (Kingman's Formula)
- Mass Balance
- Signal Resonance (Bullwhip Effect)
