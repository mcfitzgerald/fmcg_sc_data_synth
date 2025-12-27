# Architecture Overview

Prism Sim is a **Discrete-Event Simulation (DES)** framework designed to generate high-fidelity supply chain data that adheres to the fundamental laws of Supply Chain Physics.

## System Architecture

The system follows a modular, layered architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "Simulation Layer"
        O[Orchestrator]
        O --> DE[Demand Engine<br/>POSEngine]
        O --> TE[Transform Engine<br/>Production]
        O --> LE[Logistics Engine]
        O --> MRP[MRP Engine]
        O --> RE[Risk Events]
        O --> QE[Quirks Engine]
    end

    subgraph "Agent Layer"
        RA[Replenishment Agent<br/>MinMaxReplenisher]
        AA[Allocation Agent]
    end

    subgraph "Core Layer"
        SM[State Manager]
        WB[World Builder]
    end

    subgraph "Domain Models"
        N[Network<br/>Nodes, Links]
        P[Products<br/>Recipes]
        C[Config<br/>Loader]
    end

    O --> RA
    O --> AA
    O --> SM
    WB --> N
    WB --> P
    WB --> C
    SM --> N
```

## Package Structure

```mermaid
graph LR
    subgraph prism_sim
        config[config/]
        network[network/]
        product[product/]
        agents[agents/]
        simulation[simulation/]
        constants[constants.py]
    end

    config --> |loads| simulation
    network --> |defines topology| simulation
    product --> |defines goods| simulation
    agents --> |makes decisions| simulation
```

| Package | Responsibility |
|---------|---------------|
| `config/` | Configuration loading and validation |
| `network/` | Network topology primitives (Node, Link, Order, Shipment) |
| `product/` | Product definitions and recipes with weight/cube |
| `agents/` | Decision-making agents (replenishment, allocation) |
| `simulation/` | Core simulation engines and orchestration |
| `constants.py` | Shared constants and enumerations |

## Simulation Loop

The Orchestrator advances time daily, coordinating all engines in a specific sequence:

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant POS as POSEngine
    participant RA as Replenisher
    participant AA as Allocator
    participant MRP as MRPEngine
    participant TE as TransformEngine
    participant LE as LogisticsEngine
    participant Q as Quirks
    participant M as Monitor

    loop Each Day (t)
        O->>POS: Generate retail demand
        POS-->>O: Daily sales by store

        O->>RA: Check inventory, create orders
        RA-->>O: Replenishment orders

        O->>AA: Allocate available inventory
        AA-->>O: Allocation decisions

        O->>MRP: Plan production
        MRP-->>O: Production orders

        O->>TE: Execute manufacturing
        TE-->>O: Batches produced

        O->>LE: Process shipments
        LE-->>O: Shipments in-transit & arrived

        O->>Q: Apply behavioral quirks
        Q-->>O: Adjusted state

        O->>M: Validate physics & metrics
        M-->>O: Validation results
    end
```

## Data Flow

Orders and inventory flow through the network following Supply Chain Physics:

```mermaid
flowchart LR
    subgraph Demand["Demand Signal"]
        R[Retailers]
        POS[POS Engine]
    end

    subgraph Distribution["Distribution"]
        RDC[Regional DCs]
        INV[(Inventory)]
    end

    subgraph Manufacturing["Manufacturing"]
        PLANT[Plants]
        PROD[Production Lines]
    end

    subgraph Supply["Supply"]
        SUP[Suppliers]
        RAW[Raw Materials]
    end

    R -->|Sales| POS
    POS -->|Replenishment Order| RDC
    RDC -->|Inventory Check| INV
    INV -->|Backorder| PLANT
    PLANT -->|MRP| PROD
    PROD -->|Material Req| SUP
    SUP -->|Receipts| RAW
    RAW -->|Transform| PROD
    PROD -->|Finished Goods| RDC
    RDC -->|Shipment| R
```

## State Management

The simulation uses vectorized state management for performance:

```mermaid
graph TD
    subgraph StateManager
        IDX[ID-to-Index Mapping]
        INV[Inventory Tensor<br/>float64 array]
        WIP[WIP Tensor]
        BO[Backorder Tensor]
    end

    subgraph Operations
        READ[O(1) Read]
        WRITE[O(1) Write]
        BATCH[Vectorized Updates]
    end

    IDX --> READ
    IDX --> WRITE
    INV --> BATCH
    WIP --> BATCH
    BO --> BATCH
```

Key design principles:

- **O(1) Lookups**: Entity IDs map to integer indices for direct array access
- **Vectorized Operations**: NumPy arrays for bulk state transitions
- **Zero-Copy Views**: Data views passed between engines, not copies

## Physics Validation

The simulation continuously validates against Supply Chain Physics:

| Law | Implementation | Validation |
|-----|---------------|------------|
| **Little's Law** | $WIP = TH \times CT$ | Checked at each node |
| **Mass Balance** | $I_t = I_{t-1} + R_t - S_t$ | Inventory reconciliation |
| **VUT Equation** | Non-linear queue times | Congestion monitoring |
| **Capacity Limits** | Production capped at OEE | Transform engine |
| **Inventory Positivity** | Cannot ship negative | State manager guards |

## Key Concepts

### The Supply Chain Triangle

Every simulation decision impacts the balance between:

```mermaid
graph TD
    S[Service<br/>Fill Rate, OTD]
    CO[Cost<br/>Freight, Production]
    CA[Cash<br/>Inventory, WC]

    S ---|Trade-off| CO
    CO ---|Trade-off| CA
    CA ---|Trade-off| S
```

### Resilience Metrics

The simulation tracks system fragility:

- **Time-to-Survive (TTS)**: How long until stockout during disruption
- **Time-to-Recover (TTR)**: How long to restore full service

## Next Steps

- [Components Guide](components.md) - Deep dive into each module
- [API Reference](../reference/prism_sim/index.md) - Module documentation
- [Physics Theory](../planning/physics.md) - Theoretical foundations
