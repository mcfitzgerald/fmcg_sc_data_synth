# SCOR-DS Expansion Plan: The "Deep NAM" Build

**Status:** Draft
**Date:** 2025-12-27
**Parent:** `intent.md`

## 1. Executive Summary
This document details the engineering strategy to scale `prism-sim` from a 4-node prototype to a 4,500-node realistic FMCG network ("Deep NAM"). We will implement a **Hybrid Generation Architecture**:
1.  **Static Data (Levels 0-4):** Generated once via `Faker` and `NumPy` distributions (O(1) lookups).
2.  **Dynamic Data (Levels 5-14):** Generated daily by the `Orchestrator` physics engine.

## 2. Architecture: Hybrid Generators

### 2.1 Directory Structure
We will split the monolithic `WorldBuilder` into specialized generators:

```text
src/prism_sim/
├── generators/
│   ├── __init__.py
│   ├── base.py                 # Abstract Base Class
│   ├── static_pool.py          # Ported Faker pools (Names, Cities, Addresses)
│   ├── distributions.py        # Zipf, Barabási-Albert, Pareto logic
│   ├── hierarchy.py            # Product & Location hierarchies
│   └── network.py              # Node & Edge generation (The "Graph")
├── writers/
│   ├── __init__.py
│   ├── base.py                 # Writer ABC (CSV/Parquet/SQL support)
│   ├── static_writer.py        # Exports Levels 0-4 (The "World")
│   ├── dynamic_writer.py       # Exports Levels 5-14 (The "Simulation")
│   └── sql_writer.py           # Exports to seed.sql (Legacy Prototype Parity)
```

### 2.2 The Static Generators (Phase 1)
These run *before* the simulation starts to populate the world.

| Level | Domain | Generator | Logic / Distribution |
| :--- | :--- | :--- | :--- |
| **L0** | Reference | `hierarchy.py` | Static lists (Currencies, UoM, Incoterms) |
| **L1** | Network | `network.py` | **Barabási-Albert** (Hubs) + **Gaussian** (Geo-clusters) |
| **L2** | Partners | `network.py` | **Pareto** (80% spend on 20% suppliers) |
| **L3** | Product | `hierarchy.py` | **Zipf** (SKU popularity), Weight/Cube Matrix |
| **L4** | Recipes | `hierarchy.py` | BOM explosion, Line Capacities (OEE targets) |

**Key Algorithms:**
*   **`static_pool.py`:** Pre-generate 5,000 names/addresses. Use `numpy.random.choice` for O(1) assignment.
*   **`distributions.py`:**
    *   `zipf_weights(n, alpha=1.05)`: For SKU popularity.
    *   `preferential_attachment(nodes, m=2)`: For connecting Stores to DCs.

### 2.3 The Dynamic Writers (Phase 2+)
These hook into the `Orchestrator` to log physical events.

| Level | Domain | Hook Point | Trigger |
| :--- | :--- | :--- | :--- |
| **L5** | Source | `MRPEngine` | When `RawMat Inventory < SafetyStock` |
| **L6** | Make | `TransformEngine` | When `FinishedGood Inventory < SafetyStock` |
| **L8** | Demand | `POSEngine` | Daily loop (Vectorized `numpy` call) |
| **L9** | Order | `ReplenishmentAgent` | When `Store Inventory < ReorderPoint` |
| **L10** | Fulfill | `AllocationAgent` | Daily, processing Order backlog |
| **L11** | Ship | `LogisticsEngine` | When `TruckLoad >= 85%` or `Time > MaxWait` |

**Output Modes:**
1.  **File Mode:** Writes CSV/Parquet files for big data analysis.
2.  **SQL Mode:** Writes `seed.sql` `INSERT` statements (matching `reference/fmcg_example_OLD/schema.sql`).
3.  **Memory Mode:** No disk I/O. Validates statistics in-memory for rapid parameter tuning (100x speedup).

## 3. Implementation Steps

### Step 1: The Foundation (Generators)
1.  **Port `static_pool.py`:** Copy from `reference/` and type-hint.
2.  **Create `distributions.py`:** Implement the math helpers.
3.  **Refactor `world_definition.json`:** Split massive config into `scor_reference.json` (static) and keep `world_definition.json` for topology rules.

### Step 2: The Static World (Levels 0-4)
1.  **`ProductGenerator`:** Generate 50 SKUs across 3 categories (Oral, Home, Personal).
2.  **`NetworkGenerator`:** Generate 4 RDCs, 50 Suppliers, 4,500 Stores (using Archetypes).
3.  **`StaticWriter`:** Dump `products.csv`, `locations.csv`, `partners.csv`.

### Step 3: Simulation Integration
1.  **Load the Big World:** Update `WorldBuilder` to load the generated CSVs/JSONs instead of hardcoded nodes.
2.  **Scale the Arrays:** Ensure `StateManager` tensors resize dynamically to `(4500_stores, 50_skus)`.

### Step 4: Validation (The Audit)
We will validate the *generated world* before simulating.

*   **Check 1:** Do we have 4,500 stores?
*   **Check 2:** Is the Chicago DC connected to ~30% of stores? (Hubness).
*   **Check 3:** Do 20% of SKUs represent 80% of forecasted volume? (Zipf).

## 4. Benchmark Targets

| Metric | Target |
| :--- | :--- |
| **Total Rows** | ~15M (Full Run) |
| **Stores** | 4,500 |
| **SKUs** | 50 |
| **Generation Speed** | < 10s (Static World) |
| **Simulation Speed** | > 10 simulated days/sec |

## 5. Schema Alignment
The output will strictly follow the 70-table SCOR-DS schema defined in `reference/fmcg_example_OLD/schema.sql`.
