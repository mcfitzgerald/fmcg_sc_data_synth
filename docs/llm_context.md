# Prism Sim: LLM Context & Developer One-Pager

> **System Prompt Context:** This document contains the critical architectural, functional, and physical constraints of the Prism Sim project. Use this as primary context when reasoning about code changes, bug fixes, or feature expansions.

---

## 1. Project Identity
**Name:** Prism Sim
**Type:** Discrete-Event Simulation (DES) Engine for Supply Chain Digital Twins
**Core Philosophy:** **"Supply Chain Physics"** - The simulation must adhere to fundamental physical laws (Little's Law, Mass Balance, Capacity Constraints) rather than just statistical approximation.
**Goal:** Generate realistic, high-fidelity supply chain datasets that exhibit emerging behaviors (bullwhip effect, bottlenecks) for benchmarking optimization algorithms.

## 2. The "Physics" of the Simulation
The engine is validated against these non-negotiable laws. Any code change must preserve these invariants:

1.  **Little's Law:** $WIP = Throughput \times CycleTime$
    *   *Invariant:* You cannot increase throughput without increasing WIP or reducing cycle time.
2.  **Mass Balance:** $Inventory_{t} = Inventory_{t-1} + Receipts_t - Shipments_t$
    *   *Invariant:* Inventory cannot be created or destroyed, only moved or transformed.
3.  **VUT Equation (Kingman's Formula):** Cycle time increases non-linearly with utilization.
    *   *Invariant:* As capacity utilization approaches 100%, queue times approach infinity.
4.  **Capacity Constraints:**
    *   **Production:** Limited by OEE (Overall Equipment Effectiveness) and line speed.
    *   **Logistics:** Limited by truck Weight (kg) and Cube ($m^3$) - "Tetris Logic". **Strict enforcement**: items larger than a truck will raise `ValueError`.
5.  **Signal Resonance (Bullwhip):** Small demand variance at POS amplifies upstream due to batching and lead time lags.

## 3. System Architecture

### Core Components (`src/prism_sim/`)

| Component | Path | Responsibility |
| :--- | :--- | :--- |
| **Orchestrator** | `simulation/orchestrator.py` | The "God Object" driving the daily time loop. Uses **Dynamic Priming** (estimates base demand from `POSEngine`) to initialize inventory. |
| **State Manager** | `simulation/state.py` | Vectorized (NumPy) state store. **O(1)** access for Inventory, WIP, Backorders. **Single Source of Truth.** |
| **World Builder** | `simulation/builder.py` | Hydrates the simulation from `simulation_config.json`. Builds Network, Products, Agents. |
| **Demand Engine** | `simulation/demand.py` | `POSEngine`. Generates stochastic consumer demand with seasonality, trends, and noise. |
| **Replenishment** | `agents/replenishment.py` | `MinMaxReplenisher`. Decides *what* and *how much* to order based on (s, S) policies. |
| **Allocation** | `agents/allocation.py` | `AllocationAgent`. Decides *who gets what* when supply < demand (Fair Share logic). |
| **MRP Engine** | `simulation/mrp.py` | Manufacturing Requirements Planning. Explodes BOMs, generates production orders based on RDC need. |
| **Transform** | `simulation/transform.py` | Manufacturing execution. Consumes raw materials, occupies capacity, produces Finished Goods batches. |
| **Logistics** | `simulation/logistics.py` | `LogisticsEngine`. Groups orders into shipments, applies "Tetris Logic" (Weight/Cube packing), tracks transit. |
| **Quirks** | `simulation/quirks.py` | Injects human/system errors: Optimism Bias, Phantom Inventory, Port Congestion. |
| **Monitor** | `simulation/monitor.py` | `RealismMonitor`. Validates physics invariants at runtime. Calculates Triangle Metrics (Cost/Cash/Service). |

### Data Model (`src/prism_sim/product/core.py`, `network/core.py`)

*   **Node:** Generic location (Supplier, Plant, RDC, Retailer, Port).
*   **Link:** Connection between nodes with defined `lead_time` and `cost`.
*   **Product:** SKU with physical attributes (`weight_kg`, `volume_m3`, `units_per_case`).
*   **Recipe:** BOM (Bill of Materials) linking Inputs -> Output with `yield` and `rate`.
*   **Order:** Request for goods.
*   **Shipment:** Physical movement of goods (contains 1+ Orders).

## 4. Simulation Loop (Daily Cycle)

Every `tick()` (1 day) proceeds in this exact order:

1.  **Demand:** POS Engine generates sales at Retailers.
2.  **Replenishment:** Retailers/RDCs check stock vs policies -> Place Orders.
3.  **Allocation:** RDCs/Plants check stock -> Allocate to Orders (Full or Partial).
4.  **MRP:** Plants check Finished Goods demand -> Plan Production (Explode BOMs).
5.  **Production (Transform):** Plants execute production -> Consume Raw Mts -> Create Batches.
6.  **Logistics:**
    *   Pick & Pack (Bin Packing).
    *   Ship (Create Shipments).
    *   Advance In-Transit (decrement ETA).
    *   Receive (Update Inventory at destination).
7.  **Quirks & Risks:** Apply random events or deterministic disruptions.
8.  **Monitor:** Validate Mass Balance, Record History.

## 5. Configuration Paradigm
*   **Config-Driven:** No hardcoded parameters. All settings in `simulation_config.json`.
*   **Manifest-Driven:** Benchmarks and Scenarios defined in `benchmark_manifest.json`.

## 6. Key Developer Commands

```bash
# Run Simulation
poetry run python run_simulation.py

# Run Tests
poetry run pytest

# Type Check
poetry run mypy src/

# Lint
poetry run ruff check src/
```
