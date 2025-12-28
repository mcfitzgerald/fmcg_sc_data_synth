# Roadmap: Prism Digital Twin (Reboot)

This roadmap outlines the sequential milestones to transition from a statistical generator to a Discrete-Event Simulation (DES) governed by Supply Chain Physics.

## Milestone 1: Project Scaffolding & Reference
*   **Task 1.1:** [x] Initialize fresh repository with `poetry` and `git`.
*   **Task 1.2:** [x] Establish directory structure:
    *   `src/prism_sim/`: Core simulation engines.
    *   `reference/`: Legacy `fmcg_example` code for logic reuse.
    *   `tests/`: Integration tests
*   **Task 1.3:** [x] Port configuration "Gold Standards" (`benchmark_manifest.json`).

## Milestone 2: World Building & Performance Core
*   **Goal:** Initialize the "Physical Reality" (Levels 0-4) with an optimized **Modular Architecture**.
*   **Task 2.1:** [x] **Network Topology:** Instantiate `Node` objects for the "Big 4" RDCs, **DTC Fulfillment Centers**, and **eCommerce Pure Players**.
    *   *Constraint:* Ensure **Chicago Hub** handles ~40% of traditional retail volume.
    *   *Constraint:* Create **Specialty Surfactant SPOF** with a high-cost backup supplier.
*   **Task 2.2:** [x] **Product Physics:** Define the Weight/Cube attributes for the three core categories.
*   **Task 2.3:** [x] **BOM & Recipes:** Port formulas and capacities.
*   **Task 2.4:** [x] **Performance Engineering:** Initialize **Vectorized State Tensors** (`numpy`) for Inventory, Cash, and Backlog, ensuring O(1) index mapping.

## Milestone 3: The Time Loop & Demand Physics
*   **Goal:** Implement the Orchestrator and the "Pull" signal (Levels 8-9) using **Atomic Agents**.
*   **Task 3.1:** [x] **Orchestrator:** Build the daily time-stepper loop.
*   **Task 3.2:** [x] **Promo Calendar:** Port the vectorized "Lift & Hangover" logic to drive POS demand.
*   **Task 3.3:** [x] **POS Engine:** Generate daily consumer sales based on Seasonality + Promo.
*   **Task 3.4:** [x] **Replenishment Agent:** Implement a pluggable `MinMaxReplenisher` to create the **Bullwhip Effect** (Order Batching).

## Milestone 4: Fulfill & Logistics Physics
*   **Goal:** Move the goods and enforce constraints (Levels 10-11).
*   **Task 4.1:** [x] **Allocation Agent:** Implement the "Triangle Decision" logic (Fill, Short, or Transfer?) for the "Big 4" network.
*   **Task 4.2:** [x] **Tetris Engine:** Implement the Truck Bin-Packing logic (Weigh Out vs. Cube Out).
*   **Task 4.3:** [x] **Logistics Costing:** Calculate freight costs based on real shipment profiles (Weight/Distance).
*   **Task 4.4:** [x] **Transit Physics:** Enforce lead times via `Link` objects; inventory must sit in `In_Transit` state before arriving.

## Milestone 5: Manufacturing & Supply (Transform)
*   **Goal:** Make the goods (Levels 5-7).
*   **Task 5.1:** [x] **MRP Engine:** Translate DRP requirements into Production Orders.
*   **Task 5.2:** [x] **Production Physics:** Enforce Finite Capacity and Changeover Times (Little's Law).
    *   *Constraint:* Ensure **Batch B-2024-RECALL-001** is deterministically scheduled.
*   **Task 5.3:** [x] **SPOF Simulation:** Simulate the "Specialty Surfactant" bottleneck and the "Backup Supplier" margin penalty.

## Milestone 6: Validation, Quirks & Realism
*   **Goal:** Verify "Emergent Properties" and stress-test the system.
*   **Task 6.1:** [x] **Realism Monitor:** Implement the online validator for OEE, Truck Fill, and SLOB.
*   **Task 6.2:** [x] **Resilience Metrics:** Measure **Time-to-Survive (TTS)** and **Time-to-Recover (TTR)** during disruption events.
*   **Task 6.3:** [x] **Quirk Injection:** Port the specific behavioral engines from `quirks.py`:
    *   **AR(1) Port Congestion:** Auto-regressive delays.
    *   **Optimism Bias:** Forecast inflation curve.
    *   **Phantom Inventory:** Shrinkage with detection lag.
*   **Task 6.4:** [x] **Risk Scenarios:** Execute deterministic risk events (e.g., "Port Strike") from `benchmark_manifest.json`.
*   **Task 6.5:** **Legacy Validation:** Port legacy distribution checks (Zipf, Pareto, Hub Concentration) to `monitor.py`.
*   **Task 6.6:** **Physics Audit:** Automated checks for Mass Balance ($Inv_{in} == Inv_{out}$).

## Milestone 7: SCOR-DS Expansion (Deep NAM)
*   **Goal:** Scale to 4,500 nodes and generate full ~70-table schema (Levels 0-14).
*   **Task 7.1:** **Generators (Static):**
    *   Port `static_pool.py` (Faker) and `distributions.py` (Zipf/Barabási).
    *   Create `NetworkGenerator` to build the 4,500-store topology.
*   **Task 7.2:** **Writers (Static):**
    *   Implement `StaticWriter` to export World CSVs (Locations, Products, Partners).
    *   Implement `SQLWriter` to generate `seed.sql` for Postgres compatibility.
*   **Task 7.3:** **Simulation Writers (Dynamic):**
    *   Upgrade `SimulationWriter` to support streaming Parquet/CSV for high-volume tables.
    *   Implement **In-Memory Validation Mode** to skip I/O for rapid parameter tuning.
*   **Task 7.4:** **Full Scale Run:**
    *   Execute 365-day run with full topology.
    *   Validate row counts and referential integrity against `schema.sql`.

## Milestone 8: Architecture Overhaul (World Builder & Vectorization)
*   **Goal:** Transition to procedural ingredients and vectorized execution for scale.
*   **Task 8.1:** [x] **Procedural Generation:** Implement `ProductGenerator` updates for dynamic Ingredients and logic-driven BOMs.
*   **Task 8.2:** [x] **Recipe Matrix:** Implement `RecipeMatrixBuilder` for dense $O(1)$ BOM lookups.
*   **Task 8.3:** [x] **Vectorized Engines:** Refactor `MRPEngine` and `TransformEngine` to use matrix algebra for planning and execution.

## Milestone 9: Final Delivery
*   **Task 9.1:** Generate full SCOR-DS dataset (CSV/Parquet/SQL export).
*   **Task 9.2:** Generate "The Triangle Report": A summary of Service vs. Cost vs. Cash performance.