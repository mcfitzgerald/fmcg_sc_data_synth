# Roadmap: Prism Digital Twin (Reboot)

This roadmap outlines the sequential milestones to transition from a statistical generator to a Discrete-Event Simulation (DES) governed by Supply Chain Physics.

## Milestone 1: Project Scaffolding & Reference
*   **Task 1.1:** Initialize fresh repository with `poetry` and `git`.
*   **Task 1.2:** Establish directory structure:
    *   `src/prism_sim/`: Core simulation engines.
    *   `reference/`: Legacy `fmcg_example` code for logic reuse.
    *   `tests/bdd/`: Spec-driven validation.
*   **Task 1.3:** Port configuration "Gold Standards" (`benchmark_manifest.json`, `prism_fmcg.yaml`).

## Milestone 2: World Building & Performance Core
*   **Goal:** Initialize the "Physical Reality" (Levels 0-4) with an optimized **Modular Architecture**.
*   **Task 2.1:** **Network Topology:** Instantiate `Node` objects for the "Big 4" RDCs, **DTC Fulfillment Centers**, and **eCommerce Pure Players**.
    *   *Constraint:* Ensure **Chicago Hub** handles ~40% of traditional retail volume.
    *   *Constraint:* Create **Specialty Surfactant SPOF** with a high-cost backup supplier.
*   **Task 2.2:** **Product Physics:** Define the Weight/Cube attributes for the three core categories.
*   **Task 2.3:** **BOM & Recipes:** Port formulas and capacities.
*   **Task 2.4:** **Performance Engineering:** Initialize **Vectorized State Tensors** (`numpy`) for Inventory, Cash, and Backlog, ensuring O(1) index mapping.

## Milestone 3: The Time Loop & Demand Physics
*   **Goal:** Implement the Orchestrator and the "Pull" signal (Levels 8-9) using **Atomic Agents**.
*   **Task 3.1:** **Orchestrator:** Build the daily time-stepper loop.
*   **Task 3.2:** **Promo Calendar:** Port the vectorized "Lift & Hangover" logic to drive POS demand.
*   **Task 3.3:** **POS Engine:** Generate daily consumer sales based on Seasonality + Promo.
*   **Task 3.4:** **Replenishment Agent:** Implement a pluggable `MinMaxReplenisher` to create the **Bullwhip Effect** (Order Batching).

## Milestone 4: Fulfill & Logistics Physics
*   **Goal:** Move the goods and enforce constraints (Levels 10-11).
*   **Task 4.1:** **Allocation Agent:** Implement the "Triangle Decision" logic (Fill, Short, or Transfer?) for the "Big 4" network.
*   **Task 4.2:** **Tetris Engine:** Implement the Truck Bin-Packing logic (Weigh Out vs. Cube Out).
*   **Task 4.3:** **Logistics Costing:** Calculate freight costs based on real shipment profiles (Weight/Distance).
*   **Task 4.4:** **Transit Physics:** Enforce lead times via `Link` objects; inventory must sit in `In_Transit` state before arriving.

## Milestone 5: Manufacturing & Supply (Transform)
*   **Goal:** Make the goods (Levels 5-7).
*   **Task 5.1:** **MRP Engine:** Translate DRP requirements into Production Orders.
*   **Task 5.2:** **Production Physics:** Enforce Finite Capacity and Changeover Times (Little's Law).
    *   *Constraint:* Ensure **Batch B-2024-RECALL-001** is deterministically scheduled.
*   **Task 5.3:** **SPOF Simulation:** Simulate the "Specialty Surfactant" bottleneck and the "Backup Supplier" margin penalty.

## Milestone 6: Validation, Quirks & Realism
*   **Goal:** Verify "Emergent Properties" and stress-test the system.
*   **Task 6.1:** **Realism Monitor:** Implement the online validator for OEE, Truck Fill, and SLOB.
*   **Task 6.2:** **Resilience Metrics:** Measure **Time-to-Survive (TTS)** and **Time-to-Recover (TTR)** during disruption events.
*   **Task 6.3:** **Quirk Injection:** Port the specific behavioral engines from `quirks.py`:
    *   **AR(1) Port Congestion:** Auto-regressive delays.
    *   **Optimism Bias:** Forecast inflation curve.
    *   **Phantom Inventory:** Shrinkage with detection lag.
*   **Task 6.4:** **Risk Scenarios:** Execute deterministic risk events (e.g., "Port Strike") from `benchmark_manifest.json`.
*   **Task 6.5:** **Legacy Validation:** Run the original `validation.py` suite against the new simulation output.
*   **Task 6.6:** **Physics Audit:** Automated checks for Mass Balance ($Inv_{in} == Inv_{out}$).

## Milestone 7: Final Delivery
*   **Task 7.1:** Execute 365-day "Deep NAM" run.
*   **Task 7.2:** Generate full SCOR-DS dataset (CSV/Parquet export).
*   **Task 7.3:** Generate "The Triangle Report": A summary of Service vs. Cost vs. Cash performance.
