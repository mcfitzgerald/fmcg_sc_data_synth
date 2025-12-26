# Roadmap: Prism Digital Twin (Reboot)

This roadmap outlines the sequential milestones to transition from a statistical generator to a Discrete-Event Simulation (DES) governed by Supply Chain Physics.

## Milestone 1: Project Scaffolding & Reference
*   **Task 1.1:** Initialize fresh repository with `poetry` and `git`.
*   **Task 1.2:** Establish directory structure:
    *   `src/prism_sim/`: Core simulation engines.
    *   `reference/`: Legacy `fmcg_example` code for logic reuse.
    *   `tests/bdd/`: Spec-driven validation.
*   **Task 1.3:** Port configuration "Gold Standards" (`benchmark_manifest.json`, `prism_fmcg.yaml`).

## Milestone 2: Orchestrate (The Framework)
*   **Task 2.1:** Build the `Orchestrate` loop (Day-by-day time stepper).
*   **Task 2.2:** Build the `StateManager` (Tensor World):
    *   Initialize NumPy tensors for `Inventory[Location, SKU]`, `Demand`, `Backlog`.
*   **Task 2.3:** Build `ChaosLayer`: Deterministic risk event injector.
*   **Task 2.4:** Implement `PersistenceManager`: Batch flushing of state to SQLite/Postgres.

## Milestone 3: Order (The Demand Signal)
*   **Task 3.1:** Implement `OrderEngine`:
    *   Generate Daily POS Purchase signals per store/SKU.
    *   Implement Retailer Replenishment logic (Ordering from DCs).
*   **Task 3.2:** Implement `BehavioralQuirks`:
    *   Bullwhip "Whip Crack" (Order batching logic).
    *   Human Optimism Bias (Forecast inflation).
*   **Validation:** Verify Rubric #4 (Signal Resonance/Bullwhip).

## Milestone 4: Fulfill & Source (Logistics Physics)
*   **Task 4.1:** Implement `FulfillEngine`:
    *   Stock Allocation (Fair-share logic).
    *   "Tetris Engine": Truck bin-packing (Weight vs. Cube).
*   **Task 4.2:** Implement `SourceEngine`:
    *   Inbound procurement from global suppliers.
    *   Lead-time variance modeling (Ocean vs. Truck).
*   **Validation:** Verify Rubric #2 (Kingman’s Curve) and Rubric #5 (LTD Variance).

## Milestone 5: Transform (Manufacturing Physics)
*   **Task 5.1:** Implement `TransformEngine`:
    *   Gantt Scheduler (Finite capacity line scheduling).
    *   BOM Consumption (Decrementing raw material tensors).
    *   Yield loss & QC rejection logic.
*   **Validation:** Verify Rubric #3 (Mass Balance) and Rubric #1 (Little's Law).

## Milestone 6: Plan (Closing the Loop)
*   **Task 6.1:** Implement `PlanEngine`:
    *   Feedback loops: Demand sensing -> DRP -> Production Requirements.
*   **Task 6.2:** Final stability tuning: Ensure the system naturally converges to `benchmark_manifest.json` targets.

## Milestone 7: Final Delivery
*   **Task 7.1:** Execute 365-day "Deep NAM" run.
*   **Task 7.2:** Generate full SCOR-DS dataset.
*   **Task 7.3:** Run automated "Physics Audit" (Rubric Validation).
