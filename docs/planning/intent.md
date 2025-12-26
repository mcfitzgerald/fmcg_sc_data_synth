# Intent: Prism Digital Twin (Reboot)

**Goal:** Re-architect the Prism supply chain generator into a modular, high-fidelity **Discrete-Event Simulation (DES)** framework from first principles.

## 1. The Prism Narrative (Contextual Grounding)
The simulation generates data for **Prism Consumer Goods**, a high-velocity FMCG giant. The system models the "Deep NAM" (North American) division, characterized by:

*   **Dual Physics Profiles:**
    *   **Dense Liquids (e.g., Detergents):** Bottlenecked by weight limits (`PLANT-TN`, `SUP-US-CHEM`).
    *   **Bulky Paper (e.g., Tissue):** Bottlenecked by volume/cube limits (`PLANT-TX`, `SUP-US-PKG`).
*   **Network Topology:**
    *   **Hub-and-Spoke:** 40% of volume flows through the **Chicago Hub (`DC-NAM-CHI-001`)**.
    *   **SPOF Risk:** All palm oil comes from a single Malaysian supplier (`SUP-MY-PALM`) with long, high-variance ocean lead times.

## 2. SCOR-DS Domain Model
The simulation architecture mirrors the **ASCM SCOR-DS** process model:

1.  **Orchestrate (The Brain):** The global simulation loop that manages time, state persistence, and chaos injection.
2.  **Order (Demand):** Generates customer purchase signals (POS data) and retailer orders (`OrderEngine`).
3.  **Plan (Strategy):** Runs DRP/MRP logic to balance requirements vs. resources (`PlanEngine`).
4.  **Source (Input):** Manages procurement and inbound logistics from suppliers (`SourceEngine`).
5.  **Transform (Production):** Schedules production lines and manages BOM consumption (`TransformEngine`).
6.  **Fulfill (Output):** Manages allocation, picking, packing, and outbound shipping (`FulfillEngine`).
7.  **Return (Regenerate):** Handles reverse logistics and disposition (`ReturnEngine`).

## 3. The Supply Chain Physics Rubric
The simulation is validated against five non-negotiable physical laws:

1.  **Kinematic Consistency (Little’s Law):** $L = \lambda W$. Inventory, throughput, and cycle time must maintain a strict mathematical relationship.
2.  **Thermal Congestion (Kingman’s Curve):** Lead times must increase exponentially as utilization approaches 100%. No linear "3-day fixed" lead times.
3.  **Conservation of Flow (Mass Balance):** Inventory cannot teleport. Every unit is tracked from Source -> Transform -> Fulfill. $Inv_t = Inv_{t-1} + In - Out$.
4.  **Signal Resonance (The Bullwhip Effect):** Demand variance must amplify upstream. Supplier orders must be "spikier" than retailer POS data due to batching.
5.  **Statistical Envelope (LTD Variance):** Uncertainty is a cost. Higher variance lanes (e.g., Ocean) must drive higher safety stock levels or lower service levels.

## 4. Engineering Strategy
*   **Vectorized State:** Use `numpy` tensors for high-performance state management (~2,000 locations, 365 days).
*   **Quirk Layers:** Inject behavioral "pathologies" (e.g., Optimism Bias, Phantom Inventory) as explicit simulation layers to ensure **Benchmark Convergence** with `benchmark_manifest.json`.
*   **Reference Preservation:** Retain the `fmcg_example` directory as a "Reference" to port tuned business rules and named entities.
