# Intent: Prism Digital Twin (Reboot)

**Goal:** Re-architect the Prism supply chain generator into a modular, high-fidelity **Discrete-Event Simulation (DES)** framework from first principles.

## 1. The Prism Narrative (Contextual Grounding)
The simulation generates data for **Prism Consumer Goods**, a high-velocity FMCG giant. The system models the "Deep NAM" (North American) division, operating under the tension of the **Supply Chain Triangle** (Service, Cost, Cash).

### Product Portfolio & Logistics Physics
Prism manages three distinct categories, each imposing unique physical constraints on the network ("The Weight/Cube Matrix"):

1.  **Oral Care (Toothpaste):**
    *   **Profile:** High value-density, small primary packaging (tubes in cartons).
    *   **Logistics:** High pallet density, often "cubes out" a truck before weighing out due to stacking limits or mixed-pallet complexity.
2.  **Personal Wash (Bar Soap):**
    *   **Profile:** "The Brick." Extremely high density, rectangular efficiency.
    *   **Logistics:** Pure "Weigh Out" constraint. A truck full of bar soap exceeds legal weight limits long before it is visually full.
3.  **Home Care (Liquid Dish Detergent):**
    *   **Profile:** "The Fluid Heavyweight." Heavy liquid in irregular rigid plastic bottles (lower packing efficiency/air gaps).
    *   **Logistics:** High risk of damage/leakers. Bottlenecked by weight, but with lower pallet stability than soap.

### Network Topology: The "Big 4" Regional Echelons & Omnichannel
Prism operates a **Multi-Echelon Distributed Network** serving both traditional retail and direct channels:

1.  **Northeast (`RDC-NAM-NE` - Pennsylvania):** High-velocity node serving the BosWash corridor and **DTC Fulfillment Center**.
2.  **South (`RDC-NAM-ATL` - Atlanta):** Primary gateway for Southern retail archetypes and imports.
3.  **Midwest (`RDC-NAM-CHI` - Chicago):** The central heavy-hitter and primary rail-intermodal node.
4.  **West (`RDC-NAM-CAL` - Inland Empire):** The West Coast anchor, handling Asian imports and **eCommerce Pure Players**.

**The Multi-Node Tension:**
The simulation must now solve for **Inventory Imbalance** across channels. A stockout in the DTC node cannot be easily filled by a pallet sitting in a Retail DC without "Break-Bulk" penalties.

## 2. Supply Chain Resilience (TTR & TTS)
Beyond the "Triangle," the simulation measures the system's fragility using **Simchi-Levi’s Resilience Framework**:

*   **Time-to-Survive (TTS):** If the Chicago Hub goes dark, how long until shelves are empty? (Driven by Safety Stock).
*   **Time-to-Recover (TTR):** How long to restore full service after a disruption? (Driven by Lead Times & Capacity).

**The Resilience Tension:**
Increasing TTS (more stock) hurts Cash. Reducing TTR (redundant suppliers) hurts Cost. The simulation forces this trade-off.

### Realistic Risk Profile (Refined)
We reject "cartoonish" risks in favor of realistic FMCG vulnerabilities:

1.  **SPOF (Single Point of Failure):** Instead of a generic "Palm Oil" outage (too big to fail), we model a **Specialty Surfactant (`ING-SURF-SPEC`)** used in premium products. It has a single qualified supplier in Germany.
    *   *Contingency:* Backup exists in Mexico but costs +25% (Margin Hit) and has 2x variability.
2.  **Port Strike (Long Beach):** A 14-day stoppage at USLAX.
    *   *Effect:* Hits the West RDC hard. The system must decide: Air freight (Cost spike) or Stockout (Service hit)?
3.  **Cyber Event (WMS Outage):** A ransomware event locks the **Northeast DTC Node** for 48 hours.
    *   *Effect:* Instant backlog accumulation. TTR depends on "Catch-up Capacity" (Overtime labor).

## 3. Data Ontology & Richness
To match the fidelity of a real ERP, the simulation will generate data across the full **SCOR-DS** spectrum, modeled after the reference schema:

*   **Source:** Ingredients (CAS#, purity), Suppliers (Tiers, certifications), Purchase Orders, Goods Receipts (Quality status, lot tracking).
*   **Transform:** Plants, Production Lines (OEE, changeover times), Formulas (BOMs with yield), Work Orders, Batches (Genealogy).
*   **Product:** SKUs, Packaging Hierarchies (Units -> Cases -> Pallets), Substitutes.
*   **Order:** Channels (Retail vs. eCommerce), Promotions (Lift & Hangover), Detailed Order Lines.
*   **Fulfill:** Inventory (Safety Stock vs. Cycle Stock, Aging), Shipments (multi-leg routes), Pick Waves.
*   **Logistics:** Carriers, Contracts, Rates, Emission Factors (Scope 3 Carbon).
*   **Plan:** Forecasts (Statistical vs. Consensus), Capacity Plans, DRP/MRP logic.

## 4. Architecture: Hybrid DES & Data Richness

To achieve the "Reference Level" fidelity within a physics-driven simulation, we adopt a **Hybrid Architecture**. We reject the linear "batch generation" of the legacy system in favor of a time-stepped loop, but we retain the strict **Ontological Dependency** to ensure the world makes sense.

### Modular Component Design (First Principles)
We shun monolithic scripts. The system is built from **atomic, reusable components** that represent first-principle supply chain concepts. This ensures we can adapt the simulation (e.g., add a new region, change a truck type) without refactoring the core engine.

*   **`Node`:** Abstract base for Plants, DCs, and Stores. Handles Inventory ($I_t$) and Throughput limits ($\mu$).
*   **`Link`:** Abstract base for Routes. Handles Lead Time ($L$) and Variability ($V$).
*   **`Agent`:** Pluggable logic modules (e.g., `MinMaxReplenisher`, `FairShareAllocator`) that make decisions. You can swap a "Naive Agent" for an "ML Agent" without touching the physics.

### Phase 1: World Building (Static Initialization)
Before the clock starts ($t=0$), we generate the structural reality of the supply chain. This corresponds to **Levels 0-4** of the reference architecture:

*   **L0 (Reference):** Divisions, Units of Measure, Currencies.
*   **L1 (Network):** The "Big 4" RDCs, 50+ Suppliers, Ports, and Retailer Archetypes.
*   **L2 (Relationships):** Supplier Contracts (Lead Times, Costs) and Sourcing Rules.
*   **L3 (Product):** The "Weight/Cube" Matrix (Toothpaste vs. Soap vs. Detergent).
*   **L4 (Recipes):** BOMs, Formulas, and Production Line Capacities (OEE targets).

### Phase 2: The Time Loop (Dynamic Physics)
Once the world is built, the **Orchestrator** advances time daily. Data for **Levels 5-14** is not pre-generated; it *emerges* from the interaction of agents and physics:

1.  **Demand Signal (L8):** `POS_Engine` generates daily sales using the **Promo Calendar** (Lift & Hangover physics).
2.  **Inventory Check (L10):** Retailers check shelf stock. If $Inv < ReorderPoint$, an **Order (L9)** is fired.
3.  **Bullwhip Logic:** Retailer ordering logic applies batching (e.g., "Order full pallets only"), creating the **Signal Resonance**.
4.  **Allocation (L9):** The RDC attempts to fill the order. If $Inv_{RDC} < Order$, it decides: *Short the customer* or *Expedite from sister DC*? (Triangle Tension).
5.  **Logistics (L11):** `Fulfill_Engine` builds shipments. **Tetris Logic** applies here:
    *   *Soap:* Weighs out the truck.
    *   *Tissue:* Cubes out the truck.
    *   *Result:* We calculate real **Truck Fill Rate** and **Freight Cost** based on physical constraints, not statistical averages.
6.  **Production (L6):** Plants run MRP. If stock is low, `Transform_Engine` schedules a **Batch**.
    *   *Constraint:* Changeover time (Liquid $\to$ Gel) eats capacity. This enforces **Little's Law**.

### Phase 3: The "Realism Monitor" (Validation & Quirks)
We do not just "hope" the physics work. We run a real-time monitor that validates the **Emergent Properties** of the simulation against industry benchmarks. Crucially, we will **reuse the legacy validation scripts** from the reference to ensure our new DES engine produces data that is statistically indistinguishable from the tuned reference, but physically superior.

*   **Deterministic Risks:** We inject specific scenarios (e.g., "Port Strike on Day 120") defined in `benchmark_manifest.json` to test resilience.
*   **Behavioral Quirks:** We probabilistically inject human error (e.g., "Optimism Bias" in forecasts, "Phantom Inventory" at stores) to prevent the data from looking "too clean."

| Metric | Target | Physics Driver |
| :--- | :--- | :--- |
| **Schedule Adherence** | < 1.1 Days | Manufacturing capacity constraints & material availability. |
| **Truck Fill Rate** | > 85% Weight/Cube | The "Tetris Engine" bin-packing logic. |
| **SLOB Inventory** | < 30% of working capital | "Hangover" effects from promotions & forecast bias. |
| **OEE (Efficiency)** | 65% - 85% | Random breakdowns & changeover friction. |
| **Inventory Turns** | 6x - 14x | The tension between Service (Stock) and Cash (Inventory). |
| **Forecast MAPE** | 20% - 50% | The "Optimism Bias" quirk layer vs. actual POS volatility. |
| **Cost-to-Serve** | $1 - $3 per case | Emerges from freight distance + carrier rates + fuel surcharges. |

## 5. Performance Engineering
To simulate 365 days across 2,000+ nodes and 500+ SKUs (~10M data points) efficiently, we mandate:

1.  **Vectorized Operations:** Loops are banned for heavy lifting. All state transitions (e.g., `Inv_t = Inv_t-1 - Sales`) must use `numpy` vectorization.
2.  **Memory Mapping:** State tensors should be memory-mapped if they exceed RAM, though for this scale, optimized `float32` arrays should fit in memory.
3.  **Zero-Copy Views:** Pass data views, not copies, between engines.
4.  **O(1) Lookups:** All Entity IDs (SKU, Location) are mapped to integer indices (0...N) to allow direct array indexing, avoiding slow dictionary hash lookups during the inner loop.

## 6. The Financial Core: Desmet's Triangle
The simulation is not just moving boxes; it is balancing the **Supply Chain Triangle** to optimize **ROCE (Return on Capital Employed)**.
*(Rest of section remains unchanged...)*

## 7. Preserving Reference Fidelity
The reference implementation contains highly tuned business logic that must be ported to the new `QuirkLayer` and `Orchestrator` to prevent the simulation from becoming "physically accurate but behaviorally sterile."

### A. Behavioral Quirks (The "Human" Element)
These specific algorithms must be ported from `quirks.py`:
1.  **AR(1) Port Congestion:** Delays are not random; they are auto-regressive (`current = 0.7 * prev + noise`). This creates realistic "clumping" of late arrivals.
2.  **Bullwhip "Whip Crack":** Retailers do not just order more during promos; they batch orders into massive spikes (3x consumer demand).
3.  **Human Optimism Bias:** Planners consistently over-forecast *new* products by ~15%. This specific bias curve must be replicated.
4.  **Phantom Inventory:** A "Shrinkage" agent that deletes inventory, with a 14-day detection lag before the system realizes it's gone.

### B. Named Scenarios (The Narrative Anchors)
The simulation must converge to these specific "Named Entities" defined in the reference:
1.  **`DC-NAM-CHI-001` (Chicago Hub):** The topology must ensure ~40% of volume flows through here to stress-test the node.
2.  **`SUP-MY-PALM` (SPOF):** The single Malaysian Palm Oil supplier must exist with its specific Long Lead Time + High Variance profile.
3.  **`B-2024-RECALL-001`:** The `Transform_Engine` must deterministically create this contaminated batch of Sorbitol to trigger the recall trace scenario.

## 8. The Supply Chain Physics Rubric
The simulation is validated against five non-negotiable physical laws:
1.  **Kinematic Consistency:** Trucks cannot teleport. Travel time = Distance / Speed.
2.  **Mass Balance:** `Input (kg) = Output (kg) + Scrap`. Matter is neither created nor destroyed.
3.  **Little's Law:** `Inventory = Throughput * Flow Time`.
4.  **Capacity Constraints:** You cannot produce more than `Rate * Time`.
5.  **Inventory Positivity:** You cannot ship what you do not have (unless Backlog logic is explicitly enabled).

## 9. Engineering Standards & Quality
To ensure the robustness and maintainability of the simulation engine, the following engineering standards are strictly enforced:

### A. Coding Standards
*   **Modularity:** Concerns must be separated into first-principle components (e.g., `network`, `product`, `simulation`). Files exceeding 700-1000 lines should be refactored.
*   **Configuration Paradigm:** No hardcoded variables. All simulation parameters (policies, thresholds, constraints) must be loaded from external configuration files (e.g., `simulation_config.json`).
*   **Type Safety:** All code in `src/` must be fully typed and pass `mypy --strict`. `Any` should be avoided.
*   **Linting & Formatting:** Code must adhere to `ruff` standards for formatting and linting.

### B. Testing Strategy
*   **Integration over Unit:** Prefer integration tests that verify the *emergence* of supply chain behaviors (e.g., "Does the Bullwhip effect appear?") over isolated unit tests.
*   **Validation:** Use `semgrep` to detect hardcoded values and security risks.
*   **Reference Parity:** Changes must not regress the ability to reproduce the statistical properties of the reference implementation.

### C. Tooling
*   **Environment:** All Python execution is managed via `poetry`.
*   **Context7:** Use standard library documentation and tools for code generation to ensure idiomatic usage.