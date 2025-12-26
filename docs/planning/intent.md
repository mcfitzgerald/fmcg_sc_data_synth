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

### Network Topology
*   **Hub-and-Spoke:** 40% of volume flows through the **Chicago Hub (`DC-NAM-CHI-001`)**.
*   **SPOF Risk:** All palm oil (a critical ingredient for both Soap and Detergent) comes from a single Malaysian supplier (`SUP-MY-PALM`) with long, high-variance ocean lead times.

## 2. The Financial Core: Desmet's Triangle
The simulation is not just moving boxes; it is balancing the **Supply Chain Triangle** to optimize **ROCE (Return on Capital Employed)**. The system must capture the inherent trade-offs:

1.  **Service (Top):** Fill Rate / On-Time Delivery. (Drive Revenue).
2.  **Cost (Right):** Operating Expenses (Logistics, Manufacturing, COGS). (Drive EBIT).
3.  **Cash (Left):** Working Capital (Inventory). (Drive Capital Employed).

**The Simulation Challenge:**
*   High Inventory protects Service but hurts Cash (and ROCE).
*   Low Manufacturing Cost (long runs) improves Cost but hurts Cash (Cycle Stock).
*   The generated data must reflect these tensions—decisions in one node ripple through the triangle.

## 3. Data Ontology & Richness
To match the fidelity of a real ERP, the simulation will generate data across the full **SCOR-DS** spectrum, modeled after the reference schema:

*   **Source:** Ingredients (CAS#, purity), Suppliers (Tiers, certifications), Purchase Orders, Goods Receipts (Quality status, lot tracking).
*   **Transform:** Plants, Production Lines (OEE, changeover times), Formulas (BOMs with yield), Work Orders, Batches (Genealogy).
*   **Product:** SKUs, Packaging Hierarchies (Units -> Cases -> Pallets), Substitutes.
*   **Order:** Channels (Retail vs. eCommerce), Promotions (Lift & Hangover), Detailed Order Lines.
*   **Fulfill:** Inventory (Safety Stock vs. Cycle Stock, Aging), Shipments (multi-leg routes), Pick Waves.
*   **Logistics:** Carriers, Contracts, Rates, Emission Factors (Scope 3 Carbon).
*   **Plan:** Forecasts (Statistical vs. Consensus), Capacity Plans, DRP/MRP logic.

## 4. SCOR-DS Domain Model
The architecture mirrors the industry-standard process model:

1.  **Orchestrate:** The global clock and state manager.
2.  **Order:** Demand signal generation (Consumer $\to$ Retailer $\to$ Prism).
3.  **Plan:** DRP/MRP balancing supply and demand.
4.  **Source:** Procurement of raw materials.
5.  **Transform:** Manufacturing conversion (Ingredients $\to$ Finished Goods).
6.  **Fulfill:** Order processing, allocation, and shipping.
7.  **Return:** Reverse logistics and disposition.

## 5. The Supply Chain Physics Rubric
The simulation is validated against five non-negotiable physical laws:

1.  **Kinematic Consistency (Little’s Law):** $L = \lambda W$. Inventory, throughput, and cycle time must maintain a strict mathematical relationship.
2.  **Thermal Congestion (Kingman’s Curve):** Lead times must increase exponentially as utilization approaches 100%.
3.  **Conservation of Flow (Mass Balance):** Inventory cannot teleport. $Inv_t = Inv_{t-1} + In - Out$.
4.  **Signal Resonance (The Bullwhip Effect):** Demand variance must amplify upstream.
5.  **Statistical Envelope (LTD Variance):** High variance lanes must drive higher safety stock or stockouts.

## 6. Engineering Strategy
*   **Vectorized State:** Use `numpy` tensors for high-performance state management.
*   **Quirk Layers:** Inject behavioral "pathologies" (e.g., Optimism Bias) to ensure human-like system behavior.
*   **Reference Preservation:** Leverage the `fmcg_example` schema for entity relationships and attribute richness.
