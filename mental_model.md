# Prism Sim: Mental Model

This document outlines the core architectural and physical logic of the Prism Digital Twin. It describes how material, cash, and information flow through the simulated supply chain.

---

## 1. Network Topology (The World Definition)

The simulation models a multi-echelon supply chain structured into five primary layers. The topology is distance-based and demand-proportional.

### Echelons
1.  **Suppliers (Level 0):** Source of raw materials (Ingredients).
2.  **Plants (Level 1):** Manufacturing hubs where ingredients are transformed into Finished Goods (FG).
3.  **Regional Distribution Centers (RDCs - Level 2):** Manufacturer-controlled buffers that consolidate plant output.
4.  **Customer Distribution Centers (DCs - Level 3):** Retailer-controlled nodes (Walmart DC, Kroger DC, etc.) that feed specific store regions.
5.  **Stores (Level 4):** Demand endpoints where consumer consumption (POS) occurs.

### Routing Logic
*   **Plant-Direct:** Mass Retail and Club channels often route directly from **Plant → DC**.
*   **Consolidated:** Grocery, Pharmacy, E-commerce, and DTC channels route through **Plant → RDC → DC**.
*   **Final Mile:** All channels follow **DC → Store**.

---

## 2. Information Flow (The Signal)

The simulation uses a "pull" signal driven by consumer demand, with proactive forecasting and safety mechanisms.

### Demand Signal
*   **POS Demand:** Generated daily at the Store level based on base rates, seasonality, promotions, and random noise.
*   **Forecasts:** A 14-day deterministic "consensus forecast" is used by both replenishment agents and MRP to build stock ahead of peaks.
*   **Unmet Demand:** Lost sales at the shelf or allocation failures are recorded to prevent demand signal collapse.

### Ordering Logic
*   **Stores & DCs:** Use an **(s, S) Replenishment Policy** based on **Inventory Position** (On-Hand + In-Transit).
    *   `s` (Reorder Point): `Cycle Stock + Safety Stock`.
    *   `S` (Target Stock): `Demand * Target Days Supply`.
*   **MRP (Production Planning):**
    *   **A-Items:** Use **Net-Requirement Scheduling** (MPS-style). Production = `Target Inventory - Inventory Position`.
    *   **B/C-Items:** Use **DRP-Lite with Campaign Triggers**. Production fires only when `DOS < Threshold` to maximize OEE.

---

## 3. Material Flow (The Physics)

Material movement is governed by physical constraints (Mass Balance, Capacity, Lead Time).

### Manufacturing
*   **Transformation:** Ingredients are consumed via a dense **Recipe Matrix** (BOM) to produce FG.
*   **Capacity:** Modeled as **Discrete Parallel Lines**. Changeover penalties apply when switching products, making large "campaign runs" more efficient for slow-moving items.
*   **Backpressure:** Unneeded FG stays at the plant, increasing its Inventory Position and naturally suppressing further MRP orders.

### Distribution & Logistics
*   **Deployment:** Plants ship FG to RDCs/DCs based on **Need** (Target DOS - IP).
*   **Replenishment:** DCs ship to stores via **FTL/LTL** consolidation.
*   **In-Transit:** Shipments move via real Haversine distances with realized lead times tracked for safety stock calibration.
*   **Inventory Age:** Every unit is tracked via **Weighted Average Age**. FIFO logic reduces age proportionally on consumption, enabling accurate **SLOB (Slow/Obsolete)** reporting.

---

## 4. Cash Flow (The Value)

The "Supply Chain Triangle" (Service vs. Cash vs. Cost) is the primary lens for performance.

*   **Service (Revenue):** Driven by Store Service Level (OSA) and Fill Rate. Unfilled demand results in lost sales.
*   **Cash (Working Capital):** Tracked via **Inventory Turns** and the **Cash-to-Cash (C2C) Cycle** (`DIO + DSO - DPO`).
*   **Cost (Operating Expense):**
    *   **Freight:** Transportation costs and Scope 3 emissions (CO2 per case).
    *   **Production:** Ingredient costs and changeover/efficiency losses (OEE/TEEP).

---

## 5. State Management (The Single Source of Truth)

The `StateManager` holds the entire simulation state in high-performance NumPy tensors:
*   `actual_inventory` vs `perceived_inventory` (allows for Phantom Inventory quirks).
*   `inventory_age` (for SLOB).
*   `in_transit_tensor` (O(1) access to pipeline inventory).
*   `unmet_demand` (to preserve demand signals).

---

## 6. Daily Execution Loop

Every day (`tick`) follows this sequence:
1.  **Physics Audit & Aging**: Initialize day and age inventory.
2.  **POS Generation**: Consumers "arrive" at stores.
3.  **Consumption**: Consumers buy what is on the shelf.
4.  **Replenishment**: Stores/DCs calculate IP and place orders.
5.  **Allocation**: Upstream nodes fulfill orders (Fair Share if scarce).
6.  **Logistics**: Shipments created, moved, and received (Age-aware).
7.  **MRP & DRP**: Plants plan production based on orders and raw POS demand.
8.  **Production**: Plants transform ingredients into FG.
9.  **Deployment**: FG "pushed" to RDCs/DCs based on downstream need.
10. **Metrics & Logging**: State exported to Parquet for analysis.
