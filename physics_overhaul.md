# Physics Overhaul Plan: Closing the Service Level Gap (v0.17.0+)

**Goal:** Increase Service Level from 76% to >95% using first-principles supply chain physics rather than heuristic parameter tuning.

## The Problem: The "Bimodal" Failure

Current state (v0.16.0) shows a physics paradox:
- **Low Service Level (76%):** Stores stock out frequently.
- **Low Truck Fill (2.2%):** We are shipping frequent, small LTL loads.
- **Low Inventory Turns (4.7x):** We are over-stocked on the wrong items.

### Root Cause Analysis

The current Replenisher uses a simplified Safety Stock formula ($SS = z\sigma_D\sqrt{L}$) that assumes:
1.  **Lead Time is Constant ($L=3$):** It ignores the reality of FTL consolidation delays, allocation cuts, and transit variability.
2.  **Supply is Reliable:** It assumes that when an order is placed, it is filled immediately. In reality, RDC stockouts extend the effective lead time.
3.  **One Size Fits All:** It applies the same service target ($z=1.65$) to high-velocity A-items and erratic C-items.

**The Result:**
- **Popular SKUs:** Stock out because the formula underestimates the *supply risk* (delays) and *demand risk* (Zipfian spikes).
- **Niche SKUs:** Accumulate excess inventory because the "standard" targets are too high for their low velocity.

---

## The Solution: Physics-Based Replenishment

We will implement a 3-Phase Overhaul to align the simulation logic with Supply Chain Physics.

### Phase 1: Instrument "Effective Lead Time" (Measure Reality)

We cannot buffer against risks we don't measure. We must track the *actual* realized lead time for every link in the network.

**The Physics:**
$$L_{eff} = T_{review} + T_{processing} + T_{transit}$$

- **$T_{review}$:** Time between "need" and "order" (Order Cycle).
- **$T_{processing}$:** Time waiting for inventory allocation + FTL consolidation delay.
- **$T_{transit}$:** Physical travel time.

**Implementation Plan:**
1.  **Tracking:**
    - Update `Shipment` to track `order_creation_day`.
    - In `LogisticsEngine` and `Replenisher`, calculate `Actual Lead Time = Arrival Day - Order Creation Day`.
2.  **History:**
    - Store a rolling history of Lead Times per Link (Source $\to$ Target).
    - Calculate Mean Lead Time ($\mu_L$) and **Lead Time Standard Deviation ($\sigma_L$)**.

### Phase 2: The Full Safety Stock Formula

Update the Replenisher to use the "Textbook" formula for Safety Stock under uncertainty in both Demand and Supply.

**The Formula:**
$$SS = z \sqrt{ \underbrace{\bar{L}\sigma_D^2}_{\text{Demand Risk}} + \underbrace{\bar{D}^2\sigma_L^2}_{\text{Supply Risk}} }$$

- **$ar{L}\sigma_D^2$:** Protects against demand variability during the lead time.
- **$ar{D}^2\sigma_L^2$:** Protects against **Supply/Logistics variability**.
    - If FTL consolidation takes days $\to$ $\sigma_L$ increases $\to$ SS increases automatically.
    - If RDCs are often out of stock $\to$ $\sigma_L$ increases $\to$ SS increases automatically.

**Why this fixes the paradox:**
- It naturally resolves the FTL vs. LTL conflict. FTL routes (high variability) will trigger higher buffers. LTL routes (low variability) will allow leaner stocks.

### Phase 3: Dynamic Segmentation (ABC Logic)

Allocate the inventory budget where it impacts the Service Level metric the most.

**The Logic:**
- **A-Items (Top 80% Volume):** Target $z=2.33$ (99%). We *must* be in stock.
- **B-Items (Next 15%):** Target $z=1.65$ (95%).
- **C-Items (Bottom 5%):** Target $z=1.28$ (90%).

**Implementation:**
1.  **Classification:** Dynamically rank SKUs by rolling revenue/volume.
2.  **Targeting:** Assign $z$-scores based on rank.

---

## Technical Implementation Steps

### 1. `Shipment` Update
- Add `original_order_day` (int) to `Shipment` class in `network/core.py`.
- Ensure this metadata is preserved during `LogisticsEngine` processing.

### 2. `Replenisher` State
- Add `lead_time_history` tensor: `[n_nodes, n_sources, history_len]`.
- Add `record_lead_time(target_id, source_id, days)` method.
- Add `get_lead_time_stats(target_id, source_id)` returning $(\mu_L, \sigma_L)$.

### 3. `generate_orders` Overhaul
- Update the ROP calculation loop to:
    1.  Lookup $\mu_L$ and $\sigma_L$ for the specific store-supplier link.
    2.  Lookup $\sigma_D$ (Demand Std Dev).
    3.  Apply the full formula.

### 4. Config & Segmentation
- Add `service_level_targets` to `simulation_config.json`:
  ```json
  "segmentation": {
    "A": 2.33,
    "B": 1.65,
    "C": 1.28
  }
  ```
- Implement simple ABC ranking in `Replenisher`.

---

## Expected Outcome
- **Service Level:** >95% (driven by A-items being in stock).
- **Inventory Turns:** Increase to >8x (driven by reducing C-item bloat).
- **Physics Compliance:** The system will "self-heal" against logistics delays without manual tuning.
