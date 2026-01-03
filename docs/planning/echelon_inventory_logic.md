# Echelon Inventory Logic: Implementation Plan

**Goal:** Implement **Multi-Echelon Inventory Optimization (MEIO)** principles for Customer DCs to resolve the "Hoarding at RDC / Starvation at Store" imbalance.

**Current Problem (The "Signal Trap"):**
Currently, a Customer DC (e.g., `RET-DC-001`) orders based on *local* Inventory Position (IP) vs. *local* Demand.
1.  Stores run out of stock → They stop selling (Lost Sales).
2.  Stores order from DC → DC runs out of stock.
3.  DC ships what it has, then goes to zero.
4.  Stores stop receiving goods → Their "Inflow" drops.
5.  DC sees "Demand" (Orders Received) drop because stores are discouraged/empty.
6.  DC reduces its own orders to RDC.
7.  **Result:** RDC sits full, DC sits empty, Store sits empty. The "Pull Signal" dies in the middle.

**The Solution: Echelon Inventory Position**
The DC should not just look at its own four walls. It should look at the **entire downstream network**.
$$
IP_{Echelon} = IP_{DC} + \sum_{s \in Stores} IP_{Store}
$$
$$
Demand_{Echelon} = \sum_{s \in Stores} Demand_{Store}^{POS}
$$

If stores are empty, $IP_{Store}$ is low. The DC sees $IP_{Echelon}$ drop below target, so it orders from the RDC *even if the DC itself is empty and stores haven't successfully ordered yet*. It "pushes" the need upstream.

---

## 1. Architectural Changes

### A. Network Awareness (`MinMaxReplenisher`)
The Replenisher needs a map of "Who belongs to whom."
*   **Current:** `store_supplier_map` (Child -> Parent).
*   **Required:** `supplier_downstream_map` (Parent -> List[Children]).
*   We effectively already have `_customer_dc_indices` and `_downstream_store_count`, but we need explicit index mapping: `dc_idx -> [store_indices]`.

### B. Vectorized Aggregation
We cannot iterate 4,500 stores daily. We need a matrix operation.
*   **Concept:** An "Echelon Matrix" ($M_E$) of shape $[N_{DC}, N_{Node}]$.
    *   $M_E[i, j] = 1$ if Node $j$ is downstream of DC $i$.
    *   $M_E[i, j] = 1$ if Node $j$ is DC $i$ itself.
*   **Calculation:** `Echelon_IP = M_E @ All_Node_IP`
    *   This sums inventory across all downstream nodes for each DC in one massive dot product.

### C. Demand Signal (The "True" Demand)
Customer DCs should ignore "Orders Received" (which are artificially constrained by store batching/stockouts) and replenish based on **aggregated consumer POS demand**.
*   **Logic:** `Echelon_Demand = M_E @ POS_Demand_Vector`
*   This ensures the DC replenishes exactly what consumers bought, regardless of whether stores have placed orders yet.

---

## 2. Implementation Steps

### Step 1: Build the Echelon Map
In `MinMaxReplenisher.__init__`:
1.  Identify all Customer DCs (Parent Nodes).
2.  Identify all Stores (Child Nodes) for each DC.
3.  Construct a sparse boolean matrix (or just an index list if dense matrix is too big, but $50 \times 4500$ is small for numpy).

### Step 2: Calculate Echelon IP
In `MinMaxReplenisher.generate_orders`:
1.  `local_ip = on_hand + in_transit` (Current logic).
2.  `echelon_ip_vec = echelon_matrix @ local_ip` (New logic).
    *   Result: A vector of size $[N_{DC}, N_{Products}]$ representing total stock in the channel.

### Step 3: Calculate Echelon Demand & Target
1.  `echelon_demand_vec = echelon_matrix @ pos_demand_matrix`.
2.  `Echelon_Target = Echelon_Demand * (LeadTime_{RDC \to DC} + ReviewPeriod)`.
    *   *Note:* Safety stock logic needs to account for pooled variance ($\sqrt{N} \times \sigma$), effectively reducing relative safety stock needs (Risk Pooling benefit).

### Step 4: The Ordering Decision
For Customer DCs:
$$
Order = Target_{Echelon} - IP_{Echelon}
$$
*   This replaces the current `Target_{Local} - IP_{Local}` logic.
*   We effectively bypass the "Store Order" signal for the DC's replenishment and link it directly to POS.

---

## 3. Risks & Edge Cases

1.  **Double Counting:** We must ensure the DC doesn't count "Orders placed by stores" as demand *and* "POS sales" as demand. Echelon logic completely replaces the order-driven logic for the DC tier.
2.  **Latency:** POS data might technically be "delayed" in a real supply chain (EDI latency). For now, we assume real-time visibility (Digital Twin ideal).
3.  **Cross-Channel Noise:** Does `RDC-NE` replenish based on Echelon IP too?
    *   *Decision:* No. RDCs are too far upstream. They should stick to `MinMax` based on DC Orders. This keeps the "Bullwhip" alive between DC and RDC, while smoothing the Store-DC link.

## 4. Success Metrics
*   **Inventory Distribution:** Should shift from 93/3/3 (RDC/DC/Store) to something like 40/20/40.
*   **Service Level:** Should sustain >95% indefinitely.
*   **Order Stability:** DC orders to RDCs should track aggregate POS demand trends, not artifact spikes.
