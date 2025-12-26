# Data Generation Architecture

The generator follows a strictly ordered, 15-level dependency graph to ensure referential integrity.

## Dependency Levels

| Level | Domain | Description | Key Tables |
|-------|--------|-------------|------------|
| **0** | Reference | Foundational master data | `divisions`, `products`, `ingredients` |
| **1** | Master | Physical infrastructure | `plants`, `suppliers`, `route_segments` |
| **2** | Relationships | Sourcing & recipes | `formulas`, `supplier_ingredients` |
| **3** | Network | Logistics network | `retail_accounts`, `distribution_centers` |
| **4** | Product | SKU explosion | `skus`, `promotions`, `sku_costs` |
| **5** | Procurement | Inbound orders | `purchase_orders`, `work_orders` |
| **6** | Manufacturing | Production execution | `batches`, `po_lines`, `gr_lines` |
| **7** | Consumption | Ingredient usage | `batch_ingredients` |
| **8** | Demand | **(Largest)** Customer demand | `pos_sales`, `orders`, `forecasts` |
| **9** | Planning | Order management | `order_lines`, `allocations`, `pick_waves` |
| **10** | Fulfillment | Outbound & Inventory | `shipments`, `shipment_legs`, `inventory` |
| **11** | Lines | Detailed shipping + Transit | `shipment_lines`, `inventory` (transit) |
| **12** | Returns | Reverse logistics start | `rma_authorizations`, `returns` |
| **13** | Disposition | Reverse logistics end | `disposition_logs` |
| **14** | Monitoring | Analytics & Logs | `kpi_actuals`, `audit_logs`, `risk_events` |

## Core Components

### 1. GeneratorContext
The `GeneratorContext` class acts as the shared "world state". It holds:
*   **Data Storage:** A dictionary of lists for every table.
*   **ID Mapping:** Dictionaries mapping codes (e.g., "SKU-001") to integer IDs, allowing upper levels to reference lower levels O(1).
*   **Random State:** Seeded NumPy generators for reproducibility.

### 2. Vectorized Generation
For high-volume tables (Levels 8-11), Python loops are too slow. We use `vectorized.py` to generate columns as NumPy arrays.
*   **Zipf Distribution:** Used for SKU selection to match the Pareto (80/20) principle.
*   **Promo Calendar:** A pre-computed index used to apply lift/hangover multipliers to millions of POS rows instantly.
*   **POS Row Semantics:** To balance volume with massive B2B orders without generating billions of rows, a single row in `pos_sales` represents **Weekly Aggregated Sales** for a specific SKU at a specific Store. A mean of 60.0 units/row assumes hypermarket velocity (~8-9 units/day).
*   **Logistics Physics:** `ShipmentsGenerator` and `ShipmentLinesGenerator` enforce physical constraints (temporal sequence, location bounds) and calculate rate-based costs using vectorized operations.

### 3. Mass Balance Physics
The generator enforces conservation-of-mass at multiple levels:
*   **Cases-First Calculation:** Total cases are calculated upstream from shippable SKU quantities (demand-capped supply), then weights are derived FROM cases using actual SKU weights (~4-5 kg/case). This prevents the "12 kg/case hardcode" bug.
*   **Store-Bound Allocation:** Only shipments to stores (`dc_to_store`, `direct_to_store`) count toward the mass balance. Internal transfers (`plant_to_dc`, `dc_to_dc`) move goods between facilities but don't affect COGS accounting.
*   **Inventory Remainder:** Inventory = Production - Shipped. Generated in Level 10 after shipment volumes are known.

### 3b. Inventory Waterfall
The generator creates a complete inventory waterfall with three components:
*   **DC Inventory (Level 10):** Stock at distribution centers, tagged as `safety_stock` (14-day demand coverage) or `cycle_stock` (remainder).
*   **Transit Inventory (Level 11):** Goods in-transit created from `in_transit` shipment lines using `location_type='in_transit'`.
*   **Waterfall View:** `v_inventory_waterfall` SQL view aggregates by location, showing safety stock, cycle stock, transit inventory, and days of supply.

### 4. LookupCache
To avoid O(N) scans when linking tables (e.g., finding all lines for an order), `LookupCache` builds O(1) indices (hash maps) after each level is generated.

### 5. StreamingWriter
Data is written to disk using PostgreSQL `COPY` format. To prevent OOM errors, the `DependencyTracker` allows the writer to purge in-memory data for a table once all its dependent children have been generated.
