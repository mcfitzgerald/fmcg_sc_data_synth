# Plan: ERP-Style Data Export via Post-Processing

## Decision: Post-Process vs Change Sim Output

### Post-Processing (Recommended)
| Pros | Cons |
|------|------|
| Keeps simulation fast (no extra I/O during run) | Two-step process |
| Separation of concerns (sim = physics, ETL = data modeling) | Extra disk space |
| Can iterate on schema without re-running 455-day sim | - |
| Flat format remains useful for debugging/analysis | - |
| Different output schemas possible from same run | - |

### Change Sim Output Directly
| Pros | Cons |
|------|------|
| One-step process | Slower simulation (more files, more I/O) |
| No intermediate files | Couples sim to ERP schema concerns |
| - | Schema changes require modifying sim code |
| - | Harder to debug (lose flat view) |

**Verdict:** Post-processing is cleaner. The simulation's job is physics modeling, not ERP formatting.

---

## Current State

**Simulation Output (455 days = 90 burn-in + 365 sim):**
- `orders.csv` - 13.5M rows, flat (order + line combined)
- `shipments.csv` - 15M rows, flat (shipment + line combined)
- `batches.csv` - 22.8K rows
- `inventory.csv` - 154.7M rows (daily snapshots)
- `metrics.json` - Summary KPIs

**Static World:**
- `products.csv` - ~700 rows (ingredients + SKUs flattened)
- `locations.csv` - ~4,500 rows (all node types flattened)
- `recipes.csv` - ~500 rows (BOMs)
- `links.csv` - ~6,000 rows (network edges)
- `partners.csv` - ~20 rows (suppliers)

---

## Gap Analysis: 71 Tables vs What We Can Export

### Tier 1: Can Export Now (15 tables) ✅
Data already in CSVs, just needs normalization:

| Table | Source |
|-------|--------|
| suppliers | locations.csv WHERE type=SUPPLIER |
| plants | locations.csv WHERE type=PLANT |
| distribution_centers | locations.csv WHERE type=RDC/DC |
| retail_locations | locations.csv WHERE type=STORE |
| ingredients | products.csv WHERE category=INGREDIENT |
| skus | products.csv WHERE category!=INGREDIENT |
| formulas | recipes.csv (header) |
| formula_ingredients | recipes.csv (unnest ingredients dict) |
| links | links.csv |
| orders | orders.csv (dedupe headers) |
| order_lines | orders.csv (line details) |
| shipments | shipments.csv (dedupe headers) |
| shipment_lines | shipments.csv (line details) |
| batches | batches.csv |
| inventory | inventory.csv |

### Tier 2: Hidden in Sim, Not Exported (~12 tables) 🔶
Data exists in simulation but not written to CSV:

| Table | Where It Lives | Export Effort |
|-------|----------------|---------------|
| production_lines | simulation_config.json plant_parameters | Low - extract from config |
| channels | CustomerChannel enum + config | Low - generate from enum |
| batch_ingredients | TransformEngine consumes ingredients | Medium - add to writer |
| demand_forecasts | POSEngine.get_deterministic_forecast() | Medium - add to writer |
| kpi_actuals | metrics.json (partial) | Low - expand metrics |
| risk_events | RiskEventManager logs | Medium - add to writer |
| promotions | PromoCalendar (if enabled) | Medium - add to writer |

### Tier 3: Would Require Sim Changes 🔴

**REVISED - Focus on meaningful gaps:**

| Gap | Priority | Effort | Tables Added |
|-----|----------|--------|--------------|
| **Procurement** | HIGH | Medium | purchase_orders, purchase_order_lines, goods_receipts, supplier_ingredients (~4) |
| **Work Orders** | MEDIUM | Low | work_orders, work_order_materials (~2) |
| **S&OP Export** | MEDIUM | Low | demand_forecasts, supply_plans (~2) - data exists, just not exported |
| **Returns** | LOW | Medium | returns, return_lines, disposition_logs (~3) |

**SKIP (low value or complex):**
- order_allocations - happens but logging adds little analytics value
- carriers, routes, shipment_legs - simplified logistics is fine for now
- ports, pick_waves - not modeled, low priority
- divisions, retail_accounts - organizational hierarchy, can seed as Tier 4

**Procurement Detail (HIGH priority):**
- Currently: `plant_ingredient_buffer = 5M` (effectively infinite)
- Need: Track ingredient inventory → PO when low → supplier lead time → goods receipt
- Impact: Creates realistic constraints (can't produce without ingredients)

**Work Orders Detail (MEDIUM priority):**
- Currently: MRP → Batch (direct)
- Need: MRP → Work Order (planned) → Release → Batch (executed)
- Impact: Mostly data modeling, adds WO tracking layer

**S&OP Export Detail (MEDIUM priority):**
- Currently: POSEngine.get_deterministic_forecast() exists, MRP uses it
- Need: Export forecasts + aggregate production plans as formal "plan" documents
- Impact: Export-only change, no new physics

**Returns Detail (LOW priority):**
- Currently: No returns
- Need: X% of delivered shipments → return → DC → disposition (restock/scrap)
- Impact: New simple simulation loop

### Tier 4: Reference/Config Data (~24 tables) 📋
Could be generated as static seed data:

| Table | Approach |
|-------|----------|
| certifications | Generate from supplier attributes |
| packaging_types | Derive from product dimensions |
| sku_costs | Derive from cost_per_case |
| carrier_rates | Generate based on distance |
| kpi_thresholds | Copy from simulation_config |
| business_rules | Generate from config parameters |

---

## Recommended Scope (Revised)

| Phase | What | Tables | Effort |
|-------|------|--------|--------|
| **1. Export** | Tier 1 + 2: Normalize CSVs + export hidden sim data | ~27 | This PR |
| **2. Seed** | Tier 4: Generate reference/config data | ~24 | Fast follow |
| **3a. Procurement** | Add ingredient PO simulation | ~4 | Medium |
| **3b. Work Orders** | Layer WO before batches | ~2 | Low |
| **3c. S&OP Export** | Formalize forecast/plan export | ~2 | Low |
| **3d. Returns** | Lean returns simulation | ~3 | Medium |

**Total: ~62 tables** (vs 71 in original schema)

**Intentionally skipped (~9 tables):**
- order_allocations, carriers, routes, shipment_legs, ports, pick_waves
- These add complexity without proportional analytics value

---

## Implementation: Post-Processing Script

### New File: `scripts/export_erp_format.py`

A standalone script that reads flat CSVs and outputs ERP-normalized tables.

### Transformations

#### 1. Split `locations.csv` by NodeType
```
locations.csv (4,500 rows)
    ├── suppliers.csv       (~20 rows)   WHERE type = 'NodeType.SUPPLIER'
    ├── plants.csv          (~4 rows)    WHERE type = 'NodeType.PLANT'
    ├── distribution_centers.csv (~12 rows) WHERE type LIKE '%RDC%' or '%DC%'
    └── retail_locations.csv (~4,400 rows) WHERE type = 'NodeType.STORE'
```

#### 2. Split `orders.csv` into header + lines
Current: Same `order_id` repeats for each product (13.5M rows)
```
orders.csv (flat)
    ├── orders.csv (headers)      - DISTINCT order_id, day, source_id, target_id, status
    └── order_lines.csv (details) - order_id, line_number, product_id, quantity
```

#### 3. Split `shipments.csv` into header + lines
Current: Same `shipment_id` repeats for each product (15M rows)
```
shipments.csv (flat)
    ├── shipments.csv (headers)      - DISTINCT shipment_id, creation_day, arrival_day, source_id, target_id, status
    └── shipment_lines.csv (details) - shipment_id, line_number, product_id, quantity, weight_kg, volume_m3
```

#### 4. Split `products.csv` by category
```
products.csv (700 rows)
    ├── ingredients.csv (~200 rows) WHERE category = 'ProductCategory.INGREDIENT'
    └── skus.csv        (~500 rows) WHERE category != 'INGREDIENT'
```

#### 5. Generate integer ID lookup tables
Create `id_mapping.json` for FK relationships:
```json
{
  "locations": {"SUP-001": 1, "PLANT-CA": 2, ...},
  "products": {"PKG-BOTTLE-001": 1, "SKU-HOME-001": 2, ...}
}
```

### Output Structure
```
data/output/erp_format/
├── master/
│   ├── suppliers.csv
│   ├── plants.csv
│   ├── distribution_centers.csv
│   ├── retail_locations.csv
│   ├── ingredients.csv
│   ├── skus.csv
│   ├── recipes.csv (renamed from recipes.csv)
│   └── links.csv (renamed from links.csv)
├── transactional/
│   ├── orders.csv
│   ├── order_lines.csv
│   ├── shipments.csv
│   ├── shipment_lines.csv
│   ├── batches.csv
│   └── inventory.csv (sampled or partitioned)
└── reference/
    ├── id_mapping.json
    └── schema.sql (simplified, matching output)
```

---

## Simplified Target Schema (~15 tables)

Based on what we can derive from current simulation output:

```sql
-- Master Data
CREATE TABLE suppliers (id SERIAL PRIMARY KEY, code VARCHAR, name VARCHAR, ...);
CREATE TABLE plants (id SERIAL PRIMARY KEY, code VARCHAR, name VARCHAR, ...);
CREATE TABLE distribution_centers (id SERIAL PRIMARY KEY, code VARCHAR, name VARCHAR, ...);
CREATE TABLE retail_locations (id SERIAL PRIMARY KEY, code VARCHAR, name VARCHAR, ...);
CREATE TABLE ingredients (id SERIAL PRIMARY KEY, code VARCHAR, name VARCHAR, category VARCHAR, ...);
CREATE TABLE skus (id SERIAL PRIMARY KEY, code VARCHAR, name VARCHAR, category VARCHAR, ...);
CREATE TABLE recipes (id SERIAL PRIMARY KEY, product_id INT, ingredients JSONB, ...);
CREATE TABLE links (id SERIAL PRIMARY KEY, source_id INT, target_id INT, distance_km DECIMAL, ...);

-- Transactional Data
CREATE TABLE orders (id SERIAL PRIMARY KEY, order_number VARCHAR, day INT, source_id INT, target_id INT, status VARCHAR);
CREATE TABLE order_lines (order_id INT, line_number INT, product_id INT, quantity DECIMAL, PRIMARY KEY(order_id, line_number));
CREATE TABLE shipments (id SERIAL PRIMARY KEY, shipment_number VARCHAR, creation_day INT, arrival_day INT, ...);
CREATE TABLE shipment_lines (shipment_id INT, line_number INT, product_id INT, quantity DECIMAL, ...);
CREATE TABLE batches (id SERIAL PRIMARY KEY, batch_number VARCHAR, plant_id INT, product_id INT, day INT, ...);
CREATE TABLE inventory (id SERIAL PRIMARY KEY, day INT, location_id INT, product_id INT, quantity DECIMAL);
```

---

## Verification

1. Run post-processor:
   ```bash
   poetry run python scripts/export_erp_format.py --input data/output --output data/output/erp_format
   ```

2. Validate row counts:
   ```bash
   # orders + order_lines should sum to original orders.csv rows
   wc -l data/output/erp_format/transactional/orders.csv
   wc -l data/output/erp_format/transactional/order_lines.csv
   ```

3. Test Postgres load (in new repo):
   ```bash
   psql -d prism_db -f schema.sql
   psql -d prism_db -c "\COPY suppliers FROM 'suppliers.csv' CSV HEADER"
   # ... etc
   ```

---

## Files to Create
- `scripts/export_erp_format.py` - Main post-processing script
- `scripts/erp_schema.sql` - Simplified Postgres schema for export

## Files to Modify
- None (post-processing only, no sim changes)
