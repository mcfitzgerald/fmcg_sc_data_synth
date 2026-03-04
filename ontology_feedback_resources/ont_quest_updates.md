# Ontology & Benchmark Question Update Guide

**Purpose:** After completing sim-level fixes (Chunks 1-4, v0.81.0-v0.84.0), both the PCG ontology (`pcg.yaml`) and the 85 benchmark questions (`questions.md`) need updating to reflect the new data reality. This guide documents every change needed.

**How to use:** Work through this in a separate session with access to a loaded PostgreSQL database (from a fresh ERP export after the stabilization + 365-day run). Query the DB to get real entity codes and verify counts before updating.

---

## Part A: Benchmark Question Updates

### A1. Entity Code Format Mapping

The questions use fictional code formats. Below is the mapping to actual formats.

| Entity | Question Format | Actual Format | Example |
|--------|----------------|---------------|---------|
| Supplier | `SUP-0012` | `SUP-###` (3-digit) | `SUP-012` |
| Ingredient | `ING-RM-GLYC-001` | `ACT-{TYPE}-###` / `BLK-{TYPE}-###` / `PKG-{PRI\|SEC\|TER}-{TYPE}-###` | `ACT-HUMECTANT-001` |
| SKU | `SKU-OC-TPST-042` | `SKU-{CATEGORY}-###` | `SKU-ORAL-042` |
| SKU (old) | `SKU-OC-TPST-042-OLD` | No alias chain exists | N/A |
| Formula | `FRM-OC-MINT-001` | Query `formulas` table for real IDs | (integer or generated) |
| Batch | `B-2024-05831` | Query `batches` table for real batch_number | (generated format) |
| Order | `ORD-2024-08471` | Query `orders` table for real order_number | (generated format) |
| Shipment | `SHP-2024-11020` | Query `shipments` table for real shipment_id | (generated format) |
| PO | `PO-2024-00350` | Query `purchase_orders` table for real po_number | (generated format) |
| GR | `GR-2024-02200` | Query `goods_receipts` table for real gr_number | (generated format) |
| Return | `RTN-2024-00150` | Query `returns` table for real return_id | (generated format) |
| Retail Location | `RTL-MR-0042` | `STORE-RET-###-####` / `STORE-GRO-###-####` / etc. | `STORE-RET-001-0042` |
| Plant (Dallas) | "Dallas plant" | `PLANT-TX` | Texas Plant |
| Plant (Columbus) | "Columbus plant" | `PLANT-OH` | Ohio Plant |
| Plant (Sacramento) | "Sacramento plant" | `PLANT-CA` | California Plant |
| Plant (Atlanta) | "Atlanta plant" | `PLANT-GA` | Georgia Plant |
| RDC (Memphis) | "Memphis RDC" | `RDC-SO` or `RDC-SE` | Check city in locations table |
| RDC (Chicago) | "Chicago RDC" | `RDC-MW` | Midwest DC |
| RDC (Jacksonville) | "Jacksonville RDC" | `RDC-SE` | Southeast DC |
| RDC (Phoenix) | "Phoenix RDC" | `RDC-SW` or `RDC-WE` | Check city in locations table |
| Distribution Center | "DC" | `RET-DC-###` / `GRO-DC-###` / `CLUB-DC-###` / `ECOM-FC-###` / etc. | `RET-DC-001` |
| Channel | Generic | `MASS_RETAIL` / `GROCERY` / `CLUB` / `PHARMACY` / `DISTRIBUTOR` / `ECOM` / `DTC` | 7 channels |

### A2. Queries to Run for Code Substitution

Run these against the loaded PostgreSQL ERP database to get real codes:

```sql
-- Suppliers (pick ones with interesting characteristics)
SELECT supplier_code, name, city, country, kralqic_tier, active
FROM suppliers ORDER BY supplier_code LIMIT 20;

-- Ingredients by type (get real names after S1 fix)
SELECT ingredient_code, name, category, subcategory, cost_per_kg
FROM ingredients ORDER BY category, subcategory;

-- SKUs by category (get real product codes)
SELECT sku_code, name, category, segment, price_per_case, active
FROM skus WHERE category = 'ORAL_CARE' ORDER BY sku_code LIMIT 20;

-- Formulas (get real formula IDs, now includes premix/multi-level)
SELECT f.id, f.product_id, f.bom_level, f.batch_size_kg
FROM formulas f ORDER BY f.bom_level DESC, f.id LIMIT 30;

-- Batches (get real batch numbers)
SELECT batch_number, product_type, bom_level, quantity_kg
FROM batches ORDER BY batch_number LIMIT 20;

-- Orders with diverse statuses (after S4 reporting_date fix)
SELECT order_number, status, COUNT(*)
FROM orders GROUP BY order_number, status
ORDER BY order_number LIMIT 20;

-- Shipments with diverse statuses
SELECT shipment_id, status, COUNT(*)
FROM shipments GROUP BY shipment_id, status
ORDER BY shipment_id LIMIT 20;

-- Purchase orders with diverse statuses
SELECT po_number, status, COUNT(*)
FROM purchase_orders GROUP BY po_number, status
ORDER BY po_number LIMIT 20;

-- Retail locations (get real store codes)
SELECT location_code, name, city, location_type
FROM (
    SELECT * FROM retail_locations
    UNION ALL
    SELECT * FROM distribution_centers
    UNION ALL
    SELECT * FROM plants
) all_locations
ORDER BY location_type, location_code LIMIT 30;

-- Route segments (now includes lateral + multi-source)
SELECT origin_id, destination_id, origin_type, destination_type,
       distance_km, transit_time_hours
FROM route_segments
WHERE origin_type = destination_type  -- lateral links
ORDER BY distance_km LIMIT 20;

-- Lead times (now sourcing-aware, 5-73 days)
SELECT si.ingredient_id, i.name, s.name as supplier_name,
       si.lead_time_days, si.unit_cost
FROM supplier_ingredients si
JOIN suppliers s ON si.supplier_id = s.id
JOIN ingredients i ON si.ingredient_id = i.id
WHERE si.lead_time_days > 30
ORDER BY si.lead_time_days DESC;

-- BOM depth verification (new variable depth)
SELECT f.bom_level, COUNT(*) as formula_count
FROM formulas f GROUP BY f.bom_level ORDER BY f.bom_level;

-- Multi-level BOM example (find a 3-level SKU)
SELECT fi.formula_id, f.product_id, f.bom_level, fi.ingredient_id, fi.quantity_kg
FROM formula_ingredients fi
JOIN formulas f ON fi.formula_id = f.id
WHERE fi.ingredient_id LIKE 'PREMIX-%'
ORDER BY f.bom_level, fi.formula_id;

-- GL anomalies (after S6 fix)
SELECT reference_id, COUNT(*) as entries
FROM gl_journal
WHERE reference_id LIKE '%-DUP'
GROUP BY reference_id LIMIT 10;
```

### A3. Per-Question Update Notes

#### Section 1: Lookups & Joins (Q01-Q10)

| Q | Issue | Fix |
|---|-------|-----|
| Q01 | `SUP-0012` wrong format | Query for a real supplier code (e.g., `SUP-012`). Use one with interesting Kraljic tier. |
| Q02 | "ChemSource" — check if this is an actual supplier name now | Query `suppliers` for a Dallas-based supplier. S1 added real names (e.g., "Meridian Chemical Corp"). Use an actual name. No `-ALT` codes exist in current data — either remove that part or add alias friction to ERP. |
| Q03 | "open" POs — S4 added status diversity | Verify `purchase_orders` actually has `status='open'` rows. Update plant city names if needed (Dallas=PLANT-TX, Columbus=PLANT-OH, Sacramento=PLANT-CA, Atlanta=PLANT-GA). |
| Q04 | "Premium segment" — verify segment values exist in `skus` | Query `SELECT DISTINCT segment FROM skus`. Price threshold $45 should work. |
| Q05 | `ORD-2024-08471` — fictional | Pick a real order_number from the DB. Choose one with multiple lines. |
| Q06 | `FRM-OC-MINT-001` — fictional formula ID | Query for a real Oral Care formula. Also: formulas now include PREMIX-level (bom_level=2). |
| Q07 | Should work as-is | Verify channel names match (`MASS_RETAIL`, `GROCERY`, etc.). |
| Q08 | "pending, allocated, shipped, delivered" — now actually exist! | S4 added status lifecycle. This question now works. Verify status values match. |
| Q09 | Should work as-is | No code changes needed. |
| Q10 | ">30 days" threshold — now valid! | S2 added sourcing-aware lead times (5-73 days). This question now works without threshold changes. |

#### Section 2: SKU Alias Chains (Q11-Q18)

| Q | Issue | Fix |
|---|-------|-----|
| Q11-Q18 | **SKU alias/supersession chains do not exist in current data** | The sim has no SKU rename/supersession feature. These 8 questions are **unanswerable** without adding a `sku_aliases` table. **Decision needed:** (a) Add a `sku_aliases` table to the ERP generator with synthetic rename chains, or (b) Replace these 8 questions with answerable alternatives. |

**Recommendation:** Replace Section 2 with questions about the new variable BOM depth, multi-intermediate SKUs, and diamond dependencies — features that now exist and are more interesting for VKG testing.

**Alternative replacement questions (suggestions):**
- "Which SKUs use more than one bulk intermediate in their formula?"
- "Which raw materials appear in both a PREMIX sub-intermediate and a direct bulk recipe (diamond dependencies)?"
- "What is the deepest BOM path from any SKU to raw materials?"
- "Which premix sub-intermediates are shared across multiple bulk formulas?"

#### Section 3: BOM Explosion & Cost Rollup (Q19-Q28)

| Q | Issue | Fix |
|---|-------|-----|
| Q19 | `FRM-OC-MINT-001` fictional | Use a real formula ID for an Oral Care product. |
| Q20 | "Explode full BOM" — **now interesting!** | S8 added variable depth. Pick a 3-level SKU (one that goes through a PREMIX). Use real SKU code (`SKU-ORAL-###`). |
| Q21 | "sodium fluoride" — now a real ingredient name! | S1 added real names. Verify `ACT-FLUORIDE-###` has name "Sodium Fluoride" in `ingredients`. Use real SKU code. |
| Q22 | `SKU-PW-SOAP-010` → `SKU-PERSONAL-###` | Use a real Personal Wash SKU code. |
| Q23 | "citric acid" — verify this is an actual ingredient name | Query `ingredients` for real names. Pick one that appears in multiple formulas. |
| Q24 | `SKU-HM-DISH-005` → `SKU-HOME-###` | **Now interesting!** S8 added variable depth. Pick a deep-BOM SKU. Use real code. |
| Q25 | `SKU-OC-MWSH-003` → `SKU-ORAL-###` | **Now interesting!** S8 added multi-intermediate SKUs and premixes. Pick one with PREMIX in its BOM tree. |
| Q26 | Should work as-is | No code changes needed. |
| Q27 | `B-2024-05831` fictional | **Now interesting!** S3 added ±3% batch variance. Pick a real batch with measurable variance. |
| Q28 | Should work as-is | S8 increased shared ingredients (diamond deps). More interesting now. |

#### Section 4: Transport Network Analysis (Q29-Q40)

| Q | Issue | Fix |
|---|-------|-----|
| Q29 | `RTL-MR-0042` → `STORE-RET-###-####` | Use real store code. "Dallas plant" → `PLANT-TX`. |
| Q30 | Same as Q29 | Same fix. |
| Q31 | "Columbus plant to Jacksonville RDC" | `PLANT-OH` to `RDC-SE`. **Now interesting!** S7 added lateral links — may create alternate paths. |
| Q32 | Should work as-is | S7 added lateral links, so hub degree analysis is more interesting. |
| Q33 | Betweenness centrality — **now non-trivial!** | S7 added redundant routes, so shortest paths aren't unique anymore. |
| Q34 | Closeness centrality | Works as-is. More interesting with S7 topology. |
| Q35 | Eigenvector/PageRank centrality | Works as-is. |
| Q36 | Connectivity check | S7 didn't fragment the network. Should still be connected. |
| Q37 | "Atlanta plant shut down" → `PLANT-GA` | S7 multi-sourcing means some DCs have backup paths. More realistic answer. |
| Q38 | "Memphis RDC to Chicago RDC" | Map to actual RDC codes. S7 added lateral RDC links — verify these two are connected. |
| Q39 | "Sacramento plant to RTL-GR-0107, avoid Phoenix RDC" | `PLANT-CA` to real store code. Map "Phoenix RDC" to actual code. |
| Q40 | `RTL-EC-0005` → `STORE-ECOM-###-####` or similar | Use real ecommerce store code. |

#### Section 5: Cross-Domain Composition (Q41-Q50)

| Q | Issue | Fix |
|---|-------|-----|
| Q41 | `SKU-OC-TPST-001` → `SKU-ORAL-001` | Use real code. `-ALT` supplier duplicates don't exist — remove or reframe. |
| Q42 | "landed cost to Dallas plant" → `PLANT-TX` | Use real SKU and formula codes. |
| Q43 | `SKU-PW-SOAP-022` alias chain | **Alias chains don't exist.** Rewrite: e.g., "What is the total inventory of [real SKU] across all locations?" |
| Q44 | "Chicago RDC" → `RDC-MW` | "pending" orders now exist (S4). Use real codes. |
| Q45 | "palm oil" — verify ingredient name | Query `ingredients` for real names. Pick one that maps to an active chemical. |
| Q46 | `SKU-OC-MWSH-003` → real code | Use real Oral Care SKU. AP invoices exist. |
| Q47 | `RTL-MR-0100` → real store, `SKU-HM-DISH-005` → real Home Care SKU | Use real codes. |
| Q48 | "released" work orders — verify status exists | S4 added status diversity to work orders. Check actual WO statuses. "glycerin" → verify real ingredient name. |
| Q49 | `ORD-2024-12045` → real order | Pick a real order with multiple lines. |
| Q50 | Should work as-is | Category names: `ORAL_CARE`. |

#### Section 6: Multi-Graph & Polymorphism (Q51-Q60)

| Q | Issue | Fix |
|---|-------|-----|
| Q51 | `SKU-OC-TPST-001` → `SKU-ORAL-001` | Use real code. |
| Q52 | `ORD-2024-09876` → real order | Pick a real order. |
| Q53 | `RTL-PH-0200` → `STORE-PHARM-###-####` | Use real pharmacy store code. |
| Q54 | Should work as-is | Location type filtering works. |
| Q55 | "Atlanta metro area" | Query retail locations for ones near Atlanta. |
| Q56 | BOM cycle detection — **now interesting!** | S8 added diamond deps and variable depth. Cycle detection is a real test now. |
| Q57 | `B-2024-08200` → real batch | **Now interesting!** S3 added ±3% variance. Pick a real batch. |
| Q58 | `SUP-0003` → `SUP-003` | Use real code. `-ALT` duplicates don't exist — remove that part. |
| Q59 | "Dallas plant" → `PLANT-TX` | "open orders" now exist (S4). |
| Q60 | Should work as-is | Route type parsing works. |

#### Section 7: Data Integrity & Metamodel (Q61-Q68)

| Q | Issue | Fix |
|---|-------|-----|
| Q61 | "Columbus plant" → `PLANT-OH` | Verify production_lines table has active/inactive status. |
| Q62 | Should work as-is | S2 enriched lead times. Invoice variances exist (S6/v0.78.0). |
| Q63 | Should work as-is | Data integrity check. |
| Q64 | **Now answerable!** | S4 added status diversity. Was previously trivial (all CLOSED). |
| Q65 | "order 4502, line 3" → real order number + line | Pick a real order with 3+ lines. |
| Q66 | "day 180 inventory snapshot" | Verify inventory snapshots exist for this day. |
| Q67 | SKU supersession cycles — **no alias data exists** | Either add alias table or replace question. |
| Q68 | Should work as-is | Cross-document consolidation. |

#### Section 8: Lifecycle & Flow Analysis (Q69-Q76)

| Q | Issue | Fix |
|---|-------|-----|
| Q69 | `ORD-2024-07744` → real order in "pending" | **Now answerable!** S4 created pending orders. Pick one. |
| Q70 | `SHP-2024-11020` in "planned" status | **Now answerable!** S4 created diverse shipment statuses. Pick one. |
| Q71 | `ING-RM-GLYC-001` → real ingredient code | "glycerin" is now `ACT-HUMECTANT-###` (verify). Use real code. |
| Q72 | `SUP-0008` → `SUP-008` | AP invoices and payments exist. |
| Q73 | `B-2024-06500` → real batch | **Now interesting!** S3 variance means mass balance won't be exact. |
| Q74 | "day 200 GL balance" | GL anomalies exist (S6): `-DUP` entries and rounding imbalances on ~1% of days. **Now interesting!** |
| Q75 | `ORD-2024-15200` → real pending order | S4 created pending orders with unfulfilled status. |
| Q76 | "Dallas plant capacity -20%" | `PLANT-TX`. Check production_lines for capacity data. |

#### Section 9: End-to-End Traceability (Q77-Q85)

| Q | Issue | Fix |
|---|-------|-----|
| Q77 | `B-2024-04100` → real batch | Pick a batch with shipments that reached retail. |
| Q78 | `SHP-2024-08500` → real shipment | Pick a shipment with complete traceability. |
| Q79 | `GR-2024-02200` → real GR | Pick a GR for a common ingredient. "citric acid" → real name. |
| Q80 | `SUP-0019` → `SUP-019` | Contamination blast radius question. Works better with S8 diamond deps. |
| Q81 | `PO-2024-00350` → real PO | Procure-to-pay chain. AP payments exist (v0.78.0). |
| Q82 | `ORD-2024-10500` → real order | Order-to-cash chain. AR receipts exist (v0.78.0). |
| Q83 | "Personal Wash category" → `PERSONAL_WASH` | Verify category name. |
| Q84 | `RTN-2024-00150` → real return | Pick a return with full traceability. |
| Q85 | `RTL-MR-0042` → real store, `SKU-OC-TPST-001` → `SKU-ORAL-001` | "Dallas" → `PLANT-TX`. Use real codes. |

### A4. Summary of Question Disposition

| Category | Count | Action |
|----------|-------|--------|
| Code format swap only | ~45 | Replace fictional codes with real ones from DB queries |
| Now answerable (were blocked) | ~20 | Update codes, verify question still makes sense |
| Need rewrite (alias chains) | 8 | Q11-Q18: Replace with BOM depth / multi-intermediate questions |
| Need minor rewording | ~10 | Remove `-ALT` references, update ingredient names to real chemicals |
| Work as-is | ~5 | Q07, Q09, Q26, Q28, Q63 |

---

## Part B: Ontology (pcg.yaml) Updates

The ontology's **context blocks** (`business_logic`, `llm_prompt_hint`, `data_note`, state machines) contain stale information from the pre-fix data. Here's what needs updating.

### B1. Data Notes — Status Lifecycle (Chunk 2 / S4)

All `data_note` fields that say "all X have status=Y" are now wrong. S4 added reporting_date-based status diversity.

**Update these `data_note` blocks:**

| Entity | Old data_note | New data_note |
|--------|--------------|---------------|
| `PurchaseOrder` | "all POs have status=CLOSED" | "PO statuses reflect lifecycle pipeline: ~70% closed, ~15% received, ~10% approved, ~5% pending (proportions vary by reporting_date)" |
| `GoodsReceipt` | "all GRs have status=received" | "GR statuses: ~85% received, ~10% inspecting, ~5% pending" |
| `Order` | "all orders have status=CLOSED" | "Order statuses reflect pipeline: ~65% closed, ~15% shipped, ~10% allocated, ~10% pending" |
| `Shipment` | "all shipments have status=in_transit" | "Shipment statuses: ~60% delivered, ~20% in_transit, ~15% planned, ~5% loading" |
| `Batch` | "all batches have status=complete" | "Batch statuses: ~80% complete, ~10% in_progress, ~10% planned" |
| `WorkOrder` | "statuses are planned, in_progress, complete" | "Work order statuses: planned, in_progress, complete (diverse distribution, unchanged)" |
| `Return` | "all returns have status=received" | "Return statuses: ~80% received, ~10% inspecting, ~10% pending" |
| `APInvoice` | "all AP invoices have status=open" | "AP invoice statuses: ~60% paid, ~25% open, ~10% approved, ~5% pending" |
| `ARInvoice` | "disputed, open, partial" | "AR invoice statuses: ~70% paid, ~20% open, ~5% partial, ~3% disputed, ~2% bad_debt" |

> **Important:** Query the actual database to get real percentages before updating. The above are estimates based on the state machine logic.

### B2. BOM/Formula Context (Chunk 4 / S8)

**Formula class `business_logic`** (line ~341):
- Old: "545 formulas total: 500 SKU-level (bom_level=0) and 45 bulk-level (bom_level=1)"
- New: "~580 formulas total: 500 SKU-level (bom_level=0), ~49 bulk-level (bom_level=1), and ~13 premix-level (bom_level=2). Product portfolio has variable BOM depth: ~70% of SKUs are 2-level, ~20% are 3-level (via PREMIX sub-intermediates), ~10% use multiple intermediates. product_id is polymorphic..."

**Formula class `llm_prompt_hint`** (line ~342):
- Old: "For full BOM explosion: start at bom_level=0 formula, find bulk intermediate ingredients, look up their bom_level=1 formula, recurse."
- New: "For full BOM explosion: start at bom_level=0 formula, find intermediate ingredients (bulk or premix), look up their formulas at bom_level=1 or bom_level=2, recurse until only raw materials remain. BOM depth varies: some paths are 2 levels, others are 3-4. Diamond dependencies exist (same raw material reachable via multiple intermediate paths)."

**FormulaIngredient `business_logic`** (line ~363):
- Update to mention that `ingredient_id` can now point to PREMIX products (bom_level=2) in addition to bulk intermediates and raw materials.

**BulkIntermediate class `business_logic`** (line ~507):
- Old: "45 intermediate products representing the compounding stage"
- New: "~62 intermediate products (49 primary bulks at bom_level=1, 13 premix sub-intermediates at bom_level=2). Includes secondary blend intermediates for ~10% of SKUs (multi-intermediate formulas). Bridges raw materials to finished goods in variable-depth BOM (2-4 levels)."

**BulkIntermediate `llm_prompt_hint`** (line ~508):
- Old: "To find what raw materials make a bulk intermediate: join formulas (bom_level=1...)"
- New: "To find what raw materials make a bulk intermediate: join formulas matching the intermediate's bom_level (1 or 2). For bom_level=1 bulks, ingredients may include PREMIX products (bom_level=2) — recurse to get raw materials. For bom_level=2 premixes, ingredients are always raw materials."

**Batch `business_logic`** (line ~415):
- Add mention that batches now exist for PREMIX products (bom_level=2) as well as bulks and SKUs.

**Discriminator mapping** (line ~1455):
- Old: `{"0": "SKU", "1": "BulkIntermediate"}`
- New: `{"0": "SKU", "1": "BulkIntermediate", "2": "BulkIntermediate"}` (premixes are still BULK_INTERMEDIATE category)

**Formula count in polymorphic relationship** (line ~1458):
- Old: "500 formulas produce SKUs, 45 produce bulk intermediates"
- New: "500 formulas produce SKUs, ~49 produce primary bulk intermediates (bom_level=1), ~13 produce premix sub-intermediates (bom_level=2)"

**BOM explosion semantics** (line ~1479):
- Old: mentions 2-level traversal pattern
- New: mention N-level traversal — ingredient_id may point to bom_level=2 intermediates that require further recursion

### B3. Ingredient/Supplier Names (Chunk 1 / S1)

**Ingredient class `business_logic`** (line ~131):
- Old: "78 raw materials organized by category/subcategory"
- New: "78 raw materials with real chemical/packaging names (e.g., Sodium Fluoride, Glycerin, Sodium Lauryl Sulfate). Organized by category/subcategory (base_material, active_ingredient, packaging)."

**Supplier class** (if mentioned):
- Note that suppliers now have real company names (e.g., "Meridian Chemical Corp", "Pacific Packaging Solutions") instead of "Supplier N".

### B4. Lead Times (Chunk 1 / S2)

**SupplierIngredient `llm_prompt_hint`** (line ~156):
- Old: generic lead_time_days mention
- New: "lead_time_days ranges from 5-73 days, driven by sourcing origin: LOCAL sources 5-15d, REGIONAL 15-30d, GLOBAL 30-75d. For supply risk assessment, focus on GLOBAL-sourced ingredients with lead_time_days > 30."

### B5. Order Line Prices (Chunk 1 / S5)

**OrderLine / Order `llm_prompt_hint`** (line ~582):
- Old: "IMPORTANT: order_lines.unit_price is ALWAYS ZERO"
- New: "order_lines.unit_price is populated from product price_per_case (FG orders) or supplier unit_cost (PO lines). Revenue analysis can use either order_lines or ar_invoice_lines."

### B6. Batch Variance (Chunk 1 / S3)

**BatchIngredient context:**
- Add: "Actual consumed quantities include ±3% recording variance vs formula specification. This makes planned-vs-actual comparison (formula quantity vs batch_ingredient quantity) non-trivial."

### B7. Network Topology (Chunk 3 / S7)

**RouteSegment class:**
- Old: implied tree topology
- New: "Network includes hub-and-spoke base plus lateral RDC-to-RDC links (6 pairs, 12 directed edges), multi-source DCs (16 DCs served by 2 upstream RDCs, ~29% of customer DCs), and secondary-source ordering (20% of multi-source DC orders use alternate upstream). Graph is no longer a strict tree — shortest-path and centrality analyses are non-trivial."

**DistributionCenter class:**
- Add: "~29% of customer DCs are multi-sourced (served by 2 upstream RDCs). Primary vs secondary sourcing creates genuine route alternatives."

### B8. GL Anomalies (Chunk 2 / S6)

**GLJournal context:**
- Add: "~0.5% of GL postings are deliberate duplicates (reference_id suffix '-DUP'). ~1% of days have small rounding imbalances ($0.01-$1.00). These anomalies are intentional for data quality testing."

### B9. Product Count Updates

Anywhere the ontology mentions "500 SKUs" or "45 bulk intermediates" or "78 raw materials":
- SKUs: still 500 (unchanged)
- Raw materials: still 78 (unchanged)
- Bulk intermediates: ~49 primary bulks (bom_level=1) + ~13 premixes (bom_level=2) = ~62 total
- Total products: ~640 (was 623)
- Formulas: ~562 (was 545)

---

## Part C: Section 2 Replacement Suggestions

Since Q11-Q18 (SKU alias chains) are unanswerable without a `sku_aliases` table, here are replacement questions that exercise the new BOM depth features:

**Q11 (new).** Which finished SKUs have a BOM tree deeper than 2 levels? For each, show the SKU, the number of BOM levels, and the intermediate products at each level.

**Q12 (new).** For SKU [pick a 3-level SKU], explode the full BOM and show the tree structure: SKU -> bulk intermediate(s) -> premix sub-intermediate(s) -> raw materials. Include quantities at each level.

**Q13 (new).** Which raw materials are "diamond dependencies" — appearing in both a premix sub-intermediate's formula AND directly in its parent bulk intermediate's formula? These shared components create procurement concentration risk.

**Q14 (new).** Which finished SKUs use more than one bulk intermediate in their formula? For each, show the primary and secondary intermediates and their weight fractions.

**Q15 (new).** For premix sub-intermediate [pick a PREMIX], find all finished SKUs that ultimately depend on it — tracing through the multi-level BOM. How many SKUs are affected if this premix has a quality issue?

**Q16 (new).** Compare the BOM cost structure of 2-level SKUs vs 3-level SKUs: what is the average raw material cost per batch for each depth category? Does the extra processing level change the cost profile?

**Q17 (new).** Which bulk intermediates consume the most distinct raw materials across their full BOM tree (including any sub-intermediates)? Rank by total unique ingredient count.

**Q18 (new).** Starting from raw material [pick a widely-used one], trace forward through every intermediate and finished good that uses it — at any BOM level. What is the total revenue exposure if this ingredient becomes unavailable?

---

## Part D: Workflow Checklist

1. [ ] Run stabilization (3-year) + 365-day true run + ERP export
2. [ ] Load ERP CSVs into PostgreSQL
3. [ ] Run queries from Section A2 to collect real entity codes
4. [ ] Update `questions.md` with real codes (Section A3 table)
5. [ ] Decide on Section 2: replace alias chain questions or add `sku_aliases` table
6. [ ] If replacing, write new Q11-Q18 (see Part C suggestions)
7. [ ] Update `pcg.yaml` data_note blocks (Section B1)
8. [ ] Update `pcg.yaml` BOM/formula context (Section B2)
9. [ ] Update `pcg.yaml` ingredient/supplier/lead time context (B3-B4)
10. [ ] Update `pcg.yaml` order line prices and batch variance (B5-B6)
11. [ ] Update `pcg.yaml` network topology context (B7)
12. [ ] Update `pcg.yaml` GL anomaly context (B8)
13. [ ] Update `pcg.yaml` product counts (B9)
14. [ ] Re-run VKG benchmark test (85 questions against updated ontology + data)
15. [ ] Document pass/fail results in new feedback file
