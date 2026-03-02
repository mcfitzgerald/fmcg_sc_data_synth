# Benchmark First Pass Feedback

Results from running all 85 PCG benchmark questions against the live database (2026-03-02).

## Executive Summary

| Category | Questions Affected | Effort to Fix |
|---|---|---|
| Entity code format mismatches | 45 / 85 | **Trivial** — sample real codes, update questions |
| Generic ingredient names | 6 / 85 | **Trivial** — rename seed data labels |
| Status monoculture | 8 / 85 | **Sim change** — snapshot mid-run or leave docs in-flight |
| Shallow BOM tree | 4 / 85 | **Sim change** — allow nested intermediates |
| Tree-shaped transport network | 4 / 85 | **Sim change** — add lateral/redundant edges |
| Zero-value columns | 3 / 85 | **Sim change** — populate PO line costs |
| No production variance | 3 / 85 | **Sim change** — add batch consumption noise |

The ontology context blocks (v1.8.0) performed well. All three smoke tests passed on ontology guidance alone. The gaps are in the simulated data, not the ontology.

---

## 1. Entity Code Formats

Every entity type in the benchmark questions uses a fictional naming convention that doesn't match the simulator output.

### Actual Formats (from live data)

| Entity | Question Uses | Simulator Produces | Example |
|---|---|---|---|
| Supplier | `SUP-0012` | `SUP-012` | Zero-padding differs |
| Supplier (ALT) | "ChemSource" | `SUP-003-ALT` ("Supplier 3 LLC") | Names are generic |
| SKU (Oral Care) | `SKU-OC-TPST-001` | `SKU-ORAL-001` | Category abbrev differs |
| SKU (Home Care) | `SKU-HM-DISH-005` | `SKU-HOME-376` | Sequential numbering |
| SKU (Personal Wash) | `SKU-PW-SOAP-022` | `SKU-PERSONAL-229` | Sequential numbering |
| SKU (OLD suffix) | `SKU-OC-TPST-042-OLD` | `SKU-ORAL-020-OLD` | Follows base pattern |
| Formula (SKU-level) | `FRM-OC-MINT-001` | `FORM-SKU-ORAL-011` | Prefix + product ref |
| Formula (bulk-level) | — | `FORM-BULK-OC-MINT-026` | Includes variant name |
| Order | `ORD-2024-08471` | `ORD-1-CLUB-DC-001-233` | Includes seq, DC, day |
| Batch | `B-2024-05831` | `B-001-000025` | `B-{plant_seq}-{batch_seq}` |
| Shipment | `SHP-2024-11020` | `SHIP-PLANT-001-000001` | Includes plant ref |
| Goods Receipt | `GR-2024-02200` | `GR-SHP-1-SUP-001-PLANT-CA-16370` | Full provenance chain |
| Purchase Order | `PO-2024-00350` | `PO-CONS-001-001` | Consolidation-based |
| Return | `RTN-2024-00150` | `RMA-001-0-1012` | `RMA-*` format |
| Retail Location | `RTL-MR-0042` | `STORE-GRO-002-0042` | `STORE-{channel}-{num}-{seq}` |
| Ingredient | `ING-RM-GLYC-001` | `ACT-FLUORIDE-001` | Generic grade names |
| Bulk Intermediate | — | `BULK-OC-MINT-026` | Category + variant |

### Fix Options

**Option A (easiest):** Update the 85 questions to use real codes sampled from the data. No sim changes needed.

**Option B (better long-term):** Adjust the sim's naming conventions to be more human-readable and stable. For example, `ORD-2024-08471` is more natural than `ORD-1-CLUB-DC-001-233`. This would also make the benchmark questions more durable across sim reruns.

---

## 2. Ingredient / Supplier Names

### Current State

All 78 ingredients use generic grade names:
```
ACT-FLUORIDE-001    → "Fluoride Grade 1"
ACT-SURFACTANT-002  → "Surfactant Grade 2"
BLK-WATER-001       → "Water Grade 1"
PKG-PRI-TUBE-003    → "Tube Grade 3"
```

All 50+ suppliers use generic names:
```
SUP-007  → "Supplier 7"
SUP-003-ALT → "Supplier 3 LLC"
```

### Impact

6 questions reference specific chemical/material names (citric acid, palm oil, glycerin, sodium fluoride) that don't exist. The benchmark reads naturally with real names but can't resolve against the data.

### Fix

Rename the seed data. The sim doesn't care about label text — joins are all by ID. A mapping table like:

| Current | Rename To | Category |
|---|---|---|
| Fluoride Grade 1 | Sodium Fluoride | active_ingredient |
| Fluoride Grade 2 | Stannous Fluoride | active_ingredient |
| Surfactant Grade 1 | Sodium Lauryl Sulfate | active_ingredient |
| Humectant Grade 1 | Glycerin | active_ingredient |
| Oil Grade 1 | Palm Oil | bulk_material |
| Water Grade 1 | Purified Water | bulk_material |
| Thickener Grade 1 | Xanthan Gum | bulk_material |

Similarly for suppliers: "Supplier 7" → "ChemSource", "Supplier 3" → "PacificChem", etc.

---

## 3. Status Monoculture

### Current State

10 of 12 status-bearing tables have 100% single-status values:

| Table | Rows | Status | Distribution |
|---|---|---|---|
| orders | 1,413,645 | `CLOSED` | 100% |
| order_lines | 61,909,455 | `CLOSED` | 100% |
| purchase_orders | 7,618 | `CLOSED` | 100% |
| purchase_order_lines | 15,888 | `CLOSED` | 100% |
| shipments | 7,810,083 | `in_transit` | 100% |
| goods_receipts | 6,088,791 | `received` | 100% |
| returns | 71,662 | `received` | 100% |
| ap_invoices | 6,118,944 | `open` | 100% |
| ap_payments | 6,118,944 | `completed` | 100% |
| ar_receipts | 1,504,070 | `completed` | 100% |
| batches | 141,626 | `complete` | 100% |

Only two tables have variety:

| Table | Rows | Distribution |
|---|---|---|
| ar_invoices | 1,511,664 | open 97.0% / disputed 1.8% / partial 1.2% |
| work_orders | 145,369 | complete 77.1% / planned 22.3% / in_progress 0.6% |

### Impact

8+ questions are unanswerable:
- Q03: "open" POs (none exist)
- Q08: order pipeline by status (all CLOSED)
- Q44: "pending" orders (none exist)
- Q48: "released" work orders (status doesn't exist — only planned/in_progress/complete)
- Q64: order status distribution (trivially: 100% CLOSED)
- Q69: valid transitions from "pending" (no pending orders)
- Q70: shipment status transitions (all in_transit)
- Q75: allocation check (all CLOSED)

### Fix Options

**Option A (simplest):** Snapshot the database mid-simulation so documents are in various lifecycle stages. If the sim runs day-by-day, stop at day ~250 instead of day 365, leaving recent orders as pending/allocated/shipped.

**Option B (richer):** Introduce a "freeze day" parameter. Documents created after the freeze day keep their in-progress status. Earlier documents proceed to terminal states. This gives a natural pipeline: old orders are CLOSED, recent orders are in-progress.

**Option C:** Populate all lifecycle stages explicitly — a fraction of orders stuck in each state for testing.

### Additional Column Gaps

| Column | Current Value | Expected |
|---|---|---|
| `order_lines.unit_price` | 0.00 (all 61.9M rows) | Should reflect `skus.price_per_case` or `ar_invoice_lines.unit_price` |
| `purchase_order_lines.unit_cost` | 0.00 (all 15.9K rows) | Should reflect `supplier_ingredients.unit_cost` |

These zero-value columns break any PO-level cost analysis and order-level revenue analysis. Revenue is only available through `ar_invoices` / `ar_invoice_lines`, which the ontology correctly documents — but the PO cost gap has no workaround.

---

## 4. BOM Structure

### Current State

Every BOM is exactly 2 levels deep with identical structure:

```
SKU formula (bom_level=0): 500 formulas
├── Exactly 1 bulk intermediate (always)
│   └── 7-12 raw ingredients via bom_level=1 formula (45 formulas)
└── 6-7 packaging materials (direct raw ingredients)
```

Key stats:
- Level-0 formulas: 500, each with 7-8 ingredients (1 bulk + 6-7 packaging)
- Level-1 formulas: 45, each with 7-12 raw ingredients
- Bulk intermediates used as ingredients in level-1 formulas: **0** (no nesting)
- Every level-0 formula has exactly 1 bulk intermediate

### Impact

- Q24 ("how deep is the BOM?"): Always 2. Trivial.
- Q25 ("list intermediates between finished good and raw materials"): Always exactly 1.
- Q56/Q67 (cycle detection): Trivially no cycles — structure is too simple.
- BOM explosion tests can't exercise deep recursion or diamond dependencies.

### Suggested Sim Changes

1. **Allow nested intermediates.** Some bulk intermediates should themselves use other bulk intermediates as ingredients (e.g., a "Base Paste" compound that goes into a "Mint Paste" compound). This creates 3-4 level BOMs.

2. **Vary the number of intermediates.** Some SKU formulas should use 2-3 bulk intermediates (e.g., a paste intermediate + a coating intermediate), not always exactly 1.

3. **Create diamond dependencies.** Two different intermediates using the same raw material, both feeding the same SKU formula. This tests BOM explosion deduplication.

4. **Add a few deep chains.** At least one 4+ level BOM would exercise recursive handlers properly.

---

## 5. Transport Network Topology

### Current State

The network is a strict directed acyclic graph with near-tree structure:

```
Layer 0: 50 suppliers ──→ 4 plants         (100 edges, many-to-many)
Layer 1: 4 plants ──→ 6 RDCs               (24 edges, fully connected 4×6)
Layer 2: 4 plants ──→ 18 customer DCs      (direct, selective)
Layer 3: 6 RDCs ──→ 38 customer DCs        (selective, no overlap with layer 2)
Layer 4: 56 customer DCs ──→ 3,817 stores  (1:many, no overlap between DCs)
```

**Totals:** 3,997 edges, 3,933 nodes

**Key structural properties:**
- DAG (no cycles)
- Every store has exactly 1 customer_dc parent (no store served by multiple DCs)
- Every customer_dc has exactly 1 supply parent (either a plant OR an RDC, never both)
- No RDC↔RDC lateral edges
- No DC↔DC lateral edges
- From any given plant to any given store, there is exactly 1 path
- Multiple plants can reach the same store (via shared RDCs), giving "which plant" questions but not "which route" questions

### Impact

- Q31 ("multiple shortest paths"): Only 1 path exists per origin-destination. Ties don't occur.
- Q38 ("sever Memphis-Chicago RDC link"): No such link exists. RDCs don't connect to each other.
- Q39 ("avoid Phoenix RDC"): If the only path goes through Phoenix, there's no alternative — just a disconnected result.
- Resilience questions are less interesting because removing a node simply disconnects its subtree.

### Suggested Sim Changes

1. **Add RDC↔RDC lateral links.** Connect adjacent RDCs (e.g., Memphis↔Chicago, Phoenix↔Sacramento). This creates alternative paths and makes resilience questions meaningful.

2. **Allow DCs to be served by multiple RDCs.** Some customer DCs should have 2 supply parents, creating genuine route alternatives and shortest-path ties.

3. **Add direct plant→store bypass routes** for a few high-volume stores. This creates fast-lane vs standard-route tradeoffs.

4. **Add bidirectional edges** for some trunk routes (RDC↔RDC). The current graph is purely directional, which limits network analysis options.

5. **Consider a few cross-channel edges.** A grocery DC that can also serve a nearby pharmacy DC in emergencies, creating richer "what-if" analysis.

---

## 6. Additional Data Gaps

### Production Variance (Q27, Q57, Q73)

`batch_ingredients.quantity_kg` is computed as `formula_ingredients.quantity_kg × batch.quantity_kg` with only floating-point rounding (max deviation: 0.035 kg). No meaningful production variance exists. Questions asking "where did we deviate from the recipe?" always return zero.

**Fix:** Add ±2-5% random noise to batch_ingredients during generation.

### Lead Times (Q10)

Maximum supplier lead time is 14 days. Q10 asks for >30 days, returning empty results.

**Fix:** Either lower the question threshold to 10 days, or extend the lead time range in the sim to 7-60 days.

### Inventory Snapshots (Q66)

Inventory table has weekly snapshots (days 7, 14, 21, ..., 364). Q66 asks for day 180, which doesn't exist (nearest: day 175 or 182).

**Fix:** Either adjust question to use a snapshot day, or generate daily inventory snapshots (storage cost consideration).

### GL Journal (Q74)

GL journal debits and credits balance perfectly on every day. No `-DUP` duplicate entries exist. The data integrity question is trivially answered (always passes).

**Fix:** Inject a few deliberate anomalies — duplicate postings, small imbalances — to make data quality questions non-trivial.

### Missing `numpy` Dependency

PageRank and eigenvector centrality (NetworkX) require numpy, which isn't in the project's dependencies. Affects Q35.

---

## 7. What Worked Well

The ontology v1.8.0 context blocks were the star performer:

- **Q07 (revenue by channel):** Context correctly said "use `ar_invoices.channel`, NOT `order_lines.unit_price=0`" — this would have been a trap without the context.
- **Q29 (shortest route):** Corrected type discriminators (`plant/rdc/customer_dc/store` not `dc/retail`) worked perfectly for building composite node keys.
- **Q41 (BOM + supplier):** Polymorphic ingredient resolution guidance ("LEFT JOIN both tables, no discriminator") was exactly right.
- **All handler dispatch decisions were correct.** The `vg:operation_types` annotations reliably told us SQL vs traverse vs shortest_path.
- **Edge attributes, weight columns, and sql_filters** all mapped correctly to the live data.

22 of 85 questions passed clean with no issues. The ontology is doing its job as a virtual twin — the simulator just needs to produce richer data to match the ambition of the questions.

---

## Summary: Priority Fix List for Simulator

| Priority | Fix | Questions Unblocked | Effort |
|---|---|---|---|
| **P0** | Fix entity code formats in questions (or sim) | 45 | Trivial |
| **P1** | Rename ingredients/suppliers to real names | 6 | Trivial |
| **P2** | Add status lifecycle variety (snapshot mid-sim) | 8 | Medium |
| **P3** | Populate `order_lines.unit_price` and `po_lines.unit_cost` | 3 | Low |
| **P4** | Add nested bulk intermediates (3-4 level BOMs) | 4 | Medium |
| **P5** | Add RDC↔RDC lateral edges + DC multi-sourcing | 4 | Medium |
| **P6** | Add batch production variance (±2-5% noise) | 3 | Low |
| **P7** | Extend lead time range (7-60 days) | 1 | Trivial |
| **P8** | Inject GL anomalies (dupes, imbalances) | 1 | Low |
| **P9** | Add `numpy` to project dependencies | 1 | Trivial |
