# Prism Sim — Project Spec

## 1. Project Identity

Prism Sim is a discrete-event supply chain Digital Twin simulating a North American FMCG network: ~4,238 nodes, ~500 SKUs across 3 product categories, 4 plants, 6 RDCs, and ~4,200 downstream locations. Physics-first design (mass balance, capacity constraints, Little's Law). Outputs streaming Parquet for diagnostics and normalized ERP-format tables for downstream systems.

---

## 2. Validated State (v0.61.0, 365-day)

| Metric | v0.59.0 | v0.61.0 | Status |
|--------|---------|---------|--------|
| Fill Rate | 98.5% | 98.51% | GREEN |
| Inventory Turns | 10.31x | 10.41x | GREEN |
| OEE | 54.3% | 55.3% | GREEN |
| SLOB | 0.0% | 0.0% | GREEN |
| Prod/Demand | 0.98 | 1.01 | GREEN |
| Bullwhip | 0.46x | 0.68x | GREEN |
| Perfect Order | 97.5% | 97.4% | GREEN |
| Cash-to-Cash | 20.6d | 20.1d | GREEN |
| Store DOS | 6.1 | 6.1 | On target |

These are observations — not pass/fail grades against fixed targets.

**v0.60.0 + v0.61.0 + v0.62.0 changes validated:**
- DC priming matches deployment targets (was using RDC targets — 71% over-prime for A-items)
- Store priming matches channel profiles (`store_days_supply` 10→6)
- DC pipeline adjustment prevents double-stocking
- RDC push threshold lowered (40→20 DOS)
- Seasonal-aware deployment (`_compute_deployment_needs()` and `_push_excess_rdc_inventory()`)
- Tighter MRP DOS caps: A=22, B=25, C=25
- Plant FG priming raised to 3.5 DOS/plant (was 2.0) — total IP ≈17 DOS matches MRP A-item target

**Closed items (v0.68.0, converged 365-day diagnostic):**
- MRP backpressure: +0.606 correlation is seasonal confound, plant FG stable (-1.3%), caps never fire. **Closed.**
- B/C underproduction: B P/D=0.990, C=0.984 — within tolerance. Post-warm-start B≈1.009, C≈1.007. **Closed.**
- DC flow imbalance: 26.1% → **0.7% adjusted** (ECOM/DTC were measurement artifact). **Closed.**

**v0.68.0 fixes:**
- RDC target: 15→9 DOS (flow-through topology, actual ≈8.4)
- C-item buffer: 1.0→1.05 (compensates 1.6% changeover deficit)

**v0.69.0: Cost Analytics Layer (post-sim enrichment, no physics changes):**
- `Order.requested_date` populated from link lead times — enables OTIF measurement
- `cost_master.json` config for logistics, penalty, working capital, product costs
- `diagnose_cost.py` — COGS, logistics cost, carrying cost, OTIF, cost-to-serve, C2C (now in `diagnostics/cost_analysis.py`)

**v0.72.0: Unified Supply Chain Diagnostic:**
- `diagnose_supply_chain.py` — 35 questions, 8 sections, replaces 3 Tier-1 scripts
- New modules: `cost_analysis.py`, `commercial.py`, `manufacturing.py`
- Enhanced `loader.py` DataBundle with cost/price maps, batch_ingredients, channel map

**Remaining observations** (converged baseline):
- All echelons draining: P/D=0.993, -65K/day total drain (warm-start Q1 + seasonal lag)
- RDC DOS: 8.4 vs target 9.0 (previously 15.0 — now realistic)
- OEE: 54.8% (just under 55% GREEN threshold)

---

## 3. Simulation Validation

The simulation is in an iterative shake-out phase. The core engine is complete; remaining work is diagnostic: run, observe, investigate, fix, repeat.

**Development pattern:** Run 365-day sim → diagnose with diagnostic suite → find exception → fix → sometimes the fix masks a deeper logic issue → iterate.

**Definition of done:**
- Physics hold (mass balance, capacity constraints, inventory positivity)
- Emergent behaviors are realistic (bullwhip, bottlenecks, seasonal patterns)
- No hidden bandaids masking logic bugs
- Metrics stable over 365 days

**Key diagnostic tools:**
- Triangle Report (end-of-run summary)
- `scripts/analysis/diagnose_supply_chain.py` (unified 35-question diagnostic, 8 sections)
- `scripts/analysis/diagnose_erp_database.py` (ERP database diagnostic, 52 questions, 10 sections — psycopg2 against PostgreSQL)
- Standalone analyzers: `diagnose_slob.py`, `diagnose_a_item_fill.py`, `analyze_bullwhip.py`

**Known observations** (v0.67.0, converged 365-day warm-start):
- Plant FG growth: -1.3% (essentially stable; was +61.4% in v0.61.0 cold-start)
- System drain: all echelons draining at -65K/day total (P/D=0.993)
- RDC DOS: 8.4 (operating as flow-through cross-dock, not buffer warehouse)
- OEE: 54.8% (was 55.3% in v0.61.0)
- Fill rate: 98.6%, Perfect Order: 97.5%, Turns: 10.43x, C2C: 20.0d
- MRP backpressure corr: +0.606 (seasonal confound — caps never fire, plant FG stable)

---

## 3a. Design Decisions

**Production backpressure via MRP, not storage capacity (v0.63.0):** `Node.storage_capacity` was removed — production backpressure is handled by MRP DOS caps + plant FG in inventory position, not physical warehouse limits. All nodes had `storage_capacity=inf` and it was never enforced. Real FMCG plants use MRP to prevent overproduction at the planning level.

**ABC-differentiated push receive cap (v0.64.0):** `push_receive_dos_cap=12.0` (scalar) replaced with `push_receive_headroom=1.15` (multiplier on ABC deployment targets). The scalar cap blocked B/C items below their deployment targets (B=14, C=17.5 DOS vs cap=12). New caps: A≈12.1, B≈16.1, C≈20.1 DOS — derived from `dc_buffer_days × ABC_mult × headroom`.

**3-Level BOM (v0.70.0):** Replaced flat single-level BOM with 3-level structure: Level 2 (78 raw materials) → Level 1 (45 bulk intermediates) → Level 0 (500 finished SKUs). Bulk intermediates are produced at plants via dependent demand explosion from SKU production orders, same-plant assignment. TransformEngine two-pass ordering ensures bulk is produced before SKUs within each daily cycle. MRP PO generation uses two-step matrix explosion to reach leaf materials. Recipe matrix stays 2D `[n_products, n_products]`. Deployment, demand, DRP, allocation, replenishment unchanged (SKUs only).

**Cost Model Enrichment (v0.71.0):** Post-sim enrichment only — no simulation physics changes. `cost_master.json` expanded with per-route FTL/LTL logistics, echelon-specific warehouse rates, category-specific manufacturing cost structure, channel DSO. `diagnose_cost.py` upgraded from 6 to 8 sections: per-SKU COGS (was flat-category), per-echelon logistics with distance, bottom-up mfg COGS from batch_ingredients, revenue & margin by channel, channel-weighted DSO. `export_erp_format.py` cost splits now config-driven (was hardcoded 60/20/20).

**Enterprise Data Generator (v0.73.0→v0.76.1):** Replaced stale `export_erp_format.py` with DuckDB-based `scripts/erp/` package. Uses DuckDB's native parquet reader + SQL transforms + COPY TO for CSV export — processes 230M+ source rows in ~2.5 min without OOM. Financial layer adds double-entry GL journal (~47M per-shipment/batch entries, 7 event types, each with `reference_id` traceability, balanced to <$0.01), AP/AR invoices (5.4M AP + 1.5M AR headers), and deterministic transaction sequencing (`day × 10M + cat × 1M + counter`). Output: 36 tables, 368.5M total rows. Load scripts for PostgreSQL and Neo4j.

---

## 4. Likely Areas of Future Code Change

As validation iterates, these areas are probable touchpoints:

**Deployment & flow balance** (`simulation/orchestrator.py`)
- Customer DC accumulation — inflow/outflow mismatch
- RDC inventory divergence (+26K/day growth)
- Need-based deployment refinement

**MRP scheduling** (`simulation/mrp.py`)
- A-item underproduction (-2.5%) — may need buffer adjustment
- Campaign batching optimization
- Changeover frequency reduction (directly impacts OEE)

---

## 5. Enterprise Data Generator (v0.73.0)

**Status:** Complete. 36 tables, 368.5M rows, GL balanced (per-shipment detail, 100% reference_id coverage), 100% FK integrity.

**Package:** `scripts/erp/` — DuckDB-based ETL, replaces old `scripts/export_erp_format.py`.

**Run:** `poetry run python -m scripts.erp --input-dir data/output --output-dir data/output/erp`

**Output:** `data/output/erp/` — master/ (14 CSVs), transactional/ (22 CSVs), reference/, neo4j_headers/

**Load scripts:** `load_postgres.sh` (\\copy), `load_neo4j.cypher` (LOAD CSV)

**Verification (automated):**
- GL balanced: DR=CR=$759.6B (diff < $0.01)
- COGS/Revenue: 67.5%
- FK integrity: 100% on 5 spot checks
- Per-day GL balance: all 368 days balanced
- Reference ID coverage: 100% across all 7 reference_types (goods_receipt, production, shipment, freight, sale, return, plus variance/payment in friction)
- node_id coverage: ~98% for physical events (goods_receipt, production, shipment, freight, sale, return); 0% for treasury events (payment, receipt, bad_debt) — matches SAP/Oracle behavior

**v0.78.0 (friction layer):**
- Friction layer: 4 tiers — entity resolution, 3-way match, data quality, payment timing
- Config toggle: `cost_master.json` → `friction.enabled`
- New tables: `invoice_variances`, `ap_payments`, `ar_receipts`
- New GL accounts: 4200 (Discount Income), 5500 (Bad Debt)

**v0.78.1 (GL node_id + data quality fixes):**
- Freight GL split from `reference_type='shipment'` to `reference_type='freight'` — distinct financial vs physical events
- Freight GL entries get `source_sim_id` as `node_id` (15.6M rows fixed)
- Variance GL entries get plant `node_id` via AP lookup chain (1.6M rows fixed)
- Duplicate invoices now include `ap_invoice_lines` (30K invoices fixed)

**Deferred:**
- Neo4j graph optimization (relationship property enrichment)

**Reference files:**
- `scripts/export_erp_format.py` (35-table ETL)
- `scripts/erp_schema.sql` (36-table DDL)

---

## 6. Document Map

| Document | Purpose | Status |
|----------|---------|--------|
| `docs/planning/spec.md` | Goals, status, remaining work | Current |
| `docs/llm_context.md` | Architecture, code paths, config, debugging | Current |
| `docs/planning/physics.md` | SC Physics theory (Little's Law, VUT, Mass Balance) | Timeless |
| `docs/planning/triangle.md` | Desmet's SC Triangle (Service/Cost/Cash) | Timeless |
| `CLAUDE.md` | LLM agent instructions & commands | Current |
| `CHANGELOG.md` | Detailed version history | Current |
| `scripts/erp_schema.sql` | ERP DDL (36 tables, Postgres) | Current |
| `docs/planning/archive/` | Historical: intent, roadmap, investigation docs | Archived |

---

## 7. Archive Note

`docs/planning/archive/` contains historical documents from the initial build phase (Dec 2024 – Jan 2025):

- `intent.md` — Original project vision and technical spec. All milestones achieved.
- `roadmap.md` — Original task roadmap with Task IDs. All milestones complete.
- `365_day_drift_diagnosis.md` — Investigation of 365-day metric drift (Jan 2025).
- `regression_investigation.md` — Regression root cause analysis (Jan 2025).
- `calibration_diagnostic_report.md` — Calibration tuning investigation (Jan 2025).
- `notes/fresh_start_comparison.md` — Warm-start vs cold-start comparison (Dec 2024).
- `notes/schema_gap_analysis.md` — ERP schema gap analysis (Dec 2024).
