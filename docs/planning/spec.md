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

**Remaining stability concerns** (top-line metrics healthy, structural issues persist):
- All 24 deployment targets UNDER (position ratios 0.32-0.68x, target 0.8-1.1x)
- MRP backpressure correlation: +0.322 (positive = caps not constraining production)
- Customer DC flow imbalance: 26.1% inflow > outflow (but inventory draining -17.2%)

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
- `scripts/analysis/diagnose_365day.py` (3-layer pyramid: physics → operational → flow)
- Standalone analyzers: `diagnose_slob.py`, `diagnose_a_item_fill.py`, `analyze_bullwhip.py`

**Known observations** (v0.61.0, measured):
- Customer DC flow imbalance: 26.1% inflow > outflow (persistent across v0.59.0-v0.61.0)
- RDC growth: +38.3% over 365 days (push threshold fix reduced divergence rate)
- Plant FG growth: +61.4% — early overproduction (day 1-60) dominates; FG plateaus at 59-63M then declines
- MRP backpressure not engaging: production correlates positively with plant FG (+0.322)
- All deployment targets UNDER (0.32-0.68x position ratios)
- Mass balance period 0 ~155% (warm-start artifact, expected)
- Prod/Demand now 1.01 (was 0.98 in v0.59.0 — slight overcorrection)

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

## 5. ERP Export Completion

**Status:** 35/36 tables operational.

**Missing table:** `products` master (1/36) — parent entity for `skus`.

**Data quality refinements:**
- `production_lines` OEE: currently hardcoded defaults, should derive from sim
- Promotion dates: "Week X" format → ISO dates
- Inventory normalization: consistent UoM across echelons
- Plant capacity: derive from config rather than static values

**Optional enhancements:**
- `SQLWriter` for `seed.sql` (direct Postgres import)
- Referential integrity validation script

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
