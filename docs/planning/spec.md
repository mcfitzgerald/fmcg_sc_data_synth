# Prism Sim — Project Spec

## 1. Project Identity

Prism Sim is a discrete-event supply chain Digital Twin simulating a North American FMCG network: ~4,238 nodes, ~500 SKUs across 3 product categories, 4 plants, 6 RDCs, and ~4,200 downstream locations. Physics-first design (mass balance, capacity constraints, Little's Law). Outputs streaming Parquet for diagnostics and normalized ERP-format tables for downstream systems.

---

## 2. Validated State (v0.59.0, 365-day)

| Metric | Value | Notes |
|--------|-------|-------|
| Fill Rate | 98.5% | GREEN |
| Inventory Turns | 10.31x | GREEN (+49% from v0.58.0) |
| OEE | 54.3% | YELLOW |
| SLOB | 0.0% | GREEN (fixed: age tracking bugs) |
| Prod/Demand | 0.98 | GREEN |
| Bullwhip | 0.46x | GREEN |
| Perfect Order | 97.5% | GREEN |
| Cash-to-Cash | 20.6d | GREEN (-46% from v0.58.0) |
| Store DOS | 6.1 | On target (was 24.1) |

These are observations — not pass/fail grades against fixed targets.

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

**Known observations** (measured, not yet classified as issues or accepted):
- Customer DC accumulation (+25.1% inflow vs outflow)
- RDC inventory diverging (+26K/day)
- A-item production -2.5% below demand
- OEE 54.3% — slightly below target
- Mass balance period 0 ~211% (warm-start artifact, expected)
- CLUB-DC classification trap (diagnostic lesson, documented in `llm_context.md`)

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
