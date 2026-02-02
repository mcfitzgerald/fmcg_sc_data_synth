# v0.46.0 Validation State & Next Steps

> Created: 2026-02-02 after 365-day validation run.
> Purpose: Fresh-session context for diagnosing remaining inventory drift and planning v0.47.0 fixes.

---

## 1. v0.46.0 Results (365-Day Run)

### Triangle Report

| Metric | v0.45.0 (365d) | v0.46.0 (365d) | Target | Status |
|--------|----------------|----------------|--------|--------|
| A-fill | 84.1% | **85.1%** | >=85% | PASS (barely) |
| B-fill | — | **95.7%** | >=90% | PASS |
| C-fill | — | **91.5%** | >=85% | PASS |
| Turns | 4.3x | **4.29x** | 6-14x | **FAIL** |
| SLOB | 38.6% | **31.2%** | <30% | **MARGINAL** (-7.4pp) |
| OEE | — | **62.3%** | 55-85% | PASS |
| Perfect Order | — | **97.5%** | >90% | PASS |
| Truck Fill | — | **15.4%** | >85% | FAIL (known) |
| MAPE | — | **30.0%** | 20-50% | PASS |
| Cash-to-Cash | — | **81.6 days** | — | High |

### Drift Diagnostic (Production vs Demand)

| Day | Demand | Production | Ratio | Gap% |
|-----|--------|-----------|-------|------|
| 120 | 3.56M | 3.64M | 1.022 | +2.2% |
| 180 | 4.09M | 4.26M | 1.042 | +4.2% |
| 210 | 4.24M | 4.59M | 1.082 | +8.2% (peak) |
| 270 | 4.48M | 4.78M | 1.067 | +6.7% |
| 330 | 4.19M | 4.30M | 1.028 | +2.8% |
| 390 | 3.59M | 3.71M | 1.035 | +3.5% |
| 420 | 3.46M | 3.43M | **0.990** | **-1.0%** |
| 450 | 3.45M | 3.35M | **0.970** | **-3.0%** |

**Production is now tracking demand.** Ratio was 1.27-1.47x in v0.45.0; now 1.02-1.08x. Production actually goes *below* demand in late sim (0.97x at day 450), showing DOS caps and SLOB dampening kicking in.

### Production Excess by ABC Class

| Class | Total Demand | Total Prod | Excess% |
|-------|-------------|------------|---------|
| A | 1,159M | 1,187M | **+2.4%** |
| B | 212M | 234M | **+10.1%** |
| C | 76M | 77M | +1.2% |

B-items have the highest overproduction ratio. C-items are well-controlled.

### Inventory by Echelon (Day 91 -> Day 455)

| Echelon | Day 91 | Day 455 | Growth |
|---------|--------|---------|--------|
| Plant | 3 | 16 | +416% (trivial absolute) |
| MFG RDC | 25.1M | 23.9M | **-5%** (stable/declining) |
| Customer DC | 81.3M | 371.2M | **+357%** |
| Store | 52.4M | 77.8M | +48% |
| Club | 34.6M | 152.7M | **+341%** |

**RDCs are fine. The problem is 100% at Customer DCs and Club DCs.**

### SLOB Diagnostic Detail

| Echelon | Day 91 | Day 455 | Growth |
|---------|--------|---------|--------|
| RETAILER_DC | 35.0M | 196.6M | +462% |
| CLUB | 34.6M | 152.7M | +341% |
| STORE | 52.4M | 77.8M | +48% |
| MFG_RDC | 25.1M | 23.9M | -5% |
| OTHER | 18.3M | 39.9M | +118% |
| DISTRIBUTOR_DC | 5.7M | 11.0M | +93% |
| ECOM_FC | 1.5M | 1.2M | -17% |
| DTC_FC | 0.25M | 0.20M | -20% |

### InvMean Trajectory

```
Day  91: 229.5 (start)
Day 120: 213.1 (dip, spring ramp)
Day 210: 219.8 (summer peak, well controlled)
Day 270: 254.1 (post-peak, excess accumulating)
Day 330: 286.4 (winter trough, still growing)
Day 390: 315.3 (continuing)
Day 420: 318.3 (growth slowing)
Day 455: 329.2 (end, stabilizing?)
```

InvMean grew +43% over 365 days (229->329). Growth rate decelerates in last 60 days but doesn't plateau.

### Service Level Stability

- Mean service gap: 3.1% (i.e., ~97% service)
- Q1: 2.9%, Q2: 3.4%, Q3: 3.1%, Q4: 2.8%
- **Stable, no death spiral, stabilized in Q4**
- Store ending DOS: 30.7 days (up from 19.3 starting)

### Cumulative Excess

- Monotonicity: **67%** of days excess grew (was 100% in v0.45.0)
- Peak excess: 5.2% at day 270, declining to 3.5% at day 450
- **Production excess is self-correcting** — cumulative excess % is declining in later sim

---

## 2. Open Questions

### Q1: Inventory Dynamics at Customer DC and Club DC

**The core remaining problem.** RDCs are stable/declining, but Retailer DCs (+462%) and Club DCs (+341%) are ballooning.

**Hypothesis:** The replenishment engine over-orders at these echelons because:

1. **Inflow demand signal at Customer DCs** (`replenishment.py:771-783`): Customer DCs use a 7-day rolling average of **orders received from downstream stores** as their demand signal. If stores over-order (due to safety stock buffers), this inflated signal cascades up.

2. **Echelon inventory position may not be working correctly for these DCs** — v0.45.0 fixed echelon IP for RDC-sourced DCs, but plant-direct DCs (Mass Retail DCs, Club DCs) may have a different code path. The plant-direct deployment fix (v0.45.0) routes production proportionally to all deployment targets, but does the replenishment engine properly account for echelon inventory at these nodes?

3. **Channel profile settings** (`simulation_config.json`):
   - CLUB: target_days=8.0, reorder_point_days=4.0, batch_size=200
   - MASS_RETAIL: target_days=8.0, reorder_point_days=4.0, batch_size=200
   - These DCs also get ABC multipliers: A-items x1.5 on top of base days

4. **No DOS cap at distribution level** — The v0.46.0 DOS cap guard only operates in MRP (production). There is **no equivalent mechanism in replenishment** that says "stop ordering if you already have 30+ days of supply at this DC."

**Key question:** Is inventory accumulating because DCs are ordering too much, or because they're receiving too much from plants?

**Files to investigate:**
- `replenishment.py:476-518` — Customer DC logic
- `replenishment.py:771-783` — Inflow demand calculation
- `replenishment.py:1077-1131` — Echelon inventory position logic
- `orchestrator.py:173-177` — Product velocity assignment for allocation

### Q2: Are the KPI Targets Right?

**Current targets from `intent.md:102-110`:**

| Metric | Target | Question |
|--------|--------|----------|
| Inventory Turns | 6-14x | Is 6x realistic for a 4,500-node network with 500 SKUs across 7 channels? FMCG industry turns vary: retailers see 12-20x, but manufacturers see 4-8x. Our metric is **system-wide** (all echelons). |
| SLOB | <30% | Our SLOB uses age-based thresholds (A>60d, B>90d, C>120d). Is 30% the right target for a system with no clearance mechanism? With 7 channels and ~4,500 nodes, some tail inventory is structural. |
| A-fill | >=85% | Intent.md doesn't actually specify per-ABC fill targets. The 85% was set during v0.44.0 tuning. Is it the right bar? |
| Truck Fill | >85% | Currently 15.4%. This is a known issue but not the focus right now. |

**Industry context:**
- System-wide inventory turns for a vertically integrated FMCG company: 4-8x is normal
- Our 4.29x is at the low end but arguably within range for a full network including stores
- SLOB measurement: 30% of total inventory being "aged" may be structurally inevitable when stores hold 30+ DOS and items age incrementally

**Are we chasing the wrong things?** The production/demand ratio is now 1.02-1.08x — production is nearly balanced. The "inventory turns" metric may be dragged down by structural DC and store inventory that *should* exist for service level purposes. The question is whether the *growth rate* of that inventory is the real metric to track.

### Q3: A-Fill Lower Than B/C Fill — Is This Acceptable?

**Current: A=85.1%, B=95.7%, C=91.5%**

**This is the inverse of FMCG best practice**, where A-items (80% of revenue) should have the highest fill rate.

**Why it happens (three mechanisms):**

1. **Allocation priority creates a paradox** (`allocation.py:49-80`): A-items are fulfilled first when inventory is scarce. This means A-items face *sustained* rationing at constrained nodes, while B/C items experience only *intermittent* shortfalls because A-items have already consumed available stock before B/C orders are processed. The velocity-based priority means A-items always "go first" but also always "hit the wall first."

2. **Aggressive safety stock triggers more frequent orders** (`replenishment.py:228-239`): A-items have z=2.33 (vs B/C at 1.65). This creates larger, more frequent orders that stress upstream capacity more.

3. **ROP multiplier amplifies demand signal** (`mrp.py:416-445`): A-items have 1.5x ROP multiplier, B-items 1.0x, C-items 0.5x. Combined with z=2.33, A-items create disproportionate upstream demand.

**The net effect:** A-items have the highest demand velocity AND the most aggressive replenishment parameters, which means they consume upstream capacity first and face rationing more often. B/C items are less aggressive and find inventory available more often.

**Is this acceptable?** It's not ideal. In industry, A-items should be 95%+ service level. However, the gap (85% vs 96%) suggests a systemic issue, not a tuning issue. The question is whether fixing the DC-level inventory bloat (Q1) would naturally improve A-fill by making more inventory available at the right echelon.

**Relevant code locations:**
- `allocation.py:49-80` — Velocity-based order priority
- `replenishment.py:228-239` — Z-score by ABC class
- `simulation_config.json` — `abc_prioritization.a_rop_multiplier: 1.5`
- `orchestrator.py:1226-1245` — ABC fill rate calculation

### Q4: SLOB Clearance — What's Happening and Financial Impact

**Current SLOB mechanics:**

1. **Detection** (`orchestrator.py:1268-1307`): SLOB is calculated daily as:
   ```
   SLOB% = sum(inventory where weighted_age > ABC_threshold) / total_FG_inventory
   ```
   - A-items: flagged SLOB at >60 days age
   - B-items: >90 days
   - C-items: >120 days
   - Includes ALL echelons (stores, DCs, RDCs, plants)

2. **Age tracking** (`state.py:402-413, 449-481`):
   - All inventory ages +1 day each day
   - Consumed inventory reduces weighted age (FIFO approximation)
   - Fresh receipts blend age downward
   - Age is inventory-weighted average per (node, product)

3. **Production response** (`mrp.py:1069-1160`):
   - DOS cap: skip production when DOS > threshold (25/35/45 days by ABC)
   - SLOB dampening: reduce batch size by 50% when product age > SLOB threshold

4. **What does NOT happen:**
   - **No markdown/write-off** — SLOB inventory is never removed
   - **No disposition** — no mechanism to dispose of, donate, or discount aged product
   - **No replenishment suppression** — DCs continue ordering normally regardless of SLOB status
   - **No financial write-down** — no P&L impact from holding SLOB inventory

**Financial implications:**
- 31.2% SLOB = ~157M cases of the 503M total system inventory is SLOB-aged
- At ~$2-5/case, that's $314M-$785M in tied-up working capital
- Cash-to-cash cycle: 81.6 days (high, partly driven by SLOB)
- Holding costs (typically 20-25% annually) on SLOB: $63M-$196M/year

**The question:** Should the simulation implement a clearance mechanism (markdown → liquidation → write-off)? This would:
- Reduce SLOB% mechanically
- Reduce system inventory → improve turns
- Reduce InvMean
- Add financial realism (write-offs hit P&L)
- But could create a "disposal spiral" if not tuned carefully

### Q5: Recommended Next Steps — How to Address

**The remaining problems form a hierarchy:**

```
Root Cause: DC-level inventory accumulation
  ├── Retailer DCs: +462% growth (largest contributor)
  ├── Club DCs: +341% growth
  ├── Distributor DCs: +93%
  └── Stores: +48% (moderate, may be acceptable)
      │
      ├── Symptom: Inventory turns 4.29x (target 6-14x)
      ├── Symptom: SLOB 31.2% (target <30%)
      └── Symptom: InvMean monotonically increasing
```

**Production is no longer the problem.** v0.46.0 successfully brought prod/demand to 1.02-1.08x. The problem is now downstream: inventory is being distributed to DCs faster than stores consume it, or DCs are ordering more than they need.

**Potential fixes to investigate:**

#### A. DC-Level DOS Cap in Replenishment
Add a DOS cap guard in `replenishment.py` analogous to the MRP DOS cap. If a DC already has >N days of supply for a product, suppress reorder for that product.
- Pro: Direct fix, mirrors production-side logic
- Con: Could cause service level dips if set too tight
- **Files:** `replenishment.py:792-842` (order trigger logic)

#### B. Inflow Demand Signal Dampening
The inflow demand signal at Customer DCs may be inflated by downstream safety stock orders. Dampening this (e.g., using actual POS demand aggregated from stores rather than order inflow) would reduce over-ordering.
- Pro: Fixes root cause (demand signal distortion)
- Con: More complex, requires passing POS data to DC replenishment
- **Files:** `replenishment.py:771-783` (inflow demand), `replenishment.py:520-550` (expected throughput)

#### C. SLOB Clearance Mechanism
Implement markdown/write-off for inventory exceeding 2x SLOB thresholds. E.g., A-items >120d are written off, B-items >180d, C-items >240d.
- Pro: Adds financial realism, mechanically reduces SLOB%
- Con: Masks the root cause rather than fixing it
- **Files:** New logic in `orchestrator.py` or `state.py`

#### D. Replenishment Parameter Tuning
- Lower CLUB target_days from 8.0 to 6.0
- Lower MASS_RETAIL target_days from 8.0 to 6.0
- Lower A-item safety stock z-score from 2.33 to 1.96 (95th percentile → still high)
- Pro: Simple config changes
- Con: May hurt fill rates

#### E. Re-examine ABC Allocation Logic
The velocity-based allocation priority may be creating the A-fill < B/C-fill inversion. Consider:
- Pro-rata allocation (same fill ratio for all items at a node)
- ABC-tiered allocation (guarantee A-items 95% before filling B/C)
- **Files:** `allocation.py:49-80`, `allocation.py:82-163`

---

## 3. What v0.46.0 Fixed (Context)

These four fixes were applied in the most recent commit (`993739e`):

1. **DOS Cap Guard** (`mrp.py:1069-1079`): Skip production when DOS > 25/35/45 days by ABC class
2. **Seasonal Floor** (`mrp.py:1030-1036`): Demand floor tracks seasonality (was static annual avg)
3. **SLOB Dampening** (`mrp.py:1121-1123, 1158-1160`): Reduce batch 50% when product age > SLOB threshold
4. **Diagnostic Fix** (`diagnose_365day.py`): Expanded demand proxy to all 7 channels

**Evidence these fixes work:**
- Prod/demand ratio: 1.27-1.47x → 1.02-1.08x
- Cumulative excess monotonicity: 100% → 67%
- Cumulative excess %: ~15%+ → 3.5-5.2%
- Production CV tracks demand CV (ratio 1.24, was flattened)
- A-fill improved: 84.1% → 85.1%
- SLOB improved: 38.6% → 31.2%

---

## 4. Key File References

| File | What to Read | Why |
|------|-------------|-----|
| `src/prism_sim/agents/replenishment.py` | Lines 476-518 (DC logic), 771-783 (inflow demand), 1025-1048 (safety stock), 1077-1131 (echelon IP) | DC over-ordering root cause |
| `src/prism_sim/agents/allocation.py` | Lines 49-80 (priority), 82-163 (fair share) | A-fill inversion |
| `src/prism_sim/simulation/mrp.py` | Lines 1069-1160 (DOS cap + SLOB dampening) | What v0.46.0 added |
| `src/prism_sim/simulation/orchestrator.py` | Lines 1226-1245 (ABC fill), 1268-1307 (SLOB calc) | Metric calculation |
| `src/prism_sim/simulation/state.py` | Lines 402-413 (age), 449-481 (age blending), 483-504 (weighted age) | Inventory age mechanics |
| `src/prism_sim/config/simulation_config.json` | Replenishment section, MRP thresholds, ABC prioritization | Parameter values |
| `docs/planning/intent.md` | Lines 102-110 | KPI target definitions |
| `CHANGELOG.md` | v0.46.0 and v0.45.0 entries | Recent fix history |

---

## 5. Diagnostic Output Location

All diagnostic outputs from this validation run are at:
- `data/output/triangle_report.txt` — Triangle Report
- `data/output/diagnostics/` — CSV exports from SLOB diagnostic
- `data/output/metrics.json` — Simulation metrics

To regenerate diagnostics without re-running the simulation:
```bash
poetry run python scripts/analysis/diagnose_365day.py --data-dir data/output --window 30
poetry run python scripts/analysis/diagnose_slob.py data/output --csv --dos-threshold 60
poetry run python scripts/analysis/diagnose_service_level.py data/output
```
