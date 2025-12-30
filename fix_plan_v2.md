# Prism-Sim Fix Plan v2: Brownfield Data Generator

**Created:** 2025-12-30
**Updated:** 2025-12-30
**Status:** Phase A COMPLETE, Phase B IN PROGRESS

**Context:** Previous debug plan (`sparkling-nibbling-crescent.md`) was based on incorrect root cause hypotheses. Phase 1 diagnostics revealed the actual issues.

---

## Quick Start (For New Session)

### Phase A is COMPLETE. To continue:

1. **Regenerate static world** (if needed):
   ```bash
   poetry run python scripts/generate_static_world.py
   ```

2. **Run 30-day test**:
   ```bash
   poetry run python run_simulation.py --days 30 --no-logging
   ```

3. **Expected results after Phase A**:
   - Demand: ~400k cases/day
   - Capacity: ~481k cases/day
   - Production: happening (but stops ~day 20 due to bullwhip)
   - Service Level: ~88%
   - OEE: ~39%

4. **Next step: Phase B (Bullwhip Dampening)** - see section below

### Key Config Values (Post Phase A)
| Parameter | File | Value |
|-----------|------|-------|
| `base_daily_demand` | simulation_config.json | 7.0 |
| `production_hours_per_day` | simulation_config.json | 20.0 |
| `rdc_store_multiplier` | simulation_config.json | 100.0 |
| `store_days_supply` | simulation_config.json | 14.0 |
| `seasonality.amplitude` | simulation_config.json | 0.12 |
| `run_rate_cases_per_hour` (ORAL) | world_definition.json | 7500 |
| `run_rate_cases_per_hour` (PERSONAL) | world_definition.json | 9000 |
| `run_rate_cases_per_hour` (HOME) | world_definition.json | 6000 |
| `lift_multiplier` (Black Friday) | world_definition.json | 2.0 |

---

## Objective

Create a **brownfield digital twin** that generates realistic FMCG supply chain operational data with:
- **Base service level**: 92-94% (realistic, not perfect)
- **With quirks/events**: Service degrades (problems to discover)
- **With improvements**: Can achieve 95-98% (room to optimize)

The simulation produces data that looks like real enterprise ERP/WMS output. Inefficiencies are embedded but NOT labeled - they must be discovered through post-hoc analysis (VG/SQL).

---

## Current State (What's Broken)

### Simulation Behavior
- **Day 1-114**: System running, gradual degradation
- **Day 114**: PLANT-TX stops (first failure)
- **Day 155**: PLANT-OH stops
- **Day 164**: PLANT-CA stops
- **Day 171**: PLANT-GA stops (last plant)
- **Day 172+**: Complete collapse (0 production, 0 shipments)

### Root Causes Identified (Phase 1 Diagnostics)

| Issue | Current State | Impact |
|-------|---------------|--------|
| **Production capacity** | ~79k cases/day | Demand is 100-130k → structural deficit |
| **Bullwhip amplification** | 350x (60k → 21M orders) | Unrealistic, overwhelms system |
| **Ingredient inventory** | 5M units (1,923 days supply) | Way too high, hides procurement issues |
| **Replenishment policy** | Same for all 4,500 stores | Unrealistic, amplifies bullwhip |
| **Recovery mechanics** | Fill-or-Kill only | No backorder → death spiral |

### What's NOT Broken
- Supplier→Plant links (74 links exist, routing works)
- MRP purchase order generation (works when triggered)
- Allocation logic (handles capacity correctly)
- Logistics/shipment creation (works correctly)

---

## Target State

### Service Level Bands
```
98%+ │ World Class (with optimizations discovered via analysis)
95%  │ ──────── Optimized target
92-94│ BASE SIMULATION (this is our target)
85%  │ ──────── Trouble zone (with quirks/events)
<80% │ Crisis (recoverable within days)
```

### Capacity Architecture
```
Demand ceiling: ~130k cases/day (peak with seasonality)
Base capacity:  ~145-155k cases/day (10-15% headroom)
Utilization:    85-90% in steady state
```

### Key Metrics Targets
| Metric | Target | Rationale |
|--------|--------|-----------|
| Service Level | 92-94% | Realistic brownfield |
| OEE | 75-85% | Top quartile CPG benchmark |
| Inventory Turns | 8-12x | FMCG typical |
| Bullwhip Ratio | 1.5-3x per echelon | Research-backed |

---

## Research Findings

### Demand Variability (FMCG)
Source: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360835220301145), [Intuendi](https://intuendi.com/demand-variability/)

- **CoV < 0.5**: Stable, predictable demand
- **CoV 0.5-1.0**: Moderate variability
- **CoV > 1.0**: Highly volatile (red flag)
- **Seasonality**: Predictable variability, not uncertainty (±20% is typical for FMCG)

### OEE Benchmarks (CPG)
Source: [Reliable Plant](https://www.reliableplant.com/Read/21842/report-finds-'best-of-best'-plants-operate-at-93-oee), [Shoplogix](https://shoplogix.com/oee-calculation-in-cpg/)

- **Average CPG**: 40-60%
- **Typical discrete mfg**: 60%
- **Top quartile CPG**: 78%
- **World class**: 85%
- **Best of best**: 93%

### Inventory Days of Supply
Source: [Demand Planning Net](https://demandplanning.net/cpg-fmcg/), [Intuendi](https://intuendi.com/resource-center/inventory-days/)

- **FMCG finished goods**: 20-40 days DSI
- **Raw materials (commodity)**: 7-14 days
- **Raw materials (specialty)**: 30-45 days

### Colgate Peer Group
Source: [Colgate Investor Relations](https://investor.colgatepalmolive.com/news-releases/news-release-details/colgate-announces-4th-quarter-and-full-year-2024-results), [Statista](https://www.statista.com/topics/3456/colgate-palmolive/)

- Colgate 2024 Net Sales: $20.1B
- Oral/Personal/Home Care segment: $15.6B
- Volume growth: 3.7% YoY
- Global toothpaste market share: 41.4%

**Note:** Specific daily production volumes (cases/day) are proprietary and not publicly disclosed. We should calibrate our simulation to produce reasonable throughput that scales to industry revenue figures.

---

## Fix Plan

### Phase A: Capacity Rebalancing (CRITICAL)

**Goal:** Production capacity > Peak demand with 10-15% headroom

#### A.1: Increase Production Run Rates
**File:** `src/prism_sim/config/world_definition.json`

```json
"category_profiles": {
  "ORAL_CARE": {
    "run_rate_cases_per_hour": 5000,      // Was 3000
    "changeover_time_hours": 0.5
  },
  "PERSONAL_WASH": {
    "run_rate_cases_per_hour": 5500,      // Was 3600
    "changeover_time_hours": 0.75
  },
  "HOME_CARE": {
    "run_rate_cases_per_hour": 4000,      // Was 2400
    "changeover_time_hours": 1.0
  }
}
```

#### A.2: Adjust Plant Efficiency
**File:** `src/prism_sim/config/simulation_config.json`

```json
"plant_parameters": {
  "PLANT-OH": { "efficiency_factor": 0.78 },  // Was 0.70
  "PLANT-TX": { "efficiency_factor": 0.88 },  // Was 0.95 (too high!)
  "PLANT-CA": { "efficiency_factor": 0.82 },  // Was 0.85
  "PLANT-GA": { "efficiency_factor": 0.80 }   // Was 0.85
}
```

**Rationale:** Target 78-88% OEE (top quartile CPG). Lower variance between plants.

#### A.3: Validate New Capacity Math
```
PLANT-OH: 4000 * 8 * 0.78 = 24,960 cases/day
PLANT-TX: 5000 * 8 * 0.88 = 35,200 cases/day
PLANT-CA: 5250 * 8 * 0.82 = 34,440 cases/day (avg of ORAL+PERSONAL)
PLANT-GA: 4750 * 8 * 0.80 = 30,400 cases/day (avg of PERSONAL+HOME)
────────────────────────────────────────────
TOTAL: ~125,000 cases/day base capacity

With demand at 100-115k avg, this gives 8-25% headroom.
Peak demand (130k) would stress the system but not collapse it.
```

---

### Phase B: Bullwhip Dampening (HIGH)

**Goal:** Reduce order amplification from 350x to 3-9x total (realistic)

#### B.1: Reduce Replenishment Aggressiveness
**File:** `src/prism_sim/config/simulation_config.json`

```json
"agents": {
  "replenishment": {
    "target_days_supply": 10.0,       // Was 21.0 (too aggressive)
    "reorder_point_days": 4.0,        // Was 10.0 (too high)
    "min_order_qty": 50.0,            // Was 10.0
    "batch_size_cases": 100.0
  }
}
```

#### B.2: Add Channel-Specific Policies
**File:** `src/prism_sim/agents/replenishment.py`

Add channel differentiation (implementation detail):
```python
CHANNEL_POLICIES = {
    "B2M_LARGE": {      # Walmart, Target - sophisticated
        "target_days": 7,
        "reorder_point_days": 3,
        "batch_size": 500,
        "smoothing_factor": 0.3  # Dampen signals
    },
    "B2M_CLUB": {       # Costco - bulk
        "target_days": 10,
        "reorder_point_days": 4,
        "batch_size": 200,
        "smoothing_factor": 0.2
    },
    "B2M_DISTRIBUTOR": { # Traditional - less sophisticated
        "target_days": 14,
        "reorder_point_days": 5,
        "batch_size": 100,
        "smoothing_factor": 0.1
    },
    "ECOMMERCE": {      # Amazon - fast
        "target_days": 5,
        "reorder_point_days": 2,
        "batch_size": 50,
        "smoothing_factor": 0.4
    }
}
```

#### B.3: Add Demand Smoothing
**File:** `src/prism_sim/agents/replenishment.py`

Add exponential smoothing to demand signal:
```python
# Instead of raw demand:
smoothed_demand = alpha * current_demand + (1 - alpha) * previous_smoothed
# Use smoothed_demand for ROP/target calculations
```

---

### Phase C: Realistic Ingredient Inventory (MEDIUM)

**Goal:** 14-45 days of supply (not 1,923 days!)

#### C.1: Reduce Initial Inventory
**File:** `src/prism_sim/config/simulation_config.json`

```json
"manufacturing": {
  "initial_plant_inventory": {
    "default": 100000.0,              // Was 5,000,000 (reduce 50x)
    "ACT-CHEM-001": 500000.0,         // SPOF: 45 days supply
    "BLK-WATER-001": 200000.0         // Commodity: 14 days
  },
  "inventory_policies": {
    "DEFAULT": { "reorder_point_days": 10.0, "target_days_supply": 21.0 },
    "INGREDIENT": { "reorder_point_days": 14.0, "target_days_supply": 28.0 },
    "SPOF": { "reorder_point_days": 21.0, "target_days_supply": 45.0 }
  }
}
```

#### C.2: Ensure Ingredient POs Actually Trigger
With reduced inventory, the MRP's `inv_position < rop_levels` check will actually trigger, creating purchase orders. Verify this works in testing.

---

### Phase D: Recovery Mechanics (MEDIUM)

**Goal:** System can recover from gaps within days, not collapse

#### D.1: Add Backorder Handling (Optional)
Instead of Fill-or-Kill, allow partial fills to carry forward:
```python
# In allocation.py
if fill_rate < 1.0:
    unfilled_qty = order_qty * (1 - fill_rate)
    # Create backorder for next day instead of killing
```

#### D.2: Add Production Smoothing in MRP
Prevent MRP from creating huge spikes in production orders:
```python
# Cap daily production order increase to 20% over previous day
max_increase = previous_day_orders * 1.2
today_orders = min(raw_orders, max_increase)
```

---

### Phase E: Scenario Capacity (LOW - Future)

**Goal:** Enable "what-if" capacity bumps for analysis

#### E.1: Add Capacity Multiplier Config
**File:** `src/prism_sim/config/simulation_config.json`

```json
"manufacturing": {
  "capacity_scenarios": {
    "enabled": false,
    "multipliers": {
      "PLANT-OH": 1.0,
      "PLANT-TX": 1.0,
      "PLANT-CA": 1.0,
      "PLANT-GA": 1.0,
      "CONTRACT_MFG": 0.0  // 0 = disabled, 0.15 = +15% capacity
    }
  }
}
```

This allows running "what-if" scenarios without changing core logic. An analyst could say "what if we added 15% contract manufacturing capacity?" and re-run.

---

## Implementation Order

| Priority | Phase | Effort | Impact |
|----------|-------|--------|--------|
| 1 | A.1-A.3: Capacity rebalancing | Low | HIGH - Fixes collapse |
| 2 | B.1: Reduce replenishment aggressiveness | Low | HIGH - Fixes bullwhip |
| 3 | C.1: Reduce ingredient inventory | Low | MEDIUM - Adds realism |
| 4 | B.2-B.3: Channel policies + smoothing | Medium | MEDIUM - Adds realism |
| 5 | D.1-D.2: Recovery mechanics | Medium | MEDIUM - Prevents death spiral |
| 6 | E.1: Scenario capacity | Low | LOW - Future capability |

---

## Verification Criteria

After fixes, 365-day simulation should show:

| Metric | Target | Tolerance |
|--------|--------|-----------|
| Service Level | 92-94% | ±2% |
| OEE | 75-85% | ±5% |
| Inventory Turns | 8-12x | ±2x |
| Production > 0 | All 365 days | No collapse |
| Bullwhip ratio | 3-9x | Order variance / demand variance |
| Recovery time | < 7 days | From any single disruption |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/prism_sim/config/world_definition.json` | Run rates |
| `src/prism_sim/config/simulation_config.json` | Plant efficiency, replenishment params, inventory levels |
| `src/prism_sim/agents/replenishment.py` | Channel policies, demand smoothing |
| `src/prism_sim/simulation/mrp.py` | Production smoothing (optional) |
| `src/prism_sim/agents/allocation.py` | Backorder handling (optional) |

---

## Questions to Resolve Before Implementation

1. **Demand calibration**: Current avg demand is ~100k cases/day. Should this change? Need to validate against Colgate/P&G scale.

2. **Category overlap**: Currently each plant has 1-2 categories. Should we add more flexibility for "discoverable" capacity?

3. **Seasonality amplitude**: Currently ±20%. Is this realistic for oral/personal/home care?

4. **Promotion impact**: Promotions can spike demand. What's the realistic multiplier? (1.5x? 2x?)

---

## Diagnostic Scripts Created (Phase 1)

Available in `scripts/`:
- `diagnose_phase1.py` - Network topology analysis
- `diagnose_phase1_v2.py` - 30-day simulation trace
- `diagnose_phase1_v3.py` - PO pipeline verification
- `diagnose_phase1_v4.py` - Orchestrator flow trace
- `diagnose_phase1_v5.py` - MRP logic deep dive

These can be reused for verification after fixes.

---

## References

- [Demand forecasting research (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0360835220301145)
- [OEE benchmarks (Reliable Plant)](https://www.reliableplant.com/Read/21842/report-finds-'best-of-best'-plants-operate-at-93-oee)
- [Bullwhip effect research (ResearchGate)](https://www.researchgate.net/publication/372443077_Bullwhip_Effect_Demand_Variation_and_Amplification_within_Supply_Chains)
- [CPG trends 2024 (Bain)](https://www.bain.com/insights/consumer-products-report-2024-resetting-the-growth-agenda/)
- [Colgate 2024 results](https://investor.colgatepalmolive.com/news-releases/news-release-details/colgate-announces-4th-quarter-and-full-year-2024-results)

---

## Phase A Implementation Notes (Completed 2025-12-30)

### Research Question Answers

| Question | Answer |
|----------|--------|
| **Demand calibration** | Colgate NA produces ~800k-1.5M cases/day. For multi-category simulation, target 400-500k/day (subset of operations). |
| **Category overlap** | Industry uses dedicated lines within flexible plants. Current model (1-2 categories/plant) is realistic. |
| **Seasonality amplitude** | ±10-15% for oral/personal care (staple categories). Changed from ±20% to ±12%. |
| **Promotion impact** | 1.3-1.5x typical, 2x max for aggressive promos. Changed Black Friday from 3.0x to 2.0x. |

### Code Changes Made

#### 1. Configuration Updates

**simulation_config.json:**
- `base_daily_demand`: 1.0 → 7.0 (per SKU/store)
- `production_hours_per_day`: 8.0 → 20.0 (realistic 3-shift operation)
- `seasonality.amplitude`: 0.20 → 0.12
- `store_days_supply`: 28.0 → 14.0
- `rdc_store_multiplier`: 1500.0 → 100.0
- Plant efficiency factors: leveled to 78-88% OEE range

**world_definition.json:**
- `run_rate_cases_per_hour`: ORAL 3000→7500, PERSONAL 3600→9000, HOME 2400→6000
- Black Friday `lift_multiplier`: 3.0 → 2.0

#### 2. Architecture Changes (Option C: DCs + Stores)

**generators/network.py** - Major rewrite for hierarchical structure:
- 20 Retailer DCs with ~100 stores each (B2M_LARGE)
- 8 Distributor DCs with 500 small retailers each (B2M_DISTRIBUTOR)
- 30 Club stores direct to RDC (B2M_CLUB)
- 10 Ecom FCs (ECOMMERCE)
- 2 DTC FCs (DTC)
- Total: ~6,600 nodes (vs 155 before)

**simulation/builder.py:**
- Added parsing for `channel`, `store_format`, `parent_account_id` from CSV

**simulation/demand.py:**
- Skip RETAILER_DC and DISTRIBUTOR_DC in demand generation (DCs aggregate, don't generate)
- Added store format scale factors: SUPERMARKET=1.0, CONVENIENCE=0.5, CLUB=15.0, ECOM_FC=50.0

#### 3. Bug Fixes

- `sample_company()` → `sample_companies(1)[0]` in network.py
- `sample_city()` → `sample_cities(1)[0]` in network.py

### Key Learnings

1. **Run rates are baked into static world recipes**, not read dynamically. Must regenerate static world after config changes.

2. **Inventory priming drives early behavior**. With `rdc_store_multiplier=1500`, initial inventory was so high that DOS never triggered reorder, causing production=0.

3. **Demand double-counting risk**. Both DCs and stores were generating demand until we added the RETAILER_DC/DISTRIBUTOR_DC skip logic.

### Remaining Issues (Phase B)

- **Bullwhip still extreme**: Orders explode from 60k → 12.9M by day 20
- **Production stops ~day 20**: System collapses due to bullwhip
- **OEE at 39%**: Below target 75-85%, but expected given bullwhip chaos
