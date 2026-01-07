# Prism Sim v0.25.0 Forward Path

> **Purpose**: Consolidated analysis of remaining issues and physics-first approach to fixes.
> **Created**: 2026-01-07
> **Updated**: 2026-01-07 (v0.26.0 hardcode discovery)
> **Status**: Critical - Hardcode audit required before further development

---

## CRITICAL: Hardcode Violations (v0.26.0 Discovery)

### The Problem

During v0.26.0 metric fixes, we discovered **hardcoded values bypassing config**:

```python
# orchestrator.py:249-253 (BEFORE fix)
abc_target_dos = {
    0: 21.0,   # A-items: HARDCODED, ignores config
    1: 16.0,   # B-items: HARDCODED
    2: 12.0,   # C-items: HARDCODED
}
```

**Impact:**
| Parameter | Config Value | Hardcoded Value | Actual Behavior |
|-----------|--------------|-----------------|-----------------|
| store_days_supply | 4.5 days | 21.0 days | **4.7x config** |
| rdc_days_supply | 7.5 days | 21.0 days | **2.8x config** |
| Expected DOS | 18 days | - | 58 days actual |
| Inventory Turns | 20x expected | - | 6.3x actual |

This caused the simulation to hold **3x more inventory** than config specified, making the calibration script useless and all config tuning ineffective.

### Root Cause

The hardcode was added in v0.25.0 with good intentions (prevent Day 1 production spike) but violated the config-driven principle. The CLAUDE.md states "Use semgrep to detect hardcoded values" but this wasn't enforced.

### Immediate Fix Applied (v0.26.0)

Priming now derives from config with ABC velocity factors:
```python
abc_velocity_factors = {0: 1.2, 1: 1.0, 2: 0.8}  # A, B, C
abc_target_dos = {
    abc_class: store_days_supply * factor
    for abc_class, factor in abc_velocity_factors.items()
}
```

---

## P-CRITICAL: Hardcode Audit & Config Overhaul

### Phase 1: Semgrep Hardcode Scan

**Objective:** Find ALL hardcoded values that should be config-driven.

**Existing Rule:** `.semgrep/hardcodes.yaml` - catches numeric literals, strings, booleans in assignments.

**Run Scan:**
```bash
poetry run semgrep --config .semgrep/hardcodes.yaml src/prism_sim/
```

**Triage Process:**
1. Run scan, export results
2. Categorize each finding: VIOLATION (must fix) vs LEGITIMATE (constants, indices, etc.)
3. For violations: move value to config, update calibrate_config.py if derived

**Expected Violations to Find:**
- [ ] Priming buffers (FIXED in v0.26.0)
- [ ] ABC velocity factors (currently hardcoded as 1.2/1.0/0.8)
- [ ] SLOB margin (currently hardcoded as 1.5x)
- [ ] Safety stock factors
- [ ] Lead time assumptions
- [ ] Capacity utilization targets

### Phase 2: Config Schema Overhaul

**Objective:** Create a complete, validated config schema where every tunable parameter is exposed.

**New Config Sections Needed:**

```json
{
  "simulation_parameters": {
    "priming": {
      "enabled": true,
      "strategy": "demand_proportional",
      "abc_velocity_factors": {
        "A": 1.2,
        "B": 1.0,
        "C": 0.8
      },
      "echelon_targets": {
        "store_days_supply": 4.5,
        "customer_dc_days_supply": 7.5,
        "rdc_days_supply": 7.5,
        "plant_days_supply": 3.0
      },
      "network_dos_expected": "AUTO",  // Calculated from echelon targets
      "comments": {
        "network_dos_formula": "store + dc + rdc + lead_time + batch_buffer/2"
      }
    },
    "validation": {
      "slob_abc_thresholds": "AUTO",  // Derived from priming.network_dos_expected
      "slob_margin": 1.5,
      "slob_min_demand_velocity": 1.0
    }
  }
}
```

**Key Principle: Derived Values**
- Some values should be AUTO-derived from others
- `network_dos_expected` = sum of echelon targets + lead time + batch buffer
- `slob_abc_thresholds` = `network_dos_expected` × `slob_margin` × ABC factor
- Calibration script computes these; config stores the inputs

### Phase 3: Calibrate Config Overhaul

**Objective:** Make `calibrate_config.py` the single source of truth for all physics-derived parameters.

**Current State:**
- Calculates: demand, capacity, production_rate_multiplier, inventory DOS
- Added in v0.26.0: SLOB thresholds

**Required Additions:**

```python
def derive_optimal_parameters(...) -> dict:
    """
    Derive ALL physics-constrained parameters from first principles.
    """
    # === EXISTING ===
    # 1. Demand analysis
    # 2. Capacity analysis
    # 3. Production rate multiplier
    # 4. Inventory DOS targets
    # 5. SLOB thresholds (v0.26.0)

    # === NEW: Multi-Echelon Priming ===
    # 6. Network-wide DOS calculation
    network_dos = calculate_network_dos(
        store_dos=store_days_supply,
        dc_dos=customer_dc_days,
        rdc_dos=rdc_days_supply,
        lead_time=lead_time_days,
        batch_horizon=production_horizon,
    )

    # 7. Validate priming doesn't exceed triggers
    # If initial DOS < trigger_dos, Day 1 will have production spike
    # If initial DOS >> trigger_dos, excess inventory
    validate_priming_vs_triggers(
        priming_dos=network_dos,
        trigger_dos_a=trigger_a,
        trigger_dos_b=trigger_b,
        trigger_dos_c=trigger_c,
    )

    # 8. ABC velocity factors (make configurable)
    abc_velocity_factors = derive_abc_factors(
        a_service_target=0.98,  # Higher service for A-items
        b_service_target=0.95,
        c_service_target=0.90,
    )

    # 9. Safety stock by echelon
    # Higher echelons need less safety stock (demand aggregation)
    echelon_safety = calculate_echelon_safety(
        store_cv=0.3,      # High variability at store
        dc_cv=0.15,        # Aggregation reduces CV
        rdc_cv=0.10,       # More aggregation
        service_level_z=1.65,
    )

    # 10. Cross-validate: expected turns vs actual
    expected_turns = 365 / network_dos
    if abs(expected_turns - observed_turns) / expected_turns > 0.2:
        warnings.append(f"Turns mismatch: expected {expected_turns:.1f}, observed {observed_turns:.1f}")
```

**New Validations:**

```python
def validate_config_consistency(config: dict, derived: dict) -> list[str]:
    """
    Check that config values are internally consistent.
    """
    violations = []

    # 1. Priming vs Triggers
    # Initial DOS must exceed trigger to avoid Day 1 spike
    for abc in ["A", "B", "C"]:
        priming = derived["priming_dos"][abc]
        trigger = config["campaign_batching"][f"trigger_dos_{abc.lower()}"]
        if priming < trigger:
            violations.append(
                f"{abc}-item priming ({priming:.1f}d) < trigger ({trigger}d) "
                f"→ Day 1 production spike"
            )

    # 2. SLOB threshold vs Network DOS
    # SLOB threshold should be > expected DOS (else everything is SLOB)
    for abc in ["A", "B", "C"]:
        network_dos = derived["network_dos"][abc]
        slob_threshold = config["validation"]["slob_abc_thresholds"][abc]
        if slob_threshold < network_dos * 1.2:
            violations.append(
                f"{abc}-item SLOB threshold ({slob_threshold:.0f}d) too close to "
                f"expected DOS ({network_dos:.1f}d) → high SLOB %"
            )

    # 3. Multi-echelon double-counting
    # Total network DOS should equal sum of echelons (not multiply)
    store_dos = config["priming"]["echelon_targets"]["store_days_supply"]
    rdc_dos = config["priming"]["echelon_targets"]["rdc_days_supply"]
    if store_dos == rdc_dos:
        violations.append(
            f"Store DOS ({store_dos}) == RDC DOS ({rdc_dos}) - "
            f"are these intentionally equal or copy-paste error?"
        )

    # 4. Capacity vs Demand balance
    utilization = derived["capacity_utilization"]
    if utilization > 0.95:
        violations.append(f"Capacity utilization {utilization:.0%} > 95% → stockouts likely")
    if utilization < 0.50:
        violations.append(f"Capacity utilization {utilization:.0%} < 50% → low OEE")

    return violations
```

### Phase 4: Config-as-Code Enforcement

**Objective:** Prevent future hardcode violations.

**Existing Rule:** `.semgrep/hardcodes.yaml`

**Enforcement Mechanisms:**

1. **Pre-commit hook**: Run semgrep on changed files
   ```yaml
   # .pre-commit-config.yaml
   - repo: https://github.com/returntocorp/semgrep
     hooks:
       - id: semgrep
         args: ['--config', '.semgrep/hardcodes.yaml', '--error']
   ```

2. **CI check**: Fail build if new hardcodes introduced
   ```yaml
   # .github/workflows/lint.yaml
   - name: Check for hardcodes
     run: semgrep --config .semgrep/hardcodes.yaml --error src/
   ```

3. **Code review checklist**:
   - [ ] No new numeric literals in business logic
   - [ ] All thresholds read from config
   - [ ] Calibration script updated if new parameter added

---

## Updated Implementation Order

| Priority | Task | Status |
|----------|------|--------|
| P-CRITICAL | Semgrep hardcode scan | ✅ Complete (v0.27.0) |
| P-CRITICAL | Fix all hardcode violations | ✅ Complete (v0.27.0) |
| P-CRITICAL | Overhaul calibrate_config.py | ✅ Complete (v0.27.0) |
| P0 | Config schema validation | ✅ Complete (v0.27.0) |
| P1 | Pre-commit enforcement | 🔴 Not started |
| P2 | Tune config for target metrics | 🟡 Ready to start |

## v0.27.0 Hardcode Audit Summary

**Completed 2026-01-07:**

### Hardcodes Fixed (moved to config)
1. `orchestrator.py:248` - ABC velocity factors `{0: 1.2, 1: 1.0, 2: 0.8}` → `inventory.initialization.abc_velocity_factors`
2. `orchestrator.py:519` - Production order timeout `14` → `manufacturing.production_order_timeout_days`
3. `orchestrator.py:528` - Batch retention days `30` → `manufacturing.batch_retention_days`
4. `logistics.py:104` - Stale order threshold `14` → `logistics.stale_order_threshold_days`
5. `calibrate_config.py:263` - ABC velocity factors for SLOB → `validation.slob_abc_velocity_factors`
6. `calibrate_config.py:267` - SLOB margin `1.5` → `validation.slob_margin`

### Config Sections Added
```json
"inventory.initialization.abc_velocity_factors": {"A": 1.2, "B": 1.0, "C": 0.8}
"manufacturing.production_order_timeout_days": 14
"manufacturing.batch_retention_days": 30
"logistics.stale_order_threshold_days": 14
"validation.slob_margin": 1.5
"validation.slob_abc_velocity_factors": {"A": 0.7, "B": 1.0, "C": 1.5}
```

### Calibration Validations Added
1. **Multi-echelon priming validation** - Verifies network DOS (store + DC + RDC) exceeds triggers
2. **SLOB threshold vs network DOS** - Flags if thresholds too close to expected DOS
3. **Capacity utilization bounds** - Warns if <50% or >95%
4. **Expected turns sanity check** - Warns if <5x or >30x

### Remaining Legitimate Hardcodes (not violations)
- Loop counters and accumulators (0, 0.0)
- Boolean mask assignments (True/False)
- Enum/status string values
- Physical constants (Earth radius)

---

## Current State (v0.24.0 Baseline)

### 30-Day Run (Healthy)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Service Level | 86.5% | >85% | ✅ OK |
| Truck Fill | 70.0% | >85% | ⚠️ Close |
| SLOB Inventory | 29.0% | <30% | ✅ OK |

### 365-Day Run (Degraded)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Service Level | 80.2% | >85% | ❌ Gap |
| FTL Truck Fill | 31% | >85% | ❌ Gap |
| SLOB Inventory | 81% | <30% | ❌ Gap |
| OEE | 40% | 65-85% | Intentional |

**Key Insight:** Metrics are healthy at 30 days but degrade over 365 days.
This suggests a feedback loop or drift problem, not a structural issue.

---

## Root Cause Analysis

### Issue 1: FTL Truck Fill (31% vs 85% target)

**What's Happening:**
- Plant→RDC shipments ship individual batches as single-SKU trucks (~25% fill)
- RDC→Customer DC routes have insufficient daily volume per route
- With ~4500 nodes / 4 RDCs = ~1125 nodes per RDC, volume is spread thin

**Physics Constraint:**
- Truck capacity: 20,000 kg or 60 m³
- To fill a truck, need ~20,000 kg of product per route per shipment
- Current routes don't accumulate enough volume

**Root Cause:**
- NO consolidation of Plant→RDC shipments (each batch ships separately)
- Per-route volume for RDC→DC is structurally low

**Validated Fix (from industry research):**
- Plant→RDC: Consolidate multiple SKU batches into FTL trucks (Target/Walmart do this)
- RDC→DC: Cross-route consolidation or accept lower fill (these are internal transfers)

### Issue 2: SLOB Inventory (81% vs <30% target)

**What's Happening:**
- SLOB = % of inventory with DOS > 60 days
- Calculation uses `demand_floor = 0.01` for zero-demand SKUs
- Any inventory of a near-zero demand SKU → infinite DOS → flagged as SLOB

**Physics Reality:**
- 500 SKUs with ABC distribution (371 A, 94 B, 35 C)
- C-items have very low velocity by definition
- Campaign batching produces 10-day supply → DOS spikes after production
- Between replenishment cycles, DOS naturally exceeds 60 days

**Root Cause:**
The SLOB metric is measuring correctly, but the 60-day threshold is inappropriate for:
1. Slow-moving C-items (industry expects 90-120 days DOS)
2. Campaign-batched production (DOS spikes are normal)
3. Warehouse inventory (RDCs are SUPPOSED to hold 2-4 weeks stock)

**Questions to Resolve:**
1. Should SLOB be measured at store level only (not RDCs)?
2. Should threshold be ABC-differentiated (A:30d, B:60d, C:90d)?
3. Is the 0.01 demand floor creating artificial SLOB?

### Issue 3: Service Level (80.2% vs 85% target)

**What's Happening:**
- Store fill rate is 80.2%
- Gap likely due to stockouts during campaign replenishment cycles
- Trigger thresholds (14/10/7 days) + lead time (3 days) = tight window

**Root Cause:**
Production triggers may not account for full replenishment lead time.

---

## Calibration-First Principle

**CRITICAL: Before any simulation run, validate configuration against physics:**

```python
# Proposed: Add to run_simulation.py or create calibration script
def validate_config_physics(config: dict) -> tuple[bool, list[str]]:
    """
    Validate configuration parameters against supply chain physics.

    Returns:
        (is_valid, list_of_violations)

    If violations exist, either:
    1. Config is wrong (fix the config)
    2. Simulation logic is wrong (fix the code)
    """
    violations = []

    # 1. Little's Law: Inventory = Throughput × Lead Time
    # If we produce X cases/day with L days lead time, expect X*L in transit

    # 2. Campaign Batching Consistency
    # trigger_dos + lead_time < horizon_days (else stockout before replenishment)
    trigger_a = config["campaign_batching"]["trigger_dos_a"]  # 14
    lead_time = config["manufacturing"]["production_lead_time_days"]  # 3
    horizon = config["campaign_batching"]["production_horizon_days"]  # 10

    if trigger_a < lead_time:
        violations.append(f"A-item trigger ({trigger_a}d) < lead time ({lead_time}d) → stockouts")

    # 3. SLOB Threshold vs Campaign Horizon
    # If we produce 10 days supply, DOS will be ~10 days after production
    # SLOB threshold of 60 days should not flag normal inventory
    slob_threshold = config["validation"]["slob_days_threshold"]  # 60
    if horizon * 3 > slob_threshold:
        violations.append(f"3x horizon ({horizon*3}d) > SLOB threshold ({slob_threshold}d) → high SLOB")

    # 4. Truck Fill Physics
    # If average order is X kg and truck is 20,000 kg, need 20,000/X orders per truck

    return (len(violations) == 0, violations)
```

---

## Proposed Fixes (Priority Order)

### P0: Plant→RDC Truck Consolidation

**Location:** `orchestrator.py:_create_plant_shipments()`

**Current:** Each batch creates a separate shipment
**Fix:** Group batches by (plant_id, rdc_id) route, pack into FTL trucks

```python
# Pseudocode
route_batches = group_by(batches, key=lambda b: (b.plant_id, target_rdc))
for route, batches in route_batches.items():
    current_truck = []
    current_weight = 0
    for batch in batches:
        if current_weight + batch.weight > TARGET_FILL:
            ship(current_truck)
            current_truck = []
            current_weight = 0
        current_truck.append(batch)
        current_weight += batch.weight
    ship(current_truck)  # final partial
```

**Expected Impact:** FTL fill 31% → 60-70%

### P1: SLOB Metric Refinement

**Options (choose one):**

A) **ABC-differentiated thresholds** (recommended)
   - A-items: 30 days (fast movers should turn quickly)
   - B-items: 60 days (current)
   - C-items: 120 days (slow movers expected to be slow)

B) **Measure at store level only**
   - RDCs are supposed to hold inventory
   - Only flag store-level inventory as SLOB

C) **Use demand velocity floor**
   - Current: `max(demand, 0.01)`
   - Change to: `max(demand, expected_demand * 0.1)`
   - Prevents zero-demand SKUs from infinite DOS

**Expected Impact:** SLOB 81% → 40-50% (with option A)

### P2: Service Level Buffer

**Location:** `simulation_config.json:campaign_batching`

**Current:**
```json
"trigger_dos_a": 14,
"trigger_dos_b": 10,
"trigger_dos_c": 7
```

**Fix:** Add lead time buffer
```json
"trigger_dos_a": 17,  // 14 + 3 lead time
"trigger_dos_b": 13,  // 10 + 3
"trigger_dos_c": 10   // 7 + 3
```

**Expected Impact:** Service 80% → 83-85%

---

## Implementation Order

1. **Add config validation script** (establish physics baseline)
2. **Implement P0** (Plant→RDC consolidation) - biggest FTL impact
3. **Run 365-day simulation** - measure improvement
4. **Implement P1** (SLOB refinement) - reduce false positives
5. **Implement P2** (Service buffer) - if still needed after P0/P1
6. **Final validation run**

---

## Key Files

| File | Changes Needed |
|------|----------------|
| `orchestrator.py:_create_plant_shipments()` | Multi-SKU truck packing |
| `orchestrator.py:_record_daily_metrics()` | SLOB calculation refinement |
| `simulation_config.json` | Trigger threshold tuning |
| `scripts/validate_config.py` | NEW: Physics validation |

---

## Research References

- [Walmart Freight Consolidation](https://fstlogistics.com/walmart-freight-consolidation/) - Vendor pool programs
- [Target Consolidation Program](https://www.hubgroup.com/logistics-management/consolidation-fulfillment/retail-consolidation/target-consolidation-program/) - Weekly FTL consolidation
- [APQC DOS Benchmarks](https://www.apqc.org/what-we-do/benchmarking/open-standards-benchmarking/measures/inventory-days-supply) - Industry DOS targets
- [ABC Inventory Analysis](https://www.leafio.ai/blog/inventory-management-can-boost-low-demand-product-businesses/) - SLOB management
