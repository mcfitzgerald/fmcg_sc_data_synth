# Prism Sim v0.25.0 Forward Path

> **Purpose**: Consolidated analysis of remaining issues and physics-first approach to fixes.
> **Created**: 2026-01-07
> **Status**: Ready for implementation

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
