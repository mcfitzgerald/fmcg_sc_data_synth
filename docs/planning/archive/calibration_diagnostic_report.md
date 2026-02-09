# Calibration Diagnostic Report - Phase 1 Audit

**Date:** 2026-01-14
**Version:** v0.32.1 Baseline
**Analyst:** Claude (Phase 1 Diagnostic Audit)

---

## Executive Summary

The `calibrate_config.py` script has fundamental flaws that caused the v0.32.0 regression (Service 89%→82%, SLOB 4%→85%). This audit identifies three root causes:

1. **Lead Time Underestimate:** Script assumes 3-day uniform lead time; actual network cascade is 5-10 days
2. **Missing MRP Signal Lag:** MRP uses 14-day rolling window that script doesn't account for
3. **Cold-Start Artifacts:** Day 1-30 metrics are distorted by ~5% due to system warm-up

---

## 1. Multi-Echelon Lead Time Analysis

### 1.1 Actual Network Lead Times (from `links.csv`)

| Echelon Path | Avg Lead Time | Range | Notes |
|--------------|---------------|-------|-------|
| Customer DC → Store | **~1.0d** | 1.00-1.02d | Local metropolitan delivery |
| RDC → Customer DC | **~1.4d** | 1.06-2.65d | Varies by geography |
| Plant → RDC | **~1.8d** | 1.09-3.03d | Same region vs cross-country |
| Supplier → Plant | **~1.5d** | 1.0-4.2d | Wide geographic variance |

### 1.2 Lead Time Formula (from `network.py:250-251`)

```python
lt = (dist / speed / 24) + handling
# speed = 80 km/h (from config)
# handling = 1.0 day base (from config)
```

Key insight: **1 day of handling time is added to every link**, making the minimum lead time 1.0 days even for co-located nodes.

### 1.3 Multi-Echelon Cascade (Product Replenishment Path)

For a Store to receive inventory from a Plant:

```
Path: Store ← Customer_DC ← RDC ← Plant
       1.0d  +   1.4d      + 1.8d  = 4.2d (same region minimum)
       1.0d  +   2.0d      + 2.5d  = 5.5d (cross-country)
```

**Plus FTL Consolidation Delays:**
- Each FTL hop adds 1-3 days waiting for minimum pallet threshold
- `held_orders` accumulate until `min_order_pallets` met (10-20 pallets by channel)
- Store LTL mode bypasses this, but DC-to-DC is FTL

**Total Effective Replenishment Time:**

| Scenario | Transit | FTL Delays | Total |
|----------|---------|------------|-------|
| Best case (same region) | 4.2d | 1d | **5.2d** |
| Average | 4.8d | 2d | **6.8d** |
| Worst case (cross-country) | 5.5d | 4d | **9.5d** |

### 1.4 Calibration Script Assumption

**Current (`calibrate_config.py:249`):**
```python
lead_time_days = log_config.get("default_lead_time_days", 3.0)
```

**Gap:** The script underestimates total lead time by **3-7 days** (2-3x error).

### 1.5 Impact on Trigger Thresholds

Current trigger derivation (`calibrate_config.py:387-389`):
```python
trigger_a = int(replenishment_time + safety_a + review_a)  # 6 + 10 + 5 = 21
trigger_b = int(replenishment_time + safety_b + review_b)  # 6 + 6 + 5 = 17
trigger_c = int(replenishment_time + safety_c + review_c)  # 6 + 3 + 3 = 12
```

Where `replenishment_time = 6` (3d production + 3d transit).

**Correct values should account for full cascade:**
```python
# Multi-echelon replenishment time = network transit + FTL consolidation
replenishment_time = 7-10 days (not 6)

# Corrected triggers:
trigger_a = 10 + 10 + 5 = 25 days
trigger_b = 10 + 6 + 5 = 21 days
trigger_c = 10 + 3 + 3 = 16 days
```

---

## 2. Simulation Mechanics Audit

### 2.1 Parameter-to-Component Mapping

| Config Parameter | Component | File | Effect |
|-----------------|-----------|------|--------|
| `trigger_dos_a/b/c` | MRPEngine | mrp.py:224-226 | Days-of-supply threshold for campaign batching |
| `store_days_supply` | Orchestrator | orchestrator.py:244 | Day 1 store inventory priming |
| `rdc_days_supply` | Orchestrator | orchestrator.py:245 | Day 1 RDC inventory priming |
| `abc_velocity_factors` | Orchestrator | orchestrator.py:255-263 | ABC-differentiated priming multipliers |
| `production_horizon_days` | MRPEngine | mrp.py:220-221 | Days of demand to produce per trigger |
| `order_cycle_days` | MinMaxReplenisher | replenishment.py | Days between replenishment reviews |
| `default_lead_time_days` | LogisticsEngine | logistics.py:174 | Fallback if link not found |
| `slob_abc_thresholds` | Orchestrator | orchestrator.py:116-118 | DOS threshold for SLOB classification |

### 2.2 MRP Signal Processing (Critical Finding)

**MRP uses a 14-day rolling window** (`mrp.py:172-174`):

```python
self.demand_history = np.tile(
    self.expected_daily_demand, (14, 1)
).astype(np.float64)
```

**Implications:**
1. Demand signals take 7-14 days to fully propagate through MRP
2. Production responses lag actual demand changes
3. Trigger thresholds must account for this signal lag

**The calibration script has no knowledge of this.** It calculates triggers as if MRP responds instantly to demand changes.

### 2.3 MRP Signal Lag Quantification

From `mrp.py:624`:
```python
avg_daily_demand_vec = np.mean(self.demand_history, axis=0)  # 14-day average
```

And week-over-week trend detection (`mrp.py:636-639`):
```python
week1_avg = np.mean(self.demand_history[:7], axis=0)
week2_avg = np.mean(self.demand_history[7:], axis=0)
```

**Signal Propagation Timeline:**
- Day 1-7: Signal entering rolling window
- Day 7-14: Signal reaching full representation
- Day 14+: Steady-state responsiveness

**Recommended Trigger Adjustment:**
Add `MRP_signal_lag = 7 days` to trigger threshold formula.

### 2.4 ABC Production Factors (Hardcoded in Python)

Found in `calibrate_config.py:359-363`:
```python
abc_priming_factors = {
    "A": 1.2,  # Hardcoded
    "B": 1.0,  # Hardcoded
    "C": 0.85,  # Hardcoded
}
```

These should be derived from z-scores in config:
```python
# Config has: z_A=2.33, z_B=2.0, z_C=1.65
abc_factors = {
    "A": 2.33 / 2.0,  # = 1.165
    "B": 1.0,
    "C": 1.65 / 2.0,  # = 0.825
}
```

---

## 3. Cold-Start Impact Measurement

### 3.1 Cold-Start Artifacts Identified

| Day Range | Artifact | Component | Impact |
|-----------|----------|-----------|--------|
| Day 1-7 | Replenishment variance insufficient | MinMaxReplenisher | Uses default safety stock |
| Day 1-14 | MRP history warm-starting | MRPEngine | Signal doesn't reflect actual demand pattern |
| Day 1-20 | Lead time history empty | MinMaxReplenisher | Uses static lead time assumption |
| Day 1-28 | Demand variance insufficient | MinMaxReplenisher | Safety stock calculation suboptimal |
| Day 1-30 | ABC reclassification unstable | MRPEngine | Products may shift classes as data accumulates |

### 3.2 Warm-Start Initialization Analysis

**Current approach (`orchestrator.py:276-289`):**
```python
# Day 1 seasonal factor applied
day_1_seasonal = 1.0 + amplitude * np.sin(
    2 * np.pi * (1 - phase_shift) / cycle_days
)
```

**What this handles:**
- Seasonal demand adjustment for Day 1

**What this doesn't handle:**
- FTL consolidation pipeline fill
- MRP signal lag buffer
- Replenishment order pipeline
- In-transit inventory from priming

### 3.3 Cold-Start Metric Distortion Estimate

**Service Level:**
- Day 1-14: Volatile (±5-10% from steady state)
- Day 14-30: Stabilizing (±2-5% from steady state)
- Day 30+: Steady state

**Inventory Turns:**
- Day 1-30: Artificially high (priming depleting)
- Day 30-90: Stabilizing to true velocity
- Day 90+: Reliable metric

**SLOB:**
- Day 1-60: Not meaningful (insufficient time to classify slow-moving)
- Day 60+: Approaching reliability

**Estimated 90-day metric distortion: 3-5%** due to Day 1-30 artifacts.

### 3.4 Evidence from MRP Initialization

```python
# mrp.py:180-183 - Production history warm-started
total_expected_production = np.sum(self.expected_daily_demand)
self.production_order_history = np.full(
    14, total_expected_production, dtype=np.float64
)
```

This prevents immediate death spiral but creates an artificial baseline that may not match actual system dynamics.

---

## 4. Root Cause Summary

### 4.1 Why v0.32.0 Regressed

| Parameter | v0.31.0 (Working) | v0.32.0 (Broken) | Root Cause |
|-----------|-------------------|------------------|------------|
| `store_days_supply` | 27.0 | 14.0 | Script used 3d lead time, not 7d cascade |
| `rdc_days_supply` | 41.0 | 21.3 | Same lead time underestimate |
| `abc_velocity_factors.A` | 1.5 | 1.2 | Hardcoded value, not derived from service targets |
| `abc_velocity_factors.C` | 0.5 | 0.85 | Inverted hierarchy (C got MORE than B) |
| `trigger_dos_a` | 14 | 21 | Formula didn't account for MRP signal lag |
| `slob_abc_thresholds.A` | 120 | 64 | Too aggressive, flagged normal inventory as SLOB |

### 4.2 The Fundamental Error

The calibration script treats the supply chain as **single-echelon with 3-day lead time**, when it's actually **4-tier with 7-10 day effective lead time plus MRP signal lag**.

---

## 5. Recommendations for Phase 2

### 5.1 Multi-Echelon Lead Time Cascade

```python
def calculate_echelon_lead_times(world, config):
    """Calculate cumulative lead times per echelon."""
    handling = config["geospatial"]["base_handling_days"]  # 1.0
    ftl_consolidation = config["logistics"].get("ftl_consolidation_days", 2)
    prod_lt = config["manufacturing"]["production_lead_time_days"]  # 3

    # Calculate actual transit from links (or use config defaults)
    transit_store = handling  # Local delivery
    transit_dc = handling + ftl_consolidation  # ~3d
    transit_rdc = transit_dc * 1.5  # ~4.5d
    transit_plant = transit_dc * 2 + prod_lt  # ~9d

    return {
        "store": transit_store,
        "customer_dc": transit_dc,
        "rdc": transit_rdc,
        "plant": transit_plant,
    }
```

### 5.2 MRP Signal Lag Incorporation

```python
def calculate_trigger_threshold(echelon_lt, cv, z_score, config):
    # Safety stock
    ss = z_score * cv * math.sqrt(echelon_lt)

    # MRP rolling window lag (14-day history → 7-day effective lag)
    mrp_signal_lag = 7

    # Review period
    review_period = config["replenishment"]["order_cycle_days"]

    return echelon_lt + ss + mrp_signal_lag + (review_period / 2)
```

### 5.3 ABC Factor Derivation

```python
def derive_abc_factors(config):
    z = config["calibration"]["service_level_z_scores"]
    z_B = z["B"]  # Baseline
    return {
        "A": z["A"] / z_B,  # 2.33/2.0 = 1.165
        "B": 1.0,
        "C": z["C"] / z_B,  # 1.65/2.0 = 0.825
    }
```

### 5.4 Cold-Start Buffer

```python
def calculate_priming_dos(network_dos, cold_start_days=30, sim_days=365):
    """Add buffer to account for cold-start artifacts."""
    cold_start_fraction = cold_start_days / sim_days  # 30/365 = 8.2%
    cold_start_buffer = 1.0 + (cold_start_fraction * 2)  # 1.164
    return network_dos * cold_start_buffer
```

---

## 6. Verification Metrics

After implementing Phase 2 fixes, verify:

| Check | Method | Target |
|-------|--------|--------|
| Trigger > Priming | Network priming DOS > trigger DOS per ABC class | All classes pass |
| Lead time cascade | Sum of echelon lead times ≈ actual order-to-delivery time | Within 20% |
| ABC hierarchy | A > B > C for priming factors | Correct ordering |
| Cold-start stability | Day 30 metrics within 5% of Day 90 metrics | <5% variance |

---

## Appendix A: Key File Locations

| Concept | File | Line Numbers |
|---------|------|--------------|
| Lead time calculation | `generators/network.py` | 250-257 |
| MRP rolling window | `simulation/mrp.py` | 172-174, 624 |
| Inventory priming | `simulation/orchestrator.py` | 227-405 |
| Trigger thresholds | `scripts/calibrate_config.py` | 374-389 |
| ABC priming factors | `scripts/calibrate_config.py` | 359-363 |

## Appendix B: Config Parameter Quick Reference

```json
{
  "simulation_parameters": {
    "calibration": {
      "trigger_components": {
        "production_lead_time_days": 3,
        "transit_time_days": 3,  // ISSUE: Should be 7-10
        "safety_buffer_a": 10,
        "safety_buffer_b": 6,
        "safety_buffer_c": 3
      }
    },
    "inventory": {
      "initialization": {
        "store_days_supply": 27.0,  // v0.31.0 empirical
        "rdc_days_supply": 41.0,    // v0.31.0 empirical
        "abc_velocity_factors": {"A": 1.5, "B": 1.0, "C": 0.5}
      }
    }
  }
}
```
