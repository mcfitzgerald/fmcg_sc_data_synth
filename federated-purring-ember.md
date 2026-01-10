# Plan: Calibration Enhancement for Seasonal Capacity (v0.30.0)

## Executive Summary

v0.29.0 implemented flexible production capacity that tracks seasonal demand. While SLOB improved dramatically (84% → 26%), service degraded (78% → 74%). Root cause: calibration script doesn't account for seasonal capacity, leaving insufficient buffer during trough periods.

This plan enhances the calibration script to validate and optimize seasonal capacity parameters.

---

## Background Context

### What Was Implemented in v0.29.0

**Files Modified:**
- `src/prism_sim/simulation/transform.py` (lines 81-89, 168-194, 264-276, 300-311)
- `src/prism_sim/simulation/mrp.py` (lines 134-135, 569-593, 1010)
- `src/prism_sim/config/simulation_config.json` (line 91)
- `tests/test_seasonal_capacity.py` (new file, 9 tests)

**Changes:**
1. `TransformEngine._get_seasonal_capacity_factor(day)` - calculates capacity multiplier using sine wave
2. Daily capacity reset applies seasonal factor: `max_capacity × seasonal_factor`
3. OEE calculation uses effective daily capacity (prevents >100% OEE)
4. `MRPEngine._get_daily_capacity(day)` - day-aware capacity for planning
5. Config parameter: `capacity_amplitude: 0.12` (matches demand amplitude)

### Current Metrics (365-day simulation)

| Metric | v0.27.0 (before) | v0.29.0 (after) | Target |
|--------|------------------|-----------------|--------|
| SLOB | 83.8% | **26.2%** | <30% |
| Turns | 7.87x | **10.43x** | 12-16x |
| Service | 78.2% | 73.7% | >85% |
| OEE | 57.8% | 41.4% | 65-85% |

### The Problem

Service degraded because:

1. **Calibration assumes fixed capacity**: The `production_rate_multiplier = 45.7` was derived for annual average demand/capacity, not seasonal extremes.

2. **Symmetric flex removes safety buffer**: When both demand AND capacity drop by 12% during trough, there's no margin for demand variability.

3. **Real FMCG doesn't flex symmetrically**:
   - Peak: Easy to add overtime (+15% realistic)
   - Trough: Labor contracts limit reduction (-5% realistic)

---

## Seasonal Demand/Capacity Analysis

### Pattern (from `demand.seasonality` config)
```
amplitude = 0.12 (±12%)
phase_shift_days = 150
cycle_days = 365

Peak day: ~241 (factor = 1.12, demand +12%)
Trough day: ~58 (factor = 0.88, demand -12%)
```

### Simulation Timeline
```
Day 1-90:   Trough zone (demand 88-94% of base)
Day 90-180: Rising to peak
Day 180-270: Peak zone (demand 106-112% of base)
Day 270-365: Falling to trough
```

### Current Gap
```
Base demand:    ~8M cases/day
Base capacity:  ~9M cases/day (multiplier × theoretical)

At TROUGH (day 58):
  Demand:   8M × 0.88 = 7.04M
  Capacity: 9M × 0.88 = 7.92M
  Gap: +0.88M (seems OK, but no buffer for variability!)

At PEAK (day 241):
  Demand:   8M × 1.12 = 8.96M
  Capacity: 9M × 1.12 = 10.08M
  Gap: +1.12M (OK)
```

The problem: During trough, reduced capacity leaves no margin for demand spikes or supply disruptions.

---

## Implementation Plan

### Phase 1: Add Seasonal Validation to Calibration Script

**File:** `scripts/calibrate_config.py`

#### Change 1.1: Add seasonal balance validation function (after line 516)

```python
def validate_seasonal_balance(
    sim_config: dict[str, Any],
    derived: dict[str, Any],
) -> list[str]:
    """
    Validate capacity meets demand across all seasons.

    v0.30.0: Ensures seasonal capacity flex doesn't create structural
    shortfalls during peak or insufficient buffer during trough.
    """
    warnings = []

    sim_params = sim_config.get("simulation_parameters", {})
    demand_config = sim_params.get("demand", {}).get("seasonality", {})

    demand_amp = demand_config.get("amplitude", 0.12)
    capacity_amp = demand_config.get("capacity_amplitude", 0.0)

    if capacity_amp == 0:
        return warnings  # No seasonal flex, skip validation

    base_demand = derived["analysis"]["total_daily_demand"]
    base_capacity = derived["analysis"]["current_capacity"]

    # Peak validation: capacity must exceed demand with margin
    peak_demand = base_demand * (1 + demand_amp)
    peak_capacity = base_capacity * (1 + capacity_amp)
    peak_margin = (peak_capacity - peak_demand) / peak_demand

    if peak_margin < 0.05:  # Less than 5% margin
        warnings.append(
            f"Peak season: capacity margin only {peak_margin:.1%} "
            f"(demand {peak_demand:,.0f}, capacity {peak_capacity:,.0f}) "
            f"-> stockouts likely during peak"
        )

    # Trough validation: need buffer for variability
    trough_demand = base_demand * (1 - demand_amp)
    trough_capacity = base_capacity * (1 - capacity_amp)
    trough_margin = (trough_capacity - trough_demand) / trough_demand

    if trough_margin < 0.10:  # Less than 10% margin
        warnings.append(
            f"Trough season: capacity margin only {trough_margin:.1%} "
            f"(demand {trough_demand:,.0f}, capacity {trough_capacity:,.0f}) "
            f"-> insufficient buffer for demand variability"
        )

    # OEE range validation
    if base_capacity > 0:
        peak_oee = peak_demand / peak_capacity
        trough_oee = trough_demand / trough_capacity

        if peak_oee > 0.95:
            warnings.append(
                f"Peak OEE would be {peak_oee:.0%} (>95%) -> capacity-constrained"
            )
        if trough_oee < 0.40:
            warnings.append(
                f"Trough OEE would be {trough_oee:.0%} (<40%) -> very low utilization"
            )

    # Amplitude relationship check
    if capacity_amp > demand_amp:
        warnings.append(
            f"capacity_amplitude ({capacity_amp}) > demand amplitude ({demand_amp}) "
            f"-> unused capacity during peak, risk during trough"
        )

    return warnings
```

#### Change 1.2: Add seasonal capacity derivation function (after validate_seasonal_balance)

```python
def derive_seasonal_capacity_params(
    sim_config: dict[str, Any],
    target_trough_buffer: float = 0.10,
) -> dict[str, Any]:
    """
    Derive optimal capacity_amplitude based on physics.

    Key insight: capacity_amplitude should be LESS than demand_amplitude
    to maintain safety buffer during troughs.

    Real FMCG practice:
    - Peak: Easy to add overtime, temp workers
    - Trough: Labor contracts limit reduction

    Returns recommendations for seasonal capacity parameters.
    """
    sim_params = sim_config.get("simulation_parameters", {})
    demand_config = sim_params.get("demand", {}).get("seasonality", {})

    demand_amp = demand_config.get("amplitude", 0.12)
    current_capacity_amp = demand_config.get("capacity_amplitude", 0.0)

    # Derive optimal capacity amplitude
    # If demand drops 12%, capacity should drop less (e.g., 8%)
    # This maintains ~10% buffer during trough
    optimal_capacity_amp = demand_amp * (1 - target_trough_buffer)

    # Alternative: Asymmetric flex (more realistic)
    # Peak: can add 15% capacity (overtime, temp workers)
    # Trough: can only reduce 5% (labor constraints)
    asymmetric_peak = demand_amp * 1.2  # e.g., 0.144 for 0.12 demand amp
    asymmetric_trough = demand_amp * 0.4  # e.g., 0.048 for 0.12 demand amp

    return {
        "current_capacity_amplitude": current_capacity_amp,
        "recommended_symmetric": round(optimal_capacity_amp, 3),
        "recommended_asymmetric_peak": round(asymmetric_peak, 3),
        "recommended_asymmetric_trough": round(asymmetric_trough, 3),
        "demand_amplitude": demand_amp,
        "target_trough_buffer": target_trough_buffer,
    }
```

#### Change 1.3: Integrate into validate_config_consistency (line ~430)

Add call to `validate_seasonal_balance` and include results in violations list.

#### Change 1.4: Update derive_optimal_parameters to include seasonal analysis

Add seasonal capacity recommendations to the return dict (after line 420):

```python
"seasonal_capacity": derive_seasonal_capacity_params(sim_config),
```

#### Change 1.5: Update print_report to show seasonal analysis (after line 621)

```python
print("\n--- SEASONAL CAPACITY ANALYSIS ---")
seasonal = recommendations.get("seasonal_capacity", {})
if seasonal:
    print(f"Demand Amplitude: ±{seasonal['demand_amplitude']:.0%}")
    print(f"Current Capacity Amplitude: ±{seasonal['current_capacity_amplitude']:.0%}")
    print(f"Recommended (symmetric): ±{seasonal['recommended_symmetric']:.1%}")
    print(f"Recommended (asymmetric):")
    print(f"  Peak: +{seasonal['recommended_asymmetric_peak']:.1%}")
    print(f"  Trough: -{seasonal['recommended_asymmetric_trough']:.1%}")
```

#### Change 1.6: Update apply_recommendations to set capacity_amplitude

Add to `apply_recommendations` function (after line 667):

```python
# Update seasonal capacity amplitude (v0.30.0)
demand = sim_params.setdefault("demand", {})
seasonality = demand.setdefault("seasonality", {})
if "capacity_amplitude" not in seasonality or args.apply:
    seasonality["capacity_amplitude"] = rec.get("capacity_amplitude",
        seasonal.get("recommended_symmetric", 0.0))
```

---

### Phase 2: Reduce Capacity Amplitude for Better Service

**File:** `src/prism_sim/config/simulation_config.json`

Change `capacity_amplitude` from 0.12 to 0.10 (derived value):

```json
"seasonality": {
  "amplitude": 0.12,
  "capacity_amplitude": 0.10,  // Changed from 0.12 -> maintains ~10% buffer at trough
  "phase_shift_days": 150,
  "cycle_days": 365
}
```

**Rationale:**
- Trough demand: 88% of base
- Trough capacity: 90% of base (with 0.10 amplitude)
- Buffer: 2% margin + existing capacity slack
- This should improve service while maintaining SLOB benefits

---

### Phase 3: Add Asymmetric Capacity Flex (Optional Enhancement)

**Files to modify (if implementing):**
- `src/prism_sim/config/simulation_config.json`
- `src/prism_sim/simulation/transform.py`
- `src/prism_sim/simulation/mrp.py`

**Config structure:**
```json
"seasonality": {
  "amplitude": 0.12,
  "capacity_amplitude_peak": 0.15,
  "capacity_amplitude_trough": 0.05,
  "phase_shift_days": 150,
  "cycle_days": 365
}
```

**TransformEngine change:**
```python
def _get_seasonal_capacity_factor(self, day: int) -> float:
    if self._seasonal_capacity_amplitude == 0:
        return 1.0

    # Calculate base sine factor
    sine_value = np.sin(
        2 * np.pi * (day - self._seasonal_phase_shift) / self._seasonal_cycle_days
    )

    # Apply asymmetric amplitude
    if sine_value > 0:  # Peak season
        return float(1.0 + self._capacity_amplitude_peak * sine_value)
    else:  # Trough season
        return float(1.0 + self._capacity_amplitude_trough * sine_value)
```

**Note:** This is an optional enhancement. Start with symmetric 0.10 and measure impact first.

---

### Phase 4: Update Documentation

**Files:**
- `CHANGELOG.md` - Add v0.30.0 entry
- `docs/planning/intent.md` - Document seasonal capacity calibration
- `dreamy-doodling-cookie.md` - Archive/update original plan

---

## Verification Steps

### 1. Run calibration script
```bash
poetry run python scripts/calibrate_config.py
```
**Expected:** Should show seasonal capacity analysis section and any warnings about trough buffer.

### 2. Apply calibration recommendations
```bash
poetry run python scripts/calibrate_config.py --apply
```

### 3. Run 90-day validation
```bash
poetry run python run_simulation.py --days 90 --no-logging
```
**Expected:**
- Service: >80% (improved from 82.99%)
- SLOB: <15% (maintained)
- OEE: 55-70% (in target range)

### 4. Run 365-day validation
```bash
poetry run python run_simulation.py --days 365 --no-logging
```
**Expected:**
- Service: >78% (improved from 73.68%)
- SLOB: <30% (maintained)
- Turns: >10x (maintained)

### 5. Run all tests
```bash
poetry run pytest tests/ -v
```
**Expected:** All 19 tests pass (including 9 seasonal capacity tests).

### 6. Type check
```bash
poetry run mypy scripts/calibrate_config.py
```

---

## Files to Modify

| File | Changes | Complexity |
|------|---------|------------|
| `scripts/calibrate_config.py` | Add seasonal validation/derivation | Medium |
| `simulation_config.json` | Reduce capacity_amplitude to 0.10 | Trivial |
| `CHANGELOG.md` | Add v0.30.0 entry | Trivial |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Service still below target | Medium | Medium | Can further reduce capacity_amplitude to 0.08 |
| SLOB regresses | Low | Medium | Monitor; 0.10 should still prevent trough overproduction |
| Calibration breaks existing flow | Low | Low | Only adds validation, doesn't change existing logic |

---

## Key Code References

### Current seasonal factor calculation
- `transform.py:185-194` - `_get_seasonal_capacity_factor()`
- `mrp.py:586-593` - `_get_daily_capacity()`

### Calibration key functions
- `calibrate_config.py:191-422` - `derive_optimal_parameters()`
- `calibrate_config.py:425-516` - `validate_config_consistency()`
- `calibrate_config.py:635-670` - `apply_recommendations()`

### Config location
- `simulation_config.json:89-94` - `seasonality` block

---

## Summary

1. **Root cause identified**: Calibration doesn't account for seasonal capacity flex, leaving no buffer during troughs.

2. **Solution**: Enhance calibration script with seasonal validation and derive optimal `capacity_amplitude` based on physics.

3. **Quick fix**: Reduce `capacity_amplitude` from 0.12 to 0.10 for immediate service improvement.

4. **Future option**: Implement asymmetric flex (peak +15%, trough -5%) to match real FMCG practice.
