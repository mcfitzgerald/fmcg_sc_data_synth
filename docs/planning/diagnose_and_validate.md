# Diagnose and Validate: Comprehensive Plan

This document outlines the systematic approach to diagnose, fix, and validate the Prism Digital Twin simulation against physics laws and realism benchmarks.

---

## Phase 1: Baseline Capture

### 1.1 Run Full 365-Day Simulation
```bash
# Clear previous data
rm -rf data/output/*.csv data/output/*.parquet

# Run full simulation with logging
poetry run python run_simulation.py --days 365

# Verify output files exist
ls -lh data/output/
```

### 1.2 Capture Key Metrics
After simulation completes, extract and document:

| Metric | Current | Target | Source |
|--------|---------|--------|--------|
| Service Level (Fill Rate) | ? | >85% | `triangle_report.txt` |
| Inventory Turns | ? | 6-14x | `triangle_report.txt` |
| OEE | ? | 65-85% | `triangle_report.txt` |
| SLOB Inventory | ? | <30% | `triangle_report.txt` |
| Truck Fill Rate | ? | >85% | `triangle_report.txt` |
| Perfect Order Rate | ? | >90% | `triangle_report.txt` |
| MAPE (Forecast) | ? | 20-50% | `triangle_report.txt` |
| Cost per Case | ? | $1-3 | `triangle_report.txt` |

---

## Phase 2: Data Analysis

### 2.1 Inventory Distribution Analysis
Analyze where inventory is positioned across the network:

```python
# Check inventory by echelon
inventory.csv -> Group by node_type -> Sum actual_inventory
Expected: Balanced across PLANT, RDC, DC, STORE
Red Flag: >50% at any single echelon = flow blockage
```

**Key Questions:**
- [ ] Is inventory stuck at plants? (production not flowing)
- [ ] Is inventory stuck at RDCs? (push allocation not working)
- [ ] Are stores starved? (replenishment failure)

### 2.2 Flow Analysis
Trace the flow of goods through the network:

```python
# Analyze shipments by route
shipments.csv -> Group by (source_type, target_type) -> Sum quantity
Expected: PLANT→RDC→DC→STORE flow
Red Flag: Missing or low-volume routes
```

**Key Questions:**
- [ ] Are all echelon-to-echelon routes active?
- [ ] Does flow volume match demand (~460k cases/day)?
- [ ] Are lead times realistic (1-5 days per hop)?

### 2.3 Production Analysis
Verify production matches demand:

```python
# Daily production vs demand
batches.csv -> Group by day_produced -> Sum quantity
Compare to: Daily demand from simulation log (~460k)
Expected: Production ≈ Demand (±10%)
Red Flag: Production << Demand = MRP failure
```

**Key Questions:**
- [ ] Is production stable or collapsing over time?
- [ ] Does ABC classification affect production correctly?
- [ ] Are all 24 finished goods being produced?

### 2.4 SLOB Analysis
Diagnose why SLOB is high (target <30%):

```python
# Identify slow-moving inventory
inventory.csv -> Calculate days_of_supply per (node, product)
SLOB = inventory where DOS > 60 days
Expected: <30% of total inventory
Red Flag: Specific products or locations with extreme DOS
```

**Root Causes to Check:**
- [ ] C-items over-produced relative to demand
- [ ] Inventory trapped at wrong locations
- [ ] Products with zero demand still being stocked
- [ ] Initial priming too aggressive

### 2.5 Truck Fill Analysis
Diagnose why truck fill is low (target >85%):

```python
# Analyze shipment efficiency
shipments.csv -> Calculate fill_rate = actual_weight / max_weight
Expected: >85% average fill rate
Red Flag: Many small shipments (LTL inefficiency)
```

**Root Causes to Check:**
- [ ] Orders too small (not consolidated)
- [ ] Bin-packing algorithm not optimizing
- [ ] FTL vs LTL mode selection issues
- [ ] Weight vs cube constraint handling

---

## Phase 3: Physics Validation

### 3.1 Mass Balance Audit
Verify conservation of mass across the system:

```
Input (Production + Arrivals) = Output (Shipments + Sales) + Delta(Inventory)
```

**Checks:**
- [ ] No inventory created from nothing
- [ ] No inventory destroyed without shrinkage event
- [ ] Scrap/yield loss properly accounted

### 3.2 Kinematic Consistency
Verify lead times match physics:

```python
# Check lead times are realistic
links.csv -> Verify lead_time_days = (distance_km / speed_kmh / 24) + handling
Expected: 1-5 days for domestic routes
Red Flag: >10 days for any domestic route
```

**Checks:**
- [ ] All nodes have valid coordinates (not 0,0)
- [ ] Lead times calculated correctly (hours→days)
- [ ] No "teleporting" shipments (arrival before ship)

### 3.3 Little's Law Validation
Verify inventory physics:

```
Inventory = Throughput × Flow Time
```

**Checks:**
- [ ] Average inventory / daily demand ≈ average lead time
- [ ] No impossible inventory levels (negative)
- [ ] Turns align with lead time structure

### 3.4 Capacity Constraints
Verify production respects limits:

```python
# Check plant capacity utilization
batches.csv -> Daily production per plant vs capacity
Expected: <100% utilization (with some headroom)
Red Flag: >100% = physics violation
```

**Checks:**
- [ ] No plant exceeds daily capacity
- [ ] OEE reflects actual vs theoretical capacity
- [ ] Changeover time respected

---

## Phase 4: Realism Validation

### 4.1 Distribution Checks
Validate statistical properties match real-world patterns:

| Distribution | Expected | Check |
|--------------|----------|-------|
| SKU Demand | Zipfian (α≈0.5) | Top 20% SKUs = 80% volume |
| Store Size | Power Law | Few large, many small |
| Order Size | Log-normal | Most orders medium-sized |
| Lead Time | Gamma | Right-skewed with variability |

### 4.2 Hub Concentration
Verify Chicago hub handles ~40% of volume (per intent.md):

```python
# Check hub concentration
shipments.csv -> Filter routes through Chicago -> Sum quantity / Total
Expected: ~40% of traditional retail volume
```

### 4.3 Bullwhip Effect
Verify order amplification exists:

```python
# Compare order variance to demand variance
Variance(Orders) > Variance(POS Demand)
Expected: 2-5x amplification at RDC level
```

### 4.4 Seasonal Patterns
Verify seasonality is present:

```python
# Check demand by month
Group daily demand by month -> Plot
Expected: ~12% amplitude with peak around day 150
```

---

## Phase 5: Issue Diagnosis & Fix

### 5.1 SLOB Fix (Target: <30%)

**Diagnosis Steps:**
1. Identify products with DOS > 60 days
2. Identify locations with excess inventory
3. Check if demand exists for these products
4. Check if products are being over-produced

**Potential Fixes:**
- [ ] Reduce C-item production multiplier
- [ ] Implement inventory rebalancing (lateral transfers)
- [ ] Reduce initial priming levels
- [ ] Add SLOB-aware production throttling

### 5.2 Truck Fill Fix (Target: >85%)

**Diagnosis Steps:**
1. Analyze average shipment size vs truck capacity
2. Check FTL vs LTL mode selection
3. Verify bin-packing algorithm effectiveness
4. Check order consolidation logic

**Potential Fixes:**
- [ ] Increase minimum order quantities
- [ ] Implement shipment consolidation windows
- [ ] Fix bin-packing weight/cube optimization
- [ ] Adjust FTL thresholds in config

### 5.3 Service Level Tuning (Target: >85%)

**Diagnosis Steps:**
1. Identify stockout patterns by location/product
2. Check safety stock levels
3. Verify replenishment triggers (ROP)
4. Analyze demand vs supply timing

**Potential Fixes:**
- [ ] Increase safety stock multipliers
- [ ] Reduce replenishment batch sizes for faster response
- [ ] Adjust z-score for service level targets
- [ ] Enable push allocation for excess inventory

---

## Phase 6: Validation & Sign-Off

### 6.1 Re-run Simulation
After fixes, run another 365-day simulation and verify all metrics meet targets.

### 6.2 Metrics Checklist

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Service Level | >85% | | |
| Inventory Turns | 6-14x | | |
| OEE | 65-85% | | |
| SLOB | <30% | | |
| Truck Fill | >85% | | |
| Mass Balance | 0 violations | | |
| Lead Times | 1-5 days | | |

### 6.3 Physics Sign-Off
- [ ] Mass Balance: PASS
- [ ] Kinematic Consistency: PASS
- [ ] Little's Law: PASS
- [ ] Capacity Constraints: PASS
- [ ] Inventory Positivity: PASS

### 6.4 Realism Sign-Off
- [ ] Zipfian Demand: PASS
- [ ] Hub Concentration: PASS
- [ ] Bullwhip Effect: PASS
- [ ] Seasonal Patterns: PASS

---

## Appendix: Analysis Scripts

### A.1 Quick Diagnostic Script
```bash
poetry run python -c "
import pandas as pd
import numpy as np

# Load data
inv = pd.read_csv('data/output/inventory.csv')
ships = pd.read_csv('data/output/shipments.csv')
batches = pd.read_csv('data/output/batches.csv')
locs = pd.read_csv('data/output/static_world/locations.csv')

# Print summary
print('=== QUICK DIAGNOSTICS ===')
print(f'Inventory records: {len(inv):,}')
print(f'Shipment records: {len(ships):,}')
print(f'Batch records: {len(batches):,}')
print(f'Locations: {len(locs):,}')

# Last day inventory by type
loc_types = dict(zip(locs['id'], locs['type'].str.replace('NodeType.', '')))
last = inv[inv['day'] == inv['day'].max()].copy()
last['type'] = last['node_id'].map(loc_types)
print('\nInventory by echelon:')
print(last.groupby('type')['actual_inventory'].sum())
"
```

### A.2 SLOB Analysis Script
```bash
poetry run python -c "
import pandas as pd

inv = pd.read_csv('data/output/inventory.csv')
prods = pd.read_csv('data/output/static_world/products.csv')

# Calculate average daily demand per product (proxy)
# SLOB = inventory with DOS > 60

last = inv[inv['day'] == inv['day'].max()]
total_inv = last['actual_inventory'].sum()

# Estimate daily demand from products (simplified)
# Real analysis would use actual POS data
print(f'Total inventory: {total_inv:,.0f}')
print(f'SLOB analysis requires demand data - check triangle_report.txt')
"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-04 | Initial comprehensive plan |
