# Fix Plan: Plant Shipment Bug & SLOB Calculation

## Diagnosis Summary

### 365-Day Simulation Results
| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| Store Service Level | 73.46% | >95% | FAILING |
| SLOB Inventory | 70.1% | <30% | FAILING |
| Inventory Turns | 4.54x | 6-14x | LOW |
| Customer DC Inventory | 62.6M (80% of FG) | - | EXCESSIVE |

---

## Root Cause #1: Plant Shipment Bug (CRITICAL)

**File:** `src/prism_sim/simulation/orchestrator.py:700-702`

**Current Code:**
```python
rdc_ids = [
    n_id for n_id, n in self.world.nodes.items() if n.type == NodeType.DC
]
```

**Problem:** Selects ALL 44 DC nodes instead of just 4 Manufacturer RDCs:
- 4 RDC-* (Manufacturer RDCs) - should receive plant production
- 20 RET-DC-* (Retailer DCs) - should NOT receive direct shipments
- 8 DIST-DC-* (Distributor DCs) - should NOT receive direct shipments
- 10 ECOM-FC-* (Ecom FCs) - should NOT receive direct shipments
- 2 DTC-FC-* (DTC FCs) - should NOT receive direct shipments

**Impact:**
- 150M units shipped Plant→CustomerDC with ZERO orders (unauthorized PUSH)
- Only 6.6M units flowed via normal PULL orders
- 72M units accumulated at Customer DCs
- Service level degraded as normal replenishment was overwhelmed

**Fix:** Filter for RDC-* prefix only.

---

## Root Cause #2: SLOB Calculation Bug

**File:** `src/prism_sim/simulation/orchestrator.py:505-509`

**Current Code:**
```python
global_dos = total_fg_inv / max(total_demand_qty, 1.0)
is_slob = 1.0 if global_dos > self.slob_days_threshold else 0.0
```

**Problem:** Binary global check - if system DOS > 60 days, SLOB = 100%.

**Correct Approach:** Per-SKU calculation:
1. Calculate DOS per SKU: `sku_dos = sku_inventory / sku_demand`
2. Flag SKUs where `sku_dos > threshold` as SLOB
3. Calculate: `SLOB % = sum(SLOB inventory) / total inventory`

---

## Implementation Plan

### Step 1: Fix Plant Shipment Routing
**File:** `src/prism_sim/simulation/orchestrator.py`

Change line 700-702 from:
```python
rdc_ids = [
    n_id for n_id, n in self.world.nodes.items() if n.type == NodeType.DC
]
```

To:
```python
rdc_ids = [
    n_id for n_id, n in self.world.nodes.items()
    if n.type == NodeType.DC and n_id.startswith('RDC-')
]
```

### Step 2: Fix SLOB Calculation
**File:** `src/prism_sim/simulation/orchestrator.py`

Replace lines 505-509 with per-SKU SLOB logic:
```python
# SLOB % (Per-SKU calculation)
# Only finished goods can be "slow/obsolete"
if total_demand_qty > 0:
    # Calculate per-SKU days of supply
    fg_inv_per_sku = np.sum(
        self.state.actual_inventory[:, self._fg_mask], axis=0
    )
    demand_per_sku = np.sum(daily_demand[:, self._fg_mask], axis=0)

    # Avoid division by zero
    demand_per_sku_safe = np.maximum(demand_per_sku, 0.01)
    sku_dos = fg_inv_per_sku / demand_per_sku_safe

    # Flag SKUs with DOS > threshold as SLOB
    slob_mask = sku_dos > self.slob_days_threshold
    slob_inventory = fg_inv_per_sku[slob_mask].sum()

    slob_pct = slob_inventory / total_fg_inv if total_fg_inv > 0 else 0.0
    self.monitor.record_slob(slob_pct)
```

### Step 3: Run Validation
1. Run 30-day simulation to verify fix
2. Check Plant→DC shipment flow (should only go to RDC-*)
3. Verify SLOB calculation returns per-SKU percentage
4. Run full 365-day simulation to validate metrics

---

## Expected Results After Fix

| Metric | Before | Expected After |
|--------|--------|----------------|
| Store Service Level | 73.46% | >90% |
| SLOB Inventory | 70.1% | <30% |
| Inventory Turns | 4.54x | 6-10x |
| Customer DC Inventory | 80% of FG | ~30% of FG |

---

## Files to Modify
1. `src/prism_sim/simulation/orchestrator.py` - Both fixes
