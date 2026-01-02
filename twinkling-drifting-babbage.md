# Plan: Fix Service Level - Physics-Based Approach (v0.16.0)

## Problem Statement

Service Level stuck at **81%** (target: 95%+) despite demand signal fixes in v0.15.9.

## Research Findings: Multiple Root Causes

| # | Issue | Physics Violation | Impact |
|---|-------|-------------------|--------|
| 1 | **No Inventory Position** | (s,S) theory requires IP = On-Hand + In-Transit | Double-ordering oscillation |
| 2 | **No Zipfian SKU demand** | `sku_popularity_alpha: 1.05` declared but unused | MRP under-produces low-demand SKUs |
| 3 | **No formal safety stock** | ROP = demand × days (not z×σ×√LT) | No variance buffer |
| 4 | **No multi-echelon SL targets** | All nodes target same % | 0.95³ = 85% end-to-end |
| 5 | **Daily loop timing** | Demand consumed before arrivals | Transient shortages |

### Evidence from 365-Day Simulation

- **SKU-ORAL-007:** 25% of stores stocked out (produced at 1.1x orders)
- **SKU-ORAL-002:** 18% of stores stocked out (produced at 1.1x orders)
- **Store inventory:** Collapsed from 43.5% → 4.7% of finished goods
- **Store fill rate:** 99.7% (they get what they order)
- **But stores only order 45%** of true demand

---

## Recommended Approach: Layered Fixes

### Phase 1: Inventory Position Fix (Fundamental)

**Root Cause:** Stores use On-Hand only, causing double-ordering/gaps.

**Fix:** Use `Inventory Position = On-Hand + In-Transit` for (s,S) decisions.

**Files:**
- `src/prism_sim/simulation/state.py` - Add `get_in_transit_by_target()`
- `src/prism_sim/agents/replenishment.py` - Use IP in `generate_orders()`

```python
# state.py - Add method
def get_in_transit_by_target(self) -> np.ndarray:
    """Calculate in-transit inventory per target node and product."""
    in_transit = np.zeros((self.n_nodes, self.n_products), dtype=np.float64)
    for shipment in self.active_shipments:
        target_idx = self.node_id_to_idx.get(shipment.target_id)
        if target_idx is None:
            continue
        for line in shipment.lines:
            p_idx = self.product_id_to_idx.get(line.product_id)
            if p_idx is not None:
                in_transit[target_idx, p_idx] += line.quantity
    return in_transit

# replenishment.py - Modify generate_orders()
on_hand_inv = self.state.inventory[target_idx_arr, :]
in_transit_matrix = self.state.get_in_transit_by_target()
in_transit_inv = in_transit_matrix[target_idx_arr, :]
inventory_position = on_hand_inv + in_transit_inv  # Use this for (s,S)

needs_order = inventory_position < reorder_point  # Not on_hand
raw_qty = target_stock - inventory_position       # Not on_hand
```

### Phase 2: Multi-Echelon Service Level Targets (Config Change)

**Root Cause:** All nodes target 95% independently → 0.95³ ≈ 85% end-to-end.

**Fix:** Tier service levels so upstream nodes have higher targets.

**File:** `src/prism_sim/config/simulation_config.json`

```json
"replenishment": {
  "channel_profiles": {
    "PLANT": { "target_days": 28.0, "reorder_point_days": 21.0 },
    "RDC": { "target_days": 28.0, "reorder_point_days": 21.0 },
    "B2M_LARGE": { "target_days": 21.0, "reorder_point_days": 14.0 },
    "default": { "target_days": 14.0, "reorder_point_days": 10.0 }
  }
}
```

### Phase 3 (Optional): Implement Zipfian SKU Demand

**Root Cause:** `sku_popularity_alpha: 1.05` declared but unused → uniform demand.

**Fix:** Apply Zipf distribution to per-SKU demand in POSEngine.

**File:** `src/prism_sim/simulation/demand.py`

```python
# In _init_base_demand(), apply Zipf to SKU multipliers
from scipy.stats import zipf
sku_ranks = np.arange(1, n_skus + 1)
popularity_weights = 1.0 / (sku_ranks ** alpha)  # alpha from config
popularity_weights /= popularity_weights.sum()  # Normalize
self.sku_multipliers = popularity_weights * n_skus  # Scale to preserve total
```

---

## Implementation Order

Implement all three phases before validation:

1. **Phase 1:** Inventory Position fix (state.py, replenishment.py)
2. **Phase 2:** Multi-Echelon SL targets (simulation_config.json)
3. **Phase 3:** Zipfian SKU demand (demand.py)
4. **Run 365-day simulation** - Validate all fixes together
5. **Update CHANGELOG.md** - Document v0.16.0

---

## Files to Modify

| Phase | File | Change |
|-------|------|--------|
| 1 | `src/prism_sim/simulation/state.py` | Add `get_in_transit_by_target()` |
| 1 | `src/prism_sim/agents/replenishment.py` | Use Inventory Position |
| 2 | `src/prism_sim/config/simulation_config.json` | Tier SL targets |
| 3 | `src/prism_sim/simulation/demand.py` | Apply Zipf distribution |
| All | `CHANGELOG.md` | Document changes |

---

## Validation

1. **365-Day Simulation:** Service level ≥ 95%
2. **SKU Balance:** No single SKU with >10% stockout rate
3. **Order Stability:** Reduced variance in daily order volumes
4. **Physics Audit:** Mass balance preserved

---

## Theory References

- **Inventory Position:** Zipkin, "Foundations of Inventory Management"
- **Multi-Echelon SL:** Graves & Willems (2000), "Optimizing Strategic Safety Stock"
- **Zipfian Demand:** Standard FMCG practice (top 20% SKUs = 80% volume)
