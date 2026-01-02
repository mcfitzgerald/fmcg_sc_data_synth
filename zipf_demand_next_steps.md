# Zipfian SKU Demand: Next Steps for Tuning

## Current State (v0.16.0)

### What We Implemented

1. **Phase 1: Inventory Position Fix** (`state.py`, `replenishment.py`)
   - (s,S) decisions now use IP = On-Hand + In-Transit
   - Prevents double-ordering when shipments are in transit

2. **Phase 2: Multi-Echelon SL Targets** (`simulation_config.json`)
   - Manufacturing targets: 28/21 days (was 14/7)
   - Higher upstream buffers for end-to-end service level

3. **Phase 3: Channel-Aware Zipfian Demand** (`demand.py`)
   - SKUs ranked by channel segment affinity (Club→VALUE, DTC→PREMIUM)
   - Zipf formula: `weight = 1 / rank^alpha`
   - Config: `sku_popularity_alpha` (currently 0.5, was 1.05)

4. **Demand-Proportional Inventory Priming** (`orchestrator.py`)
   - Stores, Customer DCs, and RDCs now get per-SKU inventory proportional to expected demand
   - Popular SKUs get more initial inventory

### Current Results

| Config | Service Level | Notes |
|--------|---------------|-------|
| Baseline (no Zipf) | 80.6% | Phase 1+2 only |
| Zipf alpha=1.05 | 67-71% | Concentrated demand breaks system |
| Zipf alpha=0.5 | ~70% | Still below baseline |

### The Problem

Zipfian demand is **physically correct** (top 20% SKUs = 80% volume), but the supply chain's (s,S) parameters were tuned for uniform demand. When demand concentrates:
- Popular SKUs need faster replenishment
- MRP sees aggregate demand OK but per-SKU composition is wrong
- Order volumes explode (205M orders vs normal ~100M)

---

## Understanding (s,S) Replenishment Parameters

### What is (s,S)?

The (s,S) policy is a **reorder point / order-up-to** inventory policy:
- **s** = Reorder Point (ROP) - trigger level to place an order
- **S** = Order-Up-To Level (Target) - quantity to order up to

When `Inventory Position < s`, order enough to reach `S`.

### Current Parameters (simulation_config.json)

```json
"replenishment": {
  "target_days_supply": 14.0,      // S = avg_demand × 14 days
  "reorder_point_days": 10.0,      // s = avg_demand × 10 days
  "channel_profiles": {
    "B2M_LARGE": { "target_days": 21.0, "reorder_point_days": 14.0 },
    "B2M_CLUB": { "target_days": 21.0, "reorder_point_days": 14.0 },
    ...
  }
}
```

### How It Works in Code (`replenishment.py:generate_orders()`)

```python
# Calculate thresholds based on smoothed demand
avg_demand = self.smoothed_demand[target_idx_arr, :]  # 7-day rolling avg
target_stock = avg_demand * target_days    # S level
reorder_point = avg_demand * rop_days      # s level

# Use Inventory Position (On-Hand + In-Transit) for decision
inventory_position = on_hand_inv + in_transit_inv

# Order if IP < s, order up to S
needs_order = inventory_position < reorder_point
order_qty = target_stock - inventory_position
```

### Why Uniform Parameters Fail with Zipf

With **uniform demand**, all SKUs have similar velocity, so:
- 14-day target covers ~2 weeks of demand uniformly
- 10-day ROP triggers reorder before stockout

With **Zipfian demand**:
- Popular SKUs: 5x average demand velocity → 14 days becomes ~3 days effective coverage
- Unpopular SKUs: 0.2x average → 14 days becomes ~70 days coverage
- Result: Popular SKUs stockout, unpopular SKUs accumulate

### Potential Fixes

1. **SKU-specific (s,S)**: Adjust target/ROP per SKU based on demand velocity
2. **Higher safety stock**: Increase target_days across the board (but increases inventory cost)
3. **Variance-aware ROP**: Use `ROP = z×σ×√LT` formula instead of fixed days

---

## Files Modified

| File | Changes |
|------|---------|
| `src/prism_sim/simulation/state.py` | Added `get_in_transit_by_target()` |
| `src/prism_sim/agents/replenishment.py` | Uses Inventory Position for (s,S) |
| `src/prism_sim/simulation/demand.py` | Channel-aware Zipf weights, `get_base_demand_matrix()` |
| `src/prism_sim/simulation/orchestrator.py` | Demand-proportional inventory priming |
| `src/prism_sim/config/simulation_config.json` | `sku_popularity_alpha`, higher mfg targets |

---

## Next Steps to Try

### Option A: Increase Safety Stock (Quick Test)
Increase target_days across channels to see if more buffer helps:
```json
"channel_profiles": {
  "B2M_LARGE": { "target_days": 28.0, "reorder_point_days": 21.0 },  // was 21/14
  ...
}
```

### Option B: Velocity-Based (s,S) Parameters
Modify `replenishment.py` to scale target/ROP by SKU velocity:
```python
# For high-velocity SKUs, use higher target
velocity_factor = sku_demand / avg_demand  # >1 for popular, <1 for niche
adjusted_target_days = base_target_days * max(1.0, velocity_factor ** 0.5)
```

### Option C: Reduce Zipf Concentration
Try alpha=0.3 or 0.2 for a gentler distribution while still having differentiation.

### Option D: Variance-Aware Safety Stock
Implement formal safety stock: `ROP = demand × LT + z × σ × √LT`
This requires tracking demand variance per SKU.

---

## Key Code Locations for Tuning

| Concept | File | Lines |
|---------|------|-------|
| Zipf alpha parameter | `simulation_config.json` | `demand.sku_popularity_alpha` |
| Channel (s,S) parameters | `simulation_config.json` | `agents.replenishment.channel_profiles` |
| Zipf weight calculation | `demand.py` | `_build_sku_popularity_weights()` |
| (s,S) order decision | `replenishment.py` | `generate_orders()` ~L450-460 |
| Inventory priming | `orchestrator.py` | `_initialize_inventory()` |
| MRP demand signal | `mrp.py` | `generate_production_orders()` |

---

## Validation Command

```bash
poetry run python run_simulation.py --days 365 --no-logging
```

Target: Service Level ≥ 80% with Zipfian demand enabled.
