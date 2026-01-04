# Distribution Bottleneck Diagnosis Plan

## Problem Statement

v0.19.11 achieved the SLOB target (<30%) with POS-driven production, but service level remains stuck at ~64-66% despite:
- A-items being produced at 130-150% of demand when low
- Total production capacity being adequate (OEE 85%+)
- 30-day SL starting at 92.5%

**Conclusion:** The bottleneck is NOT production. Goods are produced but not reaching stores.

## Evidence

| Metric | 30-day | 365-day | Interpretation |
|--------|--------|---------|----------------|
| Service Level | 92.5% | 64.5% | System degrades over time |
| SLOB | 0% | 28.6% | Inventory accumulates correctly |
| Inventory | 811M | 630M | Inventory is being consumed |
| Production | ~400K/day | ~275K/day | Production drops over time |

Key observation: Production drops from 400K to 275K over 365 days because demand signals collapse. This is the POS-driven system working correctly - but SL drops anyway.

## Hypothesis

Inventory is stuck at RDCs instead of flowing to stores. Possible causes:

1. **Replenishment Signal Issues**
   - Customer DCs not ordering enough from RDCs
   - Stores not ordering enough from Customer DCs
   - Order signals collapsing due to downstream starvation

2. **Push Allocation Not Working**
   - `push_allocation_enabled: true` but may not be pushing enough
   - `push_threshold_dos: 21.0` may be too high
   - Push logic may not be reaching the right products

3. **Initial Inventory Distribution**
   - RDCs primed with too much inventory relative to downstream
   - `rdc_store_multiplier: 500.0` may be too high
   - Downstream nodes starving while RDCs are overstocked

4. **FTL Consolidation Delays**
   - Orders held for Full Truckload consolidation
   - Small orders from stores not shipping due to pallet minimums
   - LTL mode may not be working correctly for store deliveries

## Diagnosis Steps

### Phase 1: Inventory Distribution Analysis

```bash
# Run simulation with logging enabled to capture inventory data
poetry run python run_simulation.py --days 365 --output-dir data/results/diagnosis
```

Then analyze:
1. **Inventory by echelon over time**: Are RDCs holding while stores deplete?
2. **A-item inventory location**: Where are the fast movers sitting?
3. **Order flow by link**: Are orders flowing downstream?

Key files to examine:
- `inventory_history.csv` - Track inventory by node/product/day
- `shipments.csv` - Track shipment flows
- `orders.csv` - Track order creation and fulfillment

### Phase 2: Replenishment Signal Investigation

Check `replenishment.py` for:
1. How Customer DCs calculate demand signal (inflow vs outflow)
2. Whether stores are ordering based on actual need
3. Whether echelon logic is working correctly

Key config parameters:
```json
"echelon_safety_multiplier": 1.3,
"push_allocation_enabled": true,
"push_threshold_dos": 21.0
```

Questions:
- Is the demand signal for Customer DCs reflecting actual downstream need?
- Are DCs ordering frequently enough (check `order_cycle_days`)?
- Is the echelon IP calculation correct?

### Phase 3: Push Allocation Deep Dive

In `orchestrator.py`, find `_push_excess_rdc_inventory()`:
1. Is it being called?
2. What threshold triggers push?
3. Is it pushing A-items preferentially?
4. Is it reaching the right destinations?

Potential fixes:
- Lower `push_threshold_dos` from 21 to 14
- Add ABC-prioritized push (push A-items more aggressively)
- Increase push frequency or quantity

### Phase 4: Logistics Flow Check

In `logistics.py`:
1. Are store deliveries using LTL correctly?
2. Are FTL minimums blocking small orders?
3. Check `store_delivery_mode: "LTL"` and `ltl_min_cases: 10`

## Potential Fixes (To Test)

### Quick Wins
1. **Lower push threshold**: Change `push_threshold_dos` from 21.0 to 14.0
2. **Increase store ordering frequency**: Check if stores are ordering daily
3. **Reduce initial RDC inventory**: Lower `rdc_store_multiplier` from 500 to 100

### Structural Fixes
1. **ABC-prioritized push allocation**: Push A-items from RDCs when DOS > 14, C-items when DOS > 30
2. **Direct POS-to-replenishment signal**: Have stores calculate need from POS, not from inventory
3. **Demand signal amplification**: When downstream is starving, amplify orders upstream

### Experimental
1. **Remove Customer DC layer**: Ship directly RDC → Store to eliminate one echelon
2. **Increase store inventory targets**: Higher `store_days_supply` in initialization
3. **Reduce RDC inventory targets**: Force goods downstream faster

## Success Criteria

Target: 365-day Service Level > 85% while maintaining SLOB < 30%

Intermediate milestones:
- [ ] Identify where A-item inventory is accumulating
- [ ] Prove goods are stuck at RDCs (or disprove)
- [ ] Find a single parameter change that improves SL by >5pp
- [ ] Reach 75% SL while keeping SLOB < 35%
- [ ] Reach 85% SL while keeping SLOB < 30%

## Related Files

- `src/prism_sim/simulation/orchestrator.py` - Push allocation, daily loop
- `src/prism_sim/agents/replenishment.py` - Store/DC ordering logic
- `src/prism_sim/simulation/logistics.py` - Shipment creation, FTL/LTL
- `src/prism_sim/config/simulation_config.json` - All tunable parameters
