# Status Report: Service Level Fix Attempt (v0.18.0)

## Session Summary

### What We Fixed

1. **Plant Shipment Bug** (`orchestrator.py:699-705`) - COMPLETE
   - Plants were shipping to ALL 44 DCs instead of just 4 manufacturer RDCs
   - Fixed by adding `n_id.startswith("RDC-")` filter
   - Verified: Plant shipments now only go to RDC-NE/MW/SO/WE

2. **SLOB Calculation Bug** (`orchestrator.py:505-522`) - COMPLETE
   - Was binary (0% or 100%) based on global DOS
   - Fixed to per-SKU calculation: `sum(SLOB inventory) / total FG inventory`
   - Now correctly reports ~75% SLOB (SKUs with DOS > 60 days)

3. **Customer DC Cold-Start Demand Signal** (`replenishment.py`) - PARTIAL
   - Added expected throughput floor for customer DC ordering
   - Customer DCs now use `max(inflow_demand, expected_throughput)`
   - Added flow-based minimum order when IP > ROP but IP < Target

4. **Config Changes** (`simulation_config.json`)
   - Reduced `customer_dc_days_supply`: 21 → 10 days
   - Reduced `store_days_supply`: 21 → 14 days
   - Reduced `rdc_days_supply`: 28 → 21 days

### Current Results

| Metric | Before Fixes | After Fixes | Target |
|--------|--------------|-------------|--------|
| Store Service Level (30-day) | 97.96% | 91.09% | >95% |
| Store Service Level (365-day) | 69.53% | ~70% | >90% |
| Inventory Turns | 4.35x | 12.03x | 6-14x |
| SLOB | 70.1% (broken) | 17-77% | <30% |

**Key Observation:** 30-day results are good, but 365-day degrades significantly.

---

## The Deeper Problem: Why Doesn't the System Stabilize?

**IMPORTANT:** The bullwhip effect is **intentional realism** - real supply chains with (s,S) policies experience this. The question is NOT "how do we eliminate bullwhip" but rather "why doesn't the system reach a noisy steady state like real supply chains do?"

### Evidence of Instability (NOT bullwhip itself)

1. **Order Volumes Are Massive**
   - Daily orders: 200M+ cases
   - Daily demand: ~400k cases
   - **500x amplification** at peak

2. **Customer DC Orders Are Lumpy**
   - Day 1: 1.6M orders (bullwhip spike)
   - Days 2-3: Zero orders
   - Days 4-30: Erratic (0 to 500k)
   - Mean: 243k/day vs expected 336k/day

3. **Inventory Distribution Never Stabilizes**
   - RDCs: 92-93% of finished goods (too high)
   - Stores: 3-4% of finished goods (too low)
   - Customer DCs: 1-2% of finished goods (too low)

### Root Cause Analysis

The **(s,S) policy bullwhip is realistic and expected.** The problem is that something is preventing the system from reaching steady state.

**Possible causes of degradation (not bullwhip itself):**

1. **Demand signal attenuation** - Are customer DCs seeing true demand or a dampened version?
2. **Inventory leaking somewhere** - Is mass balance being violated?
3. **Allocation not distributing fairly** - Are some nodes being starved?
4. **Lead time mismatch** - Are orders arriving too late to meet demand?
5. **Feedback loop instability** - Is the variance-aware safety stock amplifying instead of dampening?

**The fixes we added may have introduced new issues:**
- Expected throughput floor → Could cause over-ordering in some scenarios
- Flow-based minimum → Could interfere with natural (s,S) dynamics
- Reduced initialization → May have removed necessary buffer stock

**We should probably REVERT the flow-based minimum and expected throughput fixes** and instead investigate why the vanilla (s,S) system doesn't stabilize.

---

## Potential Root Cause Fixes

### Option 1: Base-Stock Policy (Order-Up-To)
Replace (s,S) with base-stock:
```
Each period: Order = Target - IP
```
- Orders every period (smooth signal)
- Order quantity = consumption (no amplification)
- Standard in supply chain theory for reducing bullwhip

### Option 2: Continuous Review with Smaller Batches
Keep (s,S) but:
- Reduce batch sizes significantly
- Increase order frequency
- Lower (S - s) gap to reduce lumpiness

### Option 3: Echelon Stock Policy
Use system-wide inventory position:
```
Echelon IP = Local IP + All Downstream IP
```
- Prevents each node from independently amplifying
- Requires visibility across supply chain tiers

### Option 4: Demand Smoothing at Source
Instead of passing raw orders upstream:
```
Smoothed_Order = alpha * Current_Order + (1-alpha) * Previous_Smoothed
```
- Dampens spikes before they propagate
- Already partially implemented but may need tuning

---

## Questions to Investigate

1. **Is the (s,S) policy intentional for realism?**
   - Real retailers often use (s,S) and DO experience bullwhip
   - Is the simulation correctly modeling reality, or is it a bug?

2. **Where does the 200M order volume come from?**
   - Need to trace: Who is generating these massive orders?
   - Is it stores? Customer DCs? Something else?

3. **Is allocation constraining shipments?**
   - RDCs have 76M units stuck
   - Are orders being placed but not fulfilled?
   - Check allocation logic and FTL constraints

4. **Does the replenishment agent have implicit bullwhip behavior?**
   - Review `MinMaxReplenisher` for any amplification logic
   - Check if demand variance tracking is working correctly
   - Review safety stock calculations

---

## Recommended Next Steps

### Immediate (Revert & Baseline)
1. **REVERT the expected throughput and flow-based minimum fixes** in replenishment.py
2. **Keep the plant shipment fix and SLOB fix** - those were real bugs
3. **Restore original config** - 21 days for customer DCs, etc.
4. **Run 365-day baseline** to see vanilla (s,S) behavior

### Diagnostic (Find the Real Bug)
1. **Check if mass balance is violated** - is inventory leaking?
2. **Trace inventory flow** - where does produced inventory go? Why isn't it reaching stores?
3. **Analyze allocation fill rates** - are orders being partially filled?
4. **Check RDC→CustomerDC shipment volumes** - is inventory stuck at RDCs due to allocation/logistics?
5. **Review FTL consolidation logic** - could orders be held too long waiting for full trucks?

### Key Question to Answer
**Why does 93% of finished goods sit at RDCs while stores have only 3%?**

Possible answers:
- Customer DCs aren't ordering enough (demand signal issue) ← we tried fixing this
- RDCs aren't shipping enough (allocation/logistics issue) ← not yet investigated
- Shipments are delayed in transit (lead time issue) ← not yet investigated
- Something else entirely

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/prism_sim/simulation/orchestrator.py` | Plant shipment filter, SLOB calculation, pass base_demand_matrix |
| `src/prism_sim/agents/replenishment.py` | Expected throughput floor, flow-based minimum |
| `src/prism_sim/config/simulation_config.json` | Reduced initialization inventory levels |

## Files to Investigate

| File | Why |
|------|-----|
| `src/prism_sim/agents/replenishment.py` | (s,S) policy implementation, demand smoothing |
| `src/prism_sim/agents/allocation.py` | Fair share logic, order fulfillment |
| `src/prism_sim/simulation/logistics.py` | FTL constraints, shipment delays |
| `docs/planning/intent.md` | Check if bullwhip is intentional design |

---

## Session Notes

- The simulation physics (mass balance, capacity constraints) appear correct
- The issue is in the **control policy** (replenishment), not the physics
- 30-day results are acceptable; 365-day degradation suggests feedback loop
- May need to fundamentally rethink replenishment strategy vs. patching (s,S)
