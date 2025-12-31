# Forward Tasks - Phase C Continuation

## Session Summary (Dec 30, 2024 - Continued)

### Fixes Completed This Session

#### 3. Ingredient Replenishment Mismatch (CRITICAL)
**File:** `src/prism_sim/simulation/mrp.py:200-240` and `src/prism_sim/simulation/orchestrator.py`

**Root Cause:** MRP's `generate_purchase_orders()` used POS demand signal (~400k/day) for ingredient replenishment, but actual ingredient consumption was driven by production orders (amplified by bullwhip to 5-6M/day). This caused a net burn rate of ~1,380 units/day shortfall, leading to ingredient exhaustion and production collapse on days 362-365.

**Fix:** Changed `generate_purchase_orders()` to use production-based signal instead of POS demand:
```python
def generate_purchase_orders(
    self,
    current_day: int,
    active_production_orders: list[ProductionOrder],  # Changed from daily_demand
) -> list[Order]:
    # Calculate production signal from active production orders
    production_by_product = np.zeros(self.state.n_products, dtype=np.float64)
    for po in active_production_orders:
        p_idx = self.state.product_id_to_idx.get(po.product_id)
        if p_idx is not None:
            production_by_product[p_idx] += po.quantity_cases
    # ... uses production signal for ingredient ordering
```

**Orchestrator update:**
```python
ing_orders = self.mrp_engine.generate_purchase_orders(
    day, self.active_production_orders  # Production-based signal
)
```

### 365-Day Validation Results

| Metric | Before Fix (Collapse) | After Fix |
|--------|----------------------|-----------|
| Service Level | 52.54% | 58.16% |
| Manufacturing OEE | 55.1% | 61.8% |
| Production Day 365 | 0 (collapse) | 259,560 cases |
| System Survival | Collapsed day 362-365 | Full year |

**Key Improvement:** System no longer collapses. Production continues through the entire 365-day simulation.

---

## Previous Session Summary (Dec 30, 2024)

### Fixes Completed

#### 1. MRP Inventory Position Bug (CRITICAL)
**File:** `src/prism_sim/simulation/mrp.py:149-162`

**Root Cause:** MRP's `_cache_node_info()` included ALL `NodeType.DC` nodes in inventory position calculation, including customer DCs (RET-DC, DIST-DC, ECOM-FC) with ~4.5M units. This inflated Days of Supply to 11.5 days > ROP 7 days, preventing production orders and causing 94 zero-production days.

**Fix:** Only include manufacturer RDCs (nodes starting with `RDC-*`):
```python
def _cache_node_info(self) -> None:
    for node_id, node in self.world.nodes.items():
        if node.type == NodeType.DC:
            # Only include manufacturer RDCs, not customer DCs
            if node_id.startswith("RDC-"):
                self._rdc_ids.append(node_id)
```

#### 2. C.5 Smoothing History Bug
**File:** `src/prism_sim/simulation/mrp.py:295-298`

**Root Cause:** Production order history recorded pre-scaled quantities instead of post-scaled, inflating the rolling average used for smoothing.

**Fix:** Record actual (post-scaled) totals:
```python
actual_total = sum(po.quantity_cases for po in production_orders)
self.production_order_history[self._prod_hist_ptr] = actual_total
```

### Results After Fixes (30-day validation)
| Metric | Before | After |
|--------|--------|-------|
| Service Level | 51.62% | 60.19% |
| Manufacturing OEE | 44.9% | 88.2% |
| Zero-Production Days | 94 | 0 |

---

## Remaining Issues

### 1. Mass Balance Violations (Tracking Issue, Not Physics Bug)

**Symptom:** Mass balance violations at customer DCs (DIST-DC-001, etc.) showing `Expected < 0, Actual = 0`.

**Root Cause:** FTL consolidation timing mismatch:
1. Allocation decrements inventory immediately when orders are created
2. Logistics can HOLD orders if they don't meet FTL minimum pallet thresholds
3. `shipments_out` is recorded when shipments are actually created (may be days later)
4. Mass balance equation `expected = opening + receipts - shipments_out` doesn't account for "allocated but not yet shipped" inventory

**Impact:** This is an **accounting/auditing issue**, not an actual physics violation. The inventory is correctly managed; the mass balance tracker just can't account for temporal mismatches from FTL consolidation.

**Potential Fixes (not implemented):**
- Option A: Track "reserved inventory" separately from "shipped inventory"
- Option B: Modify auditor to track actual inventory deltas directly instead of reconstructing from flows
- Option C: Disable FTL consolidation (set `min_order_pallets: 0` in config)
- Option D: Accept violations as expected behavior for FTL systems

**Relevant Files:**
- `src/prism_sim/agents/allocation.py:139-144` - Inventory decrement
- `src/prism_sim/simulation/logistics.py:83-88` - FTL order holding
- `src/prism_sim/simulation/monitor.py:316-368` - Mass balance check
- `src/prism_sim/config/simulation_config.json:59-80` - FTL channel rules

### 2. 365-Day Validation - COMPLETED
✓ System survives full year with production continuing through day 365.
✓ No collapse - production at 259,560 cases on final day.

### 3. Service Level & Bullwhip (Future Optimization)
Current metrics after all fixes:
- Service Level: 58.16% (target: 98.5%)
- Bullwhip Ratio: ~15x (orders 5-6M vs demand 400k)
- Inventory Turns: Low (high SLOB)

These are tuning/optimization issues, not bugs. The system is now stable.

---

## Key Architecture Context

### Option C Network Topology
```
Plants (4) → RDC (4) → Customer DCs → Stores (4000+)
                       ├── RET-DC (Retailer DCs)
                       ├── DIST-DC (Distributor DCs)
                       └── ECOM-FC (E-commerce FCs)
```

### Node ID Prefixes
- `RDC-*` - Manufacturer Regional Distribution Centers (inventory position)
- `RET-DC-*` - Retailer Distribution Centers (customer inventory)
- `DIST-DC-*` - Distributor Distribution Centers (customer inventory)
- `ECOM-*` - E-commerce Fulfillment Centers (customer inventory)
- `PLANT-*` - Manufacturing Plants
- `STORE-*` - Retail Stores

### MRP Inventory Position Calculation
```
Inventory Position = On-Hand (RDC) + In-Transit + In-Production
Days of Supply = Inventory Position / Average Daily Demand
If DOS < ROP (7 days) → Generate Production Order
```

---

## Files Modified (All Sessions)
- `src/prism_sim/simulation/mrp.py` - MRP inventory position fix, smoothing history fix, ingredient replenishment fix
- `src/prism_sim/simulation/orchestrator.py` - Pass production orders to generate_purchase_orders
- `docs/llm_context.md` - Documentation updates
- `CHANGELOG.md` - v0.15.0, v0.15.1 release notes
- `pyproject.toml` - Version bump to 0.15.1

## Commands
```bash
# Run simulation
poetry run python run_simulation.py --days 30 --no-logging

# Run analysis scripts
poetry run python scripts/analyze_production.py
poetry run python scripts/analyze_bullwhip.py

# Run tests
poetry run pytest
```
