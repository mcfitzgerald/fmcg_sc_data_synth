# Unified DRP: Replace Push/Pull Split with Coordinated Distribution Planning

## Context

The sim has three uncoordinated inventory distribution mechanisms:
- **Deployment** (Plant→RDC): fills RDCs using recursive POS demand (~1M/day per RDC)
- **Pull** (DC→RDC orders): drains RDCs using outflow-based demand (~300K/day per RDC)
- **Push** (RDC→DC excess): should bridge the gap, but creates **ZERO shipments** (DCs already above POS-based target → dc_room=0)

Result: RDCs accumulate +628K cases/day (17.9 DOS vs 9 target), DCs starve (7.5 DOS vs 10.5 target), fill rate 80.6% vs 94% target. Additionally, 21 ECOM/DTC FCs with 807K/day aggregate demand are invisible to push (no downstream stores in topology).

**Fix:** Replace the push+pull split with a single DRP pass that plans distribution top-down — netting inventory position against target at each DC, then having RDCs proactively ship what's needed. Uses consistent demand signals throughout.

## Critical Discovery: ECOM/DTC Demand

ECOM-FCs and DTC-FCs are DC-type nodes that consume inventory directly (like stores) — they have `base_demand_matrix[fc_idx, :]` = 38,451 cases/day each, but zero downstream stores and zero outbound links. The replenisher handles them (standard (s,S) pull from RDCs), but push completely ignores them. DRP must use their own POS demand as the demand signal.

---

## Phase 1: DRP Distribution Engine (replaces push)

**Goal:** New `DRPDistributionEngine` proactively ships from RDCs to DCs based on need. Replaces `_push_excess_rdc_inventory()` at step 10a. Pull orders continue as-is (DCs still order from RDCs as a secondary mechanism).

### New file: `src/prism_sim/simulation/drp_distribution.py`

```python
class DRPDistributionEngine:
    def __init__(self, world, state, config, pos_engine, mrp_engine,
                 rdc_downstream_dcs, dc_downstream_stores,
                 dc_secondary_rdc, rdc_secondary_dcs)

    def compute_and_execute(self, day: int) -> list[Shipment]
```

**Init (one-time, reuse orchestrator topology maps):**
- Receive pre-built topology maps from orchestrator (no re-computation)
- Pre-compute `_dc_expected_demand[dc_id] → ndarray[n_products]` for each DC:
  - DCs with downstream stores: aggregate `base_demand_matrix[store_idx, :]`
  - ECOM/DTC FCs (no stores): use `base_demand_matrix[fc_idx, :]` directly
- Pre-compute `_dc_target_dos_vec[n_products]` (ABC: A=10.5, B=14, C=17.5)
- Pre-compute `_rdc_safety_dos` from config (default 3.0)
- Cache lead times per (rdc, dc) link

**`compute_and_execute(day)` algorithm — fully vectorized per RDC:**

```
for each RDC:
    seasonal = mrp_engine._get_seasonal_factor(day)
    rdc_on_hand = actual_inventory[rdc_idx, :]

    # --- Aggregate DC needs ---
    total_need = zeros(n_products)
    dc_needs = {}
    for each DC downstream of this RDC:
        demand = dc_expected_demand[dc_id] * seasonal         # [n_products]
        target = dc_target_dos_vec * demand                    # ABC-differentiated
        ip = actual_inventory[dc_idx] + in_transit[dc_idx]     # full IP (incl. pull in-transit)
        need = max(0, target - ip)                             # net requirement

        # Secondary-source fraction scaling
        if dc has secondary & this is primary: need *= (1 - frac)
        if dc has secondary & this is secondary: need *= frac

        dc_needs[dc_id] = need
        total_need += need

    # --- RDC available = on_hand - safety reserve ---
    rdc_demand = sum(dc_expected_demand[dc] for all downstream DCs) * seasonal
    rdc_safety = rdc_safety_dos * rdc_demand
    available = max(0, rdc_on_hand - rdc_safety)

    # --- Fair-share allocation (per-product) ---
    fill_ratio = where(total_need > 0, min(available / total_need, 1.0), 0.0)

    # --- Create shipments ---
    for dc_id, need in dc_needs.items():
        ship_qty = need * fill_ratio
        ship_qty[ship_qty < min_ship_cases] = 0
        if ship_qty.sum() < 100: continue
        # Create Shipment with DRP-{day}-{rdc}-{counter} ID
        # Deduct from RDC with FIFO age reduction (same pattern as push)
        # Use _populate_shipment_arrays() for perf
```

### Orchestrator changes (`orchestrator.py`)

1. **Init:** Create `DRPDistributionEngine`, pass it the topology maps
2. **Step 10a:** Replace `_push_excess_rdc_inventory(day)` call with:
   ```python
   drp_shipments = self.drp_distribution.compute_and_execute(day)
   ```
   Same post-processing (auditor, quirks, add_shipments_batch)
3. **Keep `_push_excess_rdc_inventory`** as dead code initially (remove in Phase 3 cleanup)

### Config changes (`simulation_config.json`)

Add under `agents.replenishment`:
```json
"drp_distribution": {
    "enabled": true,
    "rdc_safety_dos": 3.0,
    "min_ship_cases": 10
}
```

### Performance

- Inner loop: 6 RDCs × ~6 DCs each = ~36 iterations
- Each iteration: 5-6 numpy ops on [640]-element arrays
- Shipment creation: Python loop over non-zero products per DC (same pattern as push)
- Expected: <0.5ms/day — same order as current push

### Verification

```bash
poetry run python run_simulation.py --days 50 --no-logging
```
- RDC DOS should drop from ~18 toward 9-12 (DRP actively draining)
- DC DOS should rise from ~7.5 toward target (10.5 A-class)
- Fill rate should improve noticeably
- Mass balance clean

---

## Phase 2: Right-size Plant→RDC Deployment

**Goal:** Deployment currently fills RDCs based on recursive POS demand (~1M/day) which is 3× what DCs actually pull. Make deployment use the DRP's demand signal instead, so Plant→RDC flow matches RDC→DC flow.

### DRP addition: `get_rdc_throughput_demand(rdc_id, day)`

Returns the daily demand rate that this RDC needs to serve — sum of `dc_expected_demand × seasonal` for all DCs downstream, accounting for secondary fractions. This is **the same demand** used in `compute_and_execute`, ensuring deployment and distribution use a consistent signal.

### Orchestrator changes

**`_precompute_deployment_targets()`:** For RDC targets, replace recursive store POS aggregation with DRP throughput demand:
```python
if target_id.startswith("RDC-"):
    self._target_expected_demand[target_id] = \
        self.drp_distribution.get_rdc_throughput_demand(target_id)
```

**`_calculate_deployment_shares()`:** Recalculate shares using DRP demand for RDCs (plant-direct DCs keep POS-based shares).

### Why this works

Currently: `_target_expected_demand[RDC-WE]` = 1,053,562/day (all stores under all DCs under RDC-WE)
After: `_target_expected_demand[RDC-WE]` = ~192,000/day (just the DC-level demand, which is what the RDC actually ships downstream + ECOM/DTC demand)

Deployment fills RDC to `rdc_target_dos × rdc_demand` = 9 × 192K = 1.73M cases (not 9 × 1.05M = 9.5M).

### Verification

```bash
poetry run python run_simulation.py --days 50 --no-logging
```
- RDC inflow ≈ RDC outflow (no more accumulation)
- RDC DOS stable at ~9
- Plant FG levels stable
- DC DOS on target
- Fill rate approaching 94%

---

## Phase 3: Remove DC→RDC Pull + MRP Signal Fix + Cleanup

**Goal:** DRP is now the sole mechanism for RDC→DC flow. Remove redundant DC pull orders from the replenisher. Fix MRP demand signal. Clean up dead code.

### Replenisher changes (`replenishment.py`)

In `_identify_target_nodes()`, skip ALL DCs (including ECOM/DTC FCs) that are served by DRP:
```python
# line ~1022-1025: add condition
if node.type == NodeType.DC and "RDC" not in node.id:
    if self._drp_suppresses_dc_pull:
        continue  # DRP handles ALL DC→RDC replenishment (incl. ECOM/DTC)
```

`_drp_suppresses_dc_pull` read from config `drp_distribution.suppress_dc_pull`.

**Stores still order from DCs** — only DC/FC→RDC pull is removed. All DC types (GRO-DC, PHARM-DC, DIST-DC, ECOM-FC, DTC-FC, RET-DC, CLUB-DC) are handled by DRP.

### MRP demand signal (`mrp.py`)

Currently MRP receives DC→RDC orders as production demand signal (step 3, via `record_order_demand_batch`). When DC pull is removed, MRP loses this signal.

Add method:
```python
def record_drp_demand(self, total_drp_qty: np.ndarray) -> None:
    """Record DRP distribution quantities as demand signal."""
```

In orchestrator, after DRP execution:
```python
if drp_shipments:
    drp_total = sum ship quantities across all DRP shipments  # [n_products]
    self.mrp_engine.record_drp_demand(drp_total)
```

### Config changes

```json
"drp_distribution": {
    "enabled": true,
    "rdc_safety_dos": 3.0,
    "min_ship_cases": 10,
    "suppress_dc_pull": true
}
```

### Cleanup

- Remove `_push_excess_rdc_inventory()` method from orchestrator
- Remove `push_threshold_dos` from config
- Remove `push_allocation_enabled` from config
- Remove `_push_suppression_count` tracking

### Warm-start compatibility

DRP distribution is **stateless** — computed fresh each day from inventory tensors + topology. No changes to `warm_start.py`. Old warm-start snapshots work fine (replenisher history for DCs is loaded but unused since DCs are excluded from target nodes).

### Verification

```bash
# Full production run
poetry run python run_simulation.py --days 365 --streaming --format parquet

# Full diagnostic
poetry run python scripts/analysis/diagnose_supply_chain.py --full
```

Target metrics:
- Fill rate >= 94%
- RDC DOS ~9 (stable)
- DC DOS: A ~10.5, B ~14, C ~17.5
- Mass balance: BALANCED
- Stability: CONVERGED
- No PUSH-* shipments, no DC→RDC pull orders
- DRP-* shipments visible in shipment log

---

## Files Changed

| Phase | File | Action | Description |
|-------|------|--------|-------------|
| 1 | `src/prism_sim/simulation/drp_distribution.py` | NEW | DRPDistributionEngine |
| 1 | `src/prism_sim/simulation/orchestrator.py` | EDIT | Init DRP engine, replace step 10a |
| 1 | `src/prism_sim/config/simulation_config.json` | EDIT | Add `drp_distribution` config |
| 2 | `src/prism_sim/simulation/orchestrator.py` | EDIT | Modify `_precompute_deployment_targets` + shares |
| 2 | `src/prism_sim/simulation/drp_distribution.py` | EDIT | Add `get_rdc_throughput_demand()` |
| 3 | `src/prism_sim/agents/replenishment.py` | EDIT | Suppress DC pull when DRP enabled |
| 3 | `src/prism_sim/simulation/mrp.py` | EDIT | Add `record_drp_demand()` |
| 3 | `src/prism_sim/simulation/orchestrator.py` | EDIT | Feed DRP demand to MRP, remove push code |
| 3 | `src/prism_sim/config/simulation_config.json` | EDIT | Add `suppress_dc_pull`, remove push params |
| all | `docs/llm_context.md` | EDIT | Update daily loop, architecture |
| all | `CHANGELOG.md` | EDIT | Version bumps |

## Key Patterns Reused

- **Fair-share fill ratio** from `_create_plant_shipments` (lines 2337-2343)
- **FIFO age reduction** from push/deployment shipment creation
- **`_populate_shipment_arrays()`** from LogisticsEngine (v0.87.0 perf)
- **`add_shipments_batch()`** for in-transit tensor updates
- **ABC target DOS derivation** from `_precompute_deployment_targets` (lines 478-483)
- **Topology maps** passed by reference from orchestrator (no re-computation)
- **Seasonal factor** via `mrp_engine._get_seasonal_factor(day)`
- **Secondary-source fraction** handling from push code (lines 2637-2660)
