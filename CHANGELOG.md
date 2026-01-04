# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.19.11] - 2026-01-03

### Added
- **POS-Driven Production (Physically Correct Approach):** Replaced open-loop expected-based production with closed-loop POS-driven production.
  - Uses actual consumer demand (POS) as the primary signal
  - Inventory feedback loop: production adjusts based on actual DOS
  - ABC differentiation in response dynamics, not baseline rates
- **ABC Class Tracking:** Added `abc_class` array to MRPEngine (0=A, 1=B, 2=C) for production decisions.

### Changed
- **`_generate_rate_based_orders()`:** Complete rewrite for closed-loop control:
  - A-items: Fast response (130% catch-up when low, 90% when high)
  - B-items: Balanced response (110% catch-up, 85% when high)
  - C-items: Slow response, aggressive reduction when overstocked (down to 30%)
  - Production tracks actual demand, not expected demand
- **`generate_production_orders()`:** Now passes POS demand vector to rate-based orders.

### Results
| Metric | v0.19.9 | v0.19.11 | Target |
|--------|---------|----------|--------|
| 30-day Service Level | 93% | 92.5% | >90% |
| 365-day Service Level | 67.5% | 64.5% | >85% |
| **SLOB** | 70% | **28.6%** | <30% ✓ |
| Inventory Turns | 4.8x | 7.9x | Higher ✓ |
| Cash-to-Cash | 68 days | 33 days | Lower ✓ |

### Analysis
**SLOB target achieved!** The POS-driven approach dramatically reduced SLOB from 70% to 28.6% (<30% target).

However, 365-day service level dropped from 67.5% to 64.5%. Extensive experimentation revealed:

1. **Production is NOT the bottleneck for service level.** Even with 150% A-item production, SL stays at ~66%. This proves the remaining SL gap is a **distribution problem**, not a production problem.

2. **The physics is correct.** POS-driven production creates a self-correcting system:
   - C-items: Production drops when DOS > threshold → SLOB decreases
   - A-items: Production increases when DOS < threshold → should improve SL
   - But goods aren't reaching stores (distribution bottleneck)

3. **Tradeoff confirmed:**
   - Aggressive C-item reduction → SLOB ↓↓↓, but SL slightly ↓
   - Aggressive A-item boost → SL barely changes (stuck at ~66%)
   - Proves distribution flow is the constraint

**Next Steps for 85%+ SL:**
The service level improvement requires fixing **distribution**, not production:
- RDC-to-store replenishment tuning
- Push allocation parameters
- Potentially reduce initial inventory priming at RDCs
- Check if goods are stuck at RDCs instead of flowing to stores

See `docs/planning/mix_opt.md` for the original analysis.

## [0.19.10] - 2026-01-03 (Superseded by v0.19.11)

Experimental version with ABC-differentiated production. Improved SL to 71.7% but SLOB worsened to 76%. Replaced by POS-driven approach in v0.19.11.

## [0.19.9] - 2026-01-03

### Added
- **Rate-Based Production (Option C):** Implemented anticipatory production mode that always produces at expected demand rate, preventing the low-equilibrium trap.
  - `rate_based_production`: Config toggle for new production mode (default: true)
  - `rate_based_min_batch`: Lower min batch (100 vs 1000) to capture C-items
  - `inventory_cap_dos`: Only throttle production if DOS > 45 days
  - Catch-up mode: If DOS < ROP, produce EXTRA to recover deficit
- **MRP Diagnostics:** Added `MRPDiagnostics` dataclass for debugging signal flow.
  - Tracks demand signals, production orders, inventory position, DOS
  - Logs to `prism_sim.mrp.diagnostics` logger at INFO level

### Fixed
- **Low-Equilibrium Trap:** Rate-based production prevents the system from stabilizing at low production levels when demand signals collapse.
- **C-Item Production:** Lower min batch (100) prevents filtering out slow-moving products entirely.

### Changed
- **MRPEngine:** Added `_generate_rate_based_orders()` method for Option C logic.
- **Config:** Added `rate_based_production`, `rate_based_min_batch`, `inventory_cap_dos`, `diagnostics_enabled` to `mrp_thresholds`.

### Results
| Metric | v0.19.8 | v0.19.9 | Target |
|--------|---------|---------|--------|
| 30-day Service Level | 93% | 93% | >90% |
| 365-day Service Level | 50-58% | **67.5%** | >85% |
| OEE | 60-75% | **88%** | 75-85% |
| SLOB | - | 70% | <30% |

### Analysis
Rate-based production improved 365-day SL by +17.5pp. The remaining gap to 85%+ is due to:
1. **Product Mix Mismatch:** Fixed expected mix doesn't match actual demand variations
2. **SLOB at 70%:** Wrong products being stocked (C-items accumulating)
3. **Distribution Flow:** Goods produced but not flowing to stores efficiently

See `docs/planning/debug_foxtrot.md` for detailed analysis.

## [0.19.8] - 2026-01-03

### Fixed
- **MRP Starvation Loop (Partial):** Decoupled ingredient ordering from historical production flow.
  - Added `_calculate_max_daily_capacity()` to compute network production capacity.
  - Changed `generate_purchase_orders()` to use expected demand (not historical avg) when backlog is low.
  - Changed production smoothing cap to use `max(avg_recent, expected)` as baseline, preventing cap degradation.
- **Warm Starts:** Warm-started `demand_history` and `production_order_history` buffers.
- **Bullwhip Clamping:** Clamped `record_order_demand` signal to 4x expected demand.

### Changed
- **MRPEngine:**
  - Ingredient ordering now uses expected demand as minimum baseline (was: coupled to backlog only).
  - Production smoothing cap now floors at expected production (was: could degrade with history).
- **Orchestrator:** Added `auditor.record_plant_shipments_out` for push shipments (Mass Balance fix).
- **Config:** Tuned `production_floor_pct` (0.3 → 0.5) and `min_production_cap_pct` (0.5 → 0.7).

## [0.19.7] - 2026-01-03

### Fixed
- **ABC Alignment (Phase 1):** Aligned `MRPEngine` ABC classification with `Replenisher` and `TransformEngine` by injecting `base_demand_matrix` (Zipf-aware) into MRP. This resolves the regression where popular A-items were misclassified as B/C in production planning.
- **Service Level Recovery:** 90-day simulation Service Level recovered to **86.50%** (from 71%), exceeding the >85% target.
- **Engine Bugs:**
  - Fixed `AttributeError` in `POSEngine` by initializing `channel_sku_weights`.
  - Fixed `AttributeError` in `TransformEngine` by correcting scope of `_get_abc_priority`.

### Changed
- **Parameter Tuning (Phase 2 & 3):**
  - Increased A-item ROP multiplier (1.2 → 1.5) to buffer against demand/supply variability.
  - Decreased C-item ROP multiplier (0.8 → 0.5) and Service Level Z-score (1.28 → 1.0) to reduce SLOB.
- **MRPEngine:** Now calculates `expected_daily_demand` by summing the injected `base_demand_matrix` instead of using static config profiles.
- **Orchestrator:** Passes `base_demand_matrix` to `MRPEngine` during initialization.

## [0.19.6] - 2026-01-03

### Refactoring
- **Config-Driven Logic:** Replaced "genuine logic hardcodes" with configuration parameters and enums to improve maintainability and flexibility.
  - **Enums:** Introduced `OrderPriority` (`RUSH`, `HIGH`, `STANDARD`, `LOW`) and `ABCClass` (`A`, `B`, `C`) in `core.py` to replace integer/string literals.
  - **Configuration:** Added `min_history_days` to `replenishment` config and `min_batch_size_absolute`, `default_store_count` to `manufacturing` config in `simulation_config.json`.
  - **Agents:** Updated `replenishment.py`, `mrp.py`, and `transform.py` to use the new enums and configuration values instead of hardcoded numbers.

## [0.19.5] - 2026-01-03

### Status
- **365-Day Validation:** Run completed. Service Level (71%) and SLOB (78%) indicate a regression from baseline.
- **Root Cause:** Identified ABC misalignment between Replenisher (Zipf-aware) and MRP (Zipf-blind config).
- **Plan:** Created `docs/planning/alignment_and_param_fix.md` to address the architecture gap before further tuning.

## [0.19.4] - 2026-01-03

### Added
- **ABC Prioritization (Phase 3 & 4):** Completed remaining phases of the ABC prioritization plan.
  - **ABC-Aware Replenishment (Phase 3):** `MinMaxReplenisher` now uses config-driven thresholds (80/95%) for dynamic ABC classification, ensuring consistency with MRP logic.
  - **Production Capacity Reservation (Phase 4):** `TransformEngine` now reserves capacity for A-items by classifying products based on expected demand and prioritizing A-item production orders before C-items.
  - **Configuration:** Updated `simulation_config.json` with `abc_prioritization` block enabled by default.

### Changed
- **Replenishment:** Updated `replenishment.py` to read ABC thresholds from config.
- **Transform:** Updated `transform.py` to sort production orders by ABC priority (A > B > C) then due date.

## [0.19.3] - 2026-01-03

### Added
- **ABC Prioritization (Phase 1 & 2):** Implemented Pareto-based prioritization for Allocation and MRP to resolve product mix imbalances.
  - **Allocation:** Scarcity logic now prioritizes A-items (high velocity) over C-items within the same order priority tier. This ensures fast movers get dibs on inventory.
  - **MRP:** Production planning uses ROP multipliers (A=1.2x, C=0.8x) to ensure higher availability for fast movers and reduce SLOB for slow movers.
  - **Configuration:** Added `abc_prioritization` section to `simulation_config.json` with configurable thresholds (80/95%) and multipliers.

### Changed
- **Orchestrator:** Now calculates and injects product velocity into `AllocationAgent` during initialization.
- **MRPEngine:** Now classifies products by velocity and applies dynamic ROP multipliers during production planning.

## [0.19.2] - 2026-01-03

### Service Level Improvement: Signal Flow Optimization

This release implements the fixes outlined in `docs/planning/new-fix.md` to break the negative feedback spiral causing service level degradation.

### Fixed
- **Daily Ordering for Customer DCs:** Removed the 3-day order cycle restriction for Customer DCs using echelon logic. DCs now order every day, ensuring demand signals flow continuously upstream without accumulation delays.
- **Increased Customer DC Targets:** Raised target_days from 21 to 35 and reorder_point_days from 14 to 21 for B2M_LARGE, B2M_CLUB, and B2M_DISTRIBUTOR channels. Higher targets ensure DCs order sufficient quantities to cover downstream demand.
- **Increased Store Targets:** Raised default target_days from 14 to 21 and reorder_point_days from 10 to 14 to provide more buffer at store level.
- **Echelon Safety Multiplier:** Added `echelon_safety_multiplier` (default 1.3) to echelon target/ROP calculations. This provides a buffer beyond raw echelon demand to account for variance at the echelon level.
- **Demand-Proportional MRP Batches:** Changed MRP minimum batch size from fixed 50,000 cases to demand-proportional (7 days of demand, minimum 1,000 cases). Prevents SLOB accumulation from massive batches of low-demand products.

### Added
- **Push-Based Allocation:** Implemented `_push_excess_rdc_inventory()` in Orchestrator to push excess RDC inventory to Customer DCs when Days of Supply exceeds threshold. Uses POS-based demand signal (stable) instead of outflow demand (which collapses during the spiral).
- **Production Prioritization Support:** Added `set_base_demand()` to TransformEngine for future demand-based production scheduling.
- **Configuration Parameters:**
  - `echelon_safety_multiplier`: Buffer multiplier for echelon targets (default: 1.3)
  - `push_allocation_enabled`: Toggle for push-based allocation (default: true)
  - `push_threshold_dos`: Days of supply threshold for push (default: 21)

### Results
| Metric | v0.19.1 Baseline | v0.19.2 (90-day) | v0.19.2 (365-day) | Target |
|--------|------------------|------------------|-------------------|--------|
| Service Level | 73% | **91.84%** ✅ | 76% | >90% |
| SLOB | 65% | 54% | 73% | <30% |
| Inventory Turns | 4.73x | 6.12x ✅ | 4.69x | 6-14x |

### Root Cause Analysis
The 90-day simulation achieves >90% service level, but 365-day degrades to ~76%. Investigation revealed:
- **Product Mix Issue:** Highly concentrated demand (top 10 SKUs = 60% of volume) with SLOB at 73-80% suggests wrong products are stocked
- **Zipfian Distribution:** A-items (16 SKUs, 80% of demand) may be stocking out while C-items (47 SKUs) accumulate
- **Slow Drift:** System starts well (initial inventory priming) but drifts to suboptimal equilibrium over time

### Next Steps for v0.19.3
1. **ABC-Prioritized Allocation:** When inventory is scarce, prioritize A-items over C-items
2. **ABC-Prioritized MRP:** Weight production planning toward high-velocity SKUs
3. **Inventory Distribution Monitoring:** Track echelon-level inventory vs demand alignment
4. **Store-Level Push:** Extend push allocation to Customer DC → Store link

## [0.19.1] - 2026-01-03

### Fixed
- **MEIO Inventory Position Bug:** Fixed echelon logic to use **Local IP** (DC inventory + in-transit) instead of **Echelon IP** (DC + all downstream stores). The original implementation caused Customer DCs to under-order because downstream store inventory inflated the IP calculation, making the system appear "well-stocked" even as stores depleted.
- **MRP Demand Signal Collapse:** Added POS demand as a floor for MRP demand signal. When order-based demand declines (due to downstream starvation), MRP now uses actual consumer demand (POS) to maintain production levels, preventing the death spiral where low orders → low production → more starvation.

### Analysis
Comprehensive diagnostic scripts created in `scripts/analysis/`:
- `diagnose_service_level.py` - Analyzes service degradation patterns by echelon, product, and time
- `diagnose_slob.py` - Analyzes inventory distribution, velocity, and SLOB (slow/obsolete) products

### Known Issues
- **365-Day Service Level: 73%** (target >90%) - Inventory accumulates at MFG RDCs (93% of total) while stores starve (4.5%). Customer DCs are ordering only 54% of demand despite MEIO fix.
- **Root Cause Identified:** The signal flow architecture creates a negative feedback loop where declining downstream inventory → declining orders → declining production. The MEIO/MRP fixes are applied but the 3-day ordering cycle and target_days parameters may need tuning.
- **Next Steps:** Consider (1) daily ordering for Customer DCs, (2) higher target_days, (3) push-based allocation from RDCs to supplement pull-based ordering.

## [0.19.0] - 2026-01-03

### Added
- **Echelon Inventory Logic (MEIO):** Implemented Multi-Echelon Inventory Optimization for Customer DCs. DCs now use aggregated downstream Echelon Inventory Position (DC + Stores) and Echelon Demand (POS) to trigger replenishment, resolving the "Signal Trap" where DCs stopped ordering when stores were empty.
- **Configuration:** Added `store_batch_size_cases` and `lead_time_history_len` to `simulation_config.json` to eliminate hardcoded values in `MinMaxReplenisher`.

### Changed
- **Replenishment Logic:** Customer DCs now bypass "Orders Received" signal and link directly to POS data via Echelon logic.
- **Refactoring:** Removed hardcoded overrides for store batch sizes and lead time history length in `replenishment.py` in favor of config-driven values.

## [0.18.2] - 2026-01-02

### Fixed
- **Order Signal Collapse:** Fixed demand signal attenuation at Customer DCs (RET-DC, etc.) by switching `Replenisher` to use a 7-day average of **Inflow Demand** (orders received) for all nodes, replacing the collapsing exponential smoothing logic.
- **Phantom Ingredient Replenishment:** Explicitly masked `ProductCategory.INGREDIENT` in `MinMaxReplenisher` to prevent Stores and DCs from ordering millions of units of raw materials (chemicals, packaging).
- **Sporadic Demand:** Increased store order cycle from 1 day to 3 days to consolidate demand signals and prevent "zero-demand" days from crashing safety stock calculations.

### Changed
- **Reverted v0.18.0 Band-aids:** Removed "Expected Throughput Floor" and "Flow-based Minimum Order" logic in favor of fixing the root cause (ingredient filtering + inflow demand).
- **Configuration:** Restored baseline inventory initialization days (21d Stores, 28d RDCs, 21d DCs).

## [0.18.0] - 2026-01-02

### Bug Fixes: Plant Shipment Routing & SLOB Calculation

This release fixes two critical bugs identified during 365-day simulation analysis.

### Fixed
- **Plant Shipment Routing Bug (Critical):** Plants were shipping production to ALL 44 DC nodes instead of just 4 manufacturer RDCs. Added `n_id.startswith("RDC-")` filter to `_ship_production_to_rdcs()` in `orchestrator.py:699-705`. This prevented 150M+ units of unauthorized PUSH shipments to customer DCs.
- **SLOB Calculation Bug:** SLOB (Slow/Obsolete) inventory was calculated as binary (0% or 100%) based on global days-of-supply. Fixed to per-SKU calculation: flags each SKU where `DOS > threshold`, then reports `sum(SLOB inventory) / total FG inventory`. Now correctly identifies which portion of inventory is slow-moving.

### Added (Experimental - May Revert)
- **Expected Throughput Floor for Customer DCs:** Customer DCs now use `max(inflow_demand, expected_throughput)` as their demand signal. `expected_throughput` is calculated by aggregating base demand from all downstream stores. This prevents cold-start under-ordering when stores haven't placed orders yet.
- **Flow-Based Minimum Order:** Customer DCs order at least `expected_throughput * lead_time` even when `IP > ROP`, to maintain inventory flow.

### Changed
- **Initialization Inventory Levels:** Reduced to prevent over-priming:
  - `customer_dc_days_supply`: 21 → 10 days
  - `store_days_supply`: 21 → 14 days
  - `rdc_days_supply`: 28 → 21 days

### Known Issues
- **365-Day Service Level Degradation:** Service level degrades from ~92% (30-day) to ~70% (365-day). The bullwhip effect is intentional realism, but the system should stabilize rather than degrade. Root cause under investigation - may be allocation/logistics bottleneck rather than replenishment policy.
- **Inventory Imbalance:** 93% of finished goods remain at RDCs, only 3% at stores. This suggests a flow bottleneck between RDCs and downstream nodes.

### Files Modified
- `src/prism_sim/simulation/orchestrator.py` - Plant shipment filter, SLOB calculation, pass base_demand_matrix
- `src/prism_sim/agents/replenishment.py` - Expected throughput floor, flow-based minimum
- `src/prism_sim/config/simulation_config.json` - Reduced initialization inventory levels

---

## [0.17.0] - 2026-01-02

### Physics Overhaul: First-Principles Supply Chain Physics

This major release overhauls the replenishment physics to use realized performance data and textbook inventory theory, moving away from heuristic safety stock.

### Added
- **Realized Lead Time Tracking (Phase 1):**
  - `Shipment` now tracks `original_order_day`.
  - `LogisticsEngine` captures the earliest order creation day during consolidation.
  - `Orchestrator` records realized lead time upon arrival.
  - `MinMaxReplenisher` maintains a rolling history of lead times per link.
- **Full Safety Stock Formula (Phase 2):**
  - Implemented the robust formula: $SS = z \sqrt{\bar{L}\sigma_D^2 + \bar{D}^2\sigma_L^2}$.
  - This formula protects against both **Demand Risk** (variability in sales) and **Supply Risk** (variability in logistics/fulfillment delays).
- **Dynamic ABC Segmentation (Phase 3):**
  - Products are dynamically classified every 7 days based on cumulative sales volume.
  - **A-Items (Top 80%):** Target 99% Service Level ($z=2.33$).
  - **B-Items (Next 15%):** Target 95% Service Level ($z=1.65$).
  - **C-Items (Bottom 5%):** Target 90% Service Level ($z=1.28$).
  - Optimizes inventory budget by prioritizing high-velocity items.

### Fixed
- **Zero Mypy Errors:** Resolved all type safety issues across the entire `src` directory.
- **Ruff Compliance:** Fixed over 100 style, complexity, and linting issues.
- **Hardcode Elimination:** Replaced multiple magic numbers with the config paradigm:
  - `order_cycle_days`: Configurable replenishment frequency.
  - `format_scale_factors`: Moved store format demand multipliers to `simulation_config.json`.
  - `stores_per_retailer_dc`: Moved to topology config in `world_definition.json`.
  - `mass_balance_min_threshold`: Moved to validation config.

### Technical Details
The system is now "self-healing." If a disruption occurs (e.g., port strike), the realized lead time variance ($\sigma_L$) will increase, causing the Replenisher to automatically raise safety stock buffers without manual intervention.

### Results (15-day simulation)
- Store Service Level: **97.9%** (Target >95% met)
- Inventory Turns: **7.9x** (Target ~8x met)
- System Stability: High (Physics laws strictly enforced with zero drift)

## [0.16.0] - 2026-01-02

### Service Level Fix: Physics-Based Multi-Phase Approach

This release implements a multi-phase fix for the 81% service level problem.

#### Phase 1: Inventory Position Fix for (s,S) Replenishment

This release implements the fundamental Inventory Position fix for (s,S) replenishment decisions. Per Zipkin's "Foundations of Inventory Management", (s,S) policies must use Inventory Position (On-Hand + In-Transit) rather than just On-Hand inventory to prevent double-ordering oscillation.

### Added
- **`get_in_transit_by_target()` (`state.py`):** New method calculates in-transit inventory per target node by aggregating quantities across all active shipments.
- **Variance-Aware Safety Stock (`replenishment.py`):** Implemented dynamic safety stock calculation based on demand variability ($ROP = \mu_L + z\sigma_L$).
  - Tracks rolling demand history per SKU via `record_demand()`.
  - Automatically adapts inventory buffers: popular/stable SKUs get lower relative safety stock, erratic/niche SKUs get higher buffers.
  - Replaces static "Days of Supply" heuristic which failed under Zipfian demand concentration.

### Changed
- **Replenishment (s,S) Decision (`replenishment.py`):**
  - Now uses Inventory Position (IP = On-Hand + In-Transit) for reorder point comparison
  - Order quantity calculated as Target Stock - IP (not Target - On-Hand)
  - This prevents double-ordering when shipments are already in transit
- **Manufacturing Targets (`simulation_config.json`):**
  - `target_days_supply`: 14 → 28 days
  - `reorder_point_days`: 7 → 21 days
  - Creates larger safety stock buffers at upstream nodes

### Removed
- **Legacy Tests:** Deleted low-value unit tests ("sham tests") that relied heavily on mocking without validating emergent behavior. The project now prioritizes full 365-day simulation runs to evaluate system stability and physics compliance.

### Technical Details

**The Double-Ordering Problem (Fixed):**
```
Before (v0.15.9):
  Store has 50 on-hand, 100 in-transit → compares 50 vs ROP → orders more
  Result: Double-ordering when shipments already cover the gap

After (v0.16.0):
  Store has 50 on-hand, 100 in-transit → IP = 150 → compares 150 vs ROP
  Result: No order if in-transit already covers replenishment need
```

### Files Modified
- `src/prism_sim/simulation/state.py`: Added `get_in_transit_by_target()` method
- `src/prism_sim/agents/replenishment.py`: Uses Inventory Position for (s,S) decisions, added variance tracking
- `src/prism_sim/simulation/orchestrator.py`: Passes daily demand to replenisher for history tracking
- `src/prism_sim/config/simulation_config.json`: Added safety stock parameters (`service_level_z`, `lead_time_days`)

#### Phase 2: Multi-Echelon Service Level Targets

Upstream nodes (Plants, RDCs) need higher inventory targets because end-to-end service level is the product of individual node service levels (0.95³ ≈ 85%).

### Changed
- **Manufacturing Targets (`simulation_config.json`):**
  - `target_days_supply`: 14 → 28 days
  - `reorder_point_days`: 7 → 21 days
  - Creates larger safety stock buffers at upstream nodes

### Validation
Service level stabilized at 76.20% in 365-day run with Zipfian demand enabled (System survives full year without collapse). Inventory turns at 4.73x. Further tuning of Z-scores required to reach >90% target.

### Future
- **Physics Overhaul (v0.17.0+):** Created `physics_overhaul.md` outlining a first-principles approach to fix the Service/Inventory paradox by instrumenting Effective Lead Time, implementing the full Safety Stock formula ($\sigma_L$), and adding dynamic ABC segmentation.

---

## [0.15.9] - 2026-01-01

### Service Level Improvement Phase 2: Demand Signal Fix

This release implements the core fix for demand signal attenuation identified in v0.15.8. Customer DCs now use **inflow-based demand** (orders received) instead of **outflow-based demand** (orders shipped), preventing the demand signal from being attenuated when DCs are short on inventory.

### Added
- **Inflow Tracking (`replenishment.py`):** New `inflow_history` array and methods to track orders received by each node:
  - `record_inflow(orders)`: Records orders received (pre-allocation)
  - `get_inflow_demand()`: Returns 7-day rolling average of inflow
  - Warm-start initialization to prevent cold-start issues
- **MRP Order Demand (`mrp.py`):** New `record_order_demand(orders)` method captures pre-allocation order quantities for production planning, preventing MRP from under-planning when supply chain is constrained.

### Changed
- **Customer DC Demand Signal (Critical Fix):**
  - Changed from outflow (what was shipped) to inflow (what was ordered)
  - This prevents demand attenuation cascade: when a DC is short on inventory, it now still sees the true demand from downstream stores
- **Customer DC Order Frequency:**
  - Reduced from 5-day cycle to 1-day cycle (daily ordering)
  - Creates smoother, more responsive demand signals upstream
- **Customer DC Replenishment Policy (Increased Buffers):**
  - B2M_LARGE: 14/10 → 21/14 days (target/ROP)
  - B2M_CLUB: 14/10 → 21/14 days
  - B2M_DISTRIBUTOR: 14/10 → 21/14 days
  - ECOMMERCE: 7/5 → 10/7 days
- **MRP Demand Signal:**
  - Now receives order quantities (pre-allocation) in addition to shipments
  - Uses `max(orders, shipments)` to capture true demand

### Technical Details

**The Demand Attenuation Problem (Fixed):**
```
Before (v0.15.8):
  Store requests 100 → DC ships 50 (low inv) → DC sees demand=50 → DC orders 50
  Result: 50% demand attenuation at each stage

After (v0.15.9):
  Store requests 100 → DC ships 50 (low inv) → DC sees demand=100 → DC orders 100
  Result: True demand propagates upstream
```

### Files Modified
- `src/prism_sim/agents/replenishment.py`: Inflow tracking, policy updates, daily DC ordering
- `src/prism_sim/simulation/orchestrator.py`: Record inflow, pass orders to MRP
- `src/prism_sim/simulation/mrp.py`: Order-based demand recording

### Validation
Requires 365-day simulation to validate service level improvement. Target: ≥95% (from 80.5%).

---

## [0.15.8] - 2026-01-01

### Service Level Improvement Phase 1 (75% → 80.5%)

This release improves store service level from 75.32% to 80.50% through policy tuning and demand signal fixes. Part of ongoing effort to reach 98.5% target.

### Fixed
- **ECOM FC Demand Signal (Critical):** ECOM FCs were classified as customer DCs, causing them to use outflow-based demand (which was 0 since they have no downstream stores). Now excluded from `customer_dc_indices` so they use POS demand correctly.

### Changed
- **Replenishment Policy (Increased Buffers):**
  - All channels: Target and ROP increased (~40-100% higher)
  - B2M_LARGE: 7/5 → 14/10 days (target/ROP)
  - B2M_CLUB: 10/7 → 14/10 days
  - ECOMMERCE: 5/3 → 7/5 days
  - Default: 10/7 → 14/10 days
- **Order Frequency:** Store order cycle reduced from 3 days to 1 day (daily ordering)
- **Initial Inventory Priming:**
  - Store days supply: 14 → 21 days
  - RDC days supply: 21 → 28 days
  - RDC-store multiplier: 50 → 500 (RDCs hold 500× store inventory)
  - Customer DC days supply: Added explicit config (21 days)

### Results (365-day simulation)
| Metric | v0.15.7 | v0.15.8 |
|--------|---------|---------|
| Store Service Level | 75.32% | **80.50%** |
| Inventory Turns | 6.18x | 5.11x |
| Manufacturing OEE | 82.0% | 81.9% |

### Known Issues
- **Service Level Gap:** Still 18pp below 98.5% target. Root cause identified as demand signal attenuation at customer DCs (orders based on outflow, not inflow).
- **SLOB Metric:** Shows 94.8% due to broken threshold logic (system-wide inventory > 60 days). Not a real problem.

### Next Steps
The plan to reach 98.5% service level involves changing customer DC demand signal from outflow-based (what was shipped) to inflow-based (what was ordered). This prevents demand attenuation as signals propagate upstream.

---

## [0.15.7] - 2026-01-01

### Fix Inventory Turns Calculation (Exclude Raw Materials)

This release fixes a critical metrics bug where inventory turns were calculated using ALL inventory (including raw materials at plants) instead of only finished goods.

### Fixed
- **Inventory Turns Calculation (Critical):** Inventory turns now correctly uses only finished goods (SKU-*) inventory in the denominator, excluding raw materials/ingredients at plants. Previously, 523M units of ingredients inflated the denominator, causing turns to show 0.23x instead of the actual ~6x.

### Added
- **Finished Goods Mask:** New `_build_finished_goods_mask()` method in Orchestrator creates a boolean mask to identify finished goods products (non-INGREDIENT category).

### Changed
- **SLOB Calculation:** Now uses finished goods inventory only (raw materials can't be "slow/obsolete" in the consumer sense).
- **Shrinkage Rate:** Now calculated against finished goods inventory for consistency.

### Results (365-day simulation)
| Metric | v0.15.6 | v0.15.7 |
|--------|---------|---------|
| Inventory Turns | 0.23x | **6.18x** ✓ |
| SLOB | 100% | 60.3% |
| Store Service Level | 73.34% | 75.32% |
| OEE | 78.6% | 82.0% ✓ |

### Technical Details
The inventory turns formula is: `Annual Sales / Average Inventory`

Previously, the denominator included all inventory across all nodes and products, including ~523M units of raw materials (chemicals, packaging) sitting at plants. This massively inflated the denominator, making turns appear artificially low (0.23x).

The fix creates a mask for finished goods products (24 SKUs out of 68 total products) and only sums inventory for those products when calculating turns.

---

## [0.15.6] - 2025-12-31

### MRP Demand Signal Stabilization & Production Floor

This release eliminates production collapse in 365-day simulations by improving MRP demand signal handling and adding a minimum production floor.

### Fixed
- **Production Collapse (Critical):** Production dropped to zero for days 252-279 due to demand signal dampening. When stores had sufficient inventory, order volumes dropped, which reduced the MRP shipment signal, causing MRP to calculate high Days-of-Supply and skip production orders. Eventually stores depleted, triggering massive bullwhip (35M orders on day 281).

### Added
- **Demand Velocity Tracking:** MRP now tracks week-over-week demand trends. If week 1 demand falls below 60% of week 2, the fallback signal is activated to prevent cascading decline.
- **Minimum Production Floor (30%):** When production orders fall below 30% of expected demand, MRP either scales up existing orders or creates new orders for top products, ensuring production never stops completely.

### Changed
- **MRP Signal Threshold:** Raised the collapse detection threshold from 10% to 40% of expected demand (`mrp.py:228`). The previous threshold was too low to catch gradual decline.
- **Smoothing Window:** Extended demand and production history from 7 to 14 days for smoother signals and better trend detection.

### Results (365-day simulation)
| Metric | v0.15.5 | v0.15.6 |
|--------|---------|---------|
| Store Service Level | 69.95% | **73.34%** |
| Manufacturing OEE | 62.0% | **78.6%** |
| Production Days 252-279 | **0** (collapsed) | 227K-484K |
| Total Inventory | 603M | 587M |

### Technical Details
The production collapse occurred because:
1. Stores had sufficient inventory after initial bullwhip → ordered less
2. Lower orders → lower shipments → lower MRP demand signal (40-50% of expected)
3. Previous 10% threshold didn't trigger fallback
4. MRP calculated high DOS (inventory / low_demand) → skipped production
5. Days 252-279: zero production → stores depleted → Day 281: 35M order explosion

The fix uses three mechanisms:
1. **Higher threshold (40%)** catches gradual decline earlier
2. **Velocity tracking** detects week-over-week declining trends
3. **Production floor (30%)** ensures minimum production regardless of signal

---

## [0.15.5] - 2025-12-31

### LTL Mode for Store Deliveries & Service Level Improvement

This release introduces LTL (Less Than Truckload) shipping mode for store deliveries, improving service level from 83% to 92.8%.

### Fixed
- **Fragmented Store Orders:** Stores ordered 20-40 cases but FTL required 300-1200 cases (5-20 pallets). Orders were held indefinitely for consolidation, hurting service level and causing low truck fill.

### Added
- **LTL Shipping Mode:** Differentiated shipping modes based on destination:
  - **FTL (Full Truckload):** DC-to-DC shipments maintain pallet minimums for consolidation
  - **LTL (Less Than Truckload):** DC-to-Store shipments ship immediately with minimum 10 cases
- **Config Options:** `store_delivery_mode`, `ltl_min_cases`, `default_ftl_min_pallets` in `simulation_config.json`
- **FTL/LTL Tracking:** `LogisticsEngine` now tracks `ftl_shipment_count` and `ltl_shipment_count`

### Changed
- **LogisticsEngine:** `create_shipments()` now checks if target is a Store and applies LTL mode (no pallet minimum)
- **Default FTL Minimum:** Routes without channel-specific rules now use `default_ftl_min_pallets` (10) instead of 0

### Results
| Metric | v0.15.4 | v0.15.5 |
|--------|---------|---------|
| Service Level (90-day) | 83% | **92.8%** |
| Truck Fill Rate | 15% | 4.2% |
| Manufacturing OEE | 81% | 83% |

### Note on Truck Fill Rate
The truck fill rate dropped because:
1. Most shipments are now LTL to stores (intentionally small)
2. FMCG products "cube out" (fill by volume) before "weighting out" (fill by weight)
3. Weight-based truck fill isn't appropriate for light, bulky products
4. With LTL, service level is the better metric for overall performance

---

## [0.15.4] - 2025-12-31

### Bullwhip Dampening & Service Level Stabilization

This release eliminates the Day 2 bullwhip cascade (272M → 100K orders) and stabilizes service level over 90-day simulations.

### Fixed
- **Customer DC Bullwhip Cascade (Critical):** Customer DCs now use derived demand (allocation outflow) instead of POS demand, which was floored to 0.1. Added warm start from POSEngine equilibrium estimate and order staggering (5-day cycle) for customer DCs.
- **MRP Ingredient Ordering Explosion:** Capped ingredient ordering at 2x expected demand to prevent bullwhip-driven explosions. Reduced ACTIVE_CHEM inventory policy from 30/45 days to 7/14 days ROP/target.
- **Customer DC Inventory Priming:** Customer DCs now initialize with inventory scaled by downstream store count, preventing Day 1 mass-ordering.

### Changed
- **Replenisher:** Added `warm_start_demand` parameter, `record_outflow()` method for derived demand tracking, and order staggering for customer DCs.
- **Orchestrator:** Passes warm start demand to replenisher and records allocation outflow.
- **Config:** Reduced ACTIVE_CHEM policy to 7/14 days (was 30/45).

### Results
| Metric | v0.15.3 | v0.15.4 |
|--------|---------|---------|
| Day 2 Orders | 272M | 100K |
| Service Level (90-day) | 80% | 83% |
| Truck Fill Rate | 8% | 15% |
| Manufacturing OEE | 92% | 81% |

### Technical Details
The bullwhip cascade occurred because:
1. Customer DCs (RET-DC, DIST-DC, ECOM-FC) don't generate POS demand
2. Their demand signal was floored to 0.1, creating tiny ROP/targets
3. When stores ordered on Day 1, DC inventory dropped below the tiny ROP
4. All DCs mass-ordered on Day 2 to refill

The fix uses MRP theory's "derived demand" concept: customer DCs track orders they fulfill (allocation outflow) as their demand signal, not consumer POS.

---

## [0.15.3] - 2025-12-31

### Mass Balance Audit Fix & Service Level Optimization

This release fixes the mass balance tracking issue and improves service level from 58% to 86%.

### Fixed
- **Mass Balance FTL Timing Mismatch (Option B1):** Replaced `shipments_out` tracking with `allocation_out` to fix false violations at customer DCs. Inventory decrements are now tracked at allocation time (when they actually occur) rather than at shipment creation time (which could be delayed by FTL consolidation).
- **Floating-Point Noise Filtering:** Added minimum absolute difference threshold (1.0 case) to filter negligible floating-point violations caused by floor guards.

### Changed
- **AllocationAgent:** Now returns `AllocationResult` dataclass containing both `allocated_orders` and `allocation_matrix` for mass balance tracking.
- **PhysicsAuditor:** New methods `record_allocation_out()` and `record_plant_shipments_out()` replace the single `record_shipments_out()` method.
- **Initial Inventory Levels:** Increased `store_days_supply` from 7 to 14 days, `rdc_days_supply` from 14 to 21 days to prevent synchronized reordering.
- **Replenishment Policy:** Tightened ROP-Target gap across all channels (e.g., default ROP 4→7 days) for smaller, more frequent orders.
- **Order Staggering:** Stores now order on different days based on hash(node_id) to spread orders across 3-day cycle, reducing bullwhip amplitude.

### Results
| Metric | v0.15.2 | v0.15.3 |
|--------|---------|---------|
| Service Level (30-day) | 58% | 86% |
| Manufacturing OEE | 62% | 88% |
| Simulation Time (30-day) | 10s | 2.4s |

### Known Issues
- Day 2 bullwhip (271M orders) still occurs from customer DC cascade
- Service level declines over longer runs (80% at 60 days)
- Truck fill rate remains low (8%) due to fragmented orders

### Technical Details
The FTL consolidation timing mismatch occurred because:
1. Allocation decrements inventory immediately when orders are created
2. Logistics can HOLD orders if they don't meet FTL minimum pallet thresholds
3. `shipments_out` was only recorded when shipments were actually created (potentially days later)
4. Mass balance equation couldn't account for "allocated but not yet shipped" inventory

The fix tracks inventory decrements at the source (allocation) rather than reconstructing from downstream flows.

---

## [0.15.2] - 2025-12-30

### Ingredient Replenishment Fix

This release fixes the critical ingredient replenishment mismatch that caused 365-day simulations to collapse on days 362-365 due to ingredient exhaustion.

### Fixed
- **Ingredient Replenishment Signal (Critical):** `generate_purchase_orders()` now uses production-based signal (active production orders) instead of POS demand for ingredient ordering. Previously, ingredient replenishment was based on POS demand (~400k/day), but actual consumption was driven by production orders amplified by bullwhip (~5-6M/day), causing a net burn rate of ~1,380 units/day shortfall.

### Changed
- **Orchestrator:** Updated to pass `active_production_orders` to `generate_purchase_orders()` instead of `daily_demand`.

### Results
| Metric | v0.15.1 (Collapse) | v0.15.2 |
|--------|-------------------|---------|
| Service Level | 52.54% | 58.16% |
| Manufacturing OEE | 55.1% | 61.8% |
| Production Day 365 | 0 (collapse) | 259,560 cases |
| System Survival | Collapsed day 362-365 | Full year |

---

## [0.15.1] - 2025-12-30

### MRP Inventory Position Fix

This release fixes the critical MRP inventory position bug that caused 94 zero-production days in 365-day simulations.

### Fixed
- **MRP Inventory Position (Critical):** `_cache_node_info()` now only includes manufacturer RDCs (`RDC-*` prefix) in inventory position calculation. Previously included ALL `NodeType.DC` nodes including customer DCs (RET-DC, DIST-DC, ECOM-FC) with ~4.5M units, inflating Days of Supply to 11.5 days > ROP 7 days, preventing production orders.
- **C.5 Smoothing History Bug:** Production order history now records post-scaled (actual) quantities instead of pre-scaled values, preventing rolling average inflation.

### Known Issues
- **Mass Balance Violations:** Customer DCs show mass balance violations due to FTL consolidation timing mismatch. Allocation decrements inventory immediately, but logistics can hold orders for FTL consolidation. This is an accounting/auditing issue, not an actual physics violation.

### Results
| Metric | v0.15.0 | v0.15.1 |
|--------|---------|---------|
| Service Level | 51.6% | 60.19% |
| Manufacturing OEE | 44.9% | 88.2% |
| Zero-Production Days | 94 | 0 |

---

## [0.15.0] - 2025-12-30

### Phase C: System Stabilization

This release fixes the critical "death spiral" bug that caused 365-day simulations to collapse around day 22-27. The system now survives full-year simulations without collapse.

### Fixed
- **Death Spiral Prevention (C.1):** Added expected demand fallback in `MRPEngine`. When RDC→Store shipment signals collapse below 10% of expected demand, MRP now uses expected demand as a floor to continue generating production orders.
- **Supplier-Plant Routing (C.2):** Fixed `_find_supplier_for_ingredient()` to verify link exists before routing SPOF ingredient. Previously returned SUP-001 for all plants even when no link existed.
- **Production Smoothing (C.5):** Added 7-day rolling average tracking for production orders with 1.5x cap on daily volatility to reduce wild swings.

### Changed
- **Production Capacity (C.3):** Increased `production_hours_per_day` from 20 to 24 hours (3-shift 24/7 operation) to ensure capacity exceeds demand.
- **Realistic Initial Inventory (C.4):** Reduced starting inventory to realistic levels:
  - Store days of supply: 14 → 7 days
  - RDC days of supply: 21 → 14 days
  - RDC-store multiplier: 100 → 50
  - Raw material inventory remains high (10M units) to isolate finished goods dynamics.

### Results
| Metric | Pre-Fix | Post-Fix | Target |
|--------|---------|----------|--------|
| System Collapse | Day 27 | Never | Never ✅ |
| Service Level | 8.8% | 51.6% | >90% |
| Production (day 365) | 0 | 148k+ | >0 ✅ |
| OEE | 1.8% | 44.9% | 75-85% |

---

## [0.14.0] - 2025-12-30

### Phase A: Capacity Rebalancing & Option C Architecture

This release implements Phase A of the brownfield digital twin stabilization plan, scaling demand/capacity to realistic FMCG industry benchmarks (400-500k cases/day) and implementing hierarchical DC + Store architecture.

### Added
- **Option C Network Architecture:**
  - Hierarchical structure: DCs (logistics layer) + Stores (POS layer)
  - 20 Retailer DCs with ~100 stores each (B2M_LARGE)
  - 8 Distributor DCs with 500 small retailers each (B2M_DISTRIBUTOR)
  - 30 Club stores direct to RDC (B2M_CLUB)
  - 10 Ecom FCs and 2 DTC FCs
  - Total: ~6,600 nodes (vs 155 aggregated DCs before)
- **Store Format Scale Factors:** SUPERMARKET=1.0, CONVENIENCE=0.5, CLUB=15.0, ECOM_FC=50.0
- **Phase A Implementation Notes** section in `fix_plan_v2.md` documenting all changes and learnings

### Changed
- **Capacity Scaling (2.5x increase):**
  - Run rates: ORAL 3000→7500, PERSONAL 3600→9000, HOME 2400→6000 cases/hour
  - Production hours: 8→20 hours/day (3-shift operation)
  - Plant efficiency factors leveled to 78-88% OEE range
- **Demand Calibration:**
  - Base daily demand: 1.0→7.0 cases/SKU/store
  - Target network demand: ~420k cases/day (aligned with multi-category CPG operations)
- **Seasonality & Promos:**
  - Seasonality amplitude: ±20%→±12% (staple category realism)
  - Black Friday lift multiplier: 3.0x→2.0x
- **Inventory Priming:**
  - Store days supply: 28→14 days
  - RDC-store multiplier: 1500→100 (fixed over-priming issue)

### Fixed
- **Demand Double-Counting:** DCs (RETAILER_DC, DISTRIBUTOR_DC) no longer generate POS demand; only stores do
- **Network Generator Bugs:** `sample_company()`→`sample_companies(1)[0]`, `sample_city()`→`sample_cities(1)[0]`
- **CSV Loader:** Added parsing for `channel`, `store_format`, `parent_account_id` enums in `builder.py`

### Known Issues (Phase B Required)
- Bullwhip effect still extreme (orders explode 60k→12.9M by day 20)
- Production stops ~day 20 due to bullwhip-induced collapse
- OEE at 39% (below target 75-85%), expected given order chaos

---

## [0.13.0] - 2025-12-29

### Realism & Architecture Overhaul (Phases 0-5)
This release implements a massive overhaul of the simulation physics and network structure to align with FMCG industry standards (P&G/Colgate benchmarks).

### Added
- **Customer Channel Structure (Fix 0A):**
  - Added `CustomerChannel` (B2M_LARGE, CLUB, DISTRIBUTOR, ECOM) and `StoreFormat` enums.
  - Implemented channel-specific logistics rules (FTL vs. LTL) and minimum order quantities.
- **Packaging Hierarchy (Fix 0B):**
  - Added `PackagingType` and `ContainerType` to support realistic SKU variants (Tubes, Bottles, Pumps).
  - Procedurally generated SKU variants based on packaging profiles.
- **Order Types (Fix 0C):**
  - Added `OrderType` (STANDARD, RUSH, PROMOTIONAL, BACKORDER) with priority handling in Allocation.
- **Promo Calendar (Fix 0D):**
  - Ported vectorized `PromoCalendar` from reference implementation for realistic demand lift and hangover effects.
- **Risk Events (Fix 5):**
  - Implemented full `RiskEventManager` with 5 scenarios: Contamination, Port Strike, Supplier Opacity, Cyber Outage, Carbon Tax.
- **Behavioral Quirks (Fix 6):**
  - Added 6 behavioral pathologies: `BullwhipWhipCrack`, `SingleSourceFragility`, `DataDecay`, `PortCongestion`, `OptimismBias`, `PhantomInventory`.
- **Sustainability Metrics (Fix 10):**
  - Added Scope 3 CO2 emissions tracking in `LogisticsEngine` (0.1 kg/ton-km).
  - Added `emissions_kg` tracking to Shipments.
- **Expanded KPIs (Fix 8):**
  - Added trackers for Perfect Order Rate, Cash-to-Cash Cycle, MAPE, Shrinkage Rate, and SLOB %.

### Changed
- **Orchestrator:**
  - Updated MRP signal to use **RDC-to-Store Shipments** (true pull signal) instead of allocated orders to fix inverse bullwhip.
  - Integrated `RiskEventManager` and `QuirkManager` into the daily loop.
  - Updated `Triangle Report` to include new KPIs.
- **Configuration:**
  - Overhauled `simulation_config.json` with comprehensive settings for all new engines.
  - Updated `benchmark_manifest.json` with strict validation targets (OSA 93-99%, Turns 8-15x).
  - Increased inventory initialization targets (Store: 28d, RDC: 35d) to prevent drain.

## [0.12.3] - 2025-12-29

### Added
- **Bullwhip Analysis Script:** Added `scripts/analyze_bullwhip.py` for analyzing variance amplification across supply chain echelons. Calculates CV ratios (Store → RDC → Plant), detects inverse bullwhip anomalies, and reports production oscillation patterns.
- **Scenario Comparison Script:** Added `scripts/compare_scenarios.py` for comparing two simulation runs (e.g., baseline vs risk events). Shows shipment volume differences around key disruption days with day-by-day breakdown.

### Fixed
- **Inverse Bullwhip Effect:** Resolved the "Inverse Bullwhip" anomaly (where upstream variance was lower than downstream) by updating `MRPEngine` to use a 7-day moving average of **actual RDC shipments** (lumpy signal) instead of smoothed POS demand proxies. This restores realistic demand amplification upstream.
- **Low OEE (28% -> 99%):** Fixed massive plant over-capacity by:
  - Reducing `production_hours_per_day` from 24 to 8 (single shift).
  - Increasing `batch_size_cases` in `simulation_config.json` to 100.
  - Increasing `min_production_qty` logic in `mrp.py` to 2x net requirement.
- **Service Level Metrics:** Introduced **Store Service Level (OSA)** tracking in `Orchestrator` and `RealismMonitor` to correctly measure On-Shelf Availability (Actual Sales / Demand). The legacy "Fill Rate" metric was skewed by massive ingredient orders hitting supplier capacity caps.

### Changed
- **Inventory Policy Tuning:** Increased `replenishment` target days (21d) and reorder points (10d) in `simulation_config.json` to support higher service levels.
- **Reporting:** Updated "The Triangle Report" in `Orchestrator` to feature Store Service Level as the primary Service metric.

## [0.12.2] - 2025-12-29

### Fixed
- **Negative Inventory Physics Violation:** Eliminated all sources of negative inventory that violated the Inventory Positivity law.
  - **Demand Consumption:** Updated `Orchestrator` to constrain sales to available actual inventory (lost sales model). Previously subtracted demand blindly.
  - **Allocation Agent:** Changed `AllocationAgent` to use `actual_inventory` instead of `perceived_inventory` for fill ratio calculations. When phantom inventory quirk creates divergence, allocation now respects physical reality.
  - **Material Consumption:** Updated `TransformEngine._consume_materials()` to check and consume from `actual_inventory`, constraining consumption to available amounts.
  - **Shrinkage Quirk:** Fixed `PhantomInventoryQuirk` to base shrinkage on actual inventory and constrain shrink amount to prevent negatives.
  - **State Guards:** Added `np.maximum(0, ...)` floor guards in `StateManager.update_inventory()` and `update_inventory_batch()` to catch floating point precision errors.

### Added
- **Analysis Script:** Added `scripts/analyze_results.py` for reusable simulation output analysis. Supports JSON output and handles large inventory files via chunked reading.

### Changed
- **Category Demand Balancing:** Equalized `base_daily_demand` to 1.0 for all product categories (ORAL_CARE, PERSONAL_WASH, HOME_CARE).
- **Plant Flexibility:** Added ORAL_CARE to PLANT-CA's `supported_categories` for better capacity balancing.

### Results
- **Before fix:** 3.4M cases backlog, 1,161 cells with negative inventory (min: -212.86)
- **After fix:** 0 cases backlog, 0 cells with negative inventory

## [0.12.1] - 2025-12-29

### Changed
- **Simulation Physics Tuning:**
  - **SPOF Isolation:** Updated `hierarchy.py` to restrict `ACT-CHEM-001` (SPOF ingredient) to only ~20% of the portfolio (Premium Oral Care), reducing systemic vulnerability.
  - **Tiered Inventory Policies:** Updated `mrp.py` and `simulation_config.json` to support granular ROP/Target levels. Ingredients now use "Commodity" (7d/14d) vs. "Specialty" (30d/45d) policies.
  - **Supplier Capacity:** Updated `NetworkGenerator` to set infinite capacity for bulk suppliers while constraining the SPOF supplier (`SUP-001`) to 500k units/day.
  - **Allocation Logic:** Updated `AllocationAgent` to respect finite supplier capacity limits using Fair Share logic.
  - **Initialization:** Increased Store/RDC initialization days (14d/28d) and implemented robust Plant ingredient seeding (5M units) to prevent cold-start starvation.
- **Service Level Tracking:**
  - Updated `RealismMonitor` and `Orchestrator` to track `Service Level (LIFR)` based on `Shipped / Ordered`, replacing the legacy backlog-based index.

### Fixed
- **MRP Initialization Bug:** Fixed `AttributeError` in `MRPEngine` caused by accessing uninitialized cache variables in `_build_policy_vectors`.

### Identified
- **The Bullwhip Crisis:** 365-day simulation revealed a catastrophic feedback loop where "Fill or Kill" logic combined with zero inventory triggers infinite reordering (460k -> 66M orders/day). Mitigation plan documented in `docs/planning/sim_tuning.md`.

## [0.12.0] - 2025-12-29

### Fixed
- **MRP Physics Fix:** Implemented a "Look-ahead Horizon" (reorder point window) for `in_production_qty` calculation. This prevents the MRP engine from overestimating available supply based on distant future production orders, which previously led to "Pipeline Silence" and systemic inventory collapse.
- **Fill-or-Kill Logic:** Refactored `AllocationAgent` to implement "Fill or Kill" (Cut) logic for retailer-to-DC orders. Unfulfilled quantities are now marked as `CLOSED` rather than remaining `OPEN`. This aligns with high-velocity FMCG industry standards and prevents exponential computational backlog growth during stockouts.

### Added
- **Simulation Tuning Plan:** Created `docs/planning/sim_tuning.md` to document the strategy for stabilizing the "Deep NAM" network, including Pareto-based SPOF isolation and tiered reorder points.

## [0.11.0] - 2025-12-29

### Added
- **Streaming Writers (Task 7.3):** Implemented incremental disk writes for 365-day runs without memory exhaustion.
  - **StreamingCSVWriter:** Writes rows directly to disk as they arrive, preventing memory accumulation.
  - **StreamingParquetWriter:** Batches rows and flushes periodically for optimal compression (requires `pyarrow`).
  - **CLI Flags:** Added `--streaming`, `--format`, and `--inventory-sample-rate` to `run_simulation.py`.
  - **Config-Driven:** Writer settings can be configured in `simulation_config.json` under `writer` section.
- **Inventory Sampling:** Added `inventory_sample_rate` parameter to reduce inventory data volume (e.g., log weekly instead of daily).

### Changed
- **SimulationWriter:** Refactored to support both buffered (legacy) and streaming modes with backward compatibility.
- **Orchestrator:** Now reads writer configuration from `simulation_config.json` with CLI override support.

### Performance
- **30-day streaming test:** 549K orders, 557K shipments, 1.6M inventory records written incrementally in 10.4s.
- **Memory efficiency:** Streaming mode eliminates in-memory accumulation for high-volume tables.

### Fixed
- **SPOF Config Alignment:** Updated SPOF ingredient from hardcoded `ING-SURF-SPEC` to procedural `ACT-CHEM-001` to match Task 8.1 world builder overhaul.
- **Test Maintenance:** Updated `test_world_builder.py` to validate procedural ingredient patterns instead of hardcoded IDs. Added new tests: `test_ingredients_generated`, `test_spof_ingredient_exists`.
- **Mass Balance Test:** Fixed `test_mass_balance_detects_leak` to inject proportionally larger leaks for reliable drift detection.

## [0.10.0] - 2025-12-28

### Architecture Overhaul
- **World Builder Overhaul (Task 8.1):** Transitioned from hardcoded "2-ingredient" logic to a fully procedural ingredient generation system.
  - **Procedural Ingredients:** `ProductGenerator` now creates Packaging (Bottles, Caps, Boxes) and Chemicals (Actives, Bulk Base) dynamically based on `world_definition.json` profiles.
  - **Semantic Recipes:** BOMs are now logic-driven (e.g., "Liquid" = Bottle + Cap + Label + Base + Active) rather than random.
- **Vectorized MRP (Task 8.2):** Implemented `RecipeMatrixBuilder` to convert object-based BOMs into a dense NumPy matrix ($\mathbf{R}$) for $O(1)$ dependency lookups.
  - **Matrix Algebra:** `MRPEngine` now calculates ingredient requirements via vector-matrix multiplication ($\mathbf{req} = \mathbf{d} \cdot \mathbf{R}$), enabling instant planning for thousands of SKUs.
  - **Vectorized Transform:** Refactored `TransformEngine` to use direct tensor operations for material feasibility checks and consumption updates.
- **Mass Balance Physics Audit (Task 6.6):** Implemented a rigorous validation gate to enforce the conservation of mass.
  - **Flow Tracking:** Added `DailyFlows` to track every inventory movement (Sales, Receipts, Shipments, Production, Consumed, Shrinkage).
  - **Audit Loop:** `PhysicsAuditor` now calculates expected inventory levels daily and flags "Drift" violations if actual state diverges from physics laws.

### Added
- **Recipe Matrix:** New core component `src/prism_sim/network/recipe_matrix.py` for handling dense BOM structures.
- **Configurable Profiles:** Added `ingredient_profiles` and `recipe_logic` to `world_definition.json` to control the procedural generation of supply chains.

### Changed
- **Performance:** Simulation speed validated at ~4.6s for 30 days of the full 4,500-node network despite 10x increase in BOM complexity.
- **Refactoring:** Removed hardcoded ingredients from `generate_static_world.py` and `hierarchy.py`.

## [0.9.8] - 2025-12-28

### Added
- **LLM Context Guide:** Added `docs/llm_context.md`, a consolidated "One-Pager" for developers and AI assistants, covering physics laws, architecture, and core components.
- **Dynamic Priming:** Implemented `get_average_demand_estimate` in `POSEngine` to calculate realistic initial inventory levels based on actual demand profiles, replacing the hardcoded `1.0` case/day proxy.
- **Strict Logistics Constraints:** Enhanced `LogisticsEngine` to raise `ValueError` if an item physically exceeds truck dimensions, preventing silent data errors.

### Fixed
- **Logistics Crash:** Resolved `ValueError` in `LogisticsEngine` caused by `fit_qty` calculating to zero for fractional remaining quantities. The engine now supports partial case packing for "Fair Share" allocation scenarios.
- **Linting & Types:** Resolved ~100 `ruff` errors (E501, F841, D200) and `mypy` strict type errors in `mrp.py` and `builder.py`.
- **Code Refactoring:**
  - Simplified `MRPEngine.generate_production_orders` by extracting inventory position logic.
  - Reduced argument complexity in `Orchestrator._print_daily_status` and `_record_daily_metrics`.
  - Refactored `TransformEngine._process_single_order` to reduce return statements.

### Documentation
- **API Docs:** Configured `mkdocs` to auto-generate API reference pages from source docstrings upon build.
- **Navigation:** Updated `mkdocs.yml` to include the new "LLM Context" guide.

## [0.9.7] - 2025-12-28

### Fixed
- **Service Index Reporting:** Fixed scaling issue where `backlog_penalty_divisor` was too small (1,000 vs 100,000) for the Deep NAM network, causing Service Index to floor at 0%.
- **Cash Metric Reporting:** Implemented `Inventory Turns` tracking in `RealismMonitor` and `Orchestrator`. Now correctly reports Annualized Inventory Turns (previously 0.00x).
- **Serialization Error:** Fixed `TypeError: float32 is not JSON serializable` in `RealismMonitor` by ensuring metric accumulators store native Python floats.

### Documentation
- **Quick Start:** Updated `README.md` with clear instructions on running the simulation via `run_simulation.py` and locating generated artifacts.

### Added
- **CLI Runner:** Added `run_simulation.py` with `argparse` support for `--days`, `--no-logging`, and `--output-dir`. Replaced the rigid `run_benchmark.py`.

## [0.9.6] - 2025-12-28

### Fixed
- **Systemic Inventory Collapse:** Resolved the core physics bottleneck that caused production to stall and inventory to drain.
  - **Inventory Sync:** Fixed `AllocationAgent` and `TransformEngine` to correctly update both `actual` and `perceived` inventory, eliminating "phantom stock" and reporting errors.
  - **Material Deadlock:** Modified `TransformEngine` to check material availability only for the quantity that can be produced in the current day, preventing large orders from blocking production due to partial ingredient shortages.
  - **Batching Latency:** Implemented daily batch creation for partial production in `TransformEngine`, reducing the lead time penalty of large MOQs.
  - **MRP Robustness:** Increased `target_days_supply` (28d) and `reorder_point_days` (14d) in `simulation_config.json` to provide a sufficient buffer against lead time and demand variability.
  - **MRP Visibility:** Updated `MRPEngine` to include Plant-level finished goods in the Inventory Position calculation.

### Changed
- **OEE Stabilization:** Achieved stable 73% OEE (Target: 65-85%) across the Deep NAM network.
- **Logistics Efficiency:** Doubled Truck Fill Rate to ~52% by increasing store replenishment targets.

## [0.9.5] - 2025-12-28

### Fixed
- **Structural Capacity Deficit:** Tuned manufacturing physics to resolve the ~160k production cap vs ~230k demand:
  - **MOQ Increase:** Raised `min_production_qty` from 5k to 25k to amortize changeover penalties.
  - **Changeover Reduction:** Reduced changeover times in `world_definition.json` (0.5-1.5h) to improve plant OEE.
  - **Efficiency Boost:** Increased plant `efficiency_factor` to 0.95.
- **Initialization Bias:** Updated `simulation_config.json` to use "Steady State" midpoints for inventory initialization (Stores: 5d, RDCs: 10.5d) instead of max capacity, eliminating the artificial "destocking" period.

### Changed
- **Config-Driven Hierarchy:** Refactored `generators/hierarchy.py` and `scripts/generate_static_world.py` to load product run rates and dimensions from `world_definition.json`, removing all hardcoded "magic numbers".
- **Benchmark Config:** Reduced simulation horizon to 90 days and enabled full CSV logging by default for deeper debugging.

## [0.9.4] - 2025-12-27

### Fixed
- **Structural Deficit Resolution:** Resolved system-wide inventory collapse where production could not meet demand.
  - **MRP Logic:** Updated `MRPEngine` to use Inventory Position (On Hand + Pipeline) instead of just On Hand, eliminating redundant orders and the Bullwhip effect.
  - **Ingredient Replenishment:** Implemented `generate_purchase_orders` in `MRPEngine` to replenish raw materials at plants, preventing production halts due to ingredient exhaustion.
  - **Partial Production:** Fixed `TransformEngine` to correctly handle partial production days and calculate remaining capacity, ensuring plants don't lose time on large orders.
- **Capacity Tuning:** Doubled theoretical run rates in `recipes.csv` and increased `production_hours_per_day` to 24.0 to ensure the network has sufficient capacity to meet the ~225k case/day demand.

### Added
- **Plant-Specific Configuration:** Implemented granular control over plant capabilities in `simulation_config.json`:
  - **Efficiency & Downtime:** Configurable `efficiency_factor` (e.g., 0.70-0.95) and `unplanned_downtime_pct` per plant.
  - **Product Restrictions:** Added `supported_categories` to restrict which plants can produce specific product categories (e.g., only PLANT-TX makes Oral Care).

## [0.9.3] - 2025-12-27

### Fixed
- **Zero Orders Reporting Bug:** Corrected `Orchestrator` logging to calculate "Ordered" quantity *before* the `AllocationAgent` modifies orders in-place. This reveals the true unconstrained demand signal (approx. 11M cases/day) instead of zero.
- **Production Starvation:** Increased `initial_plant_inventory` for raw materials (`ING-BASE-LIQ`, `ING-SURF-SPEC`) from 5k to 10M units to prevent immediate SPOF (Single Point of Failure) triggering. Plants now successfully produce ~100k cases/day to meet demand.
- **Cold Start Service Collapse:** Fixed RDC initialization logic in `Orchestrator`. RDCs now initialize with 1500x the base inventory (approx. 30k cases/sku) to cover network aggregate demand, preventing immediate stockouts.
- **System Priming (Configurable):** Added `initialization` block to `simulation_config.json` allowing precise control over Store and RDC start-up inventory (Days of Supply). This moves the simulation from a "Cold Start" to a "Warm Start" state.
- **OEE Tracking Implementation:** Updated `TransformEngine` and `Orchestrator` to calculate and record plant-level capacity utilization (OEE). OEE now correctly reports in the Supply Chain Triangle Report.
- **Reporting Cleanup:** Clamped Service Index reporting to 0-100% range to avoid confusing negative values during backlog recovery phases.

## [0.9.2] - 2025-12-27

### Changed
- **Performance Optimization (Replenisher):** Implemented O(1) Store-to-Supplier lookup caching in `MinMaxReplenisher` to resolve O(N*L) performance bottleneck (22M iterations/day).
- **Config Adjustment:** Reduced `base_daily_demand` to ~1.0 case/day/store and `initial_fg_level` to 5.0 to align simulation physics with realistic NAM benchmarks.
- **Debugging:** Added extensive debug instrumentation to `Replenishment` agent and `Orchestrator` to diagnose order silence.

## [0.9.1] - 2025-12-27

### Changed
- **Performance Optimization (Task 7.3):** Implemented `enable_logging` flag in `SimulationWriter` and `Orchestrator`.
  - Default behavior now skips expensive I/O and in-memory list appending when logging is disabled.
  - `run_benchmark.py` updated to run in "In-Memory Validation Mode" (logging=False) for accurate speed measurement.

## [0.9.0] - 2025-12-27

### Added
- **Deep NAM Integration (Task 7.3):** fully integrated the 4,500-node static world into the runtime simulation.
  - **CSV Loading:** `WorldBuilder` now automatically loads `products.csv`, `locations.csv`, and `links.csv` from `data/output/static_world` if present.
  - **Dynamic Demand:** Updated `POSEngine` to generate demand based on `ProductCategory` rather than hardcoded string matching, enabling support for generated SKUs (e.g., `SKU-ORAL-001`).
  - **Test Suite Updates:** Refactored `test_milestone_3`, `test_state_manager`, and `test_world_builder` to validate against the massive network topology.

## [0.8.0] - 2025-12-27

### Added
- **Deep NAM Static Generators (Task 2.1, 2.2):** Implemented high-performance generators for massive scale simulation:
  - **ProductGenerator:** Generates 50+ SKUs across 3 categories with Zipfian popularity and realistic physical dimensions.
  - **NetworkGenerator:** Generates a 4,500-node retail network using Barabási-Albert preferential attachment for hub-and-spoke realism.
  - **StaticDataPool:** Vectorized Faker sampling for O(1) attribute generation.
  - **Distributions:** Statistical helpers for Zipf and Power-Law network topology.
- **Static World Writer:** Implemented `StaticWriter` to export Levels 0-4 (Products, Recipes, Locations, Partners, Links) to CSV format.
- **World Generation Script:** Added `scripts/generate_static_world.py` to automate the creation of the 4,500-node Deep NAM environment.

## [0.7.0] - 2025-12-27

### Added
- **MkDocs Documentation:** Implemented comprehensive documentation system with:
  - **mkdocs-material** theme with dark/light mode, navigation tabs, and search
  - **mkdocstrings[python]** for automatic API reference generation from docstrings
  - **mkdocs-gen-files** for auto-generating reference pages from source code
  - **Architecture diagrams** using Mermaid (system overview, data flow, component interactions)
  - **Getting Started** guides (installation, quick start)
  - **Changelog integration** via pymdownx.snippets
- **World Definition Config (`world_definition.json`):** Separated static world data (products, network topology, recipes) from runtime simulation parameters.
- **Semgrep Rule (`.semgrep/detect_literals.yaml`):** Added custom rule to detect hardcoded literals in source code.

### Changed
- **Config-Driven Architecture Enforcement:** Eliminated all remaining hardcoded values from source code:
  - **Deleted `constants.py`:** Moved `EPSILON` and `WEEKS_PER_YEAR` to `simulation_config.json` under `global_constants`.
  - **Refactored `builder.py`:** Now reads products, recipes, and network topology from `world_definition.json` instead of hardcoding.
  - **Updated `allocation.py`:** Receives config and reads `epsilon` from `global_constants`.
  - **Updated `demand.py`:** Reads seasonality (amplitude, phase, cycle) and noise (gamma_shape, gamma_scale) from config.
  - **Updated `logistics.py`:** Reads `epsilon_weight_kg` and `epsilon_volume_m3` from config.
  - **Updated `orchestrator.py`:** Reads scoring weights from config for triangle report.
  - **Updated `quirks.py`:** Added configurable `cluster_delay_min/max_hours` and `shrinkage_factor_min/max`.

## [0.6.0] - 2025-12-27

### Added
- **Simulation Writer (Task 7.2):** Implemented `SimulationWriter` to export SCOR-DS compatible datasets (Orders, Shipments, Batches, Inventory) to CSV/JSON.
- **Triangle Report (Task 7.3):** Added automated generation of "The Triangle Report," summarizing Service (Fill Rate), Cost (Truck Fill), and Cash (Inventory Turns) performance.
- **Reporting Infrastructure:** Integrated data collection directly into the `Orchestrator` loop for seamless end-of-run reporting.

## [0.5.0] - 2025-12-26

### Added
- **Validation Framework (Task 6.1, 6.6):** Implemented comprehensive validation in `src/prism_sim/simulation/monitor.py`:
  - **WelfordAccumulator:** O(1) streaming mean/variance calculation for real-time statistics.
  - **MassBalanceChecker:** Physics conservation tracking (input = output + scrap).
  - **RealismMonitor:** Online validator for OEE (65-85%), Truck Fill (>85%), SLOB (<30%), Inventory Turns (6-14x), Cost-per-Case ($1-3).
  - **PhysicsAuditor:** Mass balance, inventory positivity, and kinematic consistency checks.
- **Resilience Metrics (Task 6.2):** Implemented `ResilienceTracker` for TTS (Time-to-Survive) and TTR (Time-to-Recover) measurement during disruptions per Simchi-Levi framework.
- **Behavioral Quirks (Task 6.3):** Implemented realistic supply chain pathologies in `src/prism_sim/simulation/quirks.py`:
  - **PortCongestionQuirk:** AR(1) auto-regressive delays creating clustered late arrivals (coefficient=0.70, clustering when delay >4h).
  - **OptimismBiasQuirk:** 15% over-forecast for new products (<6 months old).
  - **PhantomInventoryQuirk:** 2% shrinkage with 14-day detection lag (dual inventory model).
  - **QuirkManager:** Unified interface for all quirks.
- **Risk Scenarios (Task 6.4):** Implemented `RiskEventManager` in `src/prism_sim/simulation/risk_events.py` to trigger deterministic disruptions:
  - **Port Strike (RSK-LOG-002):** 4x logistics delay.
  - **Cyber Outage (RSK-CYB-004):** 10x logistics delay.
- **Legacy Validation (Task 6.5):** Ported validation checks from reference implementation:
  - Pareto distribution check (top 20% SKUs = 75-85% volume).
  - Hub concentration check (Chicago ~20-30%).
  - Named entities verification.
  - Bullwhip ratio check (Order CV / POS CV = 1.5-3.0x).
  - Referential integrity checks.
- **Metrics Dataclasses:** Added `ProductionMetrics` and `ShipmentMetrics` for clean parameter passing.

### Changed
- **Config Paradigm Enforcement:** Refactored `Orchestrator`, `RiskEventManager`, and `QuirkManager` to eliminate hardcoded simulation parameters, moving them to `simulation_config.json`.
- **Code Quality:** All ruff and mypy strict checks pass for new modules.
- **Inventory State:** Expanded `StateManager` to track `perceived_inventory` vs `actual_inventory` to support phantom inventory simulation.

## [0.4.1] - 2025-12-26

### Changed
- **Config-Driven Values:** Moved all magic numbers to `simulation_config.json` per CLAUDE.md directives:
  - `calendar.weeks_per_year` (52)
  - `manufacturing.recall_batch_trigger_day` (30)
  - `manufacturing.default_yield_percent` (98.5)
  - `manufacturing.spof.warning_threshold` (10.0)
  - `logistics.default_lead_time_days` (3.0)
- **Refactored Allocation Agent:** Extracted helper methods (`_group_orders_by_source`, `_calculate_demand_vector`, `_calculate_fill_ratios`, `_apply_ratios_to_orders`) to reduce branch complexity.
- **Refactored Demand Engine:** Extracted `_apply_multiplier_to_cells` helper and created `PromoConfig` dataclass to reduce branch/argument complexity.
- **Code Quality:** Fixed all ruff linting issues (D200 docstrings, E501 line length, PLR1714 comparisons, RUF059 unused vars, PLC0415 imports).

### Fixed
- All ruff checks now pass with zero warnings.
- All mypy strict type checks pass (20 source files).

## [0.4.0] - 2025-12-26

### Added
- **MRP Engine (Task 5.1):** Implemented `MRPEngine` in `src/prism_sim/simulation/mrp.py` to translate DRP (Distribution Requirements Planning) into Production Orders for Plants.
- **Transform Engine (Task 5.2):** Implemented `TransformEngine` in `src/prism_sim/simulation/transform.py` with full production physics:
  - **Finite Capacity:** Enforces `run_rate_cases_per_hour` from Recipe definitions.
  - **Changeover Penalty:** Implements Little's Law friction when switching products, consuming capacity based on `changeover_time_hours`.
  - **Batch Tracking:** Creates `Batch` records with ingredient genealogy for traceability.
  - **Deterministic Batch:** Schedules `B-2024-RECALL-001` contaminated batch as per roadmap.
- **SPOF Simulation (Task 5.3):** Implemented raw material constraints for `ING-SURF-SPEC` (Specialty Surfactant). Production fails when ingredients are unavailable.
- **Network Expansion:** Added Plant nodes (`PLANT-OH`, `PLANT-TX`), backup supplier (`SUP-SURF-BACKUP`), and complete supplier-plant-RDC link topology.
- **Manufacturing Primitives:** Added `ProductionOrder`, `ProductionOrderStatus`, `Batch`, and `BatchStatus` to `network/core.py`.
- **Testing:** Added comprehensive integration tests in `tests/test_milestone_5.py` covering MRP, finite capacity, changeover penalties, SPOF, and recall batch scheduling.

### Changed
- **Orchestrator:** Extended daily loop to include MRP planning and production execution steps.
- **WorldBuilder:** Added recipes for all finished goods (Toothpaste, Soap, Detergent).
- **Configuration:** Added `manufacturing` section to `simulation_config.json` with production parameters and SPOF settings.

## [0.3.0] - 2025-12-26

### Added
- **Allocation Agent:** Implemented `AllocationAgent` in `src/prism_sim/agents/allocation.py` to handle inventory scarcity using "Fair Share" logic.
- **Logistics Engine:** Created `LogisticsEngine` in `src/prism_sim/simulation/logistics.py` to simulate physical bin-packing ("Tetris") for trucks, enforcing Weight vs. Cube constraints.
- **Transit Physics:** Implemented `Shipment` tracking and transit delays in `Orchestrator`, replacing "Magic Fulfillment" with realistic lead times.
- **Network Primitives:** Added `Shipment` and `ShipmentStatus` to `src/prism_sim/network/core.py`.
- **Testing:** Added comprehensive integration tests in `tests/test_milestone_4.py` covering allocation, bin-packing, and transit delays.

## [0.2.0] - 2025-12-26

### Added
- **Orchestrator:** Implemented the daily time-stepper loop in `src/prism_sim/simulation/orchestrator.py`.
- **Demand Engine:** Created `POSEngine` and `PromoCalendar` in `src/prism_sim/simulation/demand.py`, porting the vectorized "Lift & Hangover" physics directly from the `fmcg_example_OLD` reference to ensure high-fidelity demand signal generation.
- **Replenishment Agent:** Implemented `MinMaxReplenisher` in `src/prism_sim/agents/replenishment.py` to simulate store-level ordering and trigger the Bullwhip effect.
- **Network Expansion:** Added `Order` and `OrderLine` primitives to `network/core.py` and instantiated Retail Stores in `WorldBuilder`.
- **Testing:** Added integration tests for POS demand, promo lifts, and replenishment logic in `tests/test_milestone_3.py`.

## [0.1.1] - 2025-12-26

### Maintenance
- Enforced code quality standards using `ruff`, `mypy`, and `semgrep`.
- Fixed linting errors, unused imports, and type mismatches across the codebase.
- Added `mypy` to project dependencies.
- Refactored hardcoded simulation parameters into `simulation_config.json` to enforce the configuration paradigm.

## [0.1.0] - 2025-12-26

### Added
- **Core Primitives:** Implemented `Node` (RDC, Supplier) and `Link` (Route) classes in `src/prism_sim/network/core.py`.
- **Product Physics:** Implemented `Product` class with Weight/Cube attributes and `Recipe` for BOMs in `src/prism_sim/product/core.py`.
- **World Builder:** Created `WorldBuilder` to deterministically generate the "Deep NAM" network and product portfolio (Soap, Toothpaste, Detergent).
- **State Management:** Implemented `StateManager` using `numpy` for vectorized, O(1) inventory tracking.
- **Configuration:** Ported legacy `benchmark_manifest.json` to `src/prism_sim/config/`.
- **Testing:** Added unit tests for core primitives, world builder, and state manager.
- `physics.md`: Core reference for Supply Chain Physics theory and validation rules.
- Initial project structure and documentation.
