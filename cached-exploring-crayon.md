# Plan: Full SCOR-DS Export for Prism-Sim

## Goal
Expand prism-sim output from 4 CSV files (~3K rows) to the full SCOR-DS schema (~70 tables, millions of rows) to create a high-fidelity synthetic supply chain data foundation for a Colgate-like NAM FMCG company.

## Current State
- **Output**: 4 files (orders.csv, shipments.csv, batches.csv, inventory.csv) ~3K rows
- **World**: 4 stores, 3 SKUs, 4 RDCs, 2 plants, 3 suppliers
- **Writer**: Monolithic `SimulationWriter` with basic logging

## Target State
- **Output**: ~70 tables matching `reference/fmcg_example_OLD/schema.sql`
- **World**: Thousands of stores across 6 archetypes, 50+ SKUs, realistic NAM network
- **Architecture**: Domain-specific writers with config-driven generation

---

## Architecture: Hybrid Generation

### 1. Static Pool (Faker) - Port from Reference
Port `reference/fmcg_example_OLD/scripts/data_generation/static_pool.py` to:
```
src/prism_sim/generators/static_pool.py
```
- Pre-generate pools: names, companies, cities, geo locations, product names, etc
- Vectorized sampling with NumPy for O(1) access
- Used for entity attributes (supplier names, store addresses, etc.)

### 2. Algorithmic Generation - Distributions
Port from reference `helpers.py` and `vectorized.py`:
```
src/prism_sim/generators/distributions.py
```
- **Zipf**: SKU popularity (top 20% = 80% volume)
- **Barabási-Albert**: Preferential attachment for hub concentration
- **Seasonal curves**: Demand seasonality
- Parameters from `benchmark_manifest.json`

### 3. Entity Expansion - Config + Generation
Expand `world_definition.json` with:
- **Store Archetypes** (6 types with realistic counts):
  - MEGAMART (Walmart-like): 4,500 stores, 45% of volume
  - CLUB (Costco-like): 600 stores, 15% of volume
  - DOLLAR (Dollar General-like): 2,000 stores, 10% of volume
  - DISTRIBUTOR (mom & pop): 500 accounts → 5,000 endpoints
  - DTC (direct): 1 node, 5% of volume
  - ECOM (Amazon/digital): 200 accounts, 10% of volume

- **Products**: Define 50 SKUs across 3 categories (oral, personal wash, home care)
- **Suppliers**: 50+ with tier distribution (T1: 10, T2: 20, T3: 20+)

---

## Implementation Phases

### Phase 1: Foundation & Static Data Writers
**Files to create:**
- `src/prism_sim/writers/__init__.py`
- `src/prism_sim/writers/base.py` - BaseWriter ABC with CSV/Parquet support
- `src/prism_sim/writers/static_writer.py` - Exports world definition
- `src/prism_sim/generators/__init__.py`
- `src/prism_sim/generators/static_pool.py` - Port from reference
- `src/prism_sim/generators/distributions.py` - Zipf, Barabási-Albert
- `src/prism_sim/generators/entity_generator.py` - Store/SKU expansion

**Files to modify:**
- `src/prism_sim/config/world_definition.json` - Add archetypes, expand entities
- `src/prism_sim/config/scor_reference.json` (NEW) - Channels, divisions, ports, carriers

**Tables generated (Levels 0-4, ~40K rows):**
| Domain | Tables | Rows |
|--------|--------|------|
| Reference | divisions, channels, ports, carriers | ~100 |
| Source | ingredients, suppliers, supplier_ingredients, certifications | ~3K |
| Transform | plants, production_lines, formulas, formula_ingredients | ~500 |
| Product | products, packaging_types, skus, sku_costs | ~15K |
| Fulfill | distribution_centers, retail_accounts, retail_locations | ~15K |
| Logistics | route_segments, routes, carrier_contracts, carrier_rates | ~2K |

### Phase 2: Transform & Source Transactions
**Files to create:**
- `src/prism_sim/writers/source_writer.py` - POs, goods receipts
- `src/prism_sim/writers/transform_writer.py` - Work orders, batches, costs

**Files to modify:**
- `src/prism_sim/simulation/mrp.py` - Add PO generation for ingredients
- `src/prism_sim/simulation/transform.py` - Track work order materials
- `src/prism_sim/simulation/orchestrator.py` - Integrate new writers

**Tables generated (Levels 5-7, ~1M rows):**
| Domain | Tables | Rows |
|--------|--------|------|
| Source | purchase_orders, purchase_order_lines, goods_receipts, goods_receipt_lines | ~200K |
| Transform | work_orders, work_order_materials, batches, batch_ingredients, batch_cost_ledger | ~800K |

### Phase 3: Order & Demand (Highest Volume)
**Files to create:**
- `src/prism_sim/writers/order_writer.py` - Orders, lines, allocations
- `src/prism_sim/writers/plan_writer.py` - POS sales, forecasts

**Files to modify:**
- `src/prism_sim/simulation/demand.py` - Log POS sales per store/SKU
- `src/prism_sim/agents/replenishment.py` - Track order lines detail
- `src/prism_sim/agents/allocation.py` - Log allocation decisions

**Tables generated (Levels 8-9, ~10M+ rows):**
| Domain | Tables | Rows |
|--------|--------|------|
| Order | promotions, promotion_skus, orders, order_lines, order_allocations | ~8M |
| Plan | pos_sales, demand_forecasts, forecast_accuracy, replenishment_params | ~3M |

**Performance critical:** Use streaming writes with 10K row batches, Parquet for large tables.

### Phase 4: Fulfill & Logistics
**Files to create:**
- `src/prism_sim/writers/fulfill_writer.py` - Shipments, pick waves
- `src/prism_sim/writers/logistics_writer.py` - Legs, emissions

**Files to modify:**
- `src/prism_sim/simulation/logistics.py` - Multi-leg tracking, pick waves
- Add CO2 calculation from emission factors

**Tables generated (Levels 10-11, ~2M rows):**
| Domain | Tables | Rows |
|--------|--------|------|
| Fulfill | shipments, shipment_lines, inventory, pick_waves, pick_wave_orders | ~1.5M |
| Logistics | shipment_legs, shipment_emissions | ~500K |

### Phase 5: Returns & Orchestrate
**Files to create:**
- `src/prism_sim/writers/return_writer.py` - RMAs, returns, dispositions
- `src/prism_sim/writers/orchestrate_writer.py` - KPIs, OSA, risk events

**Files to modify:**
- `src/prism_sim/simulation/orchestrator.py` - Generate returns (2% of orders)
- `src/prism_sim/simulation/monitor.py` - Export KPI actuals format

**Tables generated (Levels 12-14, ~1.5M rows):**
| Domain | Tables | Rows |
|--------|--------|------|
| Return | rma_authorizations, returns, return_lines, disposition_logs | ~200K |
| Orchestrate | kpi_thresholds, kpi_actuals, osa_metrics, risk_events, audit_logs | ~1.3M |

---

## Critical Files Summary

### New Files
```
src/prism_sim/
├── generators/
│   ├── __init__.py
│   ├── static_pool.py          # Faker pools (port from reference)
│   ├── distributions.py        # Zipf, Barabási-Albert
│   └── entity_generator.py     # Store/SKU expansion
├── writers/
│   ├── __init__.py
│   ├── base.py                 # BaseWriter ABC
│   ├── static_writer.py        # Levels 0-4
│   ├── source_writer.py        # Domain A
│   ├── transform_writer.py     # Domain B
│   ├── order_writer.py         # Domain D (high volume)
│   ├── fulfill_writer.py       # Domain E
│   ├── logistics_writer.py     # Domain E2
│   ├── plan_writer.py          # Domain F
│   ├── return_writer.py        # Domain G
│   └── orchestrate_writer.py   # Domain H
└── config/
    └── scor_reference.json     # Static reference data
```

### Modified Files
```
src/prism_sim/
├── config/world_definition.json  # Expand entities
├── simulation/
│   ├── orchestrator.py           # Integrate all writers
│   ├── demand.py                 # Log POS sales
│   ├── mrp.py                    # Generate POs
│   ├── transform.py              # Track work order materials
│   ├── logistics.py              # Multi-leg, pick waves
│   └── monitor.py                # KPI export format
└── agents/
    ├── replenishment.py          # Order line detail
    └── allocation.py             # Allocation logging
```

---

## Validation Approach

1. **Row counts**: Match `benchmark_manifest.json.generation_targets` (±25%)
2. **Distributions**: Validate Zipf alpha, hub concentration per manifest
3. **Referential integrity**: Sample 1000 FKs per table, verify 100% valid
4. **Physics**: Mass balance, Little's Law checks from existing monitor

---

## Execution Order

1. **Phase 1** - Foundation (generators + static writers)
2. **Phase 2** - Transform/Source (manufacturing chain)
3. **Phase 3** - Order/Demand (volume driver, performance critical)
4. **Phase 4** - Fulfill/Logistics (transport network)
5. **Phase 5** - Returns/Orchestrate (KPIs, cleanup)
6. **Integration** - 365-day run, validation, documentation
