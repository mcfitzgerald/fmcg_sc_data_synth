# Validation & Chaos Engineering

The generator includes a robust validation suite to ensure synthetic data mirrors real-world supply chain patterns.

## RealismMonitor

The `RealismMonitor` runs online during generation, inspecting batches of data as they are created. It uses O(1) space algorithms to validate:

### 1. Structural Integrity
*   **Pareto Principle:** Validates that the top 20% of SKUs generate ~80% of volume.
*   **Hub Concentration:** Ensures key accounts (e.g., MegaMart) hold realistic market share (20-30%).

### 2. Kinetic Dynamics
*   **Bullwhip Effect:** Verifies that order variance is higher than POS sales variance (Target: 1.5x multiplier). Simulates promotional forward-buying (3.0x quantity batching).
*   **Friction:** Tracks transport delays by mode (Truck, Ocean, Air) to ensure realistic lead time variability.

### 3. Strategic & Financial
*   **Forecast Bias:** Tracks `(Forecast - Actual) / Actual` to detect systematic over/under-planning.
*   **Return Rate:** Monitors the ratio of `returns` to `sales` volume (Target: 2-5%).
*   **Margin Integrity:** Checks for excessive discounting (>50%) that would destroy gross margin.

### 4. Expert Reality Checks (Phase 4)

Eight advanced supply chain metrics based on industry benchmarks:

| Check | Formula | Target | Source |
|-------|---------|--------|--------|
| **Schedule Adherence** | Mean(abs(actual_start - planned_start)) | <1.1 days | Manufacturing |
| **Truck Fill Rate** | total_weight_kg / 20,000 kg | >50% | BCG Logistics |
| **SLOB Inventory** | Count(aging_bucket="90+") / total | <30% | Working Capital |
| **OEE** | Availability × Quality | 65-85% | FMCG Industry |
| **Inventory Turns** | Delivered Cases / (DC + Transit Inventory) | 6-14x | P&G Benchmark |
| **Forecast MAPE** | Mean(\|Forecast - Actual\| / Actual) | 20-50% | E2Open Study |
| **Cost-to-Serve** | Total Freight Cost / Total Cases | $1-3/case | BCG Benchmark |

### 5. Mass Balance (Physics) Validation

Conservation-of-mass checks ensure data generation respects physics:

| Balance Equation | Formula | Tolerance | Physics |
|-----------------|---------|-----------|---------|
| **Ingredient→Batch** | (output_kg - input_kg) / input_kg | <+2% | Yield loss means output < input |
| **Batch→Ship+Inv** | (shipped + inventory - produced) / produced | ±10% | Production = Distribution |
| **Order→Fulfill** | (fulfilled - ordered) / ordered | <+2% | Can't ship more than ordered |

**Key implementation details:**
- **Chemical Coherence:** SKUs are strictly linked to `batches` via matching `formula_id`. This ensures genealogy validity (e.g., you cannot have a "Shampoo" SKU backed by a "Toothpaste" batch).
- **Inventory Sourcing:** Inventory generation is deferred to Level 10 (Fulfillment) so it can be calculated as the remainder of `Production - Shipments`.
- **COGS Proxy:** Inventory Turns uses only DELIVERED store-bound shipments as the numerator to align with GAAP accounting (Sales/Inventory). In-transit shipments are counted as transit inventory, not COGS.
- **Transit Inventory:** Shipments with `status='in_transit'` generate inventory records with `location_type='in_transit'`, ensuring goods in motion are properly accounted.

## Chaos Injection

To support "Beast Mode" testing, the generator injects deterministic anomalies defined in `benchmark_manifest.json`.

### Risk Events
Deterministic scenarios that trigger specific data patterns:

| Event | Description | Effect |
|-------|-------------|--------|
| **RSK-BIO-001** | Contamination | Forces Sorbitol batch to `REJECTED` for recall tracing |
| **RSK-LOG-002** | Port Strike | Gamma (fat-tail) delays at USLAX, 3,900+ affected legs |
| **RSK-SUP-003** | Supplier Degradation | Palm oil supplier OTD drops to 40% |
| **RSK-CYB-004** | Cyber Outage | Chicago DC pick waves set to `ON_HOLD` |
| **RSK-ENV-005** | Carbon Tax | 3.0x CO2 emission multiplier applied |

### Quirks
Behavioral pathologies that occur probabilistically:

| Quirk | Description | Effect |
|-------|-------------|--------|
| **bullwhip_whip_crack** | Promotional batching | Retailers batch orders during promos (3x quantities) |
| **phantom_inventory** | Shrinkage | Random 2% inventory shrinkage injected |
| **port_congestion_flicker** | Correlated delays | 5,200+ legs with correlated port congestion |
| **single_source_fragility** | SPOF risk | Palm oil sourced from single supplier |
| **human_optimism_bias** | Forecast inflation | 15% over-forecast on 10,000+ records |
| **data_decay** | Quality degradation | 1,500+ batches rejected due to stale inputs |

## Validation Output

Sample validation output showing all checks:

```
Benchmark Comparison:
  Pareto (top 20%)       83.4%    [75%-85%]    PASS
  Hub concentration      25.9%    [20%-30%]    PASS
  POS CV                 0.665    [0.15-0.8]   PASS
  Order CV               1.029    [0.2-1.2]    PASS
  Bullwhip multiplier    1.548    [0.3-3.0]    PASS
  Promo Lift             2.331    [1.5-3.5]    PASS
  ...

  Expert Reality Checks:
  Schedule Adherence     0.998    [<1.1]       PASS
  Truck Fill Rate        63.6%    [50%-100%]   PASS
  SLOB Inventory         22.0%    [<30%]       PASS
  OEE                    65.5%    [65%-85%]    PASS
  Inventory Turns        6.606    [6.0-14.0]   PASS
  Forecast MAPE          24.8%    [20%-50%]    PASS
  Cost-to-Serve          $1.50    [$1.00-$3.00]  PASS

  Mass Balance (Physics):
  Ingredient→Batch (kg)  -1.3%    [<+2%]       PASS
  Batch→Ship+Inv (cases) -5.3%    [±10%]       PASS
  Order→Fulfill (cases)  -86.0%   [<+2%]       PASS  (Fill: 14.0%)
```

**Note:** The Mass Balance calculation allocates cases only to store-bound shipments (`dc_to_store`, `direct_to_store`) to match the COGS accounting model. Internal transfers (`plant_to_dc`, `dc_to_dc`) don't count toward shipped cases.

## Benchmark Manifest

All validation thresholds are configured in `benchmark_manifest.json`:

```json
{
  "validation_tolerances": {
    "pareto_range": [0.75, 0.85],
    "hub_concentration_range": [0.20, 0.30],
    "bullwhip_range": [0.3, 3.0],
    "oee_range": [0.65, 0.85],
    "inventory_turns_range": [6.0, 14.0],
    "mape_range": [0.20, 0.50],
    "cost_per_case_range": [1.0, 3.0],
    "mass_balance_drift_max": 0.02
  }
}
```
