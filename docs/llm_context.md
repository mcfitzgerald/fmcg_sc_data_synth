# Prism Sim: LLM Context & Developer One-Pager

> **System Prompt Context:** This document contains the critical architectural, functional, and physical constraints of the Prism Sim project. Use this as primary context when reasoning about code changes, bug fixes, or feature expansions.

**Version:** 0.12.3 | **Last Updated:** 2025-12-29

---

## 1. Project Identity

**Name:** Prism Sim
**Type:** Discrete-Event Simulation (DES) Engine for Supply Chain Digital Twins
**Scale:** 4,500+ nodes (3 Plants, 8 RDCs, 4,400+ Retail Stores, 10 Suppliers)
**Core Philosophy:** **"Supply Chain Physics"** - The simulation adheres to fundamental physical laws (Little's Law, Mass Balance, Capacity Constraints) rather than statistical approximation.
**Goal:** Generate realistic, high-fidelity supply chain datasets exhibiting emergent behaviors (bullwhip effect, bottlenecks) for benchmarking optimization algorithms.

---

## 2. The "Physics" Laws (Non-Negotiable Invariants)

Any code change must preserve these. Violations indicate bugs:

| Law | Formula | Enforcement Location |
|-----|---------|---------------------|
| **Mass Balance** | $I_t = I_{t-1} + Receipts - Shipments - Consumed + Produced$ | `monitor.py:PhysicsAuditor` |
| **Little's Law** | $WIP = Throughput \times CycleTime$ | Implicit in logistics/transform |
| **Capacity Constraints** | $Production \leq Rate \times Time \times OEE$ | `transform.py` |
| **Inventory Positivity** | $Inventory \geq 0$ (cannot ship/consume what you don't have) | `state.py`, `allocation.py`, `transform.py`, `orchestrator.py` |
| **Kinematic Consistency** | $TransitTime = Distance / Speed$ (no teleporting) | `logistics.py` |

**Inventory Positivity Enforcement (v0.12.2):**
- `orchestrator.py`: Demand consumption constrained to `min(demand, available_actual)`
- `allocation.py`: Fill ratios calculated from `actual_inventory` (not perceived)
- `transform.py`: Material consumption constrained to `min(required, available_actual)`
- `state.py`: Floor guards `np.maximum(0, ...)` on all inventory updates

---

## 3. File-to-Concept Map (Where to Find Things)

### Core Simulation Loop
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Daily orchestration** | `simulation/orchestrator.py` | `Orchestrator.run()`, `_run_day()` |
| **State tensors** | `simulation/state.py` | `StateManager` |
| **World construction** | `simulation/builder.py` | `WorldBuilder` |

### Demand & Orders
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **POS demand generation** | `simulation/demand.py` | `POSEngine.generate_daily_demand()` |
| **Replenishment orders** | `agents/replenishment.py` | `MinMaxReplenisher.generate_orders()` |
| **Order allocation** | `agents/allocation.py` | `AllocationAgent.allocate()` |

### Manufacturing
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Production planning** | `simulation/mrp.py` | `MRPEngine.generate_production_orders()` |
| **Recipe matrix (BOM)** | `network/recipe_matrix.py` | `RecipeMatrixBuilder` |
| **Production execution** | `simulation/transform.py` | `TransformEngine.execute_production()` |

### Logistics & Validation
| Concept | File | Key Classes/Functions |
|---------|------|----------------------|
| **Shipment creation** | `simulation/logistics.py` | `LogisticsEngine.create_shipments()` |
| **Physics validation** | `simulation/monitor.py` | `PhysicsAuditor`, `RealismMonitor` |
| **Behavioral quirks** | `simulation/quirks.py` | `QuirkManager` |
| **Risk events** | `simulation/risk_events.py` | `RiskEventManager` |

### Data Models
| Concept | File | Key Classes |
|---------|------|-------------|
| **Network primitives** | `network/core.py` | `Node`, `Link`, `Order`, `Shipment`, `Batch` |
| **Product definitions** | `product/core.py` | `Product`, `Recipe`, `ProductCategory` |

### Generators (Static World Creation)
| Concept | File | Key Classes |
|---------|------|-------------|
| **Product/SKU generation** | `generators/hierarchy.py` | `ProductGenerator` |
| **Network topology** | `generators/network.py` | `NetworkGenerator` |

---

## 4. State Manager: The Single Source of Truth

`StateManager` holds all simulation state as NumPy tensors for O(1) access:

```python
# Tensor Shapes
inventory: np.ndarray        # [n_nodes, n_products] - Current stock (actual)
perceived_inventory: np.ndarray  # [n_nodes, n_products] - What system "sees" (may differ due to phantom inventory)
wip: np.ndarray              # [n_nodes, n_products] - Work in process at plants

# Index Mappings (str -> int for O(1) lookup)
node_id_to_idx: dict[str, int]
product_id_to_idx: dict[str, int]
```

**Critical:** Always use `state.update_inventory(node_id, product_id, delta)` - never modify tensors directly.

---

## 5. Daily Simulation Loop (Execution Order)

Every `tick()` (1 day) in `Orchestrator._run_day()`:

```
1. ARRIVALS      → Process in-transit shipments that arrived today
2. DEMAND        → POSEngine generates retail sales (consumes store inventory)
3. REPLENISHMENT → MinMaxReplenisher creates orders (Store→RDC, RDC→Plant)
4. ALLOCATION    → AllocationAgent allocates inventory to orders (Fair Share)
5. MRP           → MRPEngine plans production orders for plants
6. PRODUCTION    → TransformEngine executes manufacturing (consumes ingredients)
7. LOGISTICS     → LogisticsEngine creates shipments, advances in-transit
8. QUIRKS        → QuirkManager applies behavioral realism
9. MONITORING    → PhysicsAuditor validates mass balance, records metrics
```

**Data Flow Diagram:**
```
Retailers ──[POS Sales]──→ Demand Signal
    ↓
    [Replenishment Orders]
    ↓
RDCs ←──[Shipments]── Plants ←──[Production Orders]── MRP
    ↓                    ↓
    [Allocation]         [BOM Explosion]
    ↓                    ↓
Inventory Tensors ←── Recipe Matrix ←── Suppliers
```

---

## 6. Recipe Matrix: Vectorized BOM

The `RecipeMatrixBuilder` creates a dense matrix $\mathbf{R}$ for instant ingredient calculations:

```
Shape: [n_finished_goods, n_ingredients]
Value R[i,j] = quantity of ingredient j needed to make 1 unit of product i

Usage: ingredient_requirements = demand_vector @ recipe_matrix
```

This enables O(1) MRP calculations for thousands of SKUs instead of iterating BOMs.

---

## 7. Inventory Policies (Tiered)

`MRPEngine` uses vectorized policy vectors based on product type:

| Category | Reorder Point (days) | Target (days) | Example Products |
|----------|---------------------|---------------|------------------|
| DEFAULT | 14 | 28 | Finished goods |
| INGREDIENT | 7 | 14 | Bulk base chemicals |
| PACKAGING | 7 | 14 | Bottles, caps, boxes |
| ACTIVE_CHEM | 30 | 45 | Specialty actives |
| SPOF | 30 | 45 | `ACT-CHEM-001` (constrained) |

Configured in `simulation_config.json` under `manufacturing.inventory_policies`.

---

## 8. The Supply Chain Triangle

Every decision impacts the balance between:

```
        SERVICE (Fill Rate, OTIF)
              /\
             /  \
            /    \
           /      \
    COST ←────────→ CASH
(Freight, Prod)   (Inventory, WC)
```

**Key Metrics Tracked:**
- **Store Service Level (OSA):** `Actual_Sales / Consumer_Demand` (Target >90%)
- **Service Level (LIFR):** `Shipped_Qty / Ordered_Qty` (Internal fill rate)
- **OEE:** Plant capacity utilization (target 65-85%)
- **Truck Fill:** Logistics efficiency (target >85%)
- **Inventory Turns:** Cash efficiency (target 6-14x annually)

---

## 9. Known Issues & Current State

### The Bullwhip Crisis (v0.12.1)
**Status:** IDENTIFIED - Mitigation planned in `docs/planning/sim_tuning.md`

**Symptom:** 365-day simulation shows orders exploding from ~460k to 66M cases/day.

**Root Cause:** "Fill or Kill" allocation + zero inventory = infinite reordering loop.
- Store orders 1000, gets 0 (killed), orders 1000 again next day
- This creates a feedback loop amplifying demand 150x

**Planned Fix:**
1. `min_fill_threshold` in `AllocationAgent` - don't ship dust
2. `max_order_cap` in `MinMaxReplenisher` - dampen panic ordering

### SPOF Isolation (v0.12.1 - IMPLEMENTED)
`ACT-CHEM-001` now isolated to ~20% of portfolio (Premium Oral Care only).
Supplier `SUP-001` constrained to 500k units/day.

---

## 10. Configuration Files

| File | Purpose |
|------|---------|
| `config/simulation_config.json` | Runtime parameters (MRP, logistics, quirks, initialization) |
| `config/world_definition.json` | Static world (products, network topology, recipe logic) |
| `config/benchmark_manifest.json` | Risk scenarios, validation targets |

**Key Config Sections:**
```json
{
  "manufacturing": {
    "target_days_supply": 28.0,
    "reorder_point_days": 14.0,
    "inventory_policies": { ... },  // Tiered ROP/Target
    "spof": { "ingredient_id": "ACT-CHEM-001" }
  },
  "simulation": {
    "inventory": {
      "initialization": {
        "store_days_supply": 14.0,
        "rdc_days_supply": 28.0
      }
    }
  }
}
```

---

## 11. Key Commands

```bash
# Run Simulation (default 90 days)
poetry run python run_simulation.py
poetry run python run_simulation.py --days 365 --streaming

# Run Tests
poetry run pytest
poetry run pytest tests/test_milestone_4.py -v

# Type Check & Lint
poetry run mypy src/
poetry run ruff check src/

# Generate Static World (4,500 nodes)
poetry run python scripts/generate_static_world.py
```

---

## 12. Debugging Checklist

When simulation behaves unexpectedly:

1. **Service Level = 0%?**
   - Check `initialization.store_days_supply` - cold start starvation?
   - Check `PhysicsAuditor` for mass balance drift
   - Look for SPOF ingredient exhaustion

2. **Orders exploding?**
   - Bullwhip feedback loop - check `MinMaxReplenisher` order volumes
   - Fill-or-Kill creating infinite retry loop

3. **Production stalled?**
   - Check ingredient inventory at plants (`state.inventory[plant_idx, :]`)
   - Check `MRPEngine` purchase orders for raw materials
   - Verify `TransformEngine` capacity vs demand

4. **OEE too low?**
   - Check changeover times in recipes
   - Check `min_production_qty` (MOQ) - too small = changeover penalty

5. **Mass Balance violations?**
   - Check `PhysicsAuditor.audit()` output
   - Look for inventory updates outside `StateManager`

6. **Negative inventory detected?** (Should not happen after v0.12.2)
   - Check if new code bypasses `StateManager.update_inventory()` methods
   - Verify `actual_inventory` is used (not `perceived_inventory`) for allocation/consumption decisions
   - Look for direct tensor manipulation without floor guards
   - Run: `inv[inv['actual_inventory'] < 0]` on inventory.csv to find violating cells

---

## 13. Architecture Diagrams

### System Layers
```
┌─────────────────────────────────────────────────────────┐
│                   SIMULATION LAYER                       │
│  Orchestrator → POSEngine → MRPEngine → TransformEngine │
│                → LogisticsEngine → QuirkManager          │
├─────────────────────────────────────────────────────────┤
│                     AGENT LAYER                          │
│         MinMaxReplenisher    AllocationAgent             │
├─────────────────────────────────────────────────────────┤
│                     CORE LAYER                           │
│              StateManager    WorldBuilder                │
├─────────────────────────────────────────────────────────┤
│                   DOMAIN MODELS                          │
│     Node, Link, Order, Shipment │ Product, Recipe        │
└─────────────────────────────────────────────────────────┘
```

### Daily Loop Sequence
```
Orchestrator
    │
    ├──→ POSEngine.generate_daily_demand()
    │         └──→ state.update_inventory() [subtract sales]
    │
    ├──→ MinMaxReplenisher.generate_orders()
    │         └──→ creates Order objects
    │
    ├──→ AllocationAgent.allocate()
    │         ├──→ Fair Share if constrained
    │         └──→ Fill-or-Kill (close unfilled)
    │
    ├──→ MRPEngine.generate_production_orders()
    │         ├──→ recipe_matrix @ demand_vector
    │         └──→ creates ProductionOrder objects
    │
    ├──→ TransformEngine.execute_production()
    │         ├──→ check materials (vectorized)
    │         ├──→ consume ingredients
    │         └──→ produce finished goods
    │
    ├──→ LogisticsEngine.create_shipments()
    │         ├──→ bin-packing (weight/cube)
    │         └──→ create Shipment objects
    │
    └──→ PhysicsAuditor.audit()
              └──→ validate mass balance
```

---

## 14. Version History (Key Milestones)

| Version | Key Changes |
|---------|-------------|
| 0.12.3 | **Inverse Bullwhip Fix** - MRP uses lumpy RDC shipment signals; Store Service Level metric added |
| 0.12.2 | **Negative inventory fix** - Inventory Positivity law enforced across all deduction paths |
| 0.12.1 | Tiered inventory policies, LIFR tracking, Bullwhip Crisis identified |
| 0.12.0 | Fill-or-Kill allocation, MRP look-ahead horizon |
| 0.11.0 | Streaming writers for 365-day runs |
| 0.10.0 | **Vectorized MRP** (Recipe Matrix), Mass Balance Audit, Procedural ingredients |
| 0.9.x | Deep NAM integration, inventory collapse fixes, OEE tracking |
| 0.8.0 | Static world generators (4,500 nodes) |
| 0.5.0 | Validation framework, quirks, risk events |
| 0.4.0 | MRP + Transform engines, SPOF simulation |
| 0.3.0 | Allocation agent, logistics (Tetris), transit physics |
| 0.2.0 | Orchestrator, POS demand, replenishment |
| 0.1.0 | Core primitives, state manager, world builder |
