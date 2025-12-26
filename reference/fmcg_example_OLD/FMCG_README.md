# FMCG Example: Prism Consumer Goods

A **Fast-Moving Consumer Goods (FMCG)** supply chain example for the Virtual Graph project, demonstrating high-velocity, massive-volume patterns inspired by Colgate-Palmolive.

## Company Profile

- **Company**: Prism Consumer Goods (PCG)
- **Revenue**: ~$15B global CPG
- **HQ**: Knoxville, TN
- **Product Lines**:
  - **PrismWhite** (Oral Care) - Toothpaste
  - **ClearWave** (Home Care) - Dish Soap
  - **AquaPure** (Personal Care) - Body Wash

### Global Structure (5 Divisions, 7 Plants)

| Division | HQ | Plants | Markets |
|----------|-----|--------|---------|
| NAM | Knoxville | 2 (Tennessee, Texas) | US, Canada |
| LATAM | São Paulo | 1 (Brazil) | Brazil, Mexico, Andean |
| APAC | Singapore | 2 (China, India) | China, India, SEA, ANZ |
| EUR | Paris | 1 (Poland) | Western EU, UK, Nordics |
| AFR-EUR | Dubai | 1 (Turkey) | MENA, Sub-Saharan, CIS |

### Channel Mix

| Channel | Volume Share | Archetypes |
|---------|--------------|------------|
| B&M Large | 40% | MegaMart, ValueClub, UrbanEssential |
| B&M Distributor | 30% | RegionalGrocers, IndieRetail |
| E-commerce | 20% | DigitalFirst, OmniRetailer |
| DTC | 10% | PrismDirect |

## Quick Start

```bash
# Start PostgreSQL (port 5433)
make fmcg-db-up

# Generate seed data (~11.4M rows in ~2-3 minutes)
make fmcg-generate

# Run tests
poetry run pytest fmcg_example/tests/ -v

# Start Neo4j for benchmarking (port 7475/7688)
docker-compose -f fmcg_example/neo4j/docker-compose.yml up -d
```

## Directory Structure

```
fmcg_example/
├── ontology/
│   └── prism_fmcg.yaml              # LinkML ontology with VG extensions (71 classes)
├── postgres/
│   ├── docker-compose.yml           # PostgreSQL container (port 5433)
│   ├── schema.sql                   # 70 tables + 8 views DDL
│   ├── seed.sql                     # Generated data (~11.4M rows)
│   └── BenchmarkManifest.json       # Ground truth for validation
├── neo4j/
│   ├── docker-compose.yml           # Neo4j container (port 7475/7688)
│   └── migrate.py                   # Ontology-driven migration
├── scripts/
│   ├── generate_data.py             # Orchestrator (644 lines after refactor)
│   ├── validate_realism.sql         # Validation queries
│   └── data_generation/             # Modular generation system
│       ├── generators/              # 15 level generators (Level 0-14)
│       ├── constants/               # Reference data (divisions, ingredients, etc.)
│       ├── vectorized.py            # NumPy-based high-speed generators
│       ├── promo_calendar.py        # Multi-promo effects system
│       ├── risk_events.py           # Chaos injection (5 risk events)
│       ├── quirks.py                # Behavioral quirks (6 patterns)
│       ├── realism_monitor.py       # Online validation (Welford, Pareto)
│       └── streaming_writer.py      # Memory-efficient COPY output
├── tests/
│   ├── test_recall_trace.py         # Beast mode: lot genealogy
│   ├── test_landed_cost.py          # Beast mode: cost rollup
│   ├── test_spof_risk.py            # Beast mode: supplier criticality
│   ├── test_osa_analysis.py         # Beast mode: OSA/DC bottlenecks
│   └── test_ontology.py             # Two-layer validation
└── FMCG_README.md                   # This file
```

## Key Differences from supply_chain_example

| Aspect | supply_chain_example | fmcg_example |
|--------|---------------------|--------------|
| Domain | Aerospace/Industrial | Consumer Goods |
| Scale | ~1.7M rows, 20 tables | ~11.4M rows, 70 tables |
| Graph Shape | Deep (25+ level BOM) | Wide (1 batch → 47K orders) |
| Stress Test | Recursive traversal depth | Horizontal explosion width |
| Target Metric | BOM cost rollup | Recall trace speed |
| SCOR Coverage | Partial | Full (Plan/Source/Transform/Order/Fulfill/Return) |

## SCOR-DS Domain Model

PCG implements the full SCOR-DS (Supply Chain Operations Reference - Digital Standard) framework:

```
                              ┌─────────────────┐
                              │      PLAN       │
                              │  demand_forecasts│
                              │  capacity_plans  │
                              │  supply_plans    │
                              └────────┬────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             │                             ▼
┌─────────────────┐                    │                    ┌─────────────────┐
│     ORDER       │                    │                    │     SOURCE      │
│  orders         │◄───── DEMAND ──────┼────── SUPPLY ─────►│  purchase_orders│
│  promotions     │                    │                    │  suppliers      │
└────────┬────────┘                    │                    └────────┬────────┘
         │                    ┌────────┴────────┐                    │
         │                    │   ORCHESTRATE   │                    │
         │                    │  kpi_thresholds │                    │
         │                    │  risk_events    │                    │
         │                    └────────┬────────┘                    │
         │                             │                             │
         ▼                             │                             ▼
┌─────────────────┐                    │                    ┌─────────────────┐
│     FULFILL     │                    │                    │    TRANSFORM    │
│  shipments      │◄───── OUTPUT ──────┼────── INPUT ──────►│  batches        │
│  inventory      │                    │                    │  formulas       │
└────────┬────────┘                    │                    └────────┬────────┘
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       │
                              ┌────────┴────────┐
                              │     RETURN      │
                              │  returns        │
                              │  disposition_logs│
                              └─────────────────┘
```

## The Desmet Triangle

Every edge in the PCG model carries three dimensions in constant tension:

| Dimension | Metrics | Tradeoff |
|-----------|---------|----------|
| **Service** | OTIF, OSA, Fill Rate | Improving service often increases cost |
| **Cost** | Landed cost, freight, handling | Cutting cost may hurt service levels |
| **Cash** | Inventory days, payment terms | Holding inventory ties up working capital |

## FMCG Benchmarks

| Metric | Industrial | FMCG (Target) |
|--------|-----------|---------------|
| Inventory Turns | 3-5/year | 8-12/year |
| Perfect Order (OTIF) | 80-85% | 95-98% |
| On-Shelf Availability | N/A | 92-95% |
| Batch Yield | 60-85% | 95-99% |
| Forecast Accuracy (MAPE) | N/A | 20-30% |

## Beast Mode Queries

| Query | Description | Handler |
|-------|-------------|---------|
| **Recall Trace** | 1 batch (`B-2024-RECALL-001`) → 47K orders | `traverse()` |
| **Landed Cost** | Full margin calculation through supply chain | `path_aggregate()` |
| **SPOF Detection** | Find single-source ingredients (`SUP-PALM-MY-001`) | `resilience_analysis()` |
| **OSA Root Cause** | Correlate low-OSA with DC bottlenecks (`DC-NAM-CHI-001`) | `centrality()` |

## Named Test Entities

Deterministic fixtures for reproducible benchmarking:

| Entity ID | Type | Purpose |
|-----------|------|---------|
| `B-2024-RECALL-001` | Batch | Contaminated batch for recall trace |
| `ACCT-MEGA-001` | Account | MegaMart hub (4,500 stores, 25% of orders) |
| `SUP-PALM-MY-001` | Supplier | Single-source Palm Oil (SPOF) |
| `DC-NAM-CHI-001` | DC | Chicago bottleneck (40% NAM volume) |
| `PROMO-BF-2024` | Promotion | Black Friday (bullwhip effect) |
| `LANE-SH-LA-001` | Route | Seasonal Shanghai→LA lane |

## Connection Settings

| Service | Host | Port | Database | User | Password |
|---------|------|------|----------|------|----------|
| PostgreSQL | localhost | **5433** | prism_fmcg | virt_graph | dev_password |
| Neo4j HTTP | localhost | **7475** | - | neo4j | dev_password |
| Neo4j Bolt | localhost | **7688** | - | neo4j | dev_password |

*Note: Ports differ from supply_chain_example (5432/7474/7687) to allow both to run simultaneously.*

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Directory Structure | ✅ Complete |
| 2 | Schema (70 tables + 8 views) | ✅ Complete |
| 3 | Ontology (71 classes, ~50 relationships) | ✅ Complete |
| 4 | Data Generator (~11.4M rows, 85K rows/sec) | ✅ Complete (v0.9.40) |
| 5 | Chaos Injection (5 risk events, 6 quirks) | ✅ Complete |
| 6 | Validation Suite (8 automated checks) | ✅ Complete |
| 7 | Beast Mode Tests | 📋 TODO |
| 8 | Neo4j Comparison | 📋 TODO |

## Specification

Full specification: [`magical-launching-forest.md`](../magical-launching-forest.md)
