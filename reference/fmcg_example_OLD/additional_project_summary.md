1. Executive Summary

We are simulating Prism Consumer Goods (PCG), a fictional \$15B global Fast-Moving Consumer Goods
(FMCG) company headquartered in Knoxville, TN. The simulation models a high-velocity,
massive-volume supply chain inspired by Colgate-Palmolive, specifically designed to stress-test
the VG/SQL (Virtual Graph over SQL) architecture.

The core objective is to demonstrate that complex graph algorithms (recursive traversals,
pathfinding, centrality) can be executed efficiently over a standard relational database
(PostgreSQL) at scale (~15M rows) without data migration to a dedicated graph database.

2. Company Profile & Scale

 * Company: Prism Consumer Goods (PCG)
 * Scale: ~14.7M rows across 70 tables (SCOR-DS model).
 * Structure:
     * 5 Divisions: NAM (North America), LATAM, APAC, EUR, AFR-EUR.
     * 7 Plants: Global manufacturing network.
     * Distribution: 25 DCs, 86 retail accounts, ~10,000 retail locations.
     * Products: 3 Lines (PrismWhite, ClearWave, AquaPure) exploding into ~2,000 SKUs.
 * Graph Characteristics:
     * Shape: "Horizontal Explosion" (High fan-out). A single production batch distributes to
       thousands of retail locations.
     * Velocity: High-frequency, low-latency transactions (contrasting with the deep, slow-moving
       BOM structures of aerospace).

3. Architecture & Technology

The system is built on the VG/SQL stack:

 1. Storage: PostgreSQL 15 (Port 5433).
 2. Schema Definition: LinkML Ontology (prism_fmcg.yaml) maps graph concepts (Nodes, Edges) to
    SQL tables and Foreign Keys. It uses custom vg: extensions for graph semantics.
 3. Execution: Python Handlers (src/virt_graph/handlers/) execute graph algorithms by generating
    optimized, batched SQL queries.
     * Key Handlers: traverse() (Recall Trace), path_aggregate() (Landed Cost), centrality()
       (Bottleneck detection).
 4. Benchmarking: Neo4j container (Ports 7475/7688) serves as the performance baseline.

4. Simulation Physics & Logic

The simulation enforces strict "Physics" to ensure the data is coherent and traversable:

A. Mass Balance (Conservation of Matter)
 * Ingredient → Batch: Inputs (kg) match outputs (kg) minus yield loss.
 * Batch → Inventory: Production = Shipped + Inventory.
 * Order → Fulfillment: Cannot ship more than ordered.
 * Recent Fix: Corrected "Inventory Turns" by strictly defining COGS as delivered shipments to
   external stores only.

B. Inventory Waterfall
 * Hybrid Model: Inventory is tracked at DCs and in-transit.
 * Classification: Stock is tagged as Safety Stock (first 14 days coverage) or Cycle Stock
   (remainder).
 * Visibility: v_inventory_waterfall view aggregates position across static and moving inventory.

C. Emergent Logistics
 * Pallet-Tier Pricing: Freight costs are calculated using realistic LTL/FTL tiers based on
   pallet count and distance, not just simple linear distance.
 * Vectorized Generation: High-volume tables (POS Sales, Order Lines, Shipment Legs) are
   generated using NumPy for performance (~85k rows/sec).

5. Chaos Engineering (Beast Mode)

To validate the system's analytical capabilities, we inject deterministic "Pathologies" and "Risk
Events":


┌─────────┬─────────────┬────────────────────────────────────────────────────────────────────────

│ Even... │ Scenario    │ Impact

├─────────┼─────────────┼────────────────────────────────────────────────────────────────────────

│ **RS... │ **Contam... │ Specific batch (B-2024-RECALL-001) of Sorbitol marked REJECTED. Tes...

│ **RS... │ Port Strike │ USLAX port incurs 4x delays (Gamma distribution). Tests resilience ...

│ **RS... │ **Suppli... │ Sole-source Palm Oil supplier (SUP-PALM-MY-001) OTD drops to 40%. T...

│ **RS... │ **Cyber ... │ Chicago DC (DC-NAM-CHI-001) WMS goes dark; pick waves ON_HOLD. Test...

│ **RS... │ Carbon Tax  │ 3x CO2 cost multiplier applied. Tests cost-to-serve aggregation.

└─────────┴─────────────┴────────────────────────────────────────────────────────────────────────


Quirks (Behavioral Anomalies):
 * Bullwhip Effect: Retailers "panic buy" (3x batching) during Black Friday (PROMO-BF-2024).
 * Phantom Inventory: 2% shrinkage injected to create data drift.
 * Optimism Bias: Planners consistently over-forecast new products by 15%.

6. Validation System

The RealismMonitor ensures generated data meets FMCG industry standards:

 * Statistical Checks:
     * Pareto Principle: Top 20% SKUs must drive ~80% volume.
     * Hub Concentration: MegaMart must hold 20-30% market share.
 * KPI Benchmarks:
     * Inventory Turns: Target 6-14x (currently ~9.6x).
     * OTIF (On-Time In-Full): Target 85-99%.
     * OSA (On-Shelf Availability): Target 92-96%.
     * Cost-to-Serve: \$1.00 - \$3.00 per case.

7. Key Testing Scenarios ("Beast Mode")

These tests validate the VG/SQL core value proposition:

 1. Recall Trace: Trace B-2024-RECALL-001 through the entire network to 47,500 affected orders in
    <5 seconds.
 2. Landed Cost: Roll up material, labor, overhead, freight, and duty costs for a specific SKU at
    a specific store.
 3. SPOF Detection: Identify single-source ingredients (like Palm Oil) instantly.
 4. Root Cause Analysis: Correlate low OSA scores with upstream DC bottlenecks.

8. Implementation Status

 * Schema: ✅ 70 Tables + 8 Views implemented.
 * Ontology: ✅ Complete (Phase 3).
 * Data Generation: ✅ Complete (Levels 0-14), optimized with vectorization.
 * Validation: ✅ Passing all structural, kinetic, and physics checks.
 * Tests: 🚧 Scaffolded. The actual test logic in fmcg_example/tests/ needs to be implemented to
   execute the handlers against the generated data.
