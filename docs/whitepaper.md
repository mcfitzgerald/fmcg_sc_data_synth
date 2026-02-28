# Prism Consumer Goods — Synthetic Enterprise Dataset

## 1. Executive Summary

This dataset represents 365 days of synthetic enterprise data for **Prism Consumer Goods (PCG)**, a fictional North American fast-moving consumer goods company. PCG manufactures and distributes oral care, personal wash, and home care products through seven channels to over 3,800 retail and fulfillment locations.

The modeled company operates at scale: $38.3 billion in annual revenue, 500 SKUs, 50 suppliers, four manufacturing plants, and a 4-echelon distribution network spanning roughly 4,200 nodes. The dataset captures the full operational footprint — procurement, manufacturing, inventory management, order fulfillment, returns, and financial accounting — across 38 normalized ERP tables totaling approximately 395 million rows.

The dataset is purpose-built for three use cases: **Virtual Knowledge Graph (VKG) testing**, where heterogeneous enterprise tables must be integrated and queried as a unified graph; **supply chain analytics**, where analysts need realistic volume, variability, and cross-functional tradeoffs; and **data integration benchmarking**, where controlled data quality issues provide a known ground truth for entity resolution, record matching, and anomaly detection.

Every row traces back to a physics-based simulation that enforces mass balance, capacity constraints, and kinematic consistency. The financial layer is generated post-hoc from operational data — the same way real ERP systems record transactions after physical events occur.

---

## 2. Conceptual Framework

The dataset is structured around two established supply chain frameworks that define what gets measured and why.

### Desmet's Supply Chain Triangle

The central tension in any supply chain is the tradeoff between three competing objectives:

- **Service** — delivering what the customer wants, when they want it. Measured by fill rate, perfect order rate, and on-time delivery.
- **Cost** — operating efficiently. Measured by cost of goods sold, freight expense, manufacturing overhead, and overall equipment effectiveness.
- **Cash** — minimizing capital tied up in the business. Measured by inventory turns, days of supply, and the cash-to-cash cycle.

Improving one corner typically degrades another. Carrying more inventory (cash) improves fill rate (service) but increases holding cost (cost). Running longer production campaigns reduces changeover waste (cost) but builds excess inventory (cash). This three-way tension is the central design principle of the dataset — every operational decision leaves a trace across all three dimensions.

The north star metric is **Return on Capital Employed (ROCE)** = EBIT / Capital Employed, which captures all three corners in a single ratio.

### SCOR-DS (Supply Chain Operations Reference — Digital Standard)

SCOR provides the process taxonomy that structures the dataset. Every supply chain operation falls into one of six domains: Source (procuring materials), Make (manufacturing products), Deliver (fulfilling orders), Return (handling reverse logistics), Plan (forecasting and scheduling), and Enable (the financial and master data that supports the other five).

The 38 ERP tables map to these domains, ensuring the dataset covers the full order-to-cash and procure-to-pay cycles with no gaps. This mapping serves as the contract between the simulation engine and the enterprise data layer — if a SCOR process exists, there is a table that captures it.

**Exhibit A: Dataset-to-Framework Map**

| SCOR Domain | Triangle Corner | Key Tables | What It Captures |
|---|---|---|---|
| Source | Cost / Cash | purchase_orders, goods_receipts, ap_invoices | Procurement cycle, supplier spend, DPO |
| Make | Cost | work_orders, batches, batch_ingredients | Production costs, yield, OEE |
| Deliver | Service / Cost | orders, shipments, shipment_lines | Fill rate, freight cost, lead times |
| Return | Service | returns, disposition_logs | Return rate, disposition |
| Plan | Service | demand_forecasts | Forecast accuracy (MAPE) |
| Finance | Cash | gl_journal, ar_invoices, ar_receipts | Revenue, DSO, C2C, working capital |

---

## 3. The Company

Prism Consumer Goods is modeled as a mid-major FMCG manufacturer — large enough to operate a multi-echelon distribution network, but not so large that the data becomes unwieldy. The company's product portfolio, channel strategy, and cost structure are calibrated against public data from Colgate-Palmolive, Procter & Gamble, and Unilever.

### Product Portfolio

PCG operates in three product categories, each with distinct manufacturing characteristics. Oral Care products (toothpaste, mouthwash) run at higher line speeds but require shorter changeovers. Home Care products (dish liquid, surface cleaners) have lower unit economics but higher volume per case. Personal Wash sits between the two.

**Exhibit B: Product Portfolio**

| Dimension | Detail |
|---|---|
| Categories | Oral Care (45%), Personal Wash (30%), Home Care (25%) |
| SKU Count | 500 finished goods |
| Packaging Formats | 28 formats across 5 container types (tubes, bottles, pumps, pouches, glass jars) |
| Value Segments | Trial (7%), Mainstream (48%), Value (28%), Premium (17%) |
| BOM Depth | 3-level: 78 raw materials → 45 bulk intermediates → 500 finished goods |

### Channel Strategy

PCG serves seven distinct channels, each with different order patterns, margin profiles, and payment terms. Mass Retail and Grocery dominate volume. E-commerce carries higher margins but also higher fulfillment costs. DTC (direct-to-consumer) is a small but high-margin channel with prepaid terms.

**Exhibit C: Channel Mix**

| Channel | Revenue Share | Payment Terms | Locations |
|---|---|---|---|
| Mass Retail | 30% | Net 45 | 1,500 stores |
| Grocery | 20% | Net 30 | 1,000 stores |
| E-commerce | 18% | Net 15 | 18 fulfillment centers |
| Club | 14% | Net 30 | 47 stores |
| Distributor | 9% | Net 30 | 900 independent retailers |
| Pharmacy | 6% | Net 30 | 375 stores |
| DTC | 3% | Prepaid | 3 fulfillment centers |

---

## 4. The Supply Chain

PCG operates a four-echelon distribution network spanning North America: suppliers, plants, regional distribution centers (RDCs), and customer-facing locations (DCs, stores, fulfillment centers).

### Network Topology

Raw materials flow inbound from 50 suppliers, segmented using a Kraljic matrix: 8 strategic suppliers (specialty chemicals with few alternatives), 12 leverage suppliers (commodity chemicals where PCG has buying power), and 30 non-critical suppliers (packaging and tertiary materials). This segmentation drives procurement strategy — strategic suppliers get longer-term contracts and higher safety stock, while non-critical suppliers compete on price.

Manufacturing is concentrated in four plants, each with category specializations: Dallas handles high-speed oral care production, Columbus runs legacy home care and personal wash lines, Sacramento provides flexible multi-category capacity, and Atlanta serves as the general-purpose overflow facility.

Finished goods flow outbound through two paths. High-volume A-items ship direct from plant to customer DCs — this "plant-direct" path is faster and cheaper but requires sufficient volume to fill trucks. The remaining volume routes through six regional distribution centers that serve as consolidation and cross-dock points. In the simulated year, approximately 56% of plant output flows through RDCs, with 44% shipping direct. This dual-path topology creates realistic complexity in inventory positioning and transportation planning.

**Exhibit D: Network Topology**

| Tier | Nodes | Locations |
|---|---|---|
| Suppliers | 50 | Kraljic-segmented: 8 strategic, 12 leverage, 30 non-critical |
| Plants | 4 | Columbus OH, Dallas TX, Atlanta GA, Sacramento CA |
| RDCs | 6 | Allentown PA, Chicago IL, Memphis TN, Jacksonville FL, Phoenix AZ, Reno NV |
| Customer DCs | 56 | Retailer, grocery, club, pharmacy, and distributor warehouses |
| Endpoints | 3,843 | Retail stores and fulfillment centers across 7 channels |
| **Total** | **~4,200** | |

### Transportation

All inter-facility freight moves by full truckload (FTL) at 20,000 kg capacity, with the exception of DC-to-store deliveries which use less-than-truckload (LTL). Lead times are distance-based — computed from actual geographic coordinates using road-network speeds of 80 km/h plus a one-day handling buffer at each node.

**Exhibit E: Lead Times**

| Leg | Days | Mode |
|---|---|---|
| Supplier → Plant | 1–7 | FTL |
| Plant → RDC | ~5 | FTL |
| RDC → Customer DC | ~3 | FTL |
| DC → Store | 1 | LTL |
| **End-to-end** | **13–21** | — |

### Manufacturing

PCG's four plants operate 15 production lines on a 24-hour continuous schedule. Each line is dedicated to specific product categories based on the equipment installed — oral care lines cannot run home care products without a full retool. Production follows a two-stage BOM: raw materials are first blended into bulk intermediates (mixing, compounding), then bulk is filled and packed into finished goods (filling, labeling, cartoning, palletizing). This two-stage structure means a single batch of bulk intermediate may feed multiple finished goods SKUs that share the same formula but differ in packaging format.

**Exhibit F: Manufacturing**

| Parameter | Value |
|---|---|
| Production lines | 15 across 4 plants |
| Operating schedule | 24 hours/day, continuous |
| Run rates | 15,000–20,000 cases/hour (varies by category) |
| Yield | 98.5% |
| Changeover time | 0.5–1.5 hours (varies by category) |
| BOM stages | Two-stage: bulk blending, then fill-and-pack |

---

## 5. Demand & Planning

### Consumer Demand

Store-level demand follows a Zipf distribution — the top 20% of SKUs account for approximately 80% of volume, consistent with the Pareto pattern observed in real FMCG data. Each store's daily demand is specific to its channel and format (a hypermarket sells more than a pharmacy), creating natural volume heterogeneity across the network. Demand exhibits seasonal variation with ±12% amplitude peaking in summer months, plus promotional lifts of up to 2× during events like Black Friday and New Year sales. Post-promotion hangover effects (demand dips of 20–40% in the week following a promotion) are also modeled.

SKUs are classified into ABC tiers by velocity: 302 A-items (fast movers), 127 B-items, and 71 C-items. This classification drives differentiated inventory policies, production scheduling priority, and safety stock levels throughout the network.

### Inventory Policy

Each echelon operates under a min-max (s,S) replenishment policy with ABC-differentiated parameters. Stores carry 6 days of supply with a 3-day reorder point. Customer DCs maintain a 7-day buffer with ABC-scaled caps. RDCs operate as flow-through points with a 9-day target. Plant finished goods inventory is managed by MRP with a 14-day planning horizon.

**Exhibit G: Inventory Policy**

| Echelon | Target DOS | Reorder Point | Safety Stock |
|---|---|---|---|
| Store | 6 days | 3 days | 1.65–2.33σ (ABC-tiered) |
| Customer DC | 10–18 days | ABC-differentiated | 7-day buffer |
| RDC | 9 days | Flow-through | Cross-dock model |
| Plant FG | 14–17 days | MRP-driven | Production smoothing |

Production planning uses a 14-day rolling MRP horizon with ABC-weighted capacity allocation: A-items receive 60% of line capacity, B-items 25%, and C-items 15%. Purchase orders for raw materials are consolidated over 2-day windows to meet minimum truckload weights.

---

## 6. How the Data Was Generated

The dataset is produced by a discrete-event simulation (DES) that models 365 days of PCG's operations at daily granularity. Understanding the generation method is important because it determines what causal relationships exist in the data and what analytical questions the dataset can support.

### The Daily Loop

Each simulated day executes a fixed sequence of operations:

1. **Generate consumer demand** — store-level sales are drawn from the Zipf/seasonal/promotional model.
2. **Replenish stores** — stores place replenishment orders to their supplying DCs based on current inventory position.
3. **Allocate inventory** — available stock is allocated to open orders using fair-share logic with ABC priority.
4. **Build shipments** — allocated orders are consolidated into truckloads using bin-packing.
5. **Plan production** — MRP calculates net requirements and schedules production batches.
6. **Execute manufacturing** — production lines run scheduled batches, consuming raw materials and producing finished goods.
7. **Deploy finished goods** — plant output is pushed to RDCs and DCs based on demand signals.

### Physics Constraints

Five non-negotiable constraints are enforced at every timestep:

- **Mass balance:** input (kg) = output (kg) + scrap. Nothing is created or destroyed.
- **Kinematic consistency:** travel time = distance / speed. Shipments cannot arrive before physics allows.
- **Little's Law:** inventory = throughput × flow time. The fundamental relationship between stock and flow.
- **Capacity constraints:** production cannot exceed line rate × available hours.
- **Inventory positivity:** a node cannot ship more than it holds. No negative inventory.

### Behavioral Realism

On top of physics, the simulation includes mechanisms for real-world messiness: phantom inventory (2% shrinkage with a 14-day detection lag), bullwhip amplification (3× order batching during promotions), forecast optimism bias (15% over-forecast for new products in their first 6 months), and port congestion (autoregressive shipment delays that cluster temporally).

### Two-Pass Architecture

The data generation follows a two-pass architecture. First, the simulation engine produces raw operational data — shipments, batches, inventory snapshots, consumer demand — enforcing all physics constraints. Second, a post-hoc process reads this operational data and generates enterprise artifacts: purchase orders, invoices, GL journal entries, and payments. This mirrors how real ERP systems record financial transactions after physical events occur. The separation ensures that financial data is always consistent with the underlying operations.

### Warm-Start Convergence

The 365-day dataset starts from a pre-converged state — inventory levels, in-transit shipments, and pipeline stock are initialized to steady-state values. This means day 1 is operationally realistic, not a cold-start ramp. A 3-day stabilization period at the beginning allows minor transients to settle before the full simulation begins.

---

## 7. Operating Performance

The simulated year produces a realistic set of operating metrics that reflect the Service–Cost–Cash tradeoffs inherent in PCG's supply chain configuration. These metrics can be independently derived from the raw data in the 38 ERP tables.

**Exhibit H: Supply Chain Triangle**

| Dimension | Metric | Value |
|---|---|---|
| **Service** | Store Fill Rate | 94.4% |
| | — A-items | 98.4% |
| | — B-items | 94.5% |
| | — C-items | 88.1% |
| | Perfect Order Rate | 97.5% |
| **Cash** | Inventory Turns | 12.25× |
| | Cash-to-Cash Cycle | 14.9 days |
| **Cost** | Truck Fill Rate (FTL) | 96.3% |
| | OEE | 53.8% |
| | Gross Margin | 23.2% |

The ABC-tiered fill rate pattern is characteristic of real FMCG operations: A-items (fast movers) receive production and allocation priority, achieving near-perfect availability, while C-items (slow movers) experience more frequent stockouts due to longer production cycles and lower safety stock coverage.

**Exhibit I: P&L Summary**

| Line | Amount |
|---|---|
| Revenue | $38.3B |
| Cost of Goods Sold | ($25.8B) |
| Freight Expense | ($1.75B) |
| Manufacturing Overhead | ($1.63B) |
| Returns & Bad Debt | ($0.21B) |
| **Gross Profit** | **$8.9B (23.2%)** |

Freight represents 4.6% of revenue — consistent with industry benchmarks for a primarily domestic FTL network. Manufacturing overhead includes labor (proportional to material cost by category) and facility costs. Bad debt ($189M) arises from the 0.5% of AR invoices that are never collected, a controlled friction parameter discussed in Section 9.

---

## 8. The Enterprise Dataset

The ERP export contains 38 normalized tables organized across nine domains, covering the full order-to-cash and procure-to-pay cycles. Master data tables (14 tables) define the structural entities — suppliers, plants, SKUs, formulas, locations, channels, chart of accounts, and the transport network. Transactional tables (24 tables) capture the operational events — orders, shipments, production batches, inventory snapshots, invoices, payments, and journal entries. The schema follows third normal form with foreign key relationships that enable cross-domain joins.

**Exhibit J: Dataset by Domain**

| Domain | Tables | Key Tables | Rows |
|---|---|---|---|
| Source (Procurement) | 7 | purchase_orders, goods_receipts, ap_invoices | 12.2M |
| Transform (Manufacturing) | 7 | work_orders, batches, batch_ingredients | 1.4M |
| Product (SKU Master) | 2 | skus, bulk_intermediates | 566 |
| Order (Demand) | 3 | orders, order_lines | 63.3M |
| Fulfill (Outbound) | 5 | shipments, shipment_lines, inventory | 178M |
| Logistics | 1 | route_segments | 4K |
| Plan | 1 | demand_forecasts | 199K |
| Return | 3 | returns, disposition_logs | 215K |
| Finance | 9 | gl_journal, ap/ar_invoices, payments | 139M |
| **Total** | **38** | — | **~395M** |

The five largest tables — inventory snapshots (98.6M), shipment lines (71.6M), order lines (61.9M), GL journal (58.9M), and AR invoice lines (58.4M) — account for 88% of total rows. These tables are the primary targets for analytical queries and integration testing.

### General Ledger

The GL journal is the financial backbone of the dataset. Every physical event — goods receipt, production batch, shipment, return, payment — generates balanced debit/credit entries that trace back to the source transaction.

**Exhibit K: GL Journal Structure**

| Property | Value |
|---|---|
| Total entries | 58.9 million |
| Balance check | $2,194.3B debits = $2,194.3B credits (balanced to the penny) |
| Event types | 7: goods_receipt, production, shipment, return, payment, receipt, bad_debt |
| Chart of accounts | 14 accounts across asset, liability, revenue, and expense categories |
| Reference traceability | 100% of entries link to source transaction via reference_id |
| Node attribution | 99% of physical events carry originating facility; treasury events unattributed by design |

---

## 9. Data Quality & Friction

The dataset intentionally includes controlled data quality issues that mirror real enterprise systems. These are not bugs — they are a feature designed to test entity resolution, record matching, anomaly detection, and data cleaning pipelines. Every friction parameter is seeded deterministically for reproducibility.

The friction layer operates across four categories:

- **Entity resolution** — duplicate supplier records and legacy SKU codes create the same real-world ambiguity that plagues master data management in large enterprises.
- **Three-way match failures** — invoiced prices and quantities that don't match purchase orders, requiring variance investigation and reconciliation.
- **Data quality gaps** — missing foreign keys, duplicate invoices, and status inconsistencies that test referential integrity checks.
- **Payment timing noise** — early/late payments around contractual terms, early-pay discounts, and bad debt that affect working capital calculations.

**Exhibit L: Friction Layer**

| Issue | Rate | Example |
|---|---|---|
| Duplicate suppliers | 12% | Same supplier, different name/code ("-ALT" suffix) |
| SKU renames | 4.2% | Legacy codes still in transaction history ("-OLD" suffix) |
| Invoice price variance | 8.0% | Invoiced unit price differs from PO price by 2–15% |
| Invoice quantity variance | 5.0% | Received quantity differs from ordered quantity by 1–10% |
| Duplicate invoices | 0.5% | Double-billed invoices ("-DUP" suffix) |
| Missing foreign keys | 1–2% | Null supplier_id on AP invoices, null references in GL |
| Bad debt | 0.5% | AR invoices that are never collected |
| Payment timing noise | ±5–7 days | Early/late payments around contractual terms |
| Early-pay discounts | 10% | 2% discount if paid within 10 days |

All friction rates are independently verifiable from the data. For example, the duplicate supplier rate can be confirmed by counting suppliers whose name contains "-ALT" relative to the total supplier count. The price variance rate can be confirmed from the invoice_variances table. Because every friction parameter is controlled and documented, the dataset provides a known ground truth for benchmarking data quality tools — analysts know exactly how many duplicates, mismatches, and gaps exist and can measure their detection rate against it.

---

## 10. Diagnostic Validation

The dataset has been validated by two independent diagnostic suites that together evaluate 99 questions across physics compliance, financial integrity, process coverage, and data quality.

**Supply chain diagnostic (35 questions):** Validates the operational data against physics laws and industry benchmarks. Checks mass balance conservation, Little's Law consistency, echelon flow stability, fill rate by ABC tier, inventory turns, OEE, bullwhip ratio, and forecast accuracy. Result: 5 GREEN / 3 YELLOW / 0 RED across the executive scorecard. All physics checks pass. The three YELLOW indicators reflect deliberate design choices — fill rate (94.4%) prioritizes inventory efficiency over 97%+ service, OEE (53.8%) reflects realistic utilization with changeovers and demand variability, and inventory turns (12.25×) slightly exceed the benchmark range.

**ERP database diagnostic (64 questions):** Validates the enterprise data layer against the operational data. Checks physical-financial reconciliation (do GL entries match shipments and batches?), SCOR process compliance (is every process domain populated?), temporal and causal integrity (do events occur in the correct sequence?), friction layer audit (are data quality issues injected at the configured rates?), and digital thread traceability (can every finished good be traced back through batches, ingredients, and suppliers?). Result: **62 PASS / 2 WARN / 0 FAIL**.

Key validations:

- GL balanced to the penny across all 365 days — zero imbalance on any single day
- Zero time-travel violations — no payment before invoice, no arrival before shipment, no goods receipt before purchase order
- 100% digital thread coverage — every GL entry links to its source transaction via reference_id
- Full procurement chain traceability — 96% of goods receipts can be traced through AP invoice to payment
- All 38 tables populated with non-zero row counts
- Friction rates within ±1% of configured targets across all nine categories
