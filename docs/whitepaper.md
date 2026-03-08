# Prism Consumer Goods — Synthetic Enterprise Dataset

## 1. Executive Summary

This dataset represents a full year of synthetic enterprise data for **Prism Consumer Goods (PCG)**, a fictional North American fast-moving consumer goods company. PCG manufactures and distributes oral care, personal wash, and home care products through seven channels to over 3,800 retail and fulfillment locations.

The modeled company operates at scale: ~640 products (500 finished goods, 62 bulk intermediates including 13 PREMIX sub-intermediates, and 78 raw materials), 50 Kraljic-segmented suppliers (~38 active), four manufacturing plants, and a 4-echelon distribution network spanning roughly 4,200 nodes with lateral RDC transshipment links and multi-source DC routing. The dataset captures the full operational footprint — procurement, manufacturing, inventory management, order fulfillment, returns, and financial accounting — across 38 normalized ERP tables totaling approximately 330-400 million rows depending on run configuration.

> **Note on metrics:** Exact row counts, financial totals, and performance metrics vary by run length, warm-start state, and configuration parameters. Values throughout this document are representative of a typical 340-365 day simulation run. The illustrative run used here covers 340 operational days with a 90-day financial tail.

The dataset is purpose-built for three use cases: **Virtual Knowledge Graph (VKG) testing**, where heterogeneous enterprise tables must be integrated and queried as a unified graph (85 benchmark questions with real entity codes are provided); **supply chain analytics**, where analysts need realistic volume, variability, and cross-functional tradeoffs; and **data integration benchmarking**, where controlled data quality issues provide a known ground truth for entity resolution, record matching, and anomaly detection.

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
| Source | Cost / Cash | purchase_orders, goods_receipts, ap_invoices, ap_payments | Procurement cycle, supplier spend, DPO |
| Make | Cost | work_orders, batches, batch_ingredients (incl. PREMIX) | Production costs, yield, OEE, variable BOM depth |
| Deliver | Service / Cost | orders, shipments, shipment_lines | Fill rate, freight cost, lead times |
| Return | Service | returns, disposition_logs | Return rate, disposition |
| Plan | Service | demand_forecasts | Forecast accuracy (MAPE) |
| Finance | Cash | gl_journal, ar_invoices, ar_receipts, invoice_variances | Revenue, DSO, C2C, working capital, 3-way match |

---

## 3. The Company

Prism Consumer Goods is modeled as a mid-major FMCG manufacturer — large enough to operate a multi-echelon distribution network, but not so large that the data becomes unwieldy. The company's product portfolio, channel strategy, and cost structure are calibrated against public data from Colgate-Palmolive, Procter & Gamble, and Unilever.

### Product Portfolio

PCG operates in three product categories, each with distinct manufacturing characteristics. Oral Care products (toothpaste, mouthwash) run at higher line speeds but require shorter changeovers. Home Care products (dish liquid, surface cleaners) have lower unit economics but higher volume per case. Personal Wash sits between the two.

**Exhibit B: Product Portfolio**

| Dimension | Detail |
|---|---|
| Categories | Oral Care (45%), Personal Wash (30%), Home Care (25%) |
| Finished Goods | 500 SKUs |
| Total Products | ~640: 500 FG + 62 bulk intermediates (incl. 13 PREMIX) + 78 raw materials |
| Packaging Formats | 28 formats across 5 container types (tubes, bottles, pumps, pouches, glass jars) |
| Value Segments | Trial (7%), Mainstream (48%), Value (28%), Premium (17%) |
| BOM Depth | Variable 2-4 level: 78 raw materials → 13 PREMIX sub-intermediates → 62 bulk intermediates → 500 finished goods |

The variable BOM depth means that approximately 70% of products follow the standard 2-level path (raw materials → bulk → finished goods), 20% follow a 3-level path through PREMIX sub-intermediates, and 10% use multi-intermediate blending with primary and secondary bulk inputs. Ten diamond dependencies arise naturally where raw materials are shared across PREMIX recipes and direct bulk formulations.

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

Raw materials flow inbound from 50 suppliers, segmented using a Kraljic matrix: 8 strategic suppliers (specialty chemicals with few alternatives), 12 leverage suppliers (commodity chemicals where PCG has buying power), and 30 non-critical suppliers (packaging and tertiary materials). This segmentation drives procurement strategy — strategic suppliers get longer-term contracts and higher safety stock, while non-critical suppliers compete on price. Of the 50 suppliers in the catalog, approximately 38 are active in a given simulation run, linked to 78 ingredients through 176 supplier-ingredient relationships.

Manufacturing is concentrated in four plants, each with category specializations: Dallas handles high-speed oral care production, Columbus runs legacy home care and personal wash lines, Sacramento provides flexible multi-category capacity, and Atlanta serves as the general-purpose overflow facility.

Finished goods flow outbound through two paths. High-volume A-items ship direct from plant to customer DCs — this "plant-direct" path is faster and cheaper but requires sufficient volume to fill trucks. The remaining volume routes through six regional distribution centers that serve as consolidation and cross-dock points. The six RDCs are connected by 12 directed lateral transshipment links (6 pairs), enabling DOS-triggered inventory rebalancing between regions. Approximately 16 customer DCs (29%) are configured as multi-source, receiving product from both a primary and secondary RDC to improve resilience. This dual-path topology with lateral links creates realistic complexity in inventory positioning and transportation planning.

**Exhibit D: Network Topology**

| Tier | Nodes | Detail |
|---|---|---|
| Suppliers | 50 | Kraljic-segmented: 8 strategic, 12 leverage, 30 non-critical (~38 active) |
| Plants | 4 | Columbus OH, Dallas TX, Atlanta GA, Sacramento CA |
| RDCs | 6 | Allentown PA, Chicago IL, Memphis TN, Jacksonville FL, Phoenix AZ, Reno NV |
| Lateral RDC Links | 12 | 6 bidirectional pairs for DOS-triggered transshipment |
| Customer DCs | 56 | Retailer, grocery, club, pharmacy, and distributor warehouses (16 multi-source) |
| Endpoints | 3,817 | Retail stores and fulfillment centers across 7 channels |
| Route Segments | 4,025 | Total directed links in the transport network |
| **Total Nodes** | **~4,200** | |

### Transportation

All inter-facility freight moves by full truckload (FTL) at 20,000 kg capacity, with the exception of DC-to-store deliveries which use less-than-truckload (LTL). Internal network lead times are distance-based — computed from actual geographic coordinates using road-network speeds of 80 km/h plus a one-day handling buffer at each node.

Supplier lead times are sourcing-tier aware: LOCAL suppliers deliver in 5-14 days, REGIONAL in 15-29 days, and GLOBAL in 30-72 days. Across the 176 supplier-ingredient relationships, there are 55 distinct lead time values spanning the full 5-72 day range, with an average of ~24 days.

**Exhibit E: Lead Times**

| Leg | Days | Mode |
|---|---|---|
| Supplier → Plant | 5–72 (tier-dependent) | FTL |
| Plant → RDC | ~2 | FTL |
| RDC → Customer DC | ~1 | FTL |
| DC → Store | 1 | LTL |
| RDC ↔ RDC (lateral) | ~1 | FTL |
| **End-to-end** | **9–76** | — |

### Manufacturing

PCG's four plants operate 15 production lines on a 24-hour continuous schedule. Each line is dedicated to specific product categories based on the equipment installed — oral care lines cannot run home care products without a full retool. Production follows a variable-depth BOM: raw materials may first be blended into PREMIX sub-intermediates (13 products at bom_level=2), then compounded into bulk intermediates (62 products at bom_level=1), and finally filled and packed into finished goods (500 SKUs). This variable-depth structure means a single raw material may feed multiple PREMIX recipes, which in turn feed multiple bulk formulas — creating diamond dependencies in the production graph. Batch ingredient quantities are recorded with a +/-3% variance from formula specifications, reflecting real-world measurement imprecision.

**Exhibit F: Manufacturing**

| Parameter | Value |
|---|---|
| Production lines | 15 across 4 plants |
| Operating schedule | 24 hours/day, continuous |
| Run rates | 15,000-20,000 cases/hour (varies by category) |
| Yield | 98.5% |
| Changeover time | 0.5-1.5 hours (varies by category) |
| BOM stages | Variable 2-4 stage: PREMIX blending → bulk compounding → fill-and-pack |
| PREMIX products | 13 sub-intermediates (bom_level=2) |
| Batch variance | +/-3% ingredient recording variance |

---

## 5. Demand & Planning

### Consumer Demand

Store-level demand follows a Zipf distribution — the top 20% of SKUs account for approximately 80% of volume, consistent with the Pareto pattern observed in real FMCG data. Each store's daily demand is specific to its channel and format (a hypermarket sells more than a pharmacy), creating natural volume heterogeneity across the network. Demand exhibits seasonal variation with +/-12% amplitude peaking in summer months, plus promotional lifts of up to 2x during events like Black Friday and New Year sales. Post-promotion hangover effects (demand dips of 20-40% in the week following a promotion) are also modeled.

SKUs are classified into ABC tiers by velocity: 302 A-items (fast movers), 127 B-items, and 71 C-items. This classification drives differentiated inventory policies, production scheduling priority, and safety stock levels throughout the network.

### Demand-Centric Replenishment

Every echelon in the network is synchronized to a "True Demand" signal — smoothed retail pull from stores — rather than relying on each node's own outflow history. This demand-centric model prevents the "death spiral" where low inventory causes low shipments, which then causes even lower replenishment orders. DCs use the smoothed retail demand signal for their inventory targets, DOS calculations, and floor gating, ensuring that ordering velocity recovers quickly after disruptions.

### Inventory Policy

Each echelon operates under a min-max (s,S) replenishment policy with ABC-differentiated parameters.

**Exhibit G: Inventory Policy**

| Echelon | Target DOS | Reorder Point | Safety Stock |
|---|---|---|---|
| Store | 4.5 days | 3 days | 1.65-2.33 sigma (ABC-tiered) |
| Customer DC | 7-day buffer | ABC-differentiated caps (A: 10.5, B: 14, C: 17.5 DOS) | True Demand-anchored |
| RDC | 9 days | Flow-through | Cross-dock model |
| Plant FG | 1.5 DOS priming | MRP-driven | Integral Netting: On_Hand = Target - Pipeline - WIP |

### Distribution Planning

Distribution Requirements Planning (DRP) is available but disabled by default for RDC-to-DC shipping. DRP uses smoothed expected demand per product, but store (s,S) ordering creates lumpy, SKU-specific order patterns. The result is a product-mix mismatch: DRP positions the right total volume but the wrong SKU mix at DCs. In testing, DC pull via (s,S) achieves ~95% fill rate, pure DRP achieves ~73%, and a hybrid mode achieves ~93%. The default configuration uses DC pull for product-mix accuracy.

### Production Planning

MRP uses a 14-day rolling horizon with ABC-weighted capacity allocation: A-items receive 60% of line capacity, B-items 25%, and C-items 15%. The MRP engine performs N-step level-by-level BOM explosion to support the variable depth product graph — PREMIX requirements are planned before bulk, which is planned before finished goods. Purchase orders for raw materials are consolidated over 2-day windows to meet minimum truckload weights.

---

## 6. How the Data Was Generated

The dataset is produced by a discrete-event simulation (DES) that models a full year of PCG's operations at daily granularity. Understanding the generation method is important because it determines what causal relationships exist in the data and what analytical questions the dataset can support.

### The Daily Loop

Each simulated day executes a fixed sequence of operations:

1. **Generate consumer demand** — store-level sales are drawn from the Zipf/seasonal/promotional model.
2. **Replenish stores** — stores place replenishment orders to their supplying DCs based on current inventory position and the True Demand signal.
3. **Allocate inventory** — available stock is allocated to open orders using fair-share logic with ABC priority.
4. **Build shipments** — allocated orders are consolidated into truckloads using bin-packing.
5. **Plan production** — MRP performs N-step BOM explosion and schedules production batches; DRP computes distribution needs for B/C items.
6. **Execute manufacturing** — production lines run scheduled batches, consuming raw materials and producing finished goods with +/-3% ingredient variance.
7. **Deploy finished goods** — plant output is pushed to RDCs and DCs based on need (deployment target room check).
8. **Push excess RDC inventory** — RDCs above target DOS push surplus to downstream DCs.
9. **Execute lateral transshipment** — RDCs below DOS threshold pull from neighboring RDCs via lateral links.
10. **Inject behavioral realism** — phantom inventory shrinkage, bullwhip amplification, forecast bias, and port congestion effects.

### Physics Constraints

Five non-negotiable constraints are enforced at every timestep:

- **Mass balance:** input (kg) = output (kg) + scrap. Nothing is created or destroyed.
- **Kinematic consistency:** travel time = distance / speed. Shipments cannot arrive before physics allows.
- **Little's Law:** inventory = throughput x flow time. The fundamental relationship between stock and flow.
- **Capacity constraints:** production cannot exceed line rate x available hours.
- **Inventory positivity:** a node cannot ship more than it holds. No negative inventory.

### Behavioral Realism

On top of physics, the simulation includes mechanisms for real-world messiness: phantom inventory (2% shrinkage with a 14-day detection lag), bullwhip amplification (3x order batching during promotions), forecast optimism bias (15% over-forecast for new products in their first 6 months), port congestion (autoregressive shipment delays that cluster temporally), batch ingredient variance (+/-3% recording deviation from formula specifications), and demand-centric signal smoothing (70% expected demand + 30% POS for DOS calculation).

### Two-Pass Architecture

The data generation follows a two-pass architecture. First, the simulation engine produces raw operational data — shipments, batches, inventory snapshots, consumer demand — enforcing all physics constraints. Second, a post-hoc "Accountant Bot" process reads this operational data and generates enterprise artifacts: purchase orders, invoices, GL journal entries, payments, and the friction layer (duplicate suppliers, 3-way match variances, bad debt). This mirrors how real ERP systems record financial transactions after physical events occur. The separation ensures that financial data is always consistent with the underlying operations while allowing controlled data quality issues to be injected independently.

### Warm-Start Convergence

The primary operational mode uses warm-start initialization from a previously converged run — inventory levels, in-transit shipments, and pipeline stock are loaded from parquet snapshots of a prior simulation. This means day 1 is operationally realistic, not a cold-start ramp. Integral Priming ensures that on-hand inventory is set to `Target - Pipeline - WIP`, preventing the double-counting that would otherwise cause a multi-month inventory drain transient. A 3-day stabilization period at the beginning allows minor transients to settle before the full simulation begins.

---

## 7. Operating Performance

The simulated year produces a realistic set of operating metrics that reflect the Service-Cost-Cash tradeoffs inherent in PCG's supply chain configuration. These metrics can be independently derived from the raw data in the 38 ERP tables.

> **Note:** Exact metrics vary by run configuration (run length, warm-start state, config parameters). Values below are representative of a typical 340-365 day run. The demand-centric replenishment model achieves near-perfect last-mile fill rate in steady state.

**Exhibit H: Supply Chain Triangle**

| Dimension | Metric | Typical Range |
|---|---|---|
| **Service** | Store Fill Rate | 95-100% |
| | Return Rate | <0.1% |
| **Cost** | COGS / Revenue | 65-70% |
| | Freight / Revenue | 4-6% |
| | Gross Margin | 20-23% |
| **Cash** | DSO | 35-42 days |
| | DPO | ~45 days (config-driven) |
| | DIO | 35-45 days |
| | Cash-to-Cash Cycle | 25-40 days |

The ABC-tiered pattern characteristic of real FMCG operations is preserved: A-items (fast movers) receive production and allocation priority, achieving near-perfect availability, while C-items (slow movers) experience more frequent stockouts due to longer production cycles and lower safety stock coverage. With the demand-centric model, overall fill rate is substantially higher than under traditional outflow-based replenishment.

**Exhibit I: P&L Summary (Illustrative ~340-Day Run)**

| Line | Amount |
|---|---|
| Revenue | ~$25B |
| Cost of Goods Sold | (~$17B) |
| Freight Expense | (~$1.3B) |
| Manufacturing Overhead | (~$1.3B) |
| Returns & Bad Debt | (~$0.14B) |
| **Gross Profit** | **~$5.3B (~21-22%)** |

Freight represents approximately 5% of revenue — consistent with industry benchmarks for a primarily domestic FTL network. Manufacturing overhead includes labor (proportional to material cost by category) and facility costs. Bad debt (~$123M) arises from the 0.5% of AR invoices that are never collected, a controlled friction parameter discussed in Section 9. Revenue scales roughly linearly with run length; a full 365-day run produces ~$38B.

---

## 8. The Enterprise Dataset

The ERP export contains 38 normalized tables organized across ten domains, covering the full order-to-cash and procure-to-pay cycles. Master data tables (14 tables) define the structural entities — suppliers, plants, SKUs, bulk intermediates, formulas, locations, channels, chart of accounts, and the transport network. Transactional tables (24 tables) capture the operational events — orders, shipments, production batches, inventory snapshots, invoices, payments, and journal entries. The schema follows third normal form with foreign key relationships that enable cross-domain joins.

**Exhibit J: Dataset by Domain (Illustrative ~340-Day Run)**

| Domain | Tables | Key Tables | Rows |
|---|---|---|---|
| Source (Procurement) | 7 | purchase_orders, goods_receipts, ap_invoices, ap_payments | ~19.2M |
| Transform (Manufacturing) | 3 | work_orders, batches, batch_ingredients | ~1.2M |
| Product (Master) | 4 | skus, bulk_intermediates, formulas, formula_ingredients | ~5.4K |
| Order (Demand) | 2 | orders, order_lines | ~57.4M |
| Fulfill (Outbound) | 3 | shipments, shipment_lines, inventory | ~148M |
| Logistics | 1 | route_segments | ~4K |
| Plan | 1 | demand_forecasts | ~191K |
| Return | 3 | returns, return_lines, disposition_logs | ~199K |
| Finance | 5 | gl_journal, ar_invoices, ar_invoice_lines, ar_receipts, invoice_variances | ~104.3M |
| Reference | 9 | suppliers, ingredients, plants, DCs, retail_locations, channels, chart_of_accounts, production_lines, supplier_ingredients | ~4K |
| **Total** | **38** | — | **~330M** |

The five largest tables — inventory snapshots (79.2M), shipment lines (62.6M), order lines (56.1M), AR invoice lines (52.7M), and GL journal (48.4M) — account for approximately 90% of total rows. These tables are the primary targets for analytical queries and integration testing. Row counts scale with run length; a full 365-day run produces ~400M rows.

### General Ledger

The GL journal is the financial backbone of the dataset. Every physical event — goods receipt, production batch, shipment, return, payment — generates balanced debit/credit entries that trace back to the source transaction.

**Exhibit K: GL Journal Structure**

| Property | Value |
|---|---|
| Total entries | ~48.4M |
| Balance check | ~$1,705B debits = ~$1,705B credits (imbalance ~$2) |
| Reference types | 11: goods_receipt, production, shipment, freight, sale, return, payment, receipt, bad_debt, price_variance, qty_variance |
| Chart of accounts | 14 accounts: 6 asset, 1 liability, 2 revenue, 5 expense |
| Reference traceability | 100% of entries link to source transaction via reference_id |
| Node attribution | ~99% of physical events carry originating facility; treasury events (payment, receipt, bad_debt) unattributed by design |

The 14-account chart of accounts includes: Cash (1000), Raw Material Inventory (1100), Work In Process (1120), Finished Goods Inventory (1130), In-Transit Inventory (1140), Accounts Receivable (1200), Accounts Payable (2100), Revenue (4100), Discount Income (4200), Cost of Goods Sold (5100), Returns Expense (5200), Freight Expense (5300), Manufacturing Overhead (5400), and Bad Debt Expense (5500).

---

## 9. Data Quality & Friction

The dataset intentionally includes controlled data quality issues that mirror real enterprise systems. These are not bugs — they are a feature designed to test entity resolution, record matching, anomaly detection, and data cleaning pipelines. Every friction parameter is seeded deterministically for reproducibility.

The friction layer operates across four tiers:

**Tier 1 — Entity Resolution:** Duplicate supplier records (12%, "-ALT" suffix) and legacy SKU codes (4.2%, "-OLD" suffix with `supersedes_sku_id` foreign key) create the same real-world ambiguity that plagues master data management in large enterprises. SKU alias chains are guaranteed to be depth-1 (no multi-generation chains), with no cycles and valid foreign keys to active target SKUs.

**Tier 2 — Three-Way Match Failures:** Invoiced prices differ from PO prices by 2-15% on 8.0% of AP invoice lines (price variance), and received quantities differ from ordered quantities by 1-10% on 5.0% of lines (quantity variance). All variances are tracked in the `invoice_variances` table with the original PO value, invoiced value, and variance amount — providing ground truth for match-failure detection algorithms.

**Tier 3 — Data Quality Gaps:** Null foreign keys on AP invoices (2% missing `gr_id`), duplicate invoices (0.5%, "-DUP" suffix, each with its own line items), and GL rounding imbalance on approximately 4 days across the full run (by design, reflecting real-world penny rounding in high-volume accounting).

**Tier 4 — Payment Timing:** Early-pay discounts (10% of AP payments receive a 2% discount for payment within 10 days), bad debt (0.5% of AR invoices are never collected, with corresponding GL bad_debt entries), and payment timing noise (+/-5-7 days around contractual terms).

**Exhibit L: Friction Layer**

| Issue | Rate | Mechanism |
|---|---|---|
| Duplicate suppliers | 12% | Same supplier, different name/code ("-ALT" suffix) |
| SKU renames | 4.2% | Legacy codes in transaction history ("-OLD" suffix, `supersedes_sku_id`) |
| Invoice price variance | 8.0% | Invoiced unit price differs from PO price by 2-15% |
| Invoice quantity variance | 5.0% | Received quantity differs from ordered quantity by 1-10% |
| Duplicate invoices | 0.5% | Double-billed invoices ("-DUP" suffix, with line items) |
| Missing foreign keys | ~2% | Null `gr_id` on AP invoices |
| GL rounding imbalance | ~4 days | Daily imbalance >$0.10 due to high-volume penny rounding |
| Bad debt | 0.5% | AR invoices never collected (GL bad_debt entries) |
| Payment timing noise | +/-5-7 days | Early/late payments around contractual terms |
| Early-pay discounts | 10% | 2% discount if paid within 10 days |

All friction rates are independently verifiable from the data and confirmed within +/-1% of configured targets. For example, the duplicate supplier rate can be confirmed by counting suppliers whose code contains "-ALT" relative to the base supplier count. The price variance rate can be confirmed from the `invoice_variances` table. Because every friction parameter is controlled and documented, the dataset provides a known ground truth for benchmarking data quality tools — analysts know exactly how many duplicates, mismatches, and gaps exist and can measure their detection rate against it.

---

## 10. Diagnostic Validation

The dataset has been validated by two independent diagnostic suites and a VKG benchmark question set that together provide comprehensive coverage of physics compliance, financial integrity, process coverage, data quality, and analytical queryability.

**Supply chain diagnostic (35 questions):** Validates the operational data against physics laws and industry benchmarks across 8 sections: physics, scorecard, service, inventory, flow, manufacturing, financial, and deep-dive. Checks mass balance conservation, Little's Law consistency, echelon flow stability, fill rate by ABC tier, inventory turns, OEE, bullwhip ratio, and forecast accuracy. All physics checks pass.

**ERP database diagnostic (59 questions):** Validates the enterprise data layer against the operational data across 10 sections: data landscape, physical-financial reconciliation, SCOR Source/Make/Deliver/Return processes, Desmet's Triangle metrics, temporal and causal integrity, friction layer audit, and digital thread traceability. Checks include GL balance verification, SCOR process compliance, temporal sequencing, friction rate auditing, and end-to-end procurement chain tracing.

**VKG benchmark questions (85 questions):** Tests the dataset's ability to support realistic analytical queries using real entity codes across 9 sections: lookups and joins, SKU aliases and BOM depth, BOM explosion and cost rollup, transport network analysis, cross-domain composition, multi-graph and polymorphism, data integrity and metamodel, lifecycle and flow analysis, and end-to-end traceability.

**Total: 35 + 59 + 85 = 179 validation questions.**

Key validations:

- GL balanced to ~$2 across 430 days (4 days with >$0.10 imbalance due to rounding noise — by design)
- Zero time-travel violations — no payment before invoice, no arrival before shipment, no goods receipt before purchase order
- 100% digital thread coverage — all 11 reference types link to source transactions via reference_id
- 96% procurement chain traceability — GR → AP invoice → payment
- All 38 tables populated with non-zero row counts
- Friction rates within +/-1% of configured targets across all categories
- Variable BOM depth validated — 13 PREMIX sub-intermediates, 9 three-level BOM chains, 10 diamond dependencies
- Real entity names — 78% of ingredients and 100% of suppliers carry descriptive real-world names
- Lead time diversity — 5-72 day range, 55 distinct values across LOCAL/REGIONAL/GLOBAL sourcing tiers
- Batch ingredient variance — +/-3% recording variance confirmed across 983K ingredient records
