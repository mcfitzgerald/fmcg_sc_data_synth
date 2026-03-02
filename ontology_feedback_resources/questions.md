# PCG Benchmark Questions

85 questions for evaluating graph-aware analytics over the PCG ERP dataset.

Questions are written from the perspective of senior supply chain, finance, and procurement leaders at Prism Consumer Goods. They use business language only — no database jargon, no handler hints, no VG/SQL terminology. The system under test must derive the query approach entirely from the ontology.

---

## Section 1: Lookups & Joins (Q01–Q10)

**Q01.** Pull the full supplier profile for supplier code SUP-0012 — name, city, country, Kraljic tier, and active status.

**Q02.** Which ingredients does our Dallas-based supplier ChemSource offer, and at what unit cost? Include any alternate supplier records that might be duplicates — we've seen "-ALT" codes creep into the master data.

**Q03.** What is our total open purchase order exposure by plant? Show only POs with status "open" and break out the total line-item value for Dallas, Columbus, Sacramento, and Atlanta.

**Q04.** List all Oral Care SKUs in the Premium segment with a price per case above $45. Include the brand and whether the SKU is currently active.

**Q05.** Show me every line on order ORD-2024-08471 — SKU, quantity in cases, unit price, and line status.

**Q06.** For formula FRM-OC-MINT-001, which suppliers can provide the required ingredients? Show the ingredient name, supplier name, and the supplier's quoted unit cost.

**Q07.** Break down our total revenue by channel for the full year. Rank channels from highest to lowest and include the order count behind each.

**Q08.** How many orders were placed between day 90 and day 120? Summarize by status — pending, allocated, shipped, delivered — so I can see the pipeline.

**Q09.** Give me a single consolidated list of every location in our network — plants, distribution centers, and retail locations — with city, country, and location type. I need the full picture.

**Q10.** Which supplier-ingredient combinations have a lead time greater than 30 days? Include the unit cost and minimum order quantity — I want to see our long-tail procurement exposure.

---

## Section 2: SKU Alias Chains (Q11–Q18)

**Q11.** SKU code SKU-OC-TPST-042-OLD was retired last year. Trace the full supersession chain forward — what is the current active code, and what were all the intermediate renames?

**Q12.** Our newest Oral Care launch is SKU-OC-MWSH-015. Work backward through the alias chain — what legacy codes does this SKU ultimately replace?

**Q13.** How many generations deep is the SKU rename chain starting from SKU-HM-DISH-008-OLD? We need to know if our data cleanup hit this one.

**Q14.** Which SKUs are original root codes — meaning they were never created as a replacement for an older SKU? Give me a count by category.

**Q15.** Which SKUs are "terminal" — the current active code at the end of a rename chain, not yet superseded by anything newer? How many of those are actually inactive?

**Q16.** Starting from SKU-PW-SOAP-022-OLD, walk the alias chain and collect the category, brand, and active status at each step. I want to see if the brand changed during the renames.

**Q17.** From SKU-OC-TPST-100, show me every SKU reachable within three hops of the supersession chain in either direction. We're auditing nearby aliases for a catalog cleanup.

**Q18.** Across all SKU alias chains in the system, what is the average chain length? What's the longest single chain? Break it down by product category.

---

## Section 3: BOM Explosion & Cost Rollup (Q19–Q28)

**Q19.** List every ingredient in the formula for our best-selling mint toothpaste (FRM-OC-MINT-001) — ingredient name, sequence, and quantity in kg per batch.

**Q20.** Explode the full bill of materials for SKU-OC-TPST-001 down to raw materials. I need to see both the bulk intermediate stage and the final raw ingredient level — the complete tree.

**Q21.** If we need to produce 100 batches of SKU-OC-TPST-001, what is the total kg of sodium fluoride required? Roll the quantity up through every level of the BOM.

**Q22.** What is the total raw material cost for a single batch of SKU-PW-SOAP-010? Walk the BOM tree and multiply quantities by each ingredient's cost per kg.

**Q23.** Which finished goods SKUs use citric acid in their formulation — either directly or through a bulk intermediate? I need the full where-used report.

**Q24.** How many BOM levels deep is the formula tree for SKU-HM-DISH-005? Is it a simple one-level recipe or does it go through intermediates?

**Q25.** For SKU-OC-MWSH-003, list all bulk intermediates that appear in the BOM tree between the finished good and the raw materials. Show the intermediate name and its BOM level.

**Q26.** Across all formulas in the system, what is the total quantity in kg for each raw ingredient? Rank by total consumption — I want to see our top-volume raw materials.

**Q27.** For batch B-2024-05831, compare what the formula said we should have used versus what we actually consumed. Show the ingredient-level variance in kg — planned versus actual.

**Q28.** Which raw ingredients appear in three or more different formulas? These shared components are our biggest volume aggregation opportunities for procurement.

---

## Section 4: Transport Network Analysis (Q29–Q40)

**Q29.** What is the shortest route by distance from the Dallas plant to retail location RTL-MR-0042 in Phoenix? Show me every hop and the cumulative kilometers.

**Q30.** Now find the fastest route from Dallas to that same Phoenix retail location — optimize for transit time in hours, not distance. Does the fastest path differ from the shortest?

**Q31.** Are there multiple routes from the Columbus plant to the Jacksonville RDC that tie for shortest distance? List all of them.

**Q32.** Which location in our transport network has the most direct route connections? In other words, which node is our highest-degree hub?

**Q33.** Which distribution centers sit on the most shortest paths between other locations? These are our critical transit hubs — if one goes down, it disrupts the most routes.

**Q34.** Which single location in the network minimizes average distance to every other location? That's our most central node for network-wide distribution.

**Q35.** Rank all locations by their overall importance in the transport network — accounting for both the number and quality of connections, not just direct links.

**Q36.** Is our entire logistics network connected? Or are there isolated clusters of locations that can't reach each other through any sequence of route segments?

**Q37.** If the Atlanta plant were forced to shut down, what happens to our network connectivity? How many locations become unreachable, and which ones are they?

**Q38.** Suppose the route segment between the Memphis RDC and the Chicago RDC is severed — a bridge closure, say. What is the impact on network connectivity and average path length?

**Q39.** Find the shortest distance route from the Sacramento plant to retail location RTL-GR-0107, but the path must avoid the Phoenix RDC entirely. We have a capacity constraint there.

**Q40.** What is the fastest route from any manufacturing plant to retail location RTL-EC-0005? The origin can be Dallas, Columbus, Sacramento, or Atlanta — find the best starting point.

---

## Section 5: Cross-Domain Composition (Q41–Q50)

**Q41.** For every ingredient in SKU-OC-TPST-001's BOM, find all suppliers who can provide it, their unit cost, and lead time. Flag any supplier records that look like duplicates — I've seen "-ALT" entries that are probably the same vendor.

**Q42.** What is the fully landed cost to produce one batch of SKU-OC-TPST-001 at the Dallas plant? That means the BOM raw material cost plus the inbound freight cost to get each ingredient from its cheapest supplier to Dallas via the shortest route.

**Q43.** SKU-PW-SOAP-022 went through several renames. Across the entire alias chain — every old code and the current code — what is the total inventory sitting in our network right now? Break it out by location. I suspect we have cases split across stale codes.

**Q44.** Which retail locations within two route hops of the Chicago RDC have orders still in "pending" status? I want to prioritize nearby fulfillment.

**Q45.** Which active work orders use formulas that contain palm oil as an ingredient? We need to assess our production exposure to a potential palm oil disruption.

**Q46.** What is the total AP invoice amount for all ingredients that go into SKU-OC-MWSH-003's bill of materials? Trace the BOM, find each ingredient's supplier, and sum their AP invoices for the year.

**Q47.** We have a rush order at retail location RTL-MR-0100. Which DC currently has inventory of SKU-HM-DISH-005, and what is the shortest route from that DC to the retail location? Find the optimal fulfillment source.

**Q48.** Show me all work orders in "released" status that use formulas containing glycerin. I need to know our active production commitment to glycerin-dependent products.

**Q49.** Trace the full supply chain for order ORD-2024-12045: show the order lines, the SKUs on each line, the formula behind each SKU, every ingredient in those formulas, and the suppliers who provide them. Give me the full six-hop picture.

**Q50.** For each finished SKU in the Oral Care category, compare the total AR invoice revenue against the total BOM raw material cost. Which SKUs have the healthiest margin, and which are underwater?

---

## Section 6: Multi-Graph & Polymorphism (Q51–Q60)

**Q51.** For SKU-OC-TPST-001, give me the grand unified sourcing view: every raw ingredient, which suppliers provide it, the cheapest inbound shipping route from each supplier to the Dallas plant, and the outbound route from Dallas to each retail location that orders this SKU. End-to-end.

**Q52.** A customer filed a quality complaint about order ORD-2024-09876. Trace backward: which production batch filled that order, what formula was used, what ingredients were consumed, and which suppliers provided them? I need the full genealogy.

**Q53.** What is the fastest route from any manufacturing plant to retail location RTL-PH-0200, weighted by transit time? The origin type is "plant" but I don't care which one — find the best.

**Q54.** Show me all inventory currently sitting at distribution center locations only — not plants, not retail. Use the location type to filter. Include the SKU, quantity, and which DC.

**Q55.** Which raw material suppliers ultimately feed products that are shipped to retail locations in the Atlanta metro area? Trace from supplier through BOM through production through fulfillment.

**Q56.** Are there any circular references in our formula hierarchy? Can a formula's ingredient list eventually loop back to itself through bulk intermediates? We need to rule out BOM cycles before the MRP run.

**Q57.** For batch B-2024-08200, compare the planned ingredient quantities from the formula against the actual quantities consumed from batch_ingredients. Show each ingredient and the delta in kg. Where did we deviate from the recipe?

**Q58.** If supplier SUP-0003 (and any of their "-ALT" duplicate records) goes offline tomorrow, which finished SKUs are affected, which open orders contain those SKUs, and which pending shipments are at risk? Quantify the blast radius by revenue exposure.

**Q59.** If the Dallas plant shuts down for two weeks, which open orders cannot be fulfilled from current inventory? Show the order numbers, SKUs, and the gap in cases.

**Q60.** Show me all shipments that originated from our plants — not DCs — even though the shipment table doesn't have a clean origin type column. You'll need to use the route context to figure out which origin IDs are plants.

---

## Section 7: Data Integrity & Metamodel (Q61–Q68)

**Q61.** List all production lines at the Columbus plant that are currently active and available for scheduling. I only want lines that are actually operational — skip anything that's been decommissioned.

**Q62.** Pull the complete commercial terms for every supplier-ingredient relationship: unit cost, lead time in days, and minimum order quantity. We're rebuilding the sourcing matrix and I need it all. Cross-reference with invoice variances — flag any where the invoiced price differs from the catalog.

**Q63.** Verify that every batch in the system produced exactly one product — no batch should be linked to two or more SKUs. Is our one-product-per-batch rule holding, or do we have data integrity issues?

**Q64.** Give me the distribution of orders across lifecycle states: how many are pending, allocated, shipped, delivered, and cancelled? I want to see if we have a bottleneck in the pipeline.

**Q65.** Pull the specific order line for order 4502, line number 3. Show the SKU, quantity, price, and status for just that one line.

**Q66.** What was our distribution center inventory snapshot on day 180? For every DC, show each SKU on hand and the quantity in cases. I need a point-in-time picture — don't mix in plant or retail inventory.

**Q67.** Run a sanity check on our SKU supersession data: are there any cycles in the rename chains? A SKU that eventually loops back to itself would corrupt our alias resolution.

**Q68.** Give me a consolidated list of every transaction document in the system — purchase orders, goods receipts, orders, shipments, returns, AP invoices, and AR invoices — with their document number and current status. I want a single view across all document types.

---

## Section 8: Lifecycle & Flow Analysis (Q69–Q76)

**Q69.** Order ORD-2024-07744 is currently in "pending" status. What are the valid next states it can transition to? Show me the allowed moves from where it sits today.

**Q70.** A warehouse manager tried to mark shipment SHP-2024-11020 as "delivered" but it's currently in "planned" status. Is that a valid transition, or does it need to go through intermediate states first? What's the correct sequence?

**Q71.** Trace the material flow for ingredient ING-RM-GLYC-001 (glycerin): start from the purchase orders, through goods receipts, into production batches, and out through shipments. Show the quantity in kg at each stage so I can see where volume accumulates or drops.

**Q72.** For supplier SUP-0008, trace the full financial flow: purchase orders, the corresponding AP invoices, and the payments against those invoices. Flag any AP invoices where the amount doesn't match the PO — I want to see our three-way match failures.

**Q73.** For batch B-2024-06500, verify mass balance: does the total weight of ingredients consumed equal the batch output quantity plus expected scrap? Show me the numbers — I want to know if we have an unexplained variance.

**Q74.** Verify that debits equal credits in the general ledger for day 200. Pull the total debit and credit amounts and show any imbalance. Also flag if you see any duplicate invoice postings — we've had "-DUP" entries slip through.

**Q75.** Can order ORD-2024-15200 be allocated right now? Check the preconditions: is it in the right status, and is there sufficient inventory for the SKUs on the order?

**Q76.** What happens to our production plan if the Dallas plant's daily capacity drops by 20%? Which work orders would be affected, and how much planned volume exceeds the new capacity ceiling?

---

## Section 9: End-to-End Traceability (Q77–Q85)

**Q77.** Starting from batch B-2024-04100, trace forward to every retail location that received product from this batch — through shipments and their delivery destinations. Note that some shipment records may have missing foreign keys, so handle those gaps gracefully.

**Q78.** We received a customer complaint about shipment SHP-2024-08500. Trace backward to the original ingredients and suppliers: which batches were on that shipment, which formulas and ingredients went into those batches, and which suppliers provided the materials? Surface any gaps from missing FK references.

**Q79.** Goods receipt GR-2024-02200 brought in a lot of citric acid. Find every production batch that consumed citric acid from that receipt. I need the lot genealogy for a potential quality hold.

**Q80.** Supplier SUP-0019 just reported a contamination in their facility. Which of our batches used their ingredients, which SKUs did those batches produce, which orders contain those SKUs, and what is the total revenue exposure in dollar terms? Give me the full blast radius.

**Q81.** Pull the complete procurement document chain for PO PO-2024-00350: the purchase order, the goods receipt(s) that received the material, the AP invoice(s) billed against it, and the payment(s) that settled those invoices. Show me the full procure-to-pay paper trail.

**Q82.** Trace the full order-to-cash chain for order ORD-2024-10500: the order, the shipment(s) that fulfilled it, the AR invoice(s) generated, and the receipt(s) collected. Flag any AR invoices that are still open or written off as bad debt — I want to see our collection exposure.

**Q83.** For every SKU in the Personal Wash category, reconcile total AP spend (what we paid suppliers for ingredients through the BOM) against total AR revenue (what customers paid us). Show the per-SKU margin, and flag any SKUs where invoiced ingredient costs have unexplained variances or where AR receipts include bad debt write-downs.

**Q84.** A return was filed under RTN-2024-00150. Trace the returned SKU back through its production history: which batch produced it, which formula was used, and which ingredients and suppliers were involved? I need to know if the defect traces to a raw material.

**Q85.** What is the cheapest possible path to get SKU-OC-TPST-001 from raw materials to retail location RTL-MR-0042? Combine the BOM ingredient costs, the cheapest supplier for each ingredient, the lowest-cost inbound transport to Dallas, the production cost, and the cheapest outbound route to the retail location. Give me the total raw-to-shelf unit cost.
