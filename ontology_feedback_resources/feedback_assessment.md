  My Take

  P0: Entity Code Formats (45 questions)

  Verdict: Fix the questions, not the sim.

  The sim's code formats actually encode provenance (ORD-1-CLUB-DC-001-233 tells you the sequence, channel, DC, and day), which is more
  realistic than sequential numbering. The question author wrote natural-sounding codes without consulting the actual data. This is a
  1-hour find-and-replace exercise — sample real codes from the DB, update the questions.

  P1: Ingredient/Supplier Names (6 questions)

  Verdict: Fix in the ERP generator or world definition, not the sim engine.

  This follows our "90% post-hoc" principle. The sim joins on IDs, not label text. Two options:

  - Cheapest: Add a name mapping dict in the ERP generator that translates "Fluoride Grade 1" → "Sodium Fluoride" during export. ~30 real
  chemical names would cover all 78 ingredients.
  - Slightly better: Update the ingredient naming logic in generators/hierarchy.py at the source. More durable, but touches the sim's world
   generation.

  For suppliers, same thing — a mapping of 50 names in the ERP generator. Companies like "ChemSource", "NovaChem", "PacificMaterials", etc.

  P2: Status Monoculture (8 questions)

  Verdict: Fix in the ERP generator. Do NOT touch the sim engine.

  This is the most important finding. The feedback's Option B ("freeze day") is exactly right, and it belongs entirely in the ERP
  generator:

  The sim correctly runs all 365 days to completion — that's the ground truth. The ERP generator should then apply a "reporting date" (say
  day 340) where:
  - Documents created before day ~300: terminal status (CLOSED, delivered, etc.)
  - Documents created day 300-330: intermediate states (shipped, in_transit)
  - Documents created day 330-340: early states (pending, allocated)
  - Documents created day 340-365: not yet visible (simulates reporting lag)

  This gives a natural pipeline distribution without touching a single line of sim physics. It's exactly how a real ERP snapshot works —
  you query at a point in time and see documents in various lifecycle stages.

  P3: Zero-Value Columns (3 questions)

  Verdict: Fix in the ERP generator. Trivial.

  We already have the data to populate these:
  - order_lines.unit_price → copy from skus.price_per_case (or from ar_invoice_lines.unit_price)
  - purchase_order_lines.unit_cost → copy from supplier_ingredients.unit_cost

  The ERP generator just didn't populate them. One-line fix per column.

  P4: BOM Structure (4 questions)

  Verdict: Legitimate gap, but medium effort. Requires generators/hierarchy.py changes.

  This is the one item that genuinely touches the sim's world generation. Our BOM is structurally monotone — always exactly 2 levels,
  always exactly 1 bulk intermediate per SKU. A real FMCG company would have:
  - Some simple 1-level BOMs (just packaging + a premix you buy)
  - Most at 2 levels (our current structure)
  - A few at 3-4 levels (a base compound → a flavored paste → a finished product)
  - Diamond dependencies (two intermediates sharing a raw material)

  However — I want to push back slightly on the priority. For VKG testing, the current 2-level BOM already exercises the recursive
  traversal handler, the polymorphic ingredient_id resolution, and the cost rollup path. Deeper BOMs make these tests harder but don't
  unlock fundamentally new query patterns. I'd rank this P4-P5, not P3.

  P5: Transport Network Topology (4 questions)

  Verdict: Legitimate gap, but the hardest fix. Requires generators/network.py + sim logistics changes.

  This one I want to be careful about. Adding lateral edges (RDC↔RDC) and multi-sourcing isn't just a generator change — it affects how the
   sim's LogisticsEngine routes shipments, how the Orchestrator pushes excess inventory, and potentially the entire flow balance. The
  current tree topology is a deliberate simplification that makes the sim's physics tractable.

  That said, we could add lateral edges only in the ERP's route_segments table without changing the sim's actual routing. The sim would
  still route on its tree, but the ERP-visible transport network would show theoretical alternative paths. This is actually realistic —
  companies know about routes they could use even if they don't use them operationally.

  P6: Production Variance (3 questions)

  Verdict: Fix in the ERP generator. Easy and high-value.

  Add ±2-5% noise to batch_ingredients.quantity_kg during ERP generation. The sim's batch_ingredients.parquet has the deterministic values;
   the ERP generator just multiplies by 1 + uniform(-0.05, 0.05). This is a 5-line change and makes Q27, Q57, Q73 non-trivial.

  P7-P9: Minor Items

  - P7 (lead times): Easiest fix is updating the question threshold to 10 days. Or extend the range in world_definition.json. Either way
  trivial.
  - P8 (GL anomalies): We already inject -DUP invoices in the friction layer. Extending this to GL (a few duplicate postings, small
  rounding imbalances) is a natural extension.
  - P9 (numpy): Already a transitive dependency via other packages, but making it explicit is trivial.

  ---
  Revised Priority Ranking

  ┌──────────┬────────────────────────────────────────────┬──────────────────────────────┬─────────────────────┬─────────┐
  │ Priority │                    Fix                     │            Where             │ Questions Unblocked │ Effort  │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P0       │ Update questions with real entity codes    │ Questions file               │ 45                  │ Trivial │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P1       │ Rename ingredients/suppliers to real names │ ERP generator or world_def   │ 6                   │ Low     │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P2       │ Status lifecycle via "reporting date"      │ ERP generator                │ 8                   │ Medium  │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P3       │ Populate unit_price / unit_cost zeros      │ ERP generator                │ 3                   │ Trivial │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P4       │ Add batch production variance ±2-5%        │ ERP generator                │ 3                   │ Trivial │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P5       │ Extend lead time range / fix Q10 threshold │ Question or world_def        │ 1                   │ Trivial │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P6       │ Inject GL anomalies                        │ ERP generator friction layer │ 1                   │ Low     │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P7       │ Add lateral route_segments (ERP-only)      │ ERP generator                │ 4                   │ Medium  │
  ├──────────┼────────────────────────────────────────────┼──────────────────────────────┼─────────────────────┼─────────┤
  │ P8       │ Nested BOM intermediates                   │ generators/hierarchy.py      │ 4                   │ High    │
  └──────────┴────────────────────────────────────────────┴──────────────────────────────┴─────────────────────┴─────────┘

  Key reordering: I moved production variance up (trivial ERP fix unlocking 3 questions) and BOM/network down (higher effort, fewer
  questions, riskier to sim stability).

  P0-P6 are all ERP generator or question-file changes that don't touch the sim engine at all. That's 67 of 85 questions unblocked without
  risking sim physics. P7-P8 are "nice to have" that make the benchmark more demanding but aren't blocking.

  What's your thinking on priority order? And shall I start working through these, beginning with the ERP generator changes?

✻ Cogitated for 1m 25s

❯ this is great!! and more or less what i was thinking. on the revised list, aligned with P0-P6. For P0, i will have a separate
  session go through and query data to get the right ids, rell me more about the tradeoffs between erp gen or world def for P1 and
  make a reco, for P5 tell me more about why the issue happened, and what's the difference between the world def fix versus question
  (and will it catch us again when we open this to broader arbitraty questions), for p6 this seems like the right thing to do. On P7,
  this is where i want to think hard about what is actually reflective of real life, if the tree topology was oversimplified for the
  sake of the simulation, how much does it differ than an actual FMCG's logistics network, we need to make a strategic decision here,
  we are aiming for tough real-life challenges, for P8, same, are FMCG boms / nesting more complex in reality? Do the benchmarking so
  we can make the strategic decision

⏺ Great questions. Let me dig into the actual data to ground my answers, then research the real-world FMCG benchmarks.

⏺ 4 Explore agents finished (ctrl+o to expand)
   ├─ Research ingredient naming in generators · 14 tool uses · 61.7k tokens
   │  ⎿  Done
   ├─ Research lead time generation config · 25 tool uses · 53.0k tokens
   │  ⎿  Done
   ├─ Research network topology generation · 22 tool uses · 76.8k tokens
   │  ⎿  Done
   └─ Research BOM generation depth · 23 tool uses · 64.0k tokens
      ⎿  Done

⏺ Good — all four research threads are back. Let me address each of your questions with what I found, plus the real-world benchmarks.

  ---
  P1: ERP Generator vs World Definition — Tradeoffs & Recommendation

  Where names live today:
  - Ingredient names: Generated in hierarchy.py from world_definition.json ingredient_profiles. Pattern: type_name.replace("_", "
  ").title() + f" Grade {i+1}" → "Fluoride Grade 1"
  - Supplier names: Hardcoded in network.py as f"Supplier {i+1}". There's even a commented-out path that would use Faker company names — it
   was deliberately simplified.

  Option A — Fix in ERP generator (mapping dict):
  - Add a NAME_MAP dict in master_tables.py that translates at export time
  - Pros: Zero risk to sim physics, zero risk to warm-start compatibility, fast
  - Cons: Two sources of truth (sim says "Fluoride Grade 1", ERP says "Sodium Fluoride"). If someone queries the sim's raw parquet, they
  see generic names. The mapping is a maintenance burden — every time you add an ingredient type to world_definition.json, you need to
  update the map.

  Option B — Fix in world_definition.json + hierarchy.py:
  - Replace types: ["FLUORIDE", "SURFACTANT", ...] with real chemical names, or add a display_names mapping in the config
  - Pros: Single source of truth. Everything from sim parquet through ERP tables uses real names. No divergence.
  - Cons: Requires regenerating the world (fresh sim run). Warm-start snapshots from old runs become incompatible (ingredient IDs change).
  Slightly more work.

  Option C — Fix in world_definition.json only (hybrid):
  - Add a display_name field to each ingredient type in the config. Generator reads it; if present, uses it instead of "Grade N". Sim IDs
  stay stable (ACT-FLUORIDE-001), but name becomes "Sodium Fluoride".
  - Pros: Single source of truth, IDs unchanged, warm-start compatible
  - Cons: Slightly more config to maintain

  Recommendation: Option C. Add display_names to the ingredient profiles in world_definition.json:

  "ACTIVE_CHEM": {
    "types": ["SURFACTANT", "FLUORIDE", "FRAGRANCE", ...],
    "display_names": {
      "SURFACTANT": "Sodium Lauryl Sulfate",
      "FLUORIDE": ["Sodium Fluoride", "Stannous Fluoride"],
      "HUMECTANT": "Glycerin",
      ...
    }
  }

  Generator uses the display name for Product.name, keeps the type key for Product.id. IDs are unchanged (ACT-FLUORIDE-001), names become
  real ("Sodium Fluoride"). Same approach for suppliers — add a company_names list to the topology config. One source of truth, no mapping
  drift, warm-start safe because IDs don't change.

  ---
  P5: Why Lead Times Max at 14 Days

  Root cause: It's a hardcoded formula in the ERP generator, not from config:

  # scripts/erp/master_tables.py, line 470
  lead_time = 3 + (hash(ing_sim + sup_sim) % 12)   # Range: 3–14 days

  The % 12 modulo was picked for convenience — nobody tuned it against real FMCG procurement data. The sim's own logistics lead times
  (transport link travel times) are physics-based (haversine distance / truck speed + handling), but the ERP's
  supplier_ingredients.lead_time_days is just a hash-derived number.

  World def fix vs question fix:

  - Question fix: Change Q10 from ">30 days" to ">10 days". Costs nothing. But it papers over an unrealistic data range — real FMCG
  procurement spans 7-90+ days depending on sourcing origin.
  - World def / ERP gen fix: Change the formula to produce a realistic range. Something like:
    - Domestic sourcing (LOCAL): 5-15 days
    - Regional sourcing: 15-30 days
    - Global sourcing (GLOBAL): 30-75 days

  The config already tags each ingredient profile with "sourcing": "LOCAL" or "sourcing": "GLOBAL". We just don't use it for lead time
  calculation.

  Will the question fix catch us again? Yes — if you open this to arbitrary questions, anyone who asks "which suppliers have long lead
  times?" or "what's our global sourcing exposure?" will hit the same ceiling. The real fix is making lead times sourcing-aware in the ERP
  generator:

  base = {"LOCAL": 5, "REGIONAL": 15, "GLOBAL": 30}[sourcing]
  variance = {"LOCAL": 10, "REGIONAL": 15, "GLOBAL": 45}[sourcing]
  lead_time = base + (hash(ing_sim + sup_sim) % variance)

  Recommendation: Fix the ERP generator. It's a 5-line change, it uses config data we already have (sourcing field), and it makes the data
  robust against arbitrary questions. Fix the question too, but as a consequence of having better data, not as a workaround.

  ---
  P7: Transport Network — How Does Real FMCG Compare?

  This is the strategic question. Let me benchmark our topology against real-world FMCG networks.

⏺ Web Search("FMCG supply chain network topology structure plants distribution centers redundancy lateral transfers 2024 2025")
  ⎿  Did 1 search in 9s

⏺ Web Search("Procter Gamble Unilever supply chain network structure distribution centers redundant routes cross-docking")
  ⎿  Did 1 search in 11s

⏺ Web Search("FMCG supply chain lateral transfers between distribution centers transshipment hub-and-spoke vs mesh network real examples")
  ⎿  Did 1 search in 9s

⏺ Web Search("FMCG bill of materials complexity levels depth CPG consumer goods multi-level BOM intermediate products real world examples")
  ⎿  Did 1 search in 11s

⏺ Web Search(""multi-level BOM" depth levels "consumer packaged goods" OR "personal care" OR "toothpaste" OR "shampoo" compounding
            intermediate bulk manufacturing process")
  ⎿  Did 3 searches in 14s

⏺ Web Search("Procter Gamble Colgate supply chain distribution center network number redundancy multi-sourcing fulfillment")
  ⎿  Did 2 searches in 10s

⏺ Now I have the full picture. Let me give you the strategic benchmarks.

  ---
  P7: Real-World FMCG Network Topology

  What Real Companies Do

  The base model is hub-and-spoke — which aligns with our tree structure. P&G operates 5 mega distribution centers in North America (Moreno
   Valley CA, Atlanta GA, Dayton OH, Dallas TX, Shippensburg PA) designed to be within 1-day transit of 80% of retailers. That's
  structurally similar to our 4 plants → 6 RDCs → 56 customer DCs → 3,817 stores.

  But real networks have three things ours lacks:

  1. Lateral transshipment (intra-tier redundancy). When a DC runs out of stock, real companies transfer from a nearby DC at the same
  echelon. Research calls this "intra-tier redundancy" and finds it equally important as inter-tier redundancy for resilience. P&G's
  control tower system provides real-time visibility specifically to enable these lateral moves. This is the RDC↔RDC and DC↔DC links we're
  missing.

  2. Multi-sourcing (multiple supply parents). In a real network, a customer DC can be served by 2+ upstream RDCs. This creates genuine
  route alternatives and means that losing one RDC doesn't orphan its entire downstream. P&G cross-docks ~60% of volume through "flow
  centers" — these facilities serve as flexible routing points, not fixed tree branches.

  3. Variable connectivity. Not all stores are equal. High-volume stores may have direct plant-to-store bypass routes. Urban clusters may
  share DCs across channels. Some retail chains operate their own DCs that receive from multiple P&G RDCs.

  Our Gap Assessment

  ┌──────────────────────────┬───────────────────────────┬───────────────────────┬─────────────┐
  │         Feature          │         Real FMCG         │        Our Sim        │     Gap     │
  ├──────────────────────────┼───────────────────────────┼───────────────────────┼─────────────┤
  │ Hub-and-spoke base       │ Yes                       │ Yes                   │ None        │
  ├──────────────────────────┼───────────────────────────┼───────────────────────┼─────────────┤
  │ Lateral DC↔DC transfers  │ Common (emergency)        │ None                  │ Significant │
  ├──────────────────────────┼───────────────────────────┼───────────────────────┼─────────────┤
  │ Multi-source DCs         │ Common (2-3 sources)      │ Exactly 1 parent      │ Significant │
  ├──────────────────────────┼───────────────────────────┼───────────────────────┼─────────────┤
  │ Plant-to-store direct    │ Selective (high-volume)   │ None                  │ Moderate    │
  ├──────────────────────────┼───────────────────────────┼───────────────────────┼─────────────┤
  │ Cross-channel sharing    │ Occasional                │ None                  │ Minor       │
  ├──────────────────────────┼───────────────────────────┼───────────────────────┼─────────────┤
  │ Route alternatives exist │ Always (roads are a mesh) │ Never (tree = 1 path) │ Significant │
  └──────────────────────────┴───────────────────────────┴───────────────────────┴─────────────┘

  Strategic Recommendation for P7

  The tree topology is an oversimplification that matters. It's not catastrophically wrong — hub-and-spoke IS the primary model — but the
  complete absence of redundancy makes our network unrealistically fragile and makes all graph analysis questions trivial (shortest path =
  only path).

  However, from the research on the sim engine: the sim itself doesn't require tree topology. The logistics engine does direct link lookups
   (route_map[(source, target)]), not tree traversals. Adding lateral edges wouldn't break the sim — it would just create additional routes
   that the replenishment engine could use.

  Recommended approach (two-tier):

  1. ERP-only route_segments (quick win): Add ~20-30 lateral route segments to the ERP's route_segments table representing known possible
  routes that the company doesn't regularly use. These show up in graph analysis without changing sim behavior. Realistic — companies know
  their backup routes even if they don't use them daily.
  2. Sim generator enhancement (longer term): Modify network.py to add:
    - 3-5 RDC↔RDC lateral links between geographically adjacent RDCs
    - ~10% of customer DCs served by 2 RDCs instead of 1
    - A handful of direct plant→high-volume-store bypass routes

  This would give the sim genuinely richer routing while keeping the hub-and-spoke base.

  Verdict: Do it, but start with the ERP-only approach (tier 1). Save sim changes for later.

  ---
  P8: Real-World FMCG BOM Complexity

  What Real Products Look Like

  For a product like toothpaste (our core Oral Care category), the real manufacturing process is:

  Level 3: Raw Materials
    ├── Sodium Fluoride, Calcium Carbonate, Sorbitol, SLS, Silica, ...
    ├── Fragrance oils, colorants, preservatives
    └── Water (purified)
           ↓
  Level 2: Bulk Intermediate — "Base Paste"
    ├── Compounding tank: mix water + abrasives + binders → base
    └── Separate: mix fragrance + colorant → flavor blend
           ↓
  Level 1: Semi-Finished — "Flavored Paste"
    ├── Combine base paste + flavor blend + active ingredients
    └── Quality hold / viscosity testing
           ↓
  Level 0: Finished SKU
    ├── Fill into tubes (primary packaging)
    ├── Carton (secondary packaging)
    └── Case-pack + palletize (tertiary packaging)

  So real toothpaste BOMs can be 3-4 levels deep when you include separate intermediate blends. But here's the nuance:

  How Much Complexity Is Typical?

  ┌────────────────────────────┬───────────────────┬─────────────────────────────────────────────────┬───────────────────────────┐
  │        Product Type        │ Typical BOM Depth │              Intermediates per SKU              │           Notes           │
  ├────────────────────────────┼───────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ Simple soap/bar            │ 2 levels          │ 1 (base compound)                               │ Our current structure     │
  ├────────────────────────────┼───────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ Toothpaste                 │ 3 levels          │ 1-2 (base paste + flavor blend)                 │ Slightly deeper than ours │
  ├────────────────────────────┼───────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ Shampoo/conditioner        │ 2-3 levels        │ 1-2 (wash base + conditioning blend)            │ Varies by formula         │
  ├────────────────────────────┼───────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ Laundry detergent (liquid) │ 3 levels          │ 2-3 (surfactant blend + enzyme mix + fragrance) │ More complex              │
  ├────────────────────────────┼───────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ Multi-component kits       │ 4+ levels         │ 3+ (pre-blends + sub-assemblies)                │ Premium products          │
  ├────────────────────────────┼───────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ Promotional multi-packs    │ +1 level          │ Adds packaging BOM on top                       │ Seasonal spikes           │
  └────────────────────────────┴───────────────────┴─────────────────────────────────────────────────┴───────────────────────────┘

  Key patterns in real FMCG BOMs that we're missing:

  1. Variable depth across the portfolio. Not everything is 2 levels. Simple products are 2, complex ones are 3-4. The variety is the
  point.
  2. Multiple intermediates per SKU. A toothpaste isn't just "1 bulk paste + packaging." It's a base paste AND a flavor/fragrance blend AND
   possibly an active ingredient premix. Our sim always uses exactly 1 bulk intermediate per SKU.
  3. Diamond dependencies. "Purified Water" goes into the base paste AND the flavor blend AND might be used directly. Two intermediates
  sharing a raw material is the norm, not the exception.
  4. Shared intermediates across SKUs. One "Mint Base Paste" feeds 10 different mint toothpaste variants (regular, whitening, sensitive,
  kids, etc.). This is the single biggest missing pattern — it's how real CPG companies get manufacturing scale.

  Our Gap Assessment

  ┌────────────────────────────────┬───────────────────────────────┬────────────────────────────────────────────────┬────────────────┐
  │            Feature             │           Real FMCG           │                    Our Sim                     │      Gap       │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ 2-level BOMs exist             │ Yes (simple products)         │ Yes (all products)                             │ Structure      │
  │                                │                               │                                                │ valid          │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ 3-level BOMs exist             │ Yes (most products)           │ No                                             │ Moderate       │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ 4+ level BOMs exist            │ Some (complex formulations)   │ No                                             │ Minor          │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ Variable depth across          │ Always                        │ Never (uniformly 2)                            │ Significant    │
  │ portfolio                      │                               │                                                │                │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ Multiple intermediates per SKU │ Common (2-3)                  │ Never (always exactly 1)                       │ Significant    │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ Diamond dependencies           │ Universal (water, common      │ None                                           │ Moderate       │
  │                                │ bases)                        │                                                │                │
  ├────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────┼────────────────┤
  │ Shared intermediates across    │ Core CPG pattern              │ Already present (45 bulks shared across 500    │ None           │
  │ SKUs                           │                               │ SKUs)                                          │                │
  └────────────────────────────────┴───────────────────────────────┴────────────────────────────────────────────────┴────────────────┘

  Strategic Recommendation for P8

  Our 2-level BOM is realistic for the simplest products but underrepresents the portfolio. The uniformity is the bigger issue than the
  depth — every single SKU having the identical BOM shape (1 bulk + N packaging) is unrealistic.

  Recommended approach:

  This requires generators/hierarchy.py changes and the MRP engine's two-step explosion would need to become N-step. The research confirms:
  - The recipe matrix (recipe_matrix.py) is already depth-agnostic
  - The transform engine sorts by bom_level generically (any integer works)
  - The MRP engine has the hardcoded two-step explosion that needs refactoring to loop through N levels

  Minimum viable change:
  1. Keep 70% of SKUs at current 2-level depth (realistic for simple products)
  2. Add 20% at 3-level depth (base paste → flavored blend → SKU) by creating ~10 "Level 2" intermediates that consume other intermediates
  3. Add 10% at 2-level but with 2-3 intermediates per SKU instead of 1
  4. Ensure diamond dependencies exist naturally (shared raw materials across intermediates)
  5. Refactor MRP's two-step explosion into an N-step loop

  Effort: Medium-High. This is a real sim change touching hierarchy.py, mrp.py, and potentially product/core.py. I'd estimate 2-3 focused
  sessions.

  Verdict: Worth doing, but it's the highest-effort item on the list. I'd tackle P0-P6 first, then come back to P7 and P8 as a second
  phase.

  ---
  Summary: Proposed Execution Order

  Phase 1 (ERP generator + questions — no sim risk):
  - P0: You handle in separate session (question code updates)
  - P1: Ingredient/supplier real names via world_definition.json display_names
  - P2: Status lifecycle via "reporting date" in ERP generator
  - P3: Populate unit_price / unit_cost zero columns
  - P4: Batch production variance ±2-5%
  - P5: Sourcing-aware lead times in ERP generator
  - P6: GL anomaly injection

  Phase 2 (ERP-only topology enrichment):
  - P7: Add ~20-30 lateral/redundant route_segments to ERP export

  Phase 3 (Sim changes):
  - P8: BOM depth variability (hierarchy.py + mrp.py refactor)
  - P7b: Sim generator network topology enrichment (optional)

  Does this phasing make sense? Ready to start on Phase 1 when you are.

  Sources:
  - https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2470348
  - https://www.toolsgroup.com/blog/procter-gamble-supply-chain-complexity/
  - https://www.everythingsupplychain.com/procter-gambles-supply-network-6-min-video-5k-views/
  - https://globallocationstrategies.com/project/proctor-gamble-warehousing-and-distribution/
  - https://www.scielo.br/j/jistm/a/ZWFwtKbtcV6Bp6c7HQfqhTw/?lang=en
  - https://www.journal.oscm-forum.org/publication/article/a-multi-echelon-location-inventory-model-with-lateral-transshipment
  - https://www.rhsmith.umd.edu/blogs/look-inside-colgate-palmolives-global-supply-chain-strategy
  - https://www.cips.org/intelligence-hub/operations-management/bills-of-material
  - https://www.mrpeasy.com/blog/bill-of-materials/
  - https://sgsystemsglobal.com/glossary/packaging-bill-of-materials-cpg-specific-bom/

  
  Good call — fix the foundation first. Here's the sim-level fix list, re-evaluated with "do it right in the sim" as the principle:
  
    ---
    S1. Ingredient & Supplier Real Names
    - Where: world_definition.json (add display_names mapping) + generators/hierarchy.py (read and apply them) + generators/network.py
    (supplier company names)
    - What: Replace "Fluoride Grade 1" → "Sodium Fluoride", "Supplier 7" → "ChemSource", etc.
    - Effort: Low
  
    S2. Sourcing-Aware Lead Times
    - Where: generators/network.py or world_definition.json config
    - What: Currently hardcoded as 3 + (hash % 12) in ERP generator but the sim's world_definition.json already tags ingredients with
    "sourcing": "LOCAL" vs "GLOBAL". Build lead time ranges from sourcing origin: LOCAL 5-15d, REGIONAL 15-30d, GLOBAL 30-75d. Propagate
    through supplier_catalog and into the sim's Link objects.
    - Effort: Low
  
    S3. Batch Production Variance
    - Where: simulation/transform.py (batch execution)
    - What: When a batch consumes ingredients, apply ±2-5% noise to quantity_kg vs formula spec. Currently deterministic. The variance should
     be in the sim's batch_ingredients.parquet output so it's ground truth, not post-hoc fiction.
    - Effort: Low
  
    S4. Status Lifecycle Diversity
    - Where: simulation/orchestrator.py + simulation/writer.py
    - What: Add a reporting_day parameter (default: day of sim end). Documents created after reporting_day - 40 keep intermediate statuses
    (pending, allocated, shipped). Documents before that reach terminal states. This means the sim's parquet output itself contains the
    lifecycle pipeline, not just the ERP layer.
    - Effort: Medium
  
    S5. Populate Zero-Value Columns
    - Where: simulation/writer.py or simulation/orchestrator.py (wherever order_lines and PO lines are written)
    - What: Stamp order_lines.unit_price from product.price_per_case and po_lines.unit_cost from the supplier-ingredient catalog at order
    creation time. These should be sim-level facts, not post-hoc backfills.
    - Effort: Low
  
    S6. GL Anomaly Injection
    - Where: Depends — if GL is generated post-hoc in ERP generator, this stays ERP-layer. But if we want ground-truth anomalies, we'd need a
     friction pass in the sim's financial output.
    - What: Inject ~0.5% duplicate GL postings and small rounding imbalances on ~1% of days.
    - Effort: Low — but need to decide: is GL a sim output or purely ERP? (Currently ERP-only, so this one may legitimately stay in ERP
    generator.)
  
    S7. Network Topology — Lateral & Redundant Routes
    - Where: generators/network.py + config/world_definition.json
    - What:
      - Add 3-5 RDC↔RDC lateral links between geographically adjacent RDCs
      - ~10% of customer DCs served by 2 RDCs instead of 1
      - A handful of direct plant→high-volume-store bypass routes
    - Why in sim not ERP: The sim's logistics engine can actually USE these routes for shipment routing, making the transport data reflect
    real operational decisions rather than theoretical paths nobody uses.
    - Effort: Medium — generator changes are straightforward, but need to verify the replenishment and logistics engines handle multi-parent
    nodes gracefully.
  
    S8. BOM Depth Variability
    - Where: generators/hierarchy.py + simulation/mrp.py + product/core.py
    - What:
      - 70% of SKUs stay at current 2-level depth
      - 20% get 3-level depth (intermediate consumes another intermediate)
      - 10% get 2+ intermediates per SKU (e.g., base paste + fragrance blend)
      - Diamond dependencies naturally emerge (shared raw materials across intermediates)
      - MRP two-step explosion → N-step loop
    - Effort: High — touches hierarchy generator, MRP engine, and product definitions
  
    ---
    Proposed order (dependencies + risk management):
  
    ┌───────┬───────────────────────┬──────────────────────────────────────────────────────┐
    │ Order │          Fix          │                      Rationale                       │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 1     │ S1 (names)            │ Zero risk, unlocks question readability, good warmup │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 2     │ S2 (lead times)       │ Zero risk, config-driven, quick                      │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 3     │ S3 (batch variance)   │ Low risk, isolated to transform.py                   │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 4     │ S5 (zero columns)     │ Low risk, data completeness                          │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 5     │ S4 (status lifecycle) │ Medium risk, touches orchestrator flow               │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 6     │ S6 (GL anomalies)     │ Low risk, may stay ERP-layer                         │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 7     │ S7 (network topology) │ Medium risk, needs logistics engine validation       │
    ├───────┼───────────────────────┼──────────────────────────────────────────────────────┤
    │ 8     │ S8 (BOM depth)        │ Highest risk, MRP refactor                           │
    └───────┴───────────────────────┴──────────────────────────────────────────────────────┘
  
    S6 is the one I'd flag — GL journal generation lives entirely in scripts/erp/, not in the sim. Making it a sim-level output would mean
    adding a financial engine to the sim, which is a much bigger architectural change than the anomaly itself warrants. I'd recommend keeping
     S6 as ERP-layer unless you feel strongly otherwise.
  
    Want to start with S1?
  
  ✻ Cogitated for 39s
  
  ❯ do you think we should make one plan to address, or chunk this (and if so, how so)
  
  ⏺ Chunk it. Three reasons:
  
    1. Blast radius isolation. If S8 (MRP refactor) introduces a bug, you don't want it tangled with S1 (naming) in the same commit history.
    Separate chunks = separate validation runs = easy bisect if something breaks.
    2. Validation gates. Each chunk needs a sim run + diagnostic check before moving on. You can't validate "did batch variance work?" and
    "did MRP N-step explosion work?" in the same run without confusion.
    3. Natural dependency ordering. S1/S2 change the world definition that S7/S8 build on. Better to land the foundation first.
  
    Proposed chunks:
  
    Chunk 1: Data Richness (S1 + S2 + S3 + S5)
  
    All low-risk, mostly generators/ and config/ changes. One sim run validates everything.
    - S1: Real ingredient/supplier names
    - S2: Sourcing-aware lead times
    - S3: Batch production variance
    - S5: Populate unit_price/unit_cost
  
    Validate: 50-day --no-logging run, spot-check parquet output for real names, lead time range, batch variance spread, non-zero prices.
  
    Chunk 2: Lifecycle (S4 + S6)
  
    Medium risk, touches orchestrator document flow.
    - S4: Status lifecycle diversity via reporting_day
    - S6: GL anomalies (assess whether this stays ERP-layer or moves to sim)
  
    Validate: 365-day streaming run, check status distributions in output, run unified diagnostic.
  
    Chunk 3: Network Topology (S7)
  
    Medium risk, isolated to generators/network.py + logistics validation.
    - Lateral RDC links, multi-source DCs, bypass routes
  
    Validate: 50-day run, verify shipments actually route over new edges, check no physics violations in diagnostic.
  
    Chunk 4: BOM Depth (S8)
  
    Highest risk, touches hierarchy.py + mrp.py + product/core.py.
    - Variable BOM depth, multiple intermediates per SKU, MRP N-step refactor
  
    Validate: 365-day streaming run, full diagnostic including BOM explosion checks, mass balance verification.
  
    ---
    S7 and S8 get their own chunks because they touch completely different subsystems (network/logistics vs product/MRP) and if both break
    simultaneously, debugging is a nightmare.
