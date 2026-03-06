# Entity Codes Reference (v0.93.0)

Generated from ERP CSVs via DuckDB. Use for ontology + question updates.

## Product Counts

| active_skus | old_sku_aliases | total_skus_with_aliases | primary_bulks | premix_count | total_intermediates | raw_materials | formulas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 21 | 521 | 49 | 13 | 62 | 78 | 562 |

## Formula Counts by bom_level

| bom_level | count |
| --- | --- |
| 0 | 500 |
| 1 | 49 |
| 2 | 13 |

## Suppliers (all)

| supplier_code | name | tier |
| --- | --- | --- |
| SUP-001 | Meridian Chemical Corp | 1 |
| SUP-002 | Pacific Packaging Solutions | 1 |
| SUP-003 | Apex Ingredients LLC | 1 |
| SUP-004 | NovaChem Industries | 1 |
| SUP-005 | TrueSource Materials | 1 |
| SUP-006 | Great Lakes Compounds | 1 |
| SUP-007 | Summit Specialty Chemicals | 1 |
| SUP-008 | PrimePack International | 1 |
| SUP-009 | Continental Raw Materials | 1 |
| SUP-010 | BrightStar Chemical Co | 1 |
| SUP-011 | Atlas Packaging Group | 1 |
| SUP-012 | Clearwater Ingredients | 1 |
| SUP-013 | Vanguard Chemical Supply | 1 |
| SUP-014 | Pinnacle Materials Inc | 1 |
| SUP-015 | Heritage Bulk Systems | 1 |
| SUP-016 | Sterling Compounds Ltd | 1 |
| SUP-017 | EcoBlend Chemicals | 1 |
| SUP-018 | Northern Resin Corp | 1 |
| SUP-019 | Cascade Packaging Inc | 1 |
| SUP-020 | SilverLine Chemical Co | 1 |
| SUP-021 | Keystone Bulk Supply | 1 |
| SUP-022 | FrontierChem Holdings | 1 |
| SUP-023 | Pacific Rim Ingredients | 1 |
| SUP-024 | Crestview Materials Group | 1 |
| SUP-025 | Liberty Chemical Works | 1 |
| SUP-026 | Granite State Compounds | 1 |
| SUP-027 | BlueSky Packaging Ltd | 1 |
| SUP-028 | Westfall Industries | 1 |
| SUP-029 | CoreMark Chemical Co | 1 |
| SUP-030 | Evergreen Supply Corp | 1 |
| SUP-031 | Phoenix Specialty Chem | 1 |
| SUP-032 | Horizon Packaging Inc | 1 |
| SUP-033 | Lakeshore Compounds LLC | 1 |
| SUP-034 | Redwood Materials Group | 1 |
| SUP-035 | Coastal Chemical Supply | 1 |
| SUP-036 | Ironbridge Ingredients | 1 |
| SUP-037 | Northstar Packaging Co | 1 |
| SUP-038 | Benchmark Chemical Corp | 1 |
| SUP-039 | Prairie Compounds Inc | 1 |
| SUP-040 | Summit Pack Solutions | 1 |
| SUP-041 | Golden Gate Chemicals | 1 |
| SUP-042 | Trident Materials LLC | 1 |
| SUP-043 | Riverbend Packaging | 1 |
| SUP-044 | CrossRoads Chemical Co | 1 |
| SUP-045 | Timberline Ingredients | 1 |
| SUP-046 | Capitol Compounds Inc | 1 |
| SUP-047 | SunCoast Packaging Ltd | 1 |
| SUP-048 | Appalachian Chemical Corp | 1 |
| SUP-049 | Midland Supply Group | 1 |
| SUP-050 | Diamond Ridge Materials | 1 |

## ALT Supplier Duplicates

| supplier_code | name |
| --- | --- |
| SUP-003-ALT | Apex Ingredients LLC LLC |
| SUP-013-ALT | VANGUARD-CHEMICAL-SUPPLY |
| SUP-016-ALT | Sterling Compounds Ltd Corp |
| SUP-023-ALT | PACIFIC RIM INGREDIENTS INC. |
| SUP-031-ALT | Phoenix Specialty Chem LLC |
| SUP-035-ALT | COASTAL-CHEMICAL-SUPPLY |

## Ingredients by Subcategory (sample)

| ingredient_code | name | category | subcategory |
| --- | --- | --- | --- |
| ACT-ABRASIVE-001 | Calcium Carbonate | INGREDIENT | active_ingredient |
| ACT-ABRASIVE-002 | Sodium Bicarbonate | INGREDIENT | active_ingredient |
| ACT-ANTIMICROBIAL-001 | Triclosan | INGREDIENT | active_ingredient |
| ACT-ANTIMICROBIAL-002 | Cetylpyridinium Chloride | INGREDIENT | active_ingredient |
| ACT-BINDER-001 | PVM/MA Copolymer | INGREDIENT | active_ingredient |
| ACT-BINDER-002 | Sodium CMC | INGREDIENT | active_ingredient |
| ACT-COLORANT-001 | Titanium Dioxide CI 77891 | INGREDIENT | active_ingredient |
| ACT-COLORANT-002 | FD&C Blue No. 1 | INGREDIENT | active_ingredient |
| ACT-FLUORIDE-001 | Sodium Fluoride | INGREDIENT | active_ingredient |
| ACT-FLUORIDE-002 | Stannous Fluoride | INGREDIENT | active_ingredient |
| ACT-FRAGRANCE-001 | Menthol Crystal | INGREDIENT | active_ingredient |
| ACT-FRAGRANCE-002 | Limonene D-Isomer | INGREDIENT | active_ingredient |
| ACT-HUMECTANT-001 | Glycerin USP | INGREDIENT | active_ingredient |
| ACT-HUMECTANT-002 | Propylene Glycol | INGREDIENT | active_ingredient |
| ACT-PRESERVATIVE-001 | Methylparaben | INGREDIENT | active_ingredient |
| ACT-PRESERVATIVE-002 | Phenoxyethanol | INGREDIENT | active_ingredient |
| ACT-SURFACTANT-001 | Sodium Lauryl Sulfate | INGREDIENT | active_ingredient |
| ACT-SURFACTANT-002 | Sodium Laureth Sulfate | INGREDIENT | active_ingredient |
| ACT-WHITENING-001 | Hydrogen Peroxide 3% | INGREDIENT | active_ingredient |
| ACT-WHITENING-002 | Sodium Tripolyphosphate | INGREDIENT | active_ingredient |
| BLK-OIL-001 | Mineral Oil NF | INGREDIENT | base_material |
| BLK-OIL-002 | Castor Oil | INGREDIENT | base_material |
| BLK-OIL-003 | Coconut Oil RBD | INGREDIENT | base_material |
| BLK-SILICATE-001 | Sodium Silicate | INGREDIENT | base_material |
| BLK-SILICATE-002 | Hydrated Silica | INGREDIENT | base_material |
| BLK-SILICATE-003 | Precipitated Silica | INGREDIENT | base_material |
| BLK-SURF_BASE-001 | Cocamidopropyl Betaine | INGREDIENT | base_material |
| BLK-SURF_BASE-002 | Sodium Cocoyl Isethionate | INGREDIENT | base_material |
| BLK-SURF_BASE-003 | Decyl Glucoside | INGREDIENT | base_material |
| BLK-THICKENER-001 | Carbomer 940 | INGREDIENT | base_material |

## SKUs by Category (sample 5 each)

| sku_code | name | category | brand |
| --- | --- | --- | --- |
| SKU-ORAL-001 | PrismWhite 400ml Pump - Enamel Protect | ORAL_CARE | PrismWhite |
| SKU-ORAL-002 | DentEx 50ml Tube - Enamel Protect | ORAL_CARE | DentEx |
| SKU-ORAL-003 | DentEx 150ml Tube - Baking Soda | ORAL_CARE | DentEx |
| SKU-ORAL-004 | DentEx 125ml Tube - Original | ORAL_CARE | DentEx |
| SKU-ORAL-005 | FreshSmile 50ml Tube - Original | ORAL_CARE | FreshSmile |
| SKU-HOME-376 | HomeGuard 200ml Bottle - Pine | HOME_CARE | HomeGuard |
| SKU-HOME-377 | PureShine 200ml Bottle - Lemon | HOME_CARE | PureShine |
| SKU-HOME-378 | PureShine 250ml Bottle - Pine | HOME_CARE | PureShine |
| SKU-HOME-379 | ClearWave 100ml Bottle - Unscented | HOME_CARE | ClearWave |
| SKU-HOME-380 | ClearWave 100ml Bottle - Pine | HOME_CARE | ClearWave |
| SKU-PERSONAL-226 | CleanEssence 100ml Bottle - Shea Butter | PERSONAL_WASH | CleanEssence |
| SKU-PERSONAL-227 | SilkTouch 1500ml Bottle - Citrus | PERSONAL_WASH | SilkTouch |
| SKU-PERSONAL-228 | SilkTouch 200ml Bottle - Shea Butter | PERSONAL_WASH | SilkTouch |
| SKU-PERSONAL-229 | AquaPure 400ml Bottle - Pomegranate | PERSONAL_WASH | AquaPure |
| SKU-PERSONAL-230 | CleanEssence 250ml Bottle - Unscented | PERSONAL_WASH | CleanEssence |

## SKU Aliases (-OLD)

| sku_code | name | category |
| --- | --- | --- |
| SKU-HOME-411-OLD | PureShine 100ml Bottle - Floral (Discontinued) | HOME_CARE |
| SKU-HOME-423-OLD | HomeGuard 100ml Bottle - Eucalyptus (Discontinued) | HOME_CARE |
| SKU-HOME-468-OLD | PureShine 200ml Bottle - Mountain Rain (Discontinued) | HOME_CARE |
| SKU-ORAL-020-OLD | FreshSmile 100ml Tube - Arctic Fresh (Discontinued) | ORAL_CARE |
| SKU-ORAL-055-OLD | DentEx 125ml Tube - Deep Clean (Discontinued) | ORAL_CARE |
| SKU-ORAL-058-OLD | FreshSmile 150ml Tube - Cool Mint (Discontinued) | ORAL_CARE |
| SKU-ORAL-059-OLD | FreshSmile 50ml Tube - Baking Soda (Discontinued) | ORAL_CARE |
| SKU-ORAL-123-OLD | FreshSmile 250ml Tube - Gum Care (Discontinued) | ORAL_CARE |
| SKU-ORAL-133-OLD | PrismWhite 200ml Tube - Arctic Fresh (Discontinued) | ORAL_CARE |
| SKU-ORAL-137-OLD | FreshSmile 50ml Tube - Fresh (Discontinued) | ORAL_CARE |
| SKU-ORAL-141-OLD | PrismWhite 175ml Tube - Baking Soda (Discontinued) | ORAL_CARE |
| SKU-ORAL-152-OLD | DentEx 250ml Pump - Spearmint (Discontinued) | ORAL_CARE |
| SKU-ORAL-167-OLD | PrismWhite 750ml Pump - Spearmint (Discontinued) | ORAL_CARE |
| SKU-ORAL-216-OLD | FreshSmile 250ml Pump - Deep Clean (Discontinued) | ORAL_CARE |
| SKU-PERSONAL-229-OLD | AquaPure 400ml Bottle - Pomegranate (Discontinued) | PERSONAL_WASH |
| SKU-PERSONAL-236-OLD | AquaPure 500ml Bottle - Almond (Discontinued) | PERSONAL_WASH |
| SKU-PERSONAL-287-OLD | CleanEssence 100ml Bottle - Rose (Discontinued) | PERSONAL_WASH |
| SKU-PERSONAL-292-OLD | CleanEssence 1000ml Bottle - Almond (Discontinued) | PERSONAL_WASH |
| SKU-PERSONAL-355-OLD | SilkTouch 500ml Bottle - Pomegranate (Discontinued) | PERSONAL_WASH |
| SKU-PERSONAL-367-OLD | CleanEssence 200ml Bottle - Almond (Discontinued) | PERSONAL_WASH |
| SKU-PERSONAL-374-OLD | CleanEssence 250ml Bottle - Tea Tree (Discontinued) | PERSONAL_WASH |

## Bulk Intermediates (all)

| bulk_code | name | bom_level |
| --- | --- | --- |
| BULK-HC-APPLE_BLOSSO-001 | Apple Blossom Home Care Compound | 1 |
| BULK-HC-CITRUS-BLEND-001 | Citrus Home Care Blend | 1 |
| BULK-HC-CITRUS_ZEST-002 | Citrus Zest Home Care Compound | 1 |
| BULK-HC-CLEAN-BLEND-002 | Clean Home Care Blend | 1 |
| BULK-HC-CLEAN_LINEN-003 | Clean Linen Home Care Compound | 1 |
| BULK-HC-EUCALYPTUS-004 | Eucalyptus Home Care Compound | 1 |
| BULK-HC-FLORAL-005 | Floral Home Care Compound | 1 |
| BULK-HC-FRESH-006 | Fresh Home Care Compound | 1 |
| BULK-HC-LAVENDER-007 | Lavender Home Care Compound | 1 |
| BULK-HC-LEMON-008 | Lemon Home Care Compound | 1 |
| BULK-HC-MOUNTAIN_RAI-009 | Mountain Rain Home Care Compound | 1 |
| BULK-HC-OCEAN_MIST-010 | Ocean Mist Home Care Compound | 1 |
| BULK-HC-ORANGE-011 | Orange Home Care Compound | 1 |
| BULK-HC-PINE-012 | Pine Home Care Compound | 1 |
| BULK-HC-SPRING_BREEZ-013 | Spring Breeze Home Care Compound | 1 |
| BULK-HC-TROPICAL-014 | Tropical Home Care Compound | 1 |
| BULK-HC-UNSCENTED-015 | Unscented Home Care Compound | 1 |
| BULK-OC-ARCTIC_FRESH-016 | Arctic Fresh Oral Care Compound | 1 |
| BULK-OC-BAKING_SODA-017 | Baking Soda Oral Care Compound | 1 |
| BULK-OC-CHARCOAL-018 | Charcoal Oral Care Compound | 1 |
| BULK-OC-CINNAMON-019 | Cinnamon Oral Care Compound | 1 |
| BULK-OC-COOL_MINT-020 | Cool Mint Oral Care Compound | 1 |
| BULK-OC-DEEP_CLEAN-021 | Deep Clean Oral Care Compound | 1 |
| BULK-OC-ENAMEL-BLEND-003 | Enamel Oral Care Blend | 1 |
| BULK-OC-ENAMEL_PROTE-022 | Enamel Protect Oral Care Compound | 1 |
| BULK-OC-FRESH-023 | Fresh Oral Care Compound | 1 |
| BULK-OC-GUM_CARE-024 | Gum Care Oral Care Compound | 1 |
| BULK-OC-HERBAL-025 | Herbal Oral Care Compound | 1 |
| BULK-OC-MINT-026 | Mint Oral Care Compound | 1 |
| BULK-OC-ORIGINAL-027 | Original Oral Care Compound | 1 |
| BULK-OC-SENSITIVE-028 | Sensitive Oral Care Compound | 1 |
| BULK-OC-SPEARMINT-029 | Spearmint Oral Care Compound | 1 |
| BULK-OC-WHITENING-030 | Whitening Oral Care Compound | 1 |
| BULK-PW-ALMOND-031 | Almond Personal Wash Compound | 1 |
| BULK-PW-ALOE-032 | Aloe Personal Wash Compound | 1 |
| BULK-PW-CHARCOAL-033 | Charcoal Personal Wash Compound | 1 |
| BULK-PW-CITRUS-034 | Citrus Personal Wash Compound | 1 |
| BULK-PW-COCONUT-035 | Coconut Personal Wash Compound | 1 |
| BULK-PW-CUCUMBER-036 | Cucumber Personal Wash Compound | 1 |
| BULK-PW-HONEY-037 | Honey Personal Wash Compound | 1 |
| BULK-PW-HONEY-BLEND-004 | Honey Personal Wash Blend | 1 |
| BULK-PW-LAVENDER-038 | Lavender Personal Wash Compound | 1 |
| BULK-PW-OCEAN_BREEZE-039 | Ocean Breeze Personal Wash Compound | 1 |
| BULK-PW-POMEGRANATE-040 | Pomegranate Personal Wash Compound | 1 |
| BULK-PW-ROSE-041 | Rose Personal Wash Compound | 1 |
| BULK-PW-SHEA_BUTTER-042 | Shea Butter Personal Wash Compound | 1 |
| BULK-PW-TEA_TREE-043 | Tea Tree Personal Wash Compound | 1 |
| BULK-PW-UNSCENTED-044 | Unscented Personal Wash Compound | 1 |
| BULK-PW-VANILLA-045 | Vanilla Personal Wash Compound | 1 |
| PREMIX-HC-LEMON-001 | Lemon Base Premix | 2 |
| PREMIX-HC-LEMON-002 | Lemon Active Premix | 2 |
| PREMIX-HC-ORANGE-003 | Orange Base Premix | 2 |
| PREMIX-HC-UNSCENTED-004 | Unscented Base Premix | 2 |
| PREMIX-HC-UNSCENTED-005 | Unscented Active Premix | 2 |
| PREMIX-OC-ORIGINAL-006 | Original Base Premix | 2 |
| PREMIX-OC-WHITENING-007 | Whitening Base Premix | 2 |
| PREMIX-PW-ALMOND-008 | Almond Base Premix | 2 |
| PREMIX-PW-ALOE-009 | Aloe Base Premix | 2 |
| PREMIX-PW-ALOE-010 | Aloe Active Premix | 2 |
| PREMIX-PW-CUCUMBER-011 | Cucumber Base Premix | 2 |
| PREMIX-PW-CUCUMBER-012 | Cucumber Active Premix | 2 |
| PREMIX-PW-SHEA-013 | Shea Base Premix | 2 |

## PREMIX Formulas (bom_level=2)

| formula_code | name | bom_level | batch_size_kg |
| --- | --- | --- | --- |
| FORM-PREMIX-HC-LEMON-001 | Formula for PREMIX-HC-LEMON-001 | 2 | 1000.0 |
| FORM-PREMIX-HC-LEMON-002 | Formula for PREMIX-HC-LEMON-002 | 2 | 1000.0 |
| FORM-PREMIX-HC-ORANGE-003 | Formula for PREMIX-HC-ORANGE-003 | 2 | 1000.0 |
| FORM-PREMIX-HC-UNSCENTED-004 | Formula for PREMIX-HC-UNSCENTED-004 | 2 | 1000.0 |
| FORM-PREMIX-HC-UNSCENTED-005 | Formula for PREMIX-HC-UNSCENTED-005 | 2 | 1000.0 |
| FORM-PREMIX-OC-ORIGINAL-006 | Formula for PREMIX-OC-ORIGINAL-006 | 2 | 1000.0 |
| FORM-PREMIX-OC-WHITENING-007 | Formula for PREMIX-OC-WHITENING-007 | 2 | 1000.0 |
| FORM-PREMIX-PW-ALMOND-008 | Formula for PREMIX-PW-ALMOND-008 | 2 | 1000.0 |
| FORM-PREMIX-PW-ALOE-009 | Formula for PREMIX-PW-ALOE-009 | 2 | 1000.0 |
| FORM-PREMIX-PW-ALOE-010 | Formula for PREMIX-PW-ALOE-010 | 2 | 1000.0 |
| FORM-PREMIX-PW-CUCUMBER-011 | Formula for PREMIX-PW-CUCUMBER-011 | 2 | 1000.0 |
| FORM-PREMIX-PW-CUCUMBER-012 | Formula for PREMIX-PW-CUCUMBER-012 | 2 | 1000.0 |
| FORM-PREMIX-PW-SHEA-013 | Formula for PREMIX-PW-SHEA-013 | 2 | 1000.0 |

## BOM Level 1 Formulas (sample)

| formula_code | name | bom_level | batch_size_kg |
| --- | --- | --- | --- |
| FORM-BULK-HC-APPLE_BLOSSO-001 | Formula for BULK-HC-APPLE_BLOSSO-001 | 1 | 1000.0 |
| FORM-BULK-HC-CITRUS-BLEND-001 | Formula for BULK-HC-CITRUS-BLEND-001 | 1 | 1000.0 |
| FORM-BULK-HC-CITRUS_ZEST-002 | Formula for BULK-HC-CITRUS_ZEST-002 | 1 | 1000.0 |
| FORM-BULK-HC-CLEAN-BLEND-002 | Formula for BULK-HC-CLEAN-BLEND-002 | 1 | 1000.0 |
| FORM-BULK-HC-CLEAN_LINEN-003 | Formula for BULK-HC-CLEAN_LINEN-003 | 1 | 1000.0 |
| FORM-BULK-HC-EUCALYPTUS-004 | Formula for BULK-HC-EUCALYPTUS-004 | 1 | 1000.0 |
| FORM-BULK-HC-FLORAL-005 | Formula for BULK-HC-FLORAL-005 | 1 | 1000.0 |
| FORM-BULK-HC-FRESH-006 | Formula for BULK-HC-FRESH-006 | 1 | 1000.0 |
| FORM-BULK-HC-LAVENDER-007 | Formula for BULK-HC-LAVENDER-007 | 1 | 1000.0 |
| FORM-BULK-HC-LEMON-008 | Formula for BULK-HC-LEMON-008 | 1 | 1000.0 |

## BOM Level 0 Formulas (sample)

| formula_code | name | bom_level |
| --- | --- | --- |
| FORM-SKU-HOME-376 | Formula for SKU-HOME-376 | 0 |
| FORM-SKU-HOME-377 | Formula for SKU-HOME-377 | 0 |
| FORM-SKU-HOME-378 | Formula for SKU-HOME-378 | 0 |
| FORM-SKU-HOME-379 | Formula for SKU-HOME-379 | 0 |
| FORM-SKU-HOME-380 | Formula for SKU-HOME-380 | 0 |
| FORM-SKU-HOME-381 | Formula for SKU-HOME-381 | 0 |
| FORM-SKU-HOME-382 | Formula for SKU-HOME-382 | 0 |
| FORM-SKU-HOME-383 | Formula for SKU-HOME-383 | 0 |
| FORM-SKU-HOME-384 | Formula for SKU-HOME-384 | 0 |
| FORM-SKU-HOME-385 | Formula for SKU-HOME-385 | 0 |

## Batch product_type Distribution

| product_type | count | pct |
| --- | --- | --- |
| finished_good | 93850 | 75.4 |
| bulk_intermediate | 28877 | 23.2 |
| premix | 1691 | 1.4 |

## Sample Batches (premix)

| batch_number | product_type | bom_level | quantity_kg | yield_percent | status |
| --- | --- | --- | --- | --- | --- |
| B-001-000001 | premix | 2 | 1250.643466007928 | 98.5 | completed |
| B-001-000096 | premix | 2 | 59754.593292588295 | 98.5 | completed |
| B-001-000097 | premix | 2 | 644.3873311562984 | 98.5 | completed |
| B-001-000216 | premix | 2 | 7245.6955167089845 | 98.5 | completed |
| B-001-000217 | premix | 2 | 7245.6955167089845 | 98.5 | completed |
| B-002-000328 | premix | 2 | 403.9234792897032 | 98.5 | completed |
| B-002-000329 | premix | 2 | 1472.4092618386737 | 98.5 | completed |
| B-002-000330 | premix | 2 | 35.6907395388985 | 98.5 | completed |
| B-002-000331 | premix | 2 | 35.6907395388985 | 98.5 | completed |
| B-002-000428 | premix | 2 | 2103.4850469733665 | 98.5 | completed |

## Sample Batches (finished_good with diverse status)

| batch_number | product_type | status | quantity_kg |
| --- | --- | --- | --- |
| B-001-000021 | finished_good | completed | 8396.41462034204 |
| B-001-000022 | finished_good | completed | 8296.3900816986 |
| B-001-000023 | finished_good | completed | 8044.331557646906 |
| B-001-000024 | finished_good | completed | 6715.850531682246 |
| B-001-000025 | finished_good | completed | 7151.78895940641 |
| B-001-000026 | finished_good | completed | 7314.586697810346 |
| B-001-000027 | finished_good | completed | 8524.5260580376 |
| B-001-000028 | finished_good | completed | 7237.143157234165 |
| B-001-000029 | finished_good | completed | 4981.31601003057 |
| B-001-000030 | finished_good | completed | 6510.489436280893 |

## Status Distribution: Purchase Orders

| status | count | pct |
| --- | --- | --- |
| closed | 5443 | 91.3 |
| open | 263 | 4.4 |
| received | 258 | 4.3 |

## Status Distribution: Goods Receipts

| status | count | pct |
| --- | --- | --- |
| posted | 4655298 | 97.3 |
| inspected | 86799 | 1.8 |
| received | 44637 | 0.9 |

## Status Distribution: Batches

| status | count | pct |
| --- | --- | --- |
| completed | 120798 | 97.1 |
| in_progress | 2912 | 2.3 |
| pending | 708 | 0.6 |

## Status Distribution: Work Orders

| status | count | pct |
| --- | --- | --- |
| complete | 127729 | 94.3 |
| in_progress | 5832 | 4.3 |
| planned | 1942 | 1.4 |

## Status Distribution: Orders

| status | count | pct |
| --- | --- | --- |
| delivered | 1161763 | 88.2 |
| shipped | 77458 | 5.9 |
| allocated | 38730 | 2.9 |
| pending | 38726 | 2.9 |

## Status Distribution: Shipments

| status | count | pct |
| --- | --- | --- |
| delivered | 6025453 | 95.5 |
| in_transit | 216040 | 3.4 |
| planned | 69236 | 1.1 |

## Status Distribution: Returns

| status | count | pct |
| --- | --- | --- |
| processed | 60571 | 91.2 |
| received | 2933 | 4.4 |
| approved | 1591 | 2.4 |
| requested | 1324 | 2.0 |

## Status Distribution: AP Invoices

| status | count | pct |
| --- | --- | --- |
| open | 4810512 | 100.0 |

## Status Distribution: AR Invoices

| status | count | pct |
| --- | --- | --- |
| open | 1289406 | 97.0 |
| disputed | 24037 | 1.8 |
| partial | 15777 | 1.2 |

## Sample Order IDs

| order_number | status | day |
| --- | --- | --- |
| ORD-1-CLUB-DC-001-1 | delivered | 1 |
| ORD-1-CLUB-DC-002-2 | delivered | 1 |
| ORD-1-CLUB-DC-003-3 | delivered | 1 |
| ORD-1-DIST-DC-001-4 | delivered | 1 |
| ORD-1-DIST-DC-002-5 | delivered | 1 |
| ORD-1-DIST-DC-003-6 | delivered | 1 |
| ORD-1-DTC-FC-001-7 | delivered | 1 |
| ORD-1-DTC-FC-002-8 | delivered | 1 |
| ORD-1-DTC-FC-003-9 | delivered | 1 |
| ORD-1-ECOM-FC-001-10 | delivered | 1 |

## Sample Shipment IDs

| shipment_number | status | ship_date |
| --- | --- | --- |
| SHIP-PLANT-001-000001 | delivered | 1 |
| SHIP-PLANT-001-000002 | delivered | 1 |
| SHIP-PLANT-001-000003 | delivered | 1 |
| SHIP-PLANT-001-000004 | delivered | 1 |
| SHIP-PLANT-001-000005 | delivered | 1 |
| SHIP-PLANT-001-000006 | delivered | 1 |
| SHIP-PLANT-001-000007 | delivered | 1 |
| SHIP-PLANT-001-000008 | delivered | 1 |
| SHIP-PLANT-001-000009 | delivered | 1 |
| SHIP-PLANT-001-000010 | delivered | 1 |

## Sample PO Numbers

| po_number | status | order_date |
| --- | --- | --- |
| PO-CONS-001-001 | closed | 1 |
| PO-CONS-001-002 | closed | 1 |
| PO-CONS-001-002 | closed | 1 |
| PO-CONS-001-005 | closed | 1 |
| PO-CONS-001-005 | closed | 1 |
| PO-CONS-001-013 | closed | 1 |
| PO-CONS-001-016 | closed | 1 |
| PO-CONS-001-017 | closed | 1 |
| PO-CONS-001-019 | closed | 1 |
| PO-CONS-001-020 | closed | 1 |

## Sample GR Numbers

| gr_number | status | receipt_date |
| --- | --- | --- |
| GR-SHP-1-SUP-001-PLANT-CA-16752 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16753 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16754 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16755 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16756 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16757 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16758 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16759 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16760 | posted | 2 |
| GR-SHP-1-SUP-001-PLANT-CA-16761 | posted | 2 |

## Sample Return IDs

| return_number | status | return_date |
| --- | --- | --- |
| RMA-001-0-1069 | processed | 1 |
| RMA-001-0-1189 | processed | 1 |
| RMA-001-0-2071 | processed | 1 |
| RMA-001-0-2381 | processed | 1 |
| RMA-001-0-2431 | processed | 1 |
| RMA-001-0-2821 | processed | 1 |
| RMA-001-0-2961 | processed | 1 |
| RMA-001-0-3131 | processed | 1 |
| RMA-001-0-3810 | processed | 1 |
| RMA-001-0-3900 | processed | 1 |

## Lead Time Ranges by Tier

| tier | min_lt | max_lt | avg_lt | links |
| --- | --- | --- | --- | --- |
| 1 | 5 | 72 | 24.3 | 176 |

## Lateral RDC-to-RDC Route Segments

| segment_code | origin_id | destination_id | distance_km | transit_time_hours |
| --- | --- | --- | --- | --- |

## Plants

| plant_code | name | city | country |
| --- | --- | --- | --- |
| PLANT-CA | California Plant | Sacramento | CA |
| PLANT-GA | Georgia Plant | Atlanta | GA |
| PLANT-OH | Ohio Plant | Columbus | OH |
| PLANT-TX | Texas Plant | Dallas | TX |

## RDCs

| dc_code | name | city | country | type |
| --- | --- | --- | --- | --- |

## DCs (sample)

| dc_code | name | city | country | type |
| --- | --- | --- | --- | --- |

## Store Code Patterns (sample by channel)

| location_code | name | channel |
| --- | --- | --- |
| STORE-CLUB-001-0001 | Club Store 1-1 | CustomerChannel.CLUB |
| STORE-CLUB-001-0002 | Club Store 1-2 | CustomerChannel.CLUB |
| STORE-CLUB-001-0003 | Club Store 1-3 | CustomerChannel.CLUB |
| STORE-CLUB-001-0004 | Club Store 1-4 | CustomerChannel.CLUB |
| STORE-CLUB-001-0005 | Club Store 1-5 | CustomerChannel.CLUB |
| STORE-CLUB-001-0006 | Club Store 1-6 | CustomerChannel.CLUB |
| STORE-CLUB-001-0007 | Club Store 1-7 | CustomerChannel.CLUB |
| STORE-CLUB-001-0008 | Club Store 1-8 | CustomerChannel.CLUB |
| STORE-CLUB-001-0009 | Club Store 1-9 | CustomerChannel.CLUB |
| STORE-CLUB-001-0010 | Club Store 1-10 | CustomerChannel.CLUB |
| STORE-CLUB-001-0011 | Club Store 1-11 | CustomerChannel.CLUB |
| STORE-CLUB-001-0012 | Club Store 1-12 | CustomerChannel.CLUB |
| STORE-CLUB-001-0013 | Club Store 1-13 | CustomerChannel.CLUB |
| STORE-CLUB-001-0014 | Club Store 1-14 | CustomerChannel.CLUB |
| STORE-CLUB-001-0015 | Club Store 1-15 | CustomerChannel.CLUB |
| STORE-CLUB-002-0001 | Club Store 2-1 | CustomerChannel.CLUB |
| STORE-CLUB-002-0002 | Club Store 2-2 | CustomerChannel.CLUB |
| STORE-CLUB-002-0003 | Club Store 2-3 | CustomerChannel.CLUB |
| STORE-CLUB-002-0004 | Club Store 2-4 | CustomerChannel.CLUB |
| STORE-CLUB-002-0005 | Club Store 2-5 | CustomerChannel.CLUB |

## GL Anomaly Counts

| dup_entries | dup_groups |
| --- | --- |
| 290615 | 121094 |

## Channels

| channel_code | name |
| --- | --- |
| CLUB | Club |
| DISTRIBUTOR | Distributor |
| DTC | Dtc |
| ECOMMERCE | Ecommerce |
| GROCERY | Grocery |
| MASS_RETAIL | Mass Retail |
| PHARMACY | Pharmacy |

## All Ingredients

| ingredient_code | name | subcategory |
| --- | --- | --- |
| ACT-ABRASIVE-001 | Calcium Carbonate | active_ingredient |
| ACT-ABRASIVE-002 | Sodium Bicarbonate | active_ingredient |
| ACT-ANTIMICROBIAL-001 | Triclosan | active_ingredient |
| ACT-ANTIMICROBIAL-002 | Cetylpyridinium Chloride | active_ingredient |
| ACT-BINDER-001 | PVM/MA Copolymer | active_ingredient |
| ACT-BINDER-002 | Sodium CMC | active_ingredient |
| ACT-COLORANT-001 | Titanium Dioxide CI 77891 | active_ingredient |
| ACT-COLORANT-002 | FD&C Blue No. 1 | active_ingredient |
| ACT-FLUORIDE-001 | Sodium Fluoride | active_ingredient |
| ACT-FLUORIDE-002 | Stannous Fluoride | active_ingredient |
| ACT-FRAGRANCE-001 | Menthol Crystal | active_ingredient |
| ACT-FRAGRANCE-002 | Limonene D-Isomer | active_ingredient |
| ACT-HUMECTANT-001 | Glycerin USP | active_ingredient |
| ACT-HUMECTANT-002 | Propylene Glycol | active_ingredient |
| ACT-PRESERVATIVE-001 | Methylparaben | active_ingredient |
| ACT-PRESERVATIVE-002 | Phenoxyethanol | active_ingredient |
| ACT-SURFACTANT-001 | Sodium Lauryl Sulfate | active_ingredient |
| ACT-SURFACTANT-002 | Sodium Laureth Sulfate | active_ingredient |
| ACT-WHITENING-001 | Hydrogen Peroxide 3% | active_ingredient |
| ACT-WHITENING-002 | Sodium Tripolyphosphate | active_ingredient |
| BLK-OIL-001 | Mineral Oil NF | base_material |
| BLK-OIL-002 | Castor Oil | base_material |
| BLK-OIL-003 | Coconut Oil RBD | base_material |
| BLK-SILICATE-001 | Sodium Silicate | base_material |
| BLK-SILICATE-002 | Hydrated Silica | base_material |
| BLK-SILICATE-003 | Precipitated Silica | base_material |
| BLK-SURF_BASE-001 | Cocamidopropyl Betaine | base_material |
| BLK-SURF_BASE-002 | Sodium Cocoyl Isethionate | base_material |
| BLK-SURF_BASE-003 | Decyl Glucoside | base_material |
| BLK-THICKENER-001 | Carbomer 940 | base_material |
| BLK-THICKENER-002 | Xanthan Gum | base_material |
| BLK-THICKENER-003 | Hydroxyethyl Cellulose | base_material |
| BLK-WATER-001 | Purified Water USP | base_material |
| BLK-WATER-002 | Deionized Water | base_material |
| BLK-WATER-003 | Distilled Water | base_material |
| BLK-WAX-001 | Carnauba Wax | base_material |
| BLK-WAX-002 | Microcrystalline Wax | base_material |
| BLK-WAX-003 | Paraffin Wax | base_material |
| PKG-PRI-BOTTLE-001 | PET Clear Bottle | packaging |
| PKG-PRI-BOTTLE-002 | HDPE Opaque Bottle | packaging |
| PKG-PRI-BOTTLE-003 | rPET Recycled Bottle | packaging |
| PKG-PRI-CAP-001 | Flip-Top Cap PP | packaging |
| PKG-PRI-CAP-002 | Screw Cap PE | packaging |
| PKG-PRI-CAP-003 | Disc-Top Cap PP | packaging |
| PKG-PRI-POUCH-001 | Stand-Up Pouch PE/PET | packaging |
| PKG-PRI-POUCH-002 | Spout Pouch Laminate | packaging |
| PKG-PRI-POUCH-003 | Flat Pouch LDPE | packaging |
| PKG-PRI-PUMP-001 | Lotion Pump 28/410 | packaging |
| PKG-PRI-PUMP-002 | Foaming Pump 43/410 | packaging |
| PKG-PRI-PUMP-003 | Treatment Pump 24/410 | packaging |
| PKG-PRI-SEAL-001 | Induction Heat Seal | packaging |
| PKG-PRI-SEAL-002 | Pressure-Sensitive Liner | packaging |
| PKG-PRI-SEAL-003 | Tamper-Evident Band | packaging |
| PKG-PRI-TUBE-001 | HDPE Laminate Tube | packaging |
| PKG-PRI-TUBE-002 | Aluminum Barrier Tube | packaging |
| PKG-PRI-TUBE-003 | ABL Squeeze Tube | packaging |
| PKG-SEC-CARTON-001 | SBS Folding Carton | packaging |
| PKG-SEC-CARTON-002 | Corrugated Tuck-Top Box | packaging |
| PKG-SEC-CARTON-003 | Kraft Window Carton | packaging |
| PKG-SEC-INSERT-001 | Coated Paper Leaflet | packaging |
| PKG-SEC-INSERT-002 | Folded Instruction Card | packaging |
| PKG-SEC-INSERT-003 | Promotional Coupon Insert | packaging |
| PKG-SEC-LABEL-001 | Pressure-Sensitive PP Label | packaging |
| PKG-SEC-LABEL-002 | Cut & Stack Paper Label | packaging |
| PKG-SEC-LABEL-003 | In-Mold Label PP | packaging |
| PKG-SEC-SLEEVE-001 | PVC Shrink Sleeve | packaging |
| PKG-SEC-SLEEVE-002 | PET Shrink Band | packaging |
| PKG-SEC-SLEEVE-003 | OPS Wrap-Around Sleeve | packaging |
| PKG-TER-CORNER_BOARD-001 | Recycled Fiber Corner Board | packaging |
| PKG-TER-CORNER_BOARD-002 | Laminated Edge Protector | packaging |
| PKG-TER-DIVIDER-001 | Corrugated Layer Pad | packaging |
| PKG-TER-DIVIDER-002 | Chipboard Cell Divider | packaging |
| PKG-TER-PALLET_WRAP-001 | Machine-Grade Stretch Film | packaging |
| PKG-TER-PALLET_WRAP-002 | Pre-Stretched Hand Wrap | packaging |
| PKG-TER-SHIPPER-001 | RSC Corrugated Shipper | packaging |
| PKG-TER-SHIPPER-002 | Die-Cut Display Shipper | packaging |
| PKG-TER-SHRINK_WRAP-001 | LDPE Stretch Wrap 80ga | packaging |
| PKG-TER-SHRINK_WRAP-002 | PVC Shrink Film 75ga | packaging |

