"""Extract real entity codes from ERP CSVs via DuckDB for ontology/question updates."""

import duckdb

M = "data/output/erp/master"
T = "data/output/erp/transactional"
R = "data/output/erp/reference"

con = duckdb.connect()

sections: list[tuple[str, str]] = []


def q(label: str, sql: str) -> None:
    """Run query and collect output."""
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    lines = [f"## {label}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    sections.append(("\n".join(lines),))


# --- Product Counts ---
q("Product Counts", f"""
SELECT
    (SELECT COUNT(*) FROM read_csv_auto('{M}/skus.csv') WHERE sku_code NOT LIKE '%-OLD') as active_skus,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/skus.csv') WHERE sku_code LIKE '%-OLD') as old_sku_aliases,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/skus.csv')) as total_skus_with_aliases,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/bulk_intermediates.csv') WHERE bom_level=1) as primary_bulks,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/bulk_intermediates.csv') WHERE bom_level=2) as premix_count,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/bulk_intermediates.csv')) as total_intermediates,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/ingredients.csv')) as raw_materials,
    (SELECT COUNT(*) FROM read_csv_auto('{M}/formulas.csv')) as formulas
""")

# --- Formula Distribution ---
q("Formula Counts by bom_level", f"""
SELECT bom_level, COUNT(*) as count
FROM read_csv_auto('{M}/formulas.csv')
GROUP BY bom_level ORDER BY bom_level
""")

# --- Suppliers (top 20 with Kraljic info) ---
q("Suppliers (all)", f"""
SELECT supplier_code, name, tier
FROM read_csv_auto('{M}/suppliers.csv')
WHERE supplier_code NOT LIKE '%-ALT'
ORDER BY supplier_code
""")

q("ALT Supplier Duplicates", f"""
SELECT supplier_code, name
FROM read_csv_auto('{M}/suppliers.csv')
WHERE supplier_code LIKE '%-ALT'
ORDER BY supplier_code
""")

# --- Ingredients by subcategory ---
q("Ingredients by Subcategory (sample)", f"""
SELECT ingredient_code, name, category, subcategory
FROM read_csv_auto('{M}/ingredients.csv')
ORDER BY subcategory, ingredient_code
LIMIT 30
""")

# --- SKUs by category (sample) ---
q("SKUs by Category (sample 5 each)", f"""
(SELECT sku_code, name, category, brand FROM read_csv_auto('{M}/skus.csv')
 WHERE category='ORAL_CARE' AND sku_code NOT LIKE '%-OLD' LIMIT 5)
UNION ALL
(SELECT sku_code, name, category, brand FROM read_csv_auto('{M}/skus.csv')
 WHERE category='HOME_CARE' AND sku_code NOT LIKE '%-OLD' LIMIT 5)
UNION ALL
(SELECT sku_code, name, category, brand FROM read_csv_auto('{M}/skus.csv')
 WHERE category='PERSONAL_WASH' AND sku_code NOT LIKE '%-OLD' LIMIT 5)
""")

# --- OLD SKU aliases ---
q("SKU Aliases (-OLD)", f"""
SELECT s.sku_code, s.name, s.category
FROM read_csv_auto('{M}/skus.csv') s
WHERE s.sku_code LIKE '%-OLD'
LIMIT 25
""")

# --- Bulk Intermediates (all) ---
q("Bulk Intermediates (all)", f"""
SELECT bulk_code, name, bom_level
FROM read_csv_auto('{M}/bulk_intermediates.csv')
ORDER BY bom_level, bulk_code
""")

# --- PREMIX Formulas ---
q("PREMIX Formulas (bom_level=2)", f"""
SELECT f.formula_code, f.name, f.bom_level, f.batch_size_kg
FROM read_csv_auto('{M}/formulas.csv') f
WHERE f.bom_level = 2
ORDER BY f.formula_code
""")

# --- Sample bom_level=1 formulas ---
q("BOM Level 1 Formulas (sample)", f"""
SELECT f.formula_code, f.name, f.bom_level, f.batch_size_kg
FROM read_csv_auto('{M}/formulas.csv') f
WHERE f.bom_level = 1
ORDER BY f.formula_code
LIMIT 10
""")

# --- Sample bom_level=0 formulas ---
q("BOM Level 0 Formulas (sample)", f"""
SELECT f.formula_code, f.name, f.bom_level
FROM read_csv_auto('{M}/formulas.csv') f
WHERE f.bom_level = 0
ORDER BY f.formula_code
LIMIT 10
""")

# --- Batch product_type distribution ---
q("Batch product_type Distribution", f"""
SELECT product_type, COUNT(*) as count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
FROM read_csv_auto('{T}/batches.csv')
GROUP BY product_type ORDER BY count DESC
""")

# --- Sample batches by type ---
q("Sample Batches (premix)", f"""
SELECT batch_number, product_type, bom_level, quantity_kg, yield_percent, status
FROM read_csv_auto('{T}/batches.csv')
WHERE product_type = 'premix'
LIMIT 10
""")

q("Sample Batches (finished_good with diverse status)", f"""
SELECT batch_number, product_type, status, quantity_kg
FROM read_csv_auto('{T}/batches.csv')
WHERE product_type = 'finished_good'
ORDER BY batch_number
LIMIT 10
""")

# --- Status distributions for all entity types ---
for entity, file, status_col in [
    ("Purchase Orders", f"{T}/purchase_orders.csv", "status"),
    ("Goods Receipts", f"{T}/goods_receipts.csv", "status"),
    ("Batches", f"{T}/batches.csv", "status"),
    ("Work Orders", f"{T}/work_orders.csv", "status"),
    ("Orders", f"{T}/orders.csv", "status"),
    ("Shipments", f"{T}/shipments.csv", "status"),
    ("Returns", f"{T}/returns.csv", "status"),
    ("AP Invoices", f"{T}/ap_invoices.csv", "status"),
    ("AR Invoices", f"{T}/ar_invoices.csv", "status"),
]:
    q(f"Status Distribution: {entity}", f"""
    SELECT {status_col}, COUNT(*) as count,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
    FROM read_csv_auto('{file}')
    GROUP BY {status_col} ORDER BY count DESC
    """)

# --- Sample entity IDs ---
q("Sample Order IDs", f"""
SELECT order_number, status, day
FROM read_csv_auto('{T}/orders.csv')
ORDER BY order_number
LIMIT 10
""")

q("Sample Shipment IDs", f"""
SELECT shipment_number, status, ship_date
FROM read_csv_auto('{T}/shipments.csv')
ORDER BY shipment_number
LIMIT 10
""")

q("Sample PO Numbers", f"""
SELECT po_number, status, order_date
FROM read_csv_auto('{T}/purchase_orders.csv')
ORDER BY po_number
LIMIT 10
""")

q("Sample GR Numbers", f"""
SELECT gr_number, status, receipt_date
FROM read_csv_auto('{T}/goods_receipts.csv')
ORDER BY gr_number
LIMIT 10
""")

q("Sample Return IDs", f"""
SELECT return_number, status, return_date
FROM read_csv_auto('{T}/returns.csv')
ORDER BY return_number
LIMIT 10
""")

# --- Lead time ranges ---
q("Lead Time Ranges by Tier", f"""
SELECT s.tier, MIN(si.lead_time_days) as min_lt, MAX(si.lead_time_days) as max_lt,
       ROUND(AVG(si.lead_time_days), 1) as avg_lt, COUNT(*) as links
FROM read_csv_auto('{M}/supplier_ingredients.csv') si
JOIN read_csv_auto('{M}/suppliers.csv') s ON si.supplier_id = s.id
GROUP BY s.tier ORDER BY s.tier
""")

# --- Lateral routes ---
q("Lateral RDC-to-RDC Route Segments", f"""
SELECT segment_code, origin_id, destination_id, distance_km, transit_time_hours
FROM read_csv_auto('{M}/route_segments.csv')
WHERE segment_code LIKE '%rdc_to_rdc%'
LIMIT 20
""")

# --- Plants ---
q("Plants", f"""
SELECT plant_code, name, city, country
FROM read_csv_auto('{M}/plants.csv')
ORDER BY plant_code
""")

# --- Distribution Centers (RDCs) ---
q("RDCs", f"""
SELECT dc_code, name, city, country, type
FROM read_csv_auto('{M}/distribution_centers.csv')
WHERE type = 'RDC'
ORDER BY dc_code
""")

# --- Distribution Centers (DCs sample) ---
q("DCs (sample)", f"""
SELECT dc_code, name, city, country, type
FROM read_csv_auto('{M}/distribution_centers.csv')
WHERE type = 'DC'
ORDER BY dc_code
LIMIT 10
""")

# --- Store patterns ---
q("Store Code Patterns (sample by channel)", f"""
SELECT rl.location_code, rl.name, rl.channel
FROM read_csv_auto('{M}/retail_locations.csv') rl
ORDER BY rl.channel, rl.location_code
LIMIT 20
""")

# --- GL anomaly counts ---
q("GL Anomaly Counts", f"""
SELECT
    (SELECT COUNT(*) FROM read_csv_auto('{T}/gl_journal.csv') WHERE reference_id LIKE '%-DUP') as dup_entries,
    (SELECT COUNT(DISTINCT reference_id) FROM read_csv_auto('{T}/gl_journal.csv') WHERE reference_id LIKE '%-DUP') as dup_groups
""")

# --- Channel list ---
q("Channels", f"""
SELECT channel_code, name FROM read_csv_auto('{M}/channels.csv') ORDER BY channel_code
""")

# --- Ingredient names (full list) ---
q("All Ingredients", f"""
SELECT ingredient_code, name, subcategory
FROM read_csv_auto('{M}/ingredients.csv')
ORDER BY ingredient_code
""")

# --- Write output ---
output = "# Entity Codes Reference (v0.93.0)\n\n"
output += "Generated from ERP CSVs via DuckDB. Use for ontology + question updates.\n\n"
for (s,) in sections:
    output += s + "\n"

with open("ontology_feedback_resources/entity_codes_reference.md", "w") as f:
    f.write(output)

print("Done. Saved to ontology_feedback_resources/entity_codes_reference.md")
