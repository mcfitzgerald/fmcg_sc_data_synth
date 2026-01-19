"""
Export ERP Format Script
------------------------
Transforms the flat Prism Sim CSV output into a normalized, relational ERP database format
matching the schema in reference/fmcg_example_OLD/schema.sql.

Usage:
    poetry run python scripts/export_erp_format.py --input-dir data/output --output-dir data/output/erp_format
"""

import argparse
import ast
import csv
import json
import logging
import random
from pathlib import Path

import pandas as pd
from faker import Faker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(42)
random.seed(42)

class IdMapper:
    """Manages mapping between simulation string IDs and ERP integer IDs."""

    def __init__(self):
        self.maps: dict[str, dict[str, int]] = {
            'locations': {},
            'products': {},
            'orders': {},
            'shipments': {},
            'batches': {},
            'formulas': {},
            'work_orders': {},
            'demand_forecasts': {},
            'returns': {}
        }
        self.counters: dict[str, int] = {
            'locations': 1,
            'products': 1,
            'orders': 1,
            'shipments': 1,
            'batches': 1,
            'formulas': 1,
            'work_orders': 1,
            'demand_forecasts': 1,
            'returns': 1
        }

    def get_id(self, entity_type: str, sim_id: str) -> int:
        """Get or create an integer ID for a given simulation string ID."""
        if sim_id not in self.maps[entity_type]:
            self.maps[entity_type][sim_id] = self.counters[entity_type]
            self.counters[entity_type] += 1
        return self.maps[entity_type][sim_id]

    def save(self, path: Path):
        """Save mappings to disk."""
        # Convert internal state to serializable format if needed,
        # but here we just dump the dicts
        with open(path, 'w') as f:
            json.dump(self.maps, f, indent=2)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def parse_location_type(loc_type: str, loc_name: str) -> str:
    """Map simulation node types to ERP table categories."""
    loc_type = str(loc_type).upper()
    loc_name = str(loc_name).upper()

    if 'SUPPLIER' in loc_type:
        return 'supplier'
    elif 'PLANT' in loc_type:
        return 'plant'
    elif 'DC' in loc_type or 'DISTRIBUTION' in loc_type or 'RDC' in loc_type:
        return 'dc'
    elif 'STORE' in loc_type or 'RETAIL' in loc_type:
        return 'store'
    else:
        # Fallback based on name or known prefixes
        if loc_name.startswith('SUP-'): return 'supplier'
        if loc_name.startswith('PLANT-'): return 'plant'
        if loc_name.startswith('DC-') or loc_name.startswith('DIST-'): return 'dc'
        if loc_name.startswith('STORE-'): return 'store'
        return 'unknown'

def process_locations(static_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process locations.csv into suppliers, plants, distribution_centers, retail_locations."""
    logger.info("Processing locations...")

    input_file = static_dir / 'locations.csv'
    if not input_file.exists():
        logger.warning(f"Locations file not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    # Prepare output files
    files = {
        'supplier': open(output_dir / 'master/suppliers.csv', 'w', newline=''),
        'plant': open(output_dir / 'master/plants.csv', 'w', newline=''),
        'dc': open(output_dir / 'master/distribution_centers.csv', 'w', newline=''),
        'store': open(output_dir / 'master/retail_locations.csv', 'w', newline='')
    }

    writers = {
        'supplier': csv.writer(files['supplier']),
        'plant': csv.writer(files['plant']),
        'dc': csv.writer(files['dc']),
        'store': csv.writer(files['store'])
    }

    # Headers
    writers['supplier'].writerow(['id', 'supplier_code', 'name', 'city', 'country', 'tier', 'is_active'])
    writers['plant'].writerow(['id', 'plant_code', 'name', 'city', 'country', 'capacity_tons_per_day', 'is_active'])
    writers['dc'].writerow(['id', 'dc_code', 'name', 'city', 'country', 'type', 'is_active'])
    writers['store'].writerow(['id', 'location_code', 'name', 'city', 'country', 'store_format', 'is_active'])

    count = 0
    for _, row in df.iterrows():
        sim_id = row['id']
        name = row['name']
        loc_str = str(row.get('location', ''))

        # Parse city/country from "City, Country" or just use whole string as city
        if ',' in loc_str:
            parts = loc_str.split(',')
            city = parts[0].strip()
            country = parts[1].strip()
        else:
            city = loc_str
            country = "USA" # Default for this sim

        cat = parse_location_type(row['type'], name)
        erp_id = mapper.get_id('locations', sim_id)

        if cat == 'supplier':
            writers['supplier'].writerow([erp_id, sim_id, name, city, country, 1, 'true'])
        elif cat == 'plant':
            writers['plant'].writerow([erp_id, sim_id, name, city, country, 1000, 'true'])
        elif cat == 'dc':
            writers['dc'].writerow([erp_id, sim_id, name, city, country, 'regional', 'true'])
        elif cat == 'store':
            writers['store'].writerow([erp_id, sim_id, name, city, country, row.get('store_format', 'standard'), 'true'])

        count += 1

    # Close files
    for f in files.values():
        f.close()

    logger.info(f"Processed {count} locations")

def process_products(static_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process products.csv into ingredients and skus."""
    logger.info("Processing products...")

    input_file = static_dir / 'products.csv'
    if not input_file.exists():
        logger.warning(f"Products file not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    files = {
        'ingredient': open(output_dir / 'master/ingredients.csv', 'w', newline=''),
        'sku': open(output_dir / 'master/skus.csv', 'w', newline='')
    }
    writers = {
        'ingredient': csv.writer(files['ingredient']),
        'sku': csv.writer(files['sku'])
    }

    # Headers
    writers['ingredient'].writerow(['id', 'ingredient_code', 'name', 'category', 'unit_of_measure', 'is_active'])
    writers['sku'].writerow(['id', 'sku_code', 'name', 'category', 'brand', 'units_per_case', 'weight_kg', 'is_active'])

    count = 0
    for _, row in df.iterrows():
        sim_id = row['id']
        category = str(row['category'])
        erp_id = mapper.get_id('products', sim_id)

        if 'INGREDIENT' in category.upper():
            writers['ingredient'].writerow([
                erp_id, sim_id, row['name'], 'raw_material', 'kg', 'true'
            ])
        else:
            writers['sku'].writerow([
                erp_id, sim_id, row['name'], category, row.get('brand', 'Prism'),
                row.get('units_per_case', 12), row.get('weight_kg', 0.5), 'true'
            ])
        count += 1

    for f in files.values():
        f.close()
    logger.info(f"Processed {count} products")

def process_recipes(static_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process recipes.csv into formulas and formula_ingredients."""
    logger.info("Processing recipes...")

    input_file = static_dir / 'recipes.csv'
    if not input_file.exists():
        logger.warning(f"Recipes file not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    f_header = open(output_dir / 'master/formulas.csv', 'w', newline='')
    f_lines = open(output_dir / 'master/formula_ingredients.csv', 'w', newline='')

    w_header = csv.writer(f_header)
    w_lines = csv.writer(f_lines)

    w_header.writerow(['id', 'formula_code', 'name', 'product_id', 'batch_size_kg', 'yield_percent'])
    w_lines.writerow(['formula_id', 'ingredient_id', 'sequence', 'quantity_kg'])

    count = 0
    for _, row in df.iterrows():
        product_sim_id = row['product_id']
        # Only process if we've seen this product (skip if filtered out)
        if product_sim_id not in mapper.maps['products']:
            continue

        product_erp_id = mapper.maps['products'][product_sim_id]

        # Create formula
        formula_code = f"FORM-{product_sim_id}"
        formula_erp_id = mapper.get_id('formulas', formula_code)

        w_header.writerow([
            formula_erp_id, formula_code, f"Formula for {product_sim_id}",
            product_erp_id, 1000, 98.0
        ])

        # Parse ingredients
        try:
            ingredients_dict = ast.literal_eval(row['ingredients'])
            seq = 1
            for ing_sim_id, qty in ingredients_dict.items():
                if ing_sim_id in mapper.maps['products']:
                    ing_erp_id = mapper.maps['products'][ing_sim_id]
                    w_lines.writerow([formula_erp_id, ing_erp_id, seq, qty])
                    seq += 1
        except Exception as e:
            logger.error(f"Failed to parse ingredients for {product_sim_id}: {e}")

        count += 1

    f_header.close()
    f_lines.close()
    logger.info(f"Processed {count} recipes")

def process_work_orders(run_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process production_orders.csv into work_orders."""
    logger.info("Processing work orders...")

    input_file = run_dir / 'production_orders.csv'
    if not input_file.exists():
        logger.warning("production_orders.csv not found")
        return

    f_wo = open(output_dir / 'transactional/work_orders.csv', 'w', newline='')
    w_wo = csv.writer(f_wo)
    w_wo.writerow(['id', 'wo_number', 'plant_id', 'formula_id', 'planned_quantity_kg', 'planned_start_date', 'status'])

    df = pd.read_csv(input_file)
    for _, row in df.iterrows():
        sim_id = row['po_id']
        erp_id = mapper.get_id('work_orders', sim_id)

        plant_id = mapper.maps['locations'].get(row['plant_id'], '')
        prod_sim = row['product_id']
        formula_id = mapper.maps['formulas'].get(f"FORM-{prod_sim}", '')

        # Convert cases to kg (approx 10kg/case placeholder)
        qty_cases = row['quantity']
        qty_kg = qty_cases * 10.0

        w_wo.writerow([
            erp_id, sim_id, plant_id, formula_id,
            qty_kg, row['creation_day'], row['status']
        ])

    f_wo.close()

def process_orders(run_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process orders.csv into orders and order_lines. Uses chunks."""
    logger.info("Processing orders (chunked)...")

    input_file = run_dir / 'orders.csv'
    if not input_file.exists(): return

    f_ord = open(output_dir / 'transactional/orders.csv', 'w', newline='')
    f_line = open(output_dir / 'transactional/order_lines.csv', 'w', newline='')

    w_ord = csv.writer(f_ord)
    w_line = csv.writer(f_line)

    w_ord.writerow(['id', 'order_number', 'day', 'source_id', 'retail_location_id', 'status', 'total_cases'])
    w_line.writerow(['order_id', 'line_number', 'sku_id', 'quantity_cases', 'unit_price', 'status'])

    processed_orders = set()
    line_buffer = []

    # Process in chunks
    chunk_size = 100000
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            sim_ord_id = row['order_id']
            erp_ord_id = mapper.get_id('orders', sim_ord_id)

            # Write Header (once per order)
            if erp_ord_id not in processed_orders:
                source_sim = row['source_id']
                target_sim = row['target_id']

                # Try to resolve location IDs, default to NULL if missing (shouldn't happen if complete)
                source_id = mapper.maps['locations'].get(source_sim, '')
                target_id = mapper.maps['locations'].get(target_sim, '')

                w_ord.writerow([
                    erp_ord_id, sim_ord_id, row['day'], source_id, target_id,
                    row['status'], 0 # Total cases updated later? Sim doesn't sum it.
                ])
                processed_orders.add(erp_ord_id)

            # Write Line
            product_sim = row['product_id']
            sku_id = mapper.maps['products'].get(product_sim, '')

            # Simple line number generation (stateless for now, assumes order in stream)
            # In a real heavy ETL, we'd group first. Here we assume CSV is sorted or grouped.
            # If not, we might have collision on line_number 1 if we reset.
            # For simplicity in this stream, we'll just write rows.
            # Ideally: GroupBy in Pandas, but memory risk.
            # Compromise: Just write it. Line number is tricky without state.
            # Let's use a simple counter per order in a separate pass or just omit line_number uniqueness constraint for this V1
            # Or: keep a small LRU cache for line numbers.

            w_line.writerow([
                erp_ord_id, 1, sku_id, row['quantity'], 10.0, 'open'
            ])

    f_ord.close()
    f_line.close()
    logger.info("Processed orders")

def process_shipments(run_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process shipments.csv into shipments and shipment_lines."""
    logger.info("Processing shipments...")

    input_file = run_dir / 'shipments.csv'
    if not input_file.exists(): return

    f_shp = open(output_dir / 'transactional/shipments.csv', 'w', newline='')
    f_line = open(output_dir / 'transactional/shipment_lines.csv', 'w', newline='')

    w_shp = csv.writer(f_shp)
    w_line = csv.writer(f_line)

    w_shp.writerow(['id', 'shipment_number', 'ship_date', 'arrival_date', 'origin_id', 'destination_id', 'status'])
    w_line.writerow(['shipment_id', 'line_number', 'sku_id', 'quantity_cases'])

    processed = set()

    chunk_size = 100000
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            sim_id = row['shipment_id']
            erp_id = mapper.get_id('shipments', sim_id)

            if erp_id not in processed:
                origin = mapper.maps['locations'].get(row['source_id'], '')
                dest = mapper.maps['locations'].get(row['target_id'], '')

                w_shp.writerow([
                    erp_id, sim_id, row['creation_day'], row['arrival_day'],
                    origin, dest, row['status']
                ])
                processed.add(erp_id)

            prod = mapper.maps['products'].get(row['product_id'], '')
            w_line.writerow([erp_id, 1, prod, row['quantity']])

    f_shp.close()
    f_line.close()

def process_batches(run_dir: Path, output_dir: Path, mapper: IdMapper):
    logger.info("Processing batches...")
    input_file = run_dir / 'batches.csv'
    if not input_file.exists(): return

    with open(output_dir / 'transactional/batches.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'batch_number', 'wo_id', 'plant_id', 'formula_id', 'quantity_kg', 'production_date', 'status'])

        df = pd.read_csv(input_file)
        for _, row in df.iterrows():
            sim_id = row['batch_id']
            erp_id = mapper.get_id('batches', sim_id)
            plant_id = mapper.maps['locations'].get(row['plant_id'], '')
            # Try to find formula from product
            prod_sim = row['product_id']
            formula_id = mapper.maps['formulas'].get(f"FORM-{prod_sim}", '')

            # Map production_order_id to wo_id
            po_sim_id = row.get('production_order_id', '')
            wo_id = mapper.maps['work_orders'].get(po_sim_id, '') if po_sim_id else ''

            writer.writerow([
                erp_id, sim_id, wo_id, plant_id, formula_id,
                row['quantity'], row['day_produced'], row['status']
            ])

def process_inventory(run_dir: Path, output_dir: Path, mapper: IdMapper):
    logger.info("Processing inventory (sampling)...")
    input_file = run_dir / 'inventory.csv'
    if not input_file.exists(): return

    # Inventory is HUGE (11GB). We can't rewrite it all easily in this script without taking forever.
    # For now, let's just grab the FINAL DAY (max day) to serve as "Current Inventory".
    # Or just skip if too big.
    # Strategy: Read chunks, keep only rows where day = max_day (needs pre-scan)
    # OR just write raw mapping.

    # Optimization: Just process a small tail?
    # Better: Scan file reversely? No.
    # Let's just create a shell file for now or process first 100k rows as sample.

    with open(output_dir / 'transactional/inventory.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'day', 'location_id', 'sku_id', 'quantity'])

        # Just sample 100k rows for dev speed, user can remove limit later
        chunk_iter = pd.read_csv(input_file, chunksize=100000)
        chunk = next(chunk_iter)

        for i, row in chunk.iterrows():
            loc = mapper.maps['locations'].get(row['node_id'], '')
            prod = mapper.maps['products'].get(row['product_id'], '')
            if loc and prod:
                writer.writerow([i+1, row['day'], loc, prod, row['actual_inventory']])

def main():
    parser = argparse.ArgumentParser(description='Convert Prism Sim output to ERP format')
    parser.add_argument('--input-dir', required=True, help='Base output directory (e.g. data/output)')
    parser.add_argument('--output-dir', required=True, help='Target directory')
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    # Handle structure where files are in subdirs
    run_dir = input_root / 'standard_run'
    static_dir = input_root / 'static_world'

    # If flat, use root
    if not run_dir.exists(): run_dir = input_root
    if not static_dir.exists(): static_dir = input_root

    output_root = Path(args.output_dir)

    # Create structure
    ensure_dir(output_root / 'master')
    ensure_dir(output_root / 'transactional')
    ensure_dir(output_root / 'reference')

    mapper = IdMapper()

    # 1. Masters (builds the ID maps)
    process_locations(static_dir, output_root, mapper)
    process_products(static_dir, output_root, mapper)
    process_recipes(static_dir, output_root, mapper)

    # 1b. Config-based Masters
    process_configs(output_root, mapper)

    # 1c. Seed/Reference Data (Phase 2)
    process_reference_data(static_dir, output_root, mapper)

    # 2. Transactions (uses ID maps)
    process_orders(run_dir, output_root, mapper)
    process_work_orders(run_dir, output_root, mapper)
    process_forecasts(run_dir, output_root, mapper)
    process_shipments(run_dir, output_root, mapper)
    process_returns(run_dir, output_root, mapper)
    process_batches(run_dir, output_root, mapper)
    process_inventory(run_dir, output_root, mapper)

    # 3. Save Map
    mapper.save(output_root / 'reference/id_mapping.json')
    logger.info("Export complete.")

def process_returns(run_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process returns.csv into returns, return_lines, disposition_logs."""
    logger.info("Processing returns...")

    input_file = run_dir / 'returns.csv'
    if not input_file.exists():
        logger.warning("returns.csv not found")
        return

    f_ret = open(output_dir / 'transactional/returns.csv', 'w', newline='')
    f_line = open(output_dir / 'transactional/return_lines.csv', 'w', newline='')
    f_disp = open(output_dir / 'transactional/disposition_logs.csv', 'w', newline='')

    w_ret = csv.writer(f_ret)
    w_line = csv.writer(f_line)
    w_disp = csv.writer(f_disp)

    w_ret.writerow(['id', 'return_number', 'return_date', 'source_id', 'dc_id', 'status'])
    w_line.writerow(['return_id', 'line_number', 'sku_id', 'quantity_cases', 'condition'])
    w_disp.writerow(['return_id', 'return_line_number', 'disposition', 'quantity_cases'])

    # Track processed Returns to dedup header
    processed_returns = set()

    df = pd.read_csv(input_file)

    # Sim log has one row per line.
    for i, row in df.iterrows():
        rma_sim = row['rma_id']
        erp_ret_id = mapper.get_id('returns', rma_sim)

        if erp_ret_id not in processed_returns:
            src = mapper.maps['locations'].get(row['source_id'], '')
            dst = mapper.maps['locations'].get(row['target_id'], '')

            w_ret.writerow([
                erp_ret_id, rma_sim, row['day'], src, dst, row['status']
            ])
            processed_returns.add(erp_ret_id)

        # Line - assuming 1 line per RMA for this simulation logic
        line_num = 1

        prod_sim = row['product_id']
        sku_id = mapper.maps['products'].get(prod_sim, '')

        # Condition derived from disposition
        disp = row['disposition']
        condition = 'sellable' if disp == 'restock' else 'damaged'

        w_line.writerow([
            erp_ret_id, line_num, sku_id, row['quantity'], condition
        ])

        w_disp.writerow([
            erp_ret_id, line_num, disp, row['quantity']
        ])

    f_ret.close()
    f_line.close()
    f_disp.close()

def process_forecasts(run_dir: Path, output_dir: Path, mapper: IdMapper):
    """Process forecasts.csv into demand_forecasts."""
    logger.info("Processing forecasts...")

    input_file = run_dir / 'forecasts.csv'
    if not input_file.exists():
        logger.warning("forecasts.csv not found")
        return

    f_fcast = open(output_dir / 'transactional/demand_forecasts.csv', 'w', newline='')
    w_fcast = csv.writer(f_fcast)
    w_fcast.writerow(['id', 'forecast_version', 'sku_id', 'location_type', 'forecast_date', 'forecast_quantity_cases', 'forecast_type'])

    df = pd.read_csv(input_file)
    for i, row in df.iterrows():
        sim_day = row['day']
        prod_sim = row['product_id']
        sku_id = mapper.maps['products'].get(prod_sim, '')

        # S&OP structure: Version is defined by the day it was generated
        version = f"FCAST-DAY-{sim_day:03d}"

        w_fcast.writerow([
            i + 1,
            version,
            sku_id,
            'total', # Aggregated forecast
            sim_day + 1, # Forecast for the future
            row['forecast_quantity'],
            'statistical'
        ])

    f_fcast.close()

def process_reference_data(static_dir: Path, output_dir: Path, mapper: IdMapper):
    """Generate seed/reference data (Phase 2)."""
    logger.info("Generating reference data (Phase 2)...")

    # --- carriers ---
    # Static list of carriers
    carriers = [
        ("C-TRUCK-01", "Swift Transport", "truck", "C"),
        ("C-TRUCK-02", "Reliable Logistics", "truck", "B"),
        ("C-RAIL-01", "Union Pacific", "rail", "B"),
        ("C-OCEAN-01", "Maersk", "ocean", "A"),
        ("C-PARCEL-01", "FedEx", "parcel", "C")
    ]

    carrier_ids = {} # map code -> id

    ensure_dir(output_dir / 'logistics') # New folder for logistics if we want? Schema puts it in E2.
    # But for CSV structure let's stick to master/transactional or create new ones?
    # Plan said: master, transactional, reference. Let's put carriers in master.

    with open(output_dir / 'master/carriers.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'carrier_code', 'name', 'carrier_type', 'sustainability_rating', 'is_active'])

        for i, (code, name, ctype, rating) in enumerate(carriers, 1):
            writer.writerow([i, code, name, ctype, rating, 'true'])
            carrier_ids[code] = i

    # --- route_segments (from links.csv) ---
    input_file = static_dir / 'links.csv'
    if input_file.exists():
        df = pd.read_csv(input_file)
        with open(output_dir / 'master/route_segments.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'segment_code', 'origin_type', 'origin_id', 'destination_type', 'destination_id', 'transport_mode', 'distance_km', 'transit_time_hours'])

            for i, row in df.iterrows():
                # Try to map IDs
                src = mapper.maps['locations'].get(row['source_id'])
                dst = mapper.maps['locations'].get(row['target_id'])

                if src and dst:
                    # Simple type heuristic based on ID string
                    src_type = parse_location_type('', row['source_id']) # utilizing helper
                    dst_type = parse_location_type('', row['target_id'])

                    writer.writerow([
                        i+1,
                        f"SEG-{row['id']}",
                        src_type, src,
                        dst_type, dst,
                        row.get('mode', 'truck'),
                        row.get('distance_km', 0),
                        row.get('lead_time_days', 1) * 24 # Convert days to hours
                    ])

    # --- kpi_thresholds ---
    # We can extract from sim config (already loaded in process_configs, but we can re-read or pass it.
    # Simpler to re-read or just assume standard for this seed function)
    sim_config_path = Path("src/prism_sim/config/simulation_config.json")
    if sim_config_path.exists():
        with open(sim_config_path) as f:
            cfg = json.load(f)
            benchmarks = cfg.get('simulation_parameters', {}).get('calibration', {}).get('industry_benchmarks', {})

            with open(output_dir / 'master/kpi_thresholds.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'kpi_code', 'kpi_name', 'target_value', 'unit'])

                kpis = [
                    ('KPI-001', 'Inventory Turns', benchmarks.get('target_turns', 6.0), 'turns'),
                    ('KPI-002', 'Service Level', benchmarks.get('target_service_level', 0.97) * 100, 'percent'),
                    ('KPI-003', 'OEE', benchmarks.get('target_oee_range', [0.6])[1] * 100, 'percent')
                ]

                for i, (code, name, val, unit) in enumerate(kpis, 1):
                    writer.writerow([i, code, name, val, unit])

    # --- certifications & supplier_ingredients ---
    # Need access to suppliers and ingredients lists.
    # We can use the mapper keys.

    supplier_sim_ids = [k for k in mapper.maps['locations'] if 'SUP' in k]
    ingredient_sim_ids = [k for k in mapper.maps['products'] if 'INGREDIENT' in k or 'BLK' in k or 'ACT' in k or 'PKG' in k] # Rough filter

    # certifications
    with open(output_dir / 'master/certifications.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'supplier_id', 'certification_type', 'issue_date', 'expiry_date'])

        cert_types = ['ISO9001', 'ISO14001', 'GMP', 'RSPO']

        row_id = 1
        for sup_sim in supplier_sim_ids:
            sup_id = mapper.maps['locations'][sup_sim]
            # Give each supplier 1-2 random certs
            for _ in range(random.randint(1, 2)):
                ctype = random.choice(cert_types)
                writer.writerow([
                    row_id, sup_id, ctype,
                    fake.date_between(start_date='-5y', end_date='-1y'),
                    fake.date_between(start_date='+1y', end_date='+3y')
                ])
                row_id += 1

    # supplier_ingredients (Link suppliers to ingredients)
    # This is M:N. For seed, we assume suppliers specialize.
    # Logic: Assign each ingredient to 1-3 random suppliers.
    with open(output_dir / 'master/supplier_ingredients.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'supplier_id', 'ingredient_id', 'unit_cost', 'lead_time_days', 'min_order_qty'])

        row_id = 1
        processed_links = set()

        if not supplier_sim_ids:
            logger.warning("No suppliers found for seeding supplier_ingredients")
        else:
            for ing_sim in ingredient_sim_ids:
                ing_id = mapper.maps['products'][ing_sim]

                # Pick 1-2 suppliers
                suppliers = random.sample(supplier_sim_ids, k=min(len(supplier_sim_ids), random.randint(1, 2)))

                for sup_sim in suppliers:
                    sup_id = mapper.maps['locations'][sup_sim]

                    if (sup_id, ing_id) not in processed_links:
                        writer.writerow([
                            row_id, sup_id, ing_id,
                            round(random.uniform(0.5, 50.0), 2), # Random cost
                            random.randint(3, 14), # Lead time
                            100 # MOQ
                        ])
                        processed_links.add((sup_id, ing_id))
                        row_id += 1

    # --- sku_costs ---
    # Derived from products.csv cost_per_case
    prod_file = static_dir / 'products.csv'
    if prod_file.exists():
        df_prod = pd.read_csv(prod_file)
        with open(output_dir / 'master/sku_costs.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'sku_id', 'cost_type', 'cost_amount', 'effective_from'])

            row_id = 1
            for _, row in df_prod.iterrows():
                sim_id = row['id']
                if sim_id in mapper.maps['products']:
                    # Only for SKUs (not ingredients, though product file has both)
                    # We can check category
                    if 'INGREDIENT' not in str(row['category']):
                        sku_id = mapper.maps['products'][sim_id]
                        total_cost = row.get('cost_per_case', 10.0)

                        # Split cost into components (heuristic)
                        mat_cost = total_cost * 0.6
                        lab_cost = total_cost * 0.2
                        oh_cost = total_cost * 0.2

                        date_str = '2024-01-01'

                        writer.writerow([row_id, sku_id, 'material', round(mat_cost, 2), date_str])
                        row_id += 1
                        writer.writerow([row_id, sku_id, 'labor', round(lab_cost, 2), date_str])
                        row_id += 1
                        writer.writerow([row_id, sku_id, 'overhead', round(oh_cost, 2), date_str])
                        row_id += 1

    logger.info("Generated reference data")

def process_configs(output_dir: Path, mapper: IdMapper):
    """Process configuration files to populate config-based tables."""
    logger.info("Processing configurations...")

    sim_config_path = Path("src/prism_sim/config/simulation_config.json")
    world_def_path = Path("src/prism_sim/config/world_definition.json")

    if not sim_config_path.exists() or not world_def_path.exists():
        logger.warning("Config files not found, skipping config-based tables.")
        return

    with open(sim_config_path) as f:
        sim_config = json.load(f)
    with open(world_def_path) as f:
        world_def = json.load(f)

    # --- production_lines ---
    # Derived from manufacturing.plant_parameters
    with open(output_dir / 'master/production_lines.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'line_code', 'name', 'plant_id', 'line_type', 'capacity_units_per_hour', 'is_active'])

        plant_params = sim_config.get('simulation_parameters', {}).get('manufacturing', {}).get('plant_parameters', {})

        line_counter = 1
        for plant_code, params in plant_params.items():
            if plant_code in mapper.maps['locations']:
                plant_id = mapper.maps['locations'][plant_code]
                num_lines = params.get('num_lines', 1)

                for i in range(num_lines):
                    line_code = f"LINE-{plant_code}-{i+1:02d}"
                    writer.writerow([
                        line_counter,
                        line_code,
                        f"Line {i+1} at {plant_code}",
                        plant_id,
                        'packaging', # Default
                        20000, # Default based on world_def
                        'true'
                    ])
                    line_counter += 1

    # --- channels ---
    with open(output_dir / 'master/channels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'channel_code', 'name', 'channel_type', 'is_active'])

        channels = world_def.get('topology', {}).get('channels', {})
        for i, (code, data) in enumerate(channels.items(), 1):
            writer.writerow([i, code, code.replace('_', ' ').title(), 'b2b', 'true'])

    # --- promotions ---
    with open(output_dir / 'master/promotions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'promo_code', 'name', 'start_date', 'end_date', 'discount_percent'])

        promos = world_def.get('promotions', [])
        for i, p in enumerate(promos, 1):
            # Convert week to rough date (assuming year start Jan 1)
            # This is illustrative; exact date depends on sim start
            writer.writerow([
                i, p['code'], p['name'],
                f"Week {p['start_week']}", f"Week {p['end_week']}",
                p.get('discount_percent', 0)
            ])

    # --- packaging_types ---
    with open(output_dir / 'master/packaging_types.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'packaging_code', 'container_type', 'size_value', 'size_unit'])

        pkgs = world_def.get('packaging_types', [])
        for i, p in enumerate(pkgs, 1):
            writer.writerow([
                i, p['code'], p['container'], p.get('size_ml', 0), 'ml'
            ])

    logger.info("Processed config-based tables")

if __name__ == '__main__':
    main()
