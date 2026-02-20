"""DuckDB-based master table generation.

Reads static world CSVs + config files and produces normalized master CSVs
with integer PKs. Populates the IdMapper for downstream transactional use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .config import ErpConfig
from .id_mapper import IdMapper

logger = logging.getLogger(__name__)

# Node prefix → echelon classification
ECHELON_PREFIXES = [
    ("PLANT-", "plant"),
    ("RDC-", "rdc"),
    ("RET-DC-", "customer_dc"),
    ("GRO-DC-", "customer_dc"),
    ("DIST-DC-", "customer_dc"),
    ("ECOM-FC-", "customer_dc"),
    ("DTC-FC-", "customer_dc"),
    ("PHARM-DC-", "customer_dc"),
    ("CLUB-DC-", "customer_dc"),
    ("STORE-", "store"),
    ("SUP-", "supplier"),
]


def classify_node(node_id: str) -> str:
    """Classify a node ID into its echelon type."""
    for prefix, echelon in ECHELON_PREFIXES:
        if node_id.startswith(prefix):
            return echelon
    return "other"


def _classify_ingredient_subcategory(ingredient_id: str) -> str:
    """Classify ingredient by prefix into subcategory."""
    if ingredient_id.startswith("BLK-"):
        return "base_material"
    if ingredient_id.startswith("ACT-"):
        return "active_ingredient"
    if ingredient_id.startswith("PKG-"):
        return "packaging"
    return "other"


def generate_master_tables(
    db: duckdb.DuckDBPyConnection,
    static_dir: Path,
    output_dir: Path,
    mapper: IdMapper,
    cfg: ErpConfig,
) -> None:
    """Generate all master CSV tables from static world data + config."""
    master_dir = output_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)

    products_csv = static_dir / "products.csv"
    locations_csv = static_dir / "locations.csv"
    links_csv = static_dir / "links.csv"
    recipes_csv = static_dir / "recipes.csv"

    # Load CSVs into DuckDB
    db.execute(
        f"CREATE TABLE raw_products AS SELECT * FROM read_csv('{products_csv}')"
    )
    db.execute(
        f"CREATE TABLE raw_locations AS SELECT * FROM read_csv('{locations_csv}')"
    )
    db.execute(f"CREATE TABLE raw_links AS SELECT * FROM read_csv('{links_csv}')")
    db.execute(f"CREATE TABLE raw_recipes AS SELECT * FROM read_csv('{recipes_csv}')")

    # ── Locations ──────────────────────────────────────────────
    _generate_locations(db, master_dir, mapper)

    # ── Products (3-level BOM) ─────────────────────────────────
    _generate_products(db, master_dir, mapper)

    # ── Formulas & Formula Ingredients ─────────────────────────
    _generate_formulas(db, master_dir, mapper)

    # ── Config-based: production lines, channels, chart of accounts ─
    _generate_production_lines(master_dir, mapper, cfg)
    _generate_channels(master_dir, cfg)
    _generate_chart_of_accounts(master_dir, cfg)

    # ── Route segments ─────────────────────────────────────────
    _generate_route_segments(db, master_dir, mapper)

    # ── Supplier-ingredient links ──────────────────────────────
    _generate_supplier_ingredients(db, master_dir, mapper)

    logger.info("Master tables complete")


def _generate_locations(
    db: duckdb.DuckDBPyConnection,
    master_dir: Path,
    mapper: IdMapper,
) -> None:
    """Split locations into suppliers, plants, distribution_centers, retail_locations."""
    rows = db.execute(
        "SELECT id, name, type, location, lat, lon, "
        "throughput_capacity, channel, store_format, parent_account_id "
        "FROM raw_locations ORDER BY id"
    ).fetchall()

    suppliers = []
    plants = []
    dcs = []
    stores = []

    for row in rows:
        sid, name, _ntype, location, lat, lon, capacity, channel, fmt, _parent = row
        echelon = classify_node(sid)
        pk = mapper.get("locations", sid)

        city, country = _parse_location(location)

        if echelon == "supplier":
            suppliers.append((pk, sid, name, city, country, lat, lon, 1, True))
        elif echelon == "plant":
            # DuckDB read_csv_auto can misinterpret 'inf' as date(9999,12,31)
            cap = capacity
            try:
                cap = float(cap)
                if cap != cap or cap == float("inf"):  # NaN or inf
                    cap = None
            except (TypeError, ValueError):
                cap = None
            plants.append(
                (pk, sid, name, city, country, lat, lon, cap, True)
            )
        elif echelon in ("rdc", "customer_dc"):
            dc_type = "rdc" if echelon == "rdc" else "customer_dc"
            dcs.append((pk, sid, name, city, country, lat, lon, dc_type, True))
        elif echelon == "store":
            stores.append(
                (pk, sid, name, city, country, lat, lon, fmt or "standard",
                 channel or "", True)
            )

    _write_csv(
        master_dir / "suppliers.csv",
        ["id", "supplier_code", "name", "city", "country", "lat", "lon",
         "tier", "is_active"],
        suppliers,
    )
    _write_csv(
        master_dir / "plants.csv",
        ["id", "plant_code", "name", "city", "country", "lat", "lon",
         "capacity_tons_per_day", "is_active"],
        plants,
    )
    _write_csv(
        master_dir / "distribution_centers.csv",
        ["id", "dc_code", "name", "city", "country", "lat", "lon",
         "type", "is_active"],
        dcs,
    )
    _write_csv(
        master_dir / "retail_locations.csv",
        ["id", "location_code", "name", "city", "country", "lat", "lon",
         "store_format", "channel", "is_active"],
        stores,
    )

    logger.info(
        "Locations: %d suppliers, %d plants, %d DCs, %d stores",
        len(suppliers), len(plants), len(dcs), len(stores),
    )


def _generate_products(
    db: duckdb.DuckDBPyConnection,
    master_dir: Path,
    mapper: IdMapper,
) -> None:
    """Split products into ingredients, bulk_intermediates, and skus."""
    rows = db.execute(
        "SELECT id, name, category, weight_kg, cost_per_case, price_per_case, "
        "brand, packaging_type_id, units_per_case, value_segment, bom_level "
        "FROM raw_products ORDER BY id"
    ).fetchall()

    ingredients = []
    bulks = []
    skus = []

    for row in rows:
        (pid, name, cat, weight, cost, price, brand,
         pkg_id, upc, vseg, bom_level) = row
        pk = mapper.get("products", pid)
        cat_clean = str(cat).replace("ProductCategory.", "")

        if bom_level == 2:
            # Raw material / ingredient
            subcat = _classify_ingredient_subcategory(pid)
            ingredients.append(
                (pk, pid, name, cat_clean, subcat, bom_level,
                 weight or 1.0, cost or 0.0, "kg", True)
            )
        elif bom_level == 1:
            # Bulk intermediate
            bulks.append(
                (pk, pid, name, cat_clean, bom_level,
                 weight or 25.0, cost or 0.0, "kg", True)
            )
        else:
            # SKU (bom_level == 0)
            skus.append(
                (pk, pid, name, cat_clean, brand or "",
                 upc or 12, weight or 5.0, cost or 10.0,
                 price or 0.0, vseg or "", True)
            )

    _write_csv(
        master_dir / "ingredients.csv",
        ["id", "ingredient_code", "name", "category", "subcategory",
         "bom_level", "weight_kg", "cost_per_kg", "unit_of_measure", "is_active"],
        ingredients,
    )
    _write_csv(
        master_dir / "bulk_intermediates.csv",
        ["id", "bulk_code", "name", "category", "bom_level",
         "weight_kg", "cost_per_kg", "unit_of_measure", "is_active"],
        bulks,
    )
    _write_csv(
        master_dir / "skus.csv",
        ["id", "sku_code", "name", "category", "brand",
         "units_per_case", "weight_kg", "cost_per_case",
         "price_per_case", "value_segment", "is_active"],
        skus,
    )

    logger.info(
        "Products: %d ingredients, %d bulks, %d SKUs",
        len(ingredients), len(bulks), len(skus),
    )


def _generate_formulas(
    db: duckdb.DuckDBPyConnection,
    master_dir: Path,
    mapper: IdMapper,
) -> None:
    """Parse recipes.csv into formulas + formula_ingredients.

    Each recipe has a product_id (BULK-* or SKU-*) and an ingredients dict.
    BULK-* recipes: bom_level=1 (BULK → raw materials)
    SKU-* recipes: bom_level=0 (SKU → BULK intermediates + packaging)
    """
    import ast

    rows = db.execute(
        "SELECT product_id, ingredients, run_rate_cases_per_hour, "
        "changeover_time_hours FROM raw_recipes ORDER BY product_id"
    ).fetchall()

    formulas = []
    formula_lines = []
    formula_counter = 0

    for product_id, ingredients_str, run_rate, changeover in rows:
        product_pk = mapper.lookup("products", product_id)
        if product_pk is None:
            continue

        formula_counter += 1
        formula_code = f"FORM-{product_id}"
        formula_pk = mapper.get("formulas", formula_code)

        bom_level = 1 if product_id.startswith("BULK-") else 0

        formulas.append(
            (formula_pk, formula_code, f"Formula for {product_id}",
             product_pk, bom_level, 1000.0, 98.0, run_rate, changeover)
        )

        # Parse ingredients dict
        try:
            ing_dict = ast.literal_eval(ingredients_str)
        except (ValueError, SyntaxError):
            logger.warning("Cannot parse ingredients for %s", product_id)
            continue

        seq = 0
        for ing_id, qty_kg in ing_dict.items():
            ing_pk = mapper.lookup("products", ing_id)
            if ing_pk is None:
                continue
            seq += 1
            formula_lines.append((formula_pk, ing_pk, seq, round(qty_kg, 6)))

    _write_csv(
        master_dir / "formulas.csv",
        ["id", "formula_code", "name", "product_id", "bom_level",
         "batch_size_kg", "yield_percent", "run_rate_cases_per_hour",
         "changeover_time_hours"],
        formulas,
    )
    _write_csv(
        master_dir / "formula_ingredients.csv",
        ["formula_id", "ingredient_id", "sequence", "quantity_kg"],
        formula_lines,
    )
    logger.info(
        "Formulas: %d formulas, %d ingredient lines",
        len(formulas), len(formula_lines),
    )


def _generate_production_lines(
    master_dir: Path,
    mapper: IdMapper,
    cfg: ErpConfig,
) -> None:
    """Generate production_lines.csv from plant_parameters config."""
    rows = []
    line_id = 0

    for plant_code, params in sorted(cfg.plant_parameters.items()):
        plant_pk = mapper.lookup("locations", plant_code)
        if plant_pk is None:
            continue
        num_lines = params.get("num_lines", 1)
        for i in range(num_lines):
            line_id += 1
            line_code = f"LINE-{plant_code}-{i + 1:02d}"
            rows.append(
                (line_id, line_code, f"Line {i + 1} at {plant_code}",
                 plant_pk, "packaging", 20000, True)
            )

    _write_csv(
        master_dir / "production_lines.csv",
        ["id", "line_code", "name", "plant_id", "line_type",
         "capacity_units_per_hour", "is_active"],
        rows,
    )
    logger.info("Production lines: %d", len(rows))


def _generate_channels(master_dir: Path, cfg: ErpConfig) -> None:
    """Generate channels.csv from world_definition channels config."""
    rows = []
    for i, (code, _data) in enumerate(sorted(cfg.channels.items()), 1):
        name = code.replace("_", " ").title()
        rows.append((i, code, name, "b2b", True))

    _write_csv(
        master_dir / "channels.csv",
        ["id", "channel_code", "name", "channel_type", "is_active"],
        rows,
    )
    logger.info("Channels: %d", len(rows))


def _generate_chart_of_accounts(master_dir: Path, cfg: ErpConfig) -> None:
    """Generate chart_of_accounts.csv from fixed seed data."""
    rows = []
    for i, acct in enumerate(cfg.chart_of_accounts, 1):
        rows.append(
            (i, acct["account_code"], acct["account_name"],
             acct["account_type"], True)
        )

    _write_csv(
        master_dir / "chart_of_accounts.csv",
        ["id", "account_code", "account_name", "account_type", "is_active"],
        rows,
    )
    logger.info("Chart of accounts: %d", len(rows))


def _generate_route_segments(
    db: duckdb.DuckDBPyConnection,
    master_dir: Path,
    mapper: IdMapper,
) -> None:
    """Generate route_segments.csv from links.csv."""
    rows_raw = db.execute(
        "SELECT id, source_id, target_id, mode, distance_km, "
        "lead_time_days FROM raw_links ORDER BY id"
    ).fetchall()

    rows = []
    for i, (link_id, src_id, tgt_id, mode, dist, lt_days) in enumerate(rows_raw, 1):
        src_pk = mapper.lookup("locations", src_id)
        tgt_pk = mapper.lookup("locations", tgt_id)
        if src_pk is None or tgt_pk is None:
            continue
        src_type = classify_node(src_id)
        tgt_type = classify_node(tgt_id)
        transit_hours = (lt_days or 1) * 24
        rows.append(
            (i, f"SEG-{link_id}", src_type, src_pk, tgt_type, tgt_pk,
             mode or "truck", dist or 0, transit_hours)
        )

    _write_csv(
        master_dir / "route_segments.csv",
        ["id", "segment_code", "origin_type", "origin_id",
         "destination_type", "destination_id", "transport_mode",
         "distance_km", "transit_time_hours"],
        rows,
    )
    logger.info("Route segments: %d", len(rows))


def _generate_supplier_ingredients(
    db: duckdb.DuckDBPyConnection,
    master_dir: Path,
    mapper: IdMapper,
) -> None:
    """Generate supplier_ingredients.csv from Kraljic catalog or round-robin fallback.

    Prefers supplier_catalog.csv from the static world if available (Kraljic
    segmentation). Falls back to round-robin distribution otherwise.
    """
    import csv as _csv

    catalog_path = Path("data/output/static_world/supplier_catalog.csv")

    # Get all supplier / ingredient PK maps
    supplier_pk_map = {
        sim_id: pk
        for sim_id, pk in mapper.all_ids("locations").items()
        if sim_id.startswith("SUP-")
    }
    ingredient_pk_map = {
        sim_id: pk
        for sim_id, pk in mapper.all_ids("products").items()
        if sim_id.startswith(("BLK-", "ACT-", "PKG-"))
    }

    if not supplier_pk_map or not ingredient_pk_map:
        logger.warning("No suppliers or ingredients to link")
        return

    rows: list[tuple] = []
    row_id = 0

    if catalog_path.exists():
        # ── Kraljic catalog path ──
        with open(catalog_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            seen: set[tuple[str, str]] = set()
            for cat_row in reader:
                sup_sim = cat_row["supplier_id"]
                ing_sim = cat_row["ingredient_id"]
                sup_pk = supplier_pk_map.get(sup_sim)
                ing_pk = ingredient_pk_map.get(ing_sim)
                if sup_pk is None or ing_pk is None:
                    continue
                pair = (sup_sim, ing_sim)
                if pair in seen:
                    continue
                seen.add(pair)
                row_id += 1
                cost = 5.0 + (hash(ing_sim) % 100) / 10.0
                lead_time = 3 + (hash(ing_sim + sup_sim) % 12)
                rows.append((row_id, sup_pk, ing_pk, round(cost, 2), lead_time, 100))
        logger.info("Loaded supplier-ingredient links from Kraljic catalog")
    else:
        # ── Round-robin fallback ──
        supplier_ids = sorted(supplier_pk_map.items())
        ingredient_ids = sorted(ingredient_pk_map.items())
        for idx, (ing_sim, ing_pk) in enumerate(ingredient_ids):
            sup1 = supplier_ids[idx % len(supplier_ids)]
            sup2 = supplier_ids[(idx + 7) % len(supplier_ids)]
            for sup_sim, sup_pk in {sup1, sup2}:
                row_id += 1
                cost = 5.0 + (hash(ing_sim) % 100) / 10.0
                lead_time = 3 + (hash(ing_sim + sup_sim) % 12)
                rows.append((row_id, sup_pk, ing_pk, round(cost, 2), lead_time, 100))
        logger.info("No supplier catalog found — used round-robin fallback")

    _write_csv(
        master_dir / "supplier_ingredients.csv",
        ["id", "supplier_id", "ingredient_id", "unit_cost",
         "lead_time_days", "min_order_qty"],
        rows,
    )
    logger.info("Supplier-ingredient links: %d", len(rows))


# ── Helpers ────────────────────────────────────────────────────

def _parse_location(loc_str: str | None) -> tuple[str, str]:
    """Parse 'City, Country' into (city, country)."""
    if not loc_str:
        return ("Unknown", "USA")
    loc = str(loc_str)
    if "," in loc:
        parts = loc.split(",", 1)
        return (parts[0].strip(), parts[1].strip())
    return (loc.strip(), "USA")


def _write_csv(
    path: Path,
    headers: list[str],
    rows: list[tuple],
) -> None:
    """Write a CSV file with headers and rows."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
