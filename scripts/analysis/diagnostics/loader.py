# ruff: noqa: E501
"""
Shared data loading, ABC classification, and node helpers.

All diagnostic modules consume a single DataBundle produced here.
Memory-safe: inventory.parquet is streamed via PyArrow row groups.
Shipments and orders are streamed with FG filtering + categorical dtypes (v0.60.0).

v0.66.0: Precomputed echelon/ABC/demand columns on shipments and orders
(built ONCE during load, eliminating redundant O(62M) .map() calls per analysis).
DOS targets derived from simulation_config.json instead of hardcoded.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# All demand-generating endpoint node prefixes (7-channel model)
# Note: "CLUB-" excluded — CLUB-DC-* are intermediate warehouses, not demand
# endpoints. Actual club stores are STORE-CLUB-* (matched by "STORE-").
DEMAND_PREFIXES = ("STORE-", "ECOM-FC-", "DTC-FC-")

# Echelon order for display
# Note: "Club" echelon is typically empty — CLUB-DC-* → "Customer DC",
# CLUB stores → STORE-CLUB-* → "Store". Kept for compatibility.
ECHELON_ORDER = ["Plant", "RDC", "Customer DC", "Store", "Club"]


# ---------------------------------------------------------------------------
# Node / product helpers
# ---------------------------------------------------------------------------

def classify_node(node_id: str) -> str:
    """Classify a node ID into its echelon tier."""
    if node_id.startswith("PLANT-"):
        return "Plant"
    if node_id.startswith("RDC-"):
        return "RDC"
    if node_id.startswith(
        ("RET-DC-", "GRO-DC-", "DIST-DC-", "ECOM-FC-", "DTC-FC-", "PHARM-DC-", "CLUB-DC-")
    ):
        return "Customer DC"
    if node_id.startswith("STORE-"):
        return "Store"
    if node_id.startswith("CLUB-"):
        return "Club"
    if node_id.startswith("SUP-"):
        return "Supplier"
    return "Other"


def is_finished_good(product_id: str) -> bool:
    """True for SKU- products (not ingredients/packaging)."""
    return product_id.startswith("SKU-")


def is_demand_endpoint(node_id: str) -> bool:
    """True for nodes that consume inventory (stores, clubs, ecom, DTC)."""
    return node_id.startswith(DEMAND_PREFIXES)


# ---------------------------------------------------------------------------
# ABC classification (per-category Pareto)
# ---------------------------------------------------------------------------

def classify_abc(
    product_volumes: pd.DataFrame,
    products: pd.DataFrame,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> dict[str, str]:
    """Per-category Pareto ABC classification from shipment volumes.

    Args:
        product_volumes: DataFrame with columns [product_id, quantity].
            Can be raw shipments or pre-aggregated per-product totals.
        products: DataFrame with columns [id, name, category].
        a_threshold: Cumulative volume fraction for A-class cutoff.
        b_threshold: Cumulative volume fraction for B-class cutoff.

    Returns dict mapping product_id -> 'A'/'B'/'C'.
    """
    fg_products = set(products[products["category"] != "INGREDIENT"]["id"])
    fg_ships = product_volumes[product_volumes["product_id"].isin(fg_products)]

    product_volume = fg_ships.groupby("product_id")["quantity"].sum().reset_index()
    product_volume.columns = ["product_id", "total_volume"]
    product_volume = product_volume.merge(
        products[["id", "category"]].rename(columns={"id": "product_id"}),
        on="product_id",
        how="left",
    )

    abc_labels: list[pd.DataFrame] = []
    for category in product_volume["category"].dropna().unique():
        cat_df = product_volume[product_volume["category"] == category].copy()
        cat_df = cat_df.sort_values("total_volume", ascending=False)
        cat_total = cat_df["total_volume"].sum()

        if cat_total == 0:
            cat_df["abc_class"] = "C"
        else:
            cum = cat_df["total_volume"].cumsum()
            cat_df["abc_class"] = "C"
            cat_df.loc[cum <= cat_total * a_threshold, "abc_class"] = "A"
            cat_df.loc[
                (cum > cat_total * a_threshold)
                & (cum <= cat_total * b_threshold),
                "abc_class",
            ] = "B"
            # First item always A
            cat_df.iloc[0, cat_df.columns.get_loc("abc_class")] = "A"

        abc_labels.append(cat_df[["product_id", "abc_class"]])

    abc_df = pd.concat(abc_labels, ignore_index=True)
    counts = abc_df["abc_class"].value_counts()
    a_n, b_n, c_n = counts.get("A", 0), counts.get("B", 0), counts.get("C", 0)
    print(f"  ABC: A={a_n}, B={b_n}, C={c_n} SKUs")
    return dict(zip(abc_df["product_id"], abc_df["abc_class"], strict=False))


# ---------------------------------------------------------------------------
# DOS Targets (config-derived, v0.66.0)
# ---------------------------------------------------------------------------

@dataclass
class DOSTargets:
    """Configured DOS targets derived from simulation_config.json.

    Eliminates hardcoded target values across diagnostic modules.
    """

    by_echelon: dict[str, dict[str, float]]  # echelon -> {A/B/C -> target DOS}
    mrp_caps: dict[str, float]  # {A/B/C -> DOS cap}


@dataclass
class SeasonalityConfig:
    """Demand seasonality parameters from simulation_config.json.

    Used to detrend diagnostic time series (stability, backpressure).
    """

    amplitude: float
    phase_shift_days: int
    cycle_days: int

    def factor(self, day: int | np.ndarray) -> float | np.ndarray:
        """Seasonal multiplier for a given day (scalar or array)."""
        return 1.0 + self.amplitude * np.sin(
            2 * math.pi * (day - self.phase_shift_days) / self.cycle_days
        )


def load_seasonality_config(config_path: Path | None = None) -> SeasonalityConfig:
    """Load seasonality parameters from simulation_config.json."""
    if config_path is None:
        config_path = (
            Path(__file__).parents[3]
            / "src"
            / "prism_sim"
            / "config"
            / "simulation_config.json"
        )

    with open(config_path) as f:
        seas = json.load(f)["simulation_parameters"]["demand"]["seasonality"]

    return SeasonalityConfig(
        amplitude=seas["amplitude"],
        phase_shift_days=int(seas["phase_shift_days"]),
        cycle_days=int(seas["cycle_days"]),
    )


def load_dos_targets(config_path: Path | None = None) -> DOSTargets:
    """Load DOS targets from simulation_config.json.

    Derives targets from the same config the simulation uses:
    - Plant: MRP horizon x ABC production buffer
    - RDC: rdc_target_dos (uniform)
    - Customer DC: dc_buffer_days x ABC multiplier (1.5/2.0/2.5)
    - Store: target_days_supply (uniform)
    - MRP caps: inventory_cap_dos_{a,b,c}
    """
    if config_path is None:
        # Locate config relative to this file (scripts/analysis/diagnostics/loader.py)
        config_path = (
            Path(__file__).parents[3]
            / "src"
            / "prism_sim"
            / "config"
            / "simulation_config.json"
        )

    with open(config_path) as f:
        config = json.load(f)["simulation_parameters"]

    replenishment = config["agents"]["replenishment"]
    mrp = config["manufacturing"]["mrp_thresholds"]
    batching = mrp["campaign_batching"]
    cal = config["calibration"]

    # Plant: MRP target = horizon x ABC buffer
    horizon = batching["production_horizon_days"]
    plant_targets = {
        "A": horizon * cal["abc_production_factors"]["a_buffer"],
        "B": horizon * cal["abc_production_factors"]["b_buffer"],
        "C": horizon * cal["abc_production_factors"]["c_production_factor"],
    }

    # RDC: uniform target
    rdc_dos = replenishment["rdc_target_dos"]
    rdc_targets = {"A": rdc_dos, "B": rdc_dos, "C": rdc_dos}

    # Customer DC: dc_buffer_days x ABC multiplier
    dc_buffer = replenishment["dc_buffer_days"]
    dc_targets = {
        "A": dc_buffer * replenishment["dc_dos_cap_mult_a"],
        "B": dc_buffer * replenishment["dc_dos_cap_mult_b"],
        "C": dc_buffer * replenishment["dc_dos_cap_mult_c"],
    }

    # Store: target_days_supply (uniform)
    store_dos = replenishment["target_days_supply"]
    store_targets = {"A": store_dos, "B": store_dos, "C": store_dos}

    # MRP DOS caps
    mrp_caps = {
        "A": float(mrp["inventory_cap_dos_a"]),
        "B": float(mrp["inventory_cap_dos_b"]),
        "C": float(mrp["inventory_cap_dos_c"]),
    }

    by_echelon = {
        "Plant": plant_targets,
        "RDC": rdc_targets,
        "Customer DC": dc_targets,
        "Store": store_targets,
        "Club": store_targets,  # Same as store (Club echelon typically empty)
    }

    return DOSTargets(by_echelon=by_echelon, mrp_caps=mrp_caps)


# ---------------------------------------------------------------------------
# Memory-safe parquet streaming (v0.60.0)
# ---------------------------------------------------------------------------

def _decode_dict_column(col: pa.Array) -> pa.Array:
    """Decode a dictionary-encoded Arrow column to plain string."""
    if pa.types.is_dictionary(col.type):
        return col.cast(pa.string())
    return col


def _load_fg_parquet(
    path: Path,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load a parquet file, keeping only finished-good rows.

    Uses bulk Arrow read + dictionary encoding for fast Categorical conversion.
    ~8x faster than row-group streaming on large files.

    Returns:
        df: DataFrame with only FG rows and categorical string columns.
        product_volumes: {product_id: total_quantity} for ABC classification.
    """
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    print(f"  Loading {path.name} ({n_rows:,} rows)...")

    # Bulk read — Arrow handles column projection + predicate pushdown
    tbl = pq.read_table(path, columns=columns)

    # FG filter in Arrow (vectorized C++)
    prod_col = _decode_dict_column(tbl.column("product_id"))
    mask = pc.starts_with(prod_col, pattern="SKU-")
    tbl = tbl.filter(mask)

    # Product volumes for ABC classification (Arrow-native groupby)
    product_volumes: dict[str, float] = {}
    if "quantity" in columns and tbl.num_rows > 0:
        vol_tbl = tbl.select(["product_id", "quantity"])
        vol_agg = vol_tbl.group_by("product_id").aggregate([("quantity", "sum")])
        pid_arr = _decode_dict_column(vol_agg.column("product_id")).to_pylist()
        qty_arr = vol_agg.column("quantity_sum").to_pylist()
        product_volumes = dict(zip(pid_arr, qty_arr, strict=False))

    # Dictionary-encode string columns → pandas Categorical (no 12GB intermediate)
    for col_name in columns:
        col_arr = tbl.column(col_name)
        if col_arr.type == pa.string() or (
            pa.types.is_dictionary(col_arr.type)
            and col_arr.type.value_type == pa.string()
        ):
            if not pa.types.is_dictionary(col_arr.type):
                idx = tbl.schema.get_field_index(col_name)
                tbl = tbl.set_column(idx, col_name, col_arr.dictionary_encode())

    # Arrow dict → pandas Categorical automatically
    df = tbl.to_pandas()

    # Downcast numerics (int16 sufficient for day values up to 32767)
    for col_name in ("creation_day", "arrival_day", "day"):
        if col_name in df.columns and df[col_name].dtype in (np.float64, np.int64):
            df[col_name] = df[col_name].astype(np.int16)
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].astype(np.float32)

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"    Done: {tbl.num_rows:,} FG rows ({mem_mb:.0f} MB)")
    return df, product_volumes


# ---------------------------------------------------------------------------
# Precomputed enrichment (v0.66.0)
# ---------------------------------------------------------------------------

def _cat_unique(series: pd.Series) -> set[str]:
    """Get unique string values from a Series (fast path for categoricals)."""
    if hasattr(series, "cat"):
        return set(str(x) for x in series.cat.categories)
    return set(str(x) for x in series.unique())


def _enrich_dataframes(
    shipments: pd.DataFrame,
    orders: pd.DataFrame,
    abc_map: dict[str, str],
) -> None:
    """Add precomputed echelon/ABC/demand columns to shipments and orders (in-place).

    Uses categorical .map() which maps only ~4200 unique categories,
    not all 62M rows individually. Total cost: O(categories) not O(rows).
    """
    t0 = time.time()
    print("  Precomputing echelon/ABC/demand columns...")

    # Build lookup dicts from unique node IDs (~4200 unique)
    all_node_ids = (
        _cat_unique(shipments["source_id"])
        | _cat_unique(shipments["target_id"])
        | _cat_unique(orders["source_id"])
        | _cat_unique(orders["target_id"])
    )
    node_ech = {nid: classify_node(nid) for nid in all_node_ids}
    node_demand = {nid: is_demand_endpoint(nid) for nid in all_node_ids}

    # Shipments: echelon, demand, ABC
    shipments["source_echelon"] = (
        shipments["source_id"].map(node_ech).astype("category")
    )
    shipments["target_echelon"] = (
        shipments["target_id"].map(node_ech).astype("category")
    )
    shipments["is_demand_endpoint"] = shipments["target_id"].map(node_demand)
    shipments["abc_class"] = (
        shipments["product_id"].map(abc_map).fillna("C").astype("category")
    )

    # Orders: echelon
    orders["source_echelon"] = orders["source_id"].map(node_ech).astype("category")
    orders["target_echelon"] = orders["target_id"].map(node_ech).astype("category")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s ({len(all_node_ids)} unique nodes)")


# ---------------------------------------------------------------------------
# Inventory streaming (memory-safe)
# ---------------------------------------------------------------------------

def stream_inventory_by_echelon(
    data_dir: Path,
    abc_map: dict[str, str],
) -> pd.DataFrame:
    """Stream inventory.parquet by row group, returning per-(day, echelon) totals.

    Builds a node→echelon dict once from the first RG's unique nodes, then
    uses Arrow FG filter + pandas categorical map per RG for fast aggregation.

    Returns DataFrame with columns: day, echelon, total, A, B, C.
    """
    inv_path = data_dir / "inventory.parquet"
    if not inv_path.exists():
        print("  WARNING: inventory.parquet not found — skipping inventory streaming")
        return pd.DataFrame(columns=["day", "echelon", "total", "A", "B", "C"])

    pf = pq.ParquetFile(inv_path)
    n_rg = pf.metadata.num_row_groups
    n_rows = pf.metadata.num_rows
    print(f"  Streaming inventory.parquet ({n_rows:,} rows, {n_rg} RGs)...")

    columns = ["day", "node_id", "product_id", "actual_inventory"]

    # Pre-build node→echelon dict from locations.csv (covers ALL nodes).
    # First RG only has ~2098 of 3933 nodes — later days have nodes absent on day 11.
    loc_path = data_dir / "static_world" / "locations.csv"
    loc_ids = pd.read_csv(loc_path, usecols=["id"])["id"]
    node_ech = {nid: classify_node(str(nid)) for nid in loc_ids}

    partial_frames: list[pd.DataFrame] = []
    rg_loaded = 0

    for rg_idx in range(n_rg):
        tbl = pf.read_row_group(rg_idx, columns=columns)
        rg_loaded += 1

        # FG filter in Arrow (vectorized C++)
        prod_col = _decode_dict_column(tbl.column("product_id"))
        tbl = tbl.filter(pc.starts_with(prod_col, "SKU-"))
        if tbl.num_rows == 0:
            continue

        # Convert to pandas with dict-encode for fast categorical map
        for col_name in ("node_id", "product_id"):
            arr = tbl.column(col_name)
            if not pa.types.is_dictionary(arr.type):
                idx = tbl.schema.get_field_index(col_name)
                tbl = tbl.set_column(idx, col_name, _decode_dict_column(arr).dictionary_encode())

        chunk = tbl.to_pandas()

        # Categorical map: only touches ~4200 unique categories, not ~1M rows
        chunk["echelon"] = chunk["node_id"].map(node_ech)
        chunk["abc"] = chunk["product_id"].map(abc_map).fillna("C")

        # GroupBy (day, echelon) with ABC pivot
        grp = chunk.groupby(["day", "echelon"], observed=True)
        totals = grp["actual_inventory"].sum().reset_index()
        totals.columns = ["day", "echelon", "total"]

        abc_grp = chunk.groupby(["day", "echelon", "abc"], observed=True)["actual_inventory"].sum().reset_index()
        abc_grp.columns = ["day", "echelon", "abc", "inv"]
        pivot = abc_grp.pivot_table(index=["day", "echelon"], columns="abc", values="inv", fill_value=0.0).reset_index()
        # Ensure all ABC columns exist
        for cls in ("A", "B", "C"):
            if cls not in pivot.columns:
                pivot[cls] = 0.0

        chunk_df = totals.merge(pivot[["day", "echelon", "A", "B", "C"]], on=["day", "echelon"], how="left")
        for cls in ("A", "B", "C"):
            if cls not in chunk_df.columns:
                chunk_df[cls] = 0.0
            chunk_df[cls] = chunk_df[cls].fillna(0.0)

        partial_frames.append(chunk_df)

        if rg_loaded % 100 == 0:
            print(f"    ... {rg_loaded}/{n_rg} row groups processed")

    if not partial_frames:
        print(f"    Done: {rg_loaded} RGs, 0 entries")
        return pd.DataFrame(columns=["day", "echelon", "total", "A", "B", "C"])

    # Combine all RG results and re-aggregate (RGs may split the same day)
    combined = pd.concat(partial_frames, ignore_index=True)
    result = (
        combined.groupby(["day", "echelon"])[["total", "A", "B", "C"]]
        .sum()
        .reset_index()
    )
    result = result.sort_values(["day", "echelon"]).reset_index(drop=True)

    print(f"    Done: {rg_loaded} RGs, {len(result)} (day, echelon) entries")
    return result


# ---------------------------------------------------------------------------
# DataBundle
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """All data needed by diagnostic modules.

    v0.72.0: Extended with cost/price maps, batch ingredients, channel map,
    distance map, cost master, and channel economics for unified diagnostics.

    v0.66.0: Shipments and orders have precomputed echelon/ABC/demand columns
    (source_echelon, target_echelon, is_demand_endpoint, abc_class).
    fg_batches pre-filtered for finished goods. DOS targets config-derived.
    """

    products: pd.DataFrame
    shipments: pd.DataFrame  # Pre-enriched: echelon, demand, ABC
    batches: pd.DataFrame  # All batches (including ingredients)
    fg_batches: pd.DataFrame  # FG batches only (product_id starts with SKU-)
    production_orders: pd.DataFrame
    forecasts: pd.DataFrame
    orders: pd.DataFrame  # Pre-enriched: source_echelon, target_echelon
    returns: pd.DataFrame
    metrics: dict
    locations: pd.DataFrame
    links: pd.DataFrame
    abc_map: dict[str, str]
    inv_by_echelon: pd.DataFrame
    dos_targets: DOSTargets
    seasonality: SeasonalityConfig
    sim_days: int = 0
    data_dir: Path = field(default_factory=lambda: Path("data/output"))

    # v0.72.0: Cost/commercial enrichment
    batch_ingredients: pd.DataFrame = field(default_factory=pd.DataFrame)
    sku_cost_map: dict[str, float] = field(default_factory=dict)
    sku_price_map: dict[str, float] = field(default_factory=dict)
    ing_cost_map: dict[str, float] = field(default_factory=dict)
    sku_cat_map: dict[str, str] = field(default_factory=dict)
    channel_map: dict[str, str] = field(default_factory=dict)
    dist_map: dict[tuple[str, str], float] = field(default_factory=dict)
    cost_master: dict = field(default_factory=dict)
    channel_econ: dict = field(default_factory=dict)


def _load_cost_master() -> dict:
    """Load cost_master.json from config directory."""
    path = (
        Path(__file__).parents[3]
        / "src" / "prism_sim" / "config" / "cost_master.json"
    )
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_world_definition() -> dict:
    """Load world_definition.json for channel economics."""
    path = (
        Path(__file__).parents[3]
        / "src" / "prism_sim" / "config" / "world_definition.json"
    )
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _pre_aggregate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregate orders if duplicates exist, else just tag with line_count.

    Groups by (day, source_id, target_id, product_id, status) + enrichment
    columns, summing quantity and counting original lines.
    Skips the expensive groupby if orders are already unique (common for DES
    simulators that create one order per link/product/day).
    """
    t0 = time.time()
    n_raw = len(orders)

    groupby_cols = ["day", "source_id", "target_id", "product_id", "status"]
    for col in ("source_echelon", "target_echelon"):
        if col in orders.columns:
            groupby_cols.append(col)

    # Quick check: sample first 100K rows to estimate compression ratio
    sample_n = min(100_000, n_raw)
    n_unique = len(orders.head(sample_n).drop_duplicates(subset=groupby_cols))
    if n_unique > sample_n * 0.8:
        # Minimal deduplication — skip expensive groupby
        orders["line_count"] = np.int8(1)
        elapsed = time.time() - t0
        print(f"  Orders already unique ({n_unique}/{sample_n} in sample), "
              f"added line_count ({elapsed:.1f}s)")
        return orders

    agg = orders.groupby(groupby_cols, observed=True).agg(
        quantity=("quantity", "sum"),
        line_count=("quantity", "size"),
    ).reset_index()

    # Preserve compact dtypes from loader
    for col_name in ("day",):
        if col_name in agg.columns:
            agg[col_name] = agg[col_name].astype(np.int16)
    if "quantity" in agg.columns:
        agg["quantity"] = agg["quantity"].astype(np.float32)
    if "line_count" in agg.columns:
        agg["line_count"] = agg["line_count"].astype(np.int32)

    elapsed = time.time() - t0
    mem_mb = agg.memory_usage(deep=True).sum() / 1e6
    print(f"  Pre-aggregated orders: {n_raw:,} -> {len(agg):,} rows "
          f"({mem_mb:.0f} MB, {elapsed:.1f}s)")
    return agg


def load_all_data(data_dir: Path) -> DataBundle:
    """Load all parquet + static files into a DataBundle.

    v0.72.0: Extended with cost/price maps, batch ingredients, channel map,
    distance map, cost master, and channel economics for unified diagnostics.

    v0.66.0: Precomputes echelon/ABC/demand columns on shipments and orders
    during load (built ONCE). DOS targets derived from simulation_config.json.
    FG batches pre-filtered.
    """
    print("Loading data...")

    products = pd.read_csv(
        data_dir / "static_world" / "products.csv",
        usecols=["id", "name", "category", "cost_per_case", "price_per_case",
                 "weight_kg"],
    )
    products["category"] = products["category"].str.replace(
        "ProductCategory.", "", regex=False
    )

    locations = pd.read_csv(data_dir / "static_world" / "locations.csv")
    links = pd.read_csv(data_dir / "static_world" / "links.csv")

    # v0.60.0: Stream shipments with FG filter + categorical dtypes
    # Only load columns used by diagnostics (drop shipment_id, weight, volume,
    # emissions which are never referenced)
    ship_columns = [
        "product_id", "source_id", "target_id",
        "creation_day", "arrival_day", "quantity",
    ]
    shipments, product_volumes = _load_fg_parquet(
        data_dir / "shipments.parquet", ship_columns
    )

    print("  Loading batches.parquet...")
    batches = pd.read_parquet(data_dir / "batches.parquet")
    print(f"    {len(batches):,} rows")

    # Pre-filter FG batches (used by most analysis functions)
    fg_mask = batches["product_id"].apply(is_finished_good)
    fg_batches = batches[fg_mask].copy()
    print(f"    {len(fg_batches):,} FG batches")

    print("  Loading production_orders.parquet...")
    production_orders = pd.read_parquet(data_dir / "production_orders.parquet")
    print(f"    {len(production_orders):,} rows")

    print("  Loading forecasts.parquet...")
    forecasts = pd.read_parquet(data_dir / "forecasts.parquet")
    print(f"    {len(forecasts):,} rows")

    # v0.60.0: Stream orders with FG filter + categorical dtypes
    order_columns = [
        "product_id", "source_id", "target_id",
        "day", "quantity", "status",
    ]
    orders, _ = _load_fg_parquet(
        data_dir / "orders.parquet", order_columns
    )

    print("  Loading returns.parquet...")
    returns = pd.read_parquet(data_dir / "returns.parquet")
    print(f"    {len(returns):,} rows")

    # v0.72.0: Batch ingredients (1.1M rows — small enough to load fully)
    bi_path = data_dir / "batch_ingredients.parquet"
    if bi_path.exists():
        print("  Loading batch_ingredients.parquet...")
        batch_ingredients = pd.read_parquet(bi_path)
        print(f"    {len(batch_ingredients):,} rows")
    else:
        batch_ingredients = pd.DataFrame()

    with open(data_dir / "metrics.json") as f:
        metrics = json.load(f)

    # v0.72.0: Build cost/price/ingredient maps from products.csv
    cost_master = _load_cost_master()
    product_costs_cfg = cost_master.get("product_costs", {})
    default_cost = product_costs_cfg.get("default", 9.0)

    sku_cost_map: dict[str, float] = {}
    sku_price_map: dict[str, float] = {}
    ing_cost_map: dict[str, float] = {}
    sku_cat_map: dict[str, str] = {}

    for _, row in products.iterrows():
        pid = row["id"]
        cat = row["category"]
        cost = row.get("cost_per_case", 0)
        price = row.get("price_per_case", 0)
        if pid.startswith("SKU-"):
            sku_cost_map[pid] = (
                float(cost) if pd.notna(cost) and cost > 0
                else product_costs_cfg.get(cat, default_cost)
            )
            sku_price_map[pid] = float(price) if pd.notna(price) and price > 0 else 0.0
            sku_cat_map[pid] = cat
        elif pd.notna(cost) and cost > 0:
            ing_cost_map[pid] = float(cost)

    # v0.72.0: Channel map from locations.csv
    channel_map: dict[str, str] = {}
    if "channel" in locations.columns:
        for _, row in locations.iterrows():
            ch = str(row.get("channel", ""))
            if ch and ch != "nan":
                channel_map[row["id"]] = ch.replace("CustomerChannel.", "")

    # v0.72.0: Distance map from links.csv
    dist_map: dict[tuple[str, str], float] = {}
    if "distance_km" in links.columns:
        for _, row in links.iterrows():
            dist_map[(row["source_id"], row["target_id"])] = row["distance_km"]

    # v0.72.0: Channel economics from world_definition.json
    world_def = _load_world_definition()
    channel_econ = world_def.get("channel_economics", {})

    # ABC classification from product volumes accumulated during streaming
    print("\nClassifying ABC...")
    vol_df = pd.DataFrame(
        list(product_volumes.items()),
        columns=["product_id", "quantity"],
    )
    abc_map = classify_abc(vol_df, products)

    # v0.66.0: Precompute echelon/ABC/demand columns ONCE
    print("\nEnriching data...")
    _enrich_dataframes(shipments, orders, abc_map)

    # v0.74.0: Pre-aggregate orders to reduce memory (~62M → ~5M rows)
    print("\nPre-aggregating orders...")
    orders = _pre_aggregate_orders(orders)

    # Stream inventory
    print("\nStreaming inventory by echelon...")
    inv_by_echelon = stream_inventory_by_echelon(data_dir, abc_map)

    # Compute sim_days from shipment range
    max_day = int(shipments["creation_day"].max()) if len(shipments) > 0 else 365
    min_day = int(shipments["creation_day"].min()) if len(shipments) > 0 else 0
    sim_days = max_day - min_day + 1

    # v0.66.0: Config-derived DOS targets
    dos_targets = load_dos_targets()

    # v0.69.4: Seasonality config for detrending
    seasonality = load_seasonality_config()

    return DataBundle(
        products=products,
        shipments=shipments,
        batches=batches,
        fg_batches=fg_batches,
        production_orders=production_orders,
        forecasts=forecasts,
        orders=orders,
        returns=returns,
        metrics=metrics,
        locations=locations,
        links=links,
        abc_map=abc_map,
        inv_by_echelon=inv_by_echelon,
        dos_targets=dos_targets,
        seasonality=seasonality,
        sim_days=sim_days,
        data_dir=data_dir,
        batch_ingredients=batch_ingredients,
        sku_cost_map=sku_cost_map,
        sku_price_map=sku_price_map,
        ing_cost_map=ing_cost_map,
        sku_cat_map=sku_cat_map,
        channel_map=channel_map,
        dist_map=dist_map,
        cost_master=cost_master,
        channel_econ=channel_econ,
    )
