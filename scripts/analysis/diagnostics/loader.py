"""
Shared data loading, ABC classification, and node helpers.

All diagnostic modules consume a single DataBundle produced here.
Memory-safe: inventory.parquet is streamed via PyArrow row groups.
Shipments and orders are streamed with FG filtering + categorical dtypes (v0.60.0).
"""

from __future__ import annotations

import json
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
        product_volumes = dict(zip(pid_arr, qty_arr))

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

    # Downcast numerics
    for col_name in ("creation_day", "arrival_day", "day"):
        if col_name in df.columns and df[col_name].dtype in (np.float64, np.int64):
            df[col_name] = df[col_name].astype(np.int32)
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].astype(np.float32)

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"    Done: {tbl.num_rows:,} FG rows ({mem_mb:.0f} MB)")
    return df, product_volumes


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
    """All data needed by diagnostic modules."""

    products: pd.DataFrame
    shipments: pd.DataFrame
    batches: pd.DataFrame
    production_orders: pd.DataFrame
    forecasts: pd.DataFrame
    orders: pd.DataFrame
    returns: pd.DataFrame
    metrics: dict
    locations: pd.DataFrame
    links: pd.DataFrame
    abc_map: dict[str, str]
    inv_by_echelon: pd.DataFrame
    sim_days: int = 0
    data_dir: Path = field(default_factory=lambda: Path("data/output"))


def load_all_data(data_dir: Path) -> DataBundle:
    """Load all parquet + static files into a DataBundle.

    v0.65.0: Bulk Arrow read + dictionary encoding for shipments/orders.
    Inventory streamed with Arrow-native filtering and groupby.
    """
    print("Loading data...")

    products = pd.read_csv(
        data_dir / "static_world" / "products.csv",
        usecols=["id", "name", "category"],
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

    with open(data_dir / "metrics.json") as f:
        metrics = json.load(f)

    # ABC classification from product volumes accumulated during streaming
    print("\nClassifying ABC...")
    vol_df = pd.DataFrame(
        list(product_volumes.items()),
        columns=["product_id", "quantity"],
    )
    abc_map = classify_abc(vol_df, products)

    # Stream inventory
    print("\nStreaming inventory by echelon...")
    inv_by_echelon = stream_inventory_by_echelon(data_dir, abc_map)

    # Compute sim_days from shipment range
    max_day = int(shipments["creation_day"].max()) if len(shipments) > 0 else 365
    min_day = int(shipments["creation_day"].min()) if len(shipments) > 0 else 0
    sim_days = max_day - min_day + 1

    return DataBundle(
        products=products,
        shipments=shipments,
        batches=batches,
        production_orders=production_orders,
        forecasts=forecasts,
        orders=orders,
        returns=returns,
        metrics=metrics,
        locations=locations,
        links=links,
        abc_map=abc_map,
        inv_by_echelon=inv_by_echelon,
        sim_days=sim_days,
        data_dir=data_dir,
    )
