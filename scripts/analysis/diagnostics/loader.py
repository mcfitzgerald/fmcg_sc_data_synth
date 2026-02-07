"""
Shared data loading, ABC classification, and node helpers.

All diagnostic modules consume a single DataBundle produced here.
Memory-safe: inventory.parquet is streamed via PyArrow row groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# All demand-generating endpoint node prefixes (7-channel model)
DEMAND_PREFIXES = ("STORE-", "CLUB-", "ECOM-FC-", "DTC-FC-")

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
        ("RET-DC-", "DIST-DC-", "ECOM-FC-", "DTC-FC-", "PHARM-DC-", "CLUB-DC-")
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
    shipments: pd.DataFrame,
    products: pd.DataFrame,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> dict[str, str]:
    """Per-category Pareto ABC classification from shipment volumes.

    Returns dict mapping product_id -> 'A'/'B'/'C'.
    """
    fg_products = set(products[products["category"] != "INGREDIENT"]["id"])
    fg_ships = shipments[shipments["product_id"].isin(fg_products)]

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
# Inventory streaming (memory-safe)
# ---------------------------------------------------------------------------

def stream_inventory_by_echelon(
    data_dir: Path,
    abc_map: dict[str, str],
) -> pd.DataFrame:
    """Stream inventory.parquet by row group, returning per-(day, echelon) totals.

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
    accum: dict[tuple[int, str], dict[str, float]] = {}
    rg_loaded = 0

    for rg_idx in range(n_rg):
        chunk = pf.read_row_group(rg_idx, columns=columns).to_pandas()
        rg_loaded += 1

        for col in ("node_id", "product_id"):
            if hasattr(chunk[col], "cat"):
                chunk[col] = chunk[col].astype(str)

        fg = chunk[chunk["product_id"].apply(is_finished_good)]
        if len(fg) == 0:
            continue

        fg = fg.copy()
        fg["echelon"] = fg["node_id"].map(classify_node)
        fg["abc"] = fg["product_id"].map(abc_map)

        grp = fg.groupby(["day", "echelon"])
        totals = grp["actual_inventory"].sum()
        for (day, ech), inv_total in totals.items():
            key = (int(day), ech)
            if key not in accum:
                accum[key] = {"total": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
            accum[key]["total"] += inv_total

        for cls in ("A", "B", "C"):
            cls_fg = fg[fg["abc"] == cls]
            if len(cls_fg) == 0:
                continue
            cls_totals = cls_fg.groupby(["day", "echelon"])["actual_inventory"].sum()
            for (day, ech), inv_total in cls_totals.items():
                key = (int(day), ech)
                if key not in accum:
                    accum[key] = {"total": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
                accum[key][cls] += inv_total

        if rg_loaded % 50 == 0:
            print(f"    ... {rg_loaded}/{n_rg} row groups processed")

    print(f"    Done: {rg_loaded} RGs, {len(accum)} (day, echelon) entries")

    rows = [{"day": d, "echelon": e, **vals} for (d, e), vals in accum.items()]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["day", "echelon", "total", "A", "B", "C"])
    return df.sort_values(["day", "echelon"]).reset_index(drop=True)


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
    """Load all parquet + static files into a DataBundle."""
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

    print("  Loading shipments.parquet...")
    shipments = pd.read_parquet(data_dir / "shipments.parquet")
    print(f"    {len(shipments):,} rows")

    print("  Loading batches.parquet...")
    batches = pd.read_parquet(data_dir / "batches.parquet")
    print(f"    {len(batches):,} rows")

    print("  Loading production_orders.parquet...")
    production_orders = pd.read_parquet(data_dir / "production_orders.parquet")
    print(f"    {len(production_orders):,} rows")

    print("  Loading forecasts.parquet...")
    forecasts = pd.read_parquet(data_dir / "forecasts.parquet")
    print(f"    {len(forecasts):,} rows")

    print("  Loading orders.parquet...")
    orders = pd.read_parquet(data_dir / "orders.parquet")
    print(f"    {len(orders):,} rows")

    print("  Loading returns.parquet...")
    returns = pd.read_parquet(data_dir / "returns.parquet")
    print(f"    {len(returns):,} rows")

    with open(data_dir / "metrics.json") as f:
        metrics = json.load(f)

    # ABC classification
    print("\nClassifying ABC...")
    abc_map = classify_abc(shipments, products)

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
