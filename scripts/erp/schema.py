"""Single source of truth for ERP database schema.

Defines all 38 tables with columns, types, constraints, and indexes.
Used to:
  1. Generate DuckDB DDL (erp_schema_duckdb.sql) during ERP export
  2. Validate CSV headers match schema definitions
  3. Column selection for DuckDB → file export
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Column:
    """Column definition."""

    name: str
    type: str  # PostgreSQL type
    constraints: str = ""  # e.g. "NOT NULL", "UNIQUE NOT NULL", "DEFAULT true"
    comment: str = ""  # inline SQL comment


@dataclass
class ForeignKey:
    """Foreign key constraint."""

    column: str
    ref_table: str
    ref_column: str = "id"


@dataclass
class Table:
    """Table definition."""

    name: str
    columns: list[Column]
    domain: str = ""  # section label
    primary_key: list[str] | None = None  # composite PK (omit for SERIAL id)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    indexes: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        """Column names in order — matches CSV header."""
        return [c.name for c in self.columns]

    @property
    def has_serial_pk(self) -> bool:
        """True if first column is a SERIAL PRIMARY KEY."""
        return self.primary_key is None and self.columns[0].name == "id"


# ---------------------------------------------------------------------------
# Column shorthand helpers
# ---------------------------------------------------------------------------

def _id() -> Column:
    return Column("id", "SERIAL", "PRIMARY KEY")


def _code(name: str, length: int = 30) -> Column:
    return Column(name, f"VARCHAR({length})", "UNIQUE NOT NULL")


def _name(length: int = 200) -> Column:
    return Column("name", f"VARCHAR({length})", "NOT NULL")


def _varchar(name: str, length: int = 30, default: str = "") -> Column:
    c = f"DEFAULT '{default}'" if default else ""
    return Column(name, f"VARCHAR({length})", c)


def _int(name: str, not_null: bool = False, default: int | None = None) -> Column:
    parts = []
    if not_null:
        parts.append("NOT NULL")
    if default is not None:
        parts.append(f"DEFAULT {default}")
    return Column(name, "INTEGER", " ".join(parts))


def _decimal(name: str, p: int = 12, s: int = 2, not_null: bool = False,
             default: float | None = None) -> Column:
    parts = []
    if not_null:
        parts.append("NOT NULL")
    if default is not None:
        parts.append(f"DEFAULT {default}")
    return Column(name, f"DECIMAL({p},{s})", " ".join(parts))


def _bool(name: str, default: bool = True) -> Column:
    return Column(name, "BOOLEAN", f"DEFAULT {'true' if default else 'false'}")


def _bigint(name: str, not_null: bool = True) -> Column:
    return Column(name, "BIGINT", "NOT NULL" if not_null else "")


def _latlon() -> list[Column]:
    return [Column("lat", "DECIMAL(10,6)"), Column("lon", "DECIMAL(10,6)")]


def _city_country() -> list[Column]:
    return [
        Column("city", "VARCHAR(100)", "NOT NULL"),
        Column("country", "VARCHAR(100)", "NOT NULL"),
    ]


def _seq() -> Column:
    return _bigint("transaction_sequence_id")


def _status(default: str = "pending") -> Column:
    return _varchar("status", 20, default)


# ---------------------------------------------------------------------------
# Table definitions — order matters (dependencies)
# ---------------------------------------------------------------------------

TABLES: list[Table] = [
    # ── DOMAIN A: SOURCE (Procurement & Inbound) ──────────────
    Table(
        name="suppliers",
        domain="DOMAIN A: SOURCE (Procurement & Inbound)",
        columns=[
            _id(), _code("supplier_code"),
            _name(),
            Column("city", "VARCHAR(100)"),
            Column("country", "VARCHAR(100)", "NOT NULL"),
            *_latlon(),
            _int("tier", not_null=True, default=1),
            _bool("is_active"),
        ],
    ),
    Table(
        name="ingredients",
        columns=[
            _id(), _code("ingredient_code"),
            _name(),
            Column("category", "VARCHAR(50)", "NOT NULL"),
            Column("subcategory", "VARCHAR(30)", "",
                   "base_material, active_ingredient, packaging"),
            _int("bom_level", default=2),
            _decimal("weight_kg", 12, 4),
            _decimal("cost_per_kg", 12, 4),
            Column("unit_of_measure", "VARCHAR(20)", "NOT NULL DEFAULT 'kg'"),
            _bool("is_active"),
        ],
    ),
    Table(
        name="supplier_ingredients",
        columns=[
            _id(),
            _int("supplier_id", not_null=True),
            _int("ingredient_id", not_null=True),
            _decimal("unit_cost", 12, 4, not_null=True),
            _int("lead_time_days", not_null=True),
            _decimal("min_order_qty", 12, 2, not_null=True),
        ],
        foreign_keys=[
            ForeignKey("supplier_id", "suppliers"),
            ForeignKey("ingredient_id", "ingredients"),
        ],
    ),
    Table(
        name="purchase_orders",
        columns=[
            _id(),
            Column("po_number", "VARCHAR(30)", "NOT NULL"),
            _int("supplier_id"),
            _int("plant_id"),
            _int("order_date", not_null=True),
            _status("open"),
            _seq(),
        ],
        foreign_keys=[ForeignKey("supplier_id", "suppliers")],
    ),
    Table(
        name="purchase_order_lines",
        primary_key=["po_id", "line_number"],
        columns=[
            _int("po_id", not_null=True),
            _int("line_number", not_null=True),
            _int("ingredient_id", not_null=True),
            _decimal("quantity_kg", 12, 2, not_null=True),
            _decimal("unit_cost", 10, 4),
            _status("open"),
        ],
        foreign_keys=[ForeignKey("po_id", "purchase_orders")],
    ),
    Table(
        name="goods_receipts",
        columns=[
            _id(),
            _code("gr_number", 50),
            _int("shipment_id"),
            _int("plant_id"),
            _int("receipt_date", not_null=True),
            _status("received"),
            _seq(),
        ],
    ),
    Table(
        name="goods_receipt_lines",
        primary_key=["gr_id", "line_number"],
        columns=[
            _int("gr_id", not_null=True),
            _int("line_number", not_null=True),
            _int("ingredient_id", not_null=True),
            _decimal("quantity_kg", 12, 2, not_null=True),
        ],
        foreign_keys=[ForeignKey("gr_id", "goods_receipts")],
    ),

    # ── DOMAIN B: TRANSFORM (Manufacturing) ───────────────────
    Table(
        name="plants",
        domain="DOMAIN B: TRANSFORM (Manufacturing)",
        columns=[
            _id(), _code("plant_code", 20),
            _name(),
            *_city_country(),
            *_latlon(),
            _decimal("capacity_tons_per_day", 10, 2),
            _bool("is_active"),
        ],
    ),
    Table(
        name="production_lines",
        columns=[
            _id(), _code("line_code"),
            _name(),
            _int("plant_id", not_null=True),
            Column("line_type", "VARCHAR(50)", "NOT NULL"),
            _int("capacity_units_per_hour", not_null=True),
            _bool("is_active"),
        ],
        foreign_keys=[ForeignKey("plant_id", "plants")],
    ),
    Table(
        name="formulas",
        columns=[
            _id(), _code("formula_code", 50),
            _name(),
            _int("product_id"),
            Column("bom_level", "INTEGER", "NOT NULL DEFAULT 0",
                   "0=SKU->BULK, 1=BULK->RM/PREMIX, 2=PREMIX->RM"),
            _decimal("batch_size_kg", 10, 2, not_null=True),
            _decimal("yield_percent", 5, 2, default=98.00),
            _decimal("run_rate_cases_per_hour", 10, 2),
            _decimal("changeover_time_hours", 5, 2),
        ],
    ),
    Table(
        name="formula_ingredients",
        primary_key=["formula_id", "ingredient_id", "sequence"],
        columns=[
            _int("formula_id", not_null=True),
            _int("ingredient_id", not_null=True),
            _int("sequence", not_null=True),
            _decimal("quantity_kg", 10, 6, not_null=True),
        ],
        foreign_keys=[ForeignKey("formula_id", "formulas")],
    ),
    Table(
        name="work_orders",
        columns=[
            _id(), _code("wo_number"),
            _int("plant_id", not_null=True),
            _int("formula_id", not_null=True),
            _decimal("planned_quantity_kg", 12, 2, not_null=True),
            _int("planned_start_date", not_null=True),
            _int("due_date"),
            _status("planned"),
            _seq(),
        ],
    ),
    Table(
        name="batches",
        columns=[
            _id(), _code("batch_number"),
            _int("wo_id"),
            _int("plant_id"),
            _int("formula_id"),
            _int("product_id"),
            _decimal("quantity_kg", 12, 2, not_null=True),
            _decimal("yield_percent", 5, 2),
            _int("production_date", not_null=True),
            _status("pending"),
            Column("product_type", "VARCHAR(20)", "",
                   "finished_good, bulk_intermediate, or premix"),
            _int("bom_level", default=0),
            _seq(),
        ],
        foreign_keys=[ForeignKey("plant_id", "plants"),
                      ForeignKey("formula_id", "formulas")],
    ),
    Table(
        name="batch_ingredients",
        columns=[
            _id(),
            _int("batch_id", not_null=True),
            _int("ingredient_id", not_null=True),
            _decimal("quantity_kg", 12, 4, not_null=True),
        ],
        foreign_keys=[ForeignKey("batch_id", "batches")],
    ),

    # ── DOMAIN C: PRODUCT (SKU Master) ────────────────────────
    Table(
        name="bulk_intermediates",
        domain="DOMAIN C: PRODUCT (SKU Master)",
        columns=[
            _id(), _code("bulk_code", 50),
            _name(),
            _varchar("category", 50),
            Column("bom_level", "INTEGER", "NOT NULL DEFAULT 1",
                   "1=primary bulk, 2=premix sub-intermediate"),
            _decimal("weight_kg", 12, 4),
            _decimal("cost_per_kg", 12, 4),
            Column("unit_of_measure", "VARCHAR(20)", "DEFAULT 'kg'"),
            _bool("is_active"),
        ],
    ),
    Table(
        name="skus",
        columns=[
            _id(), _code("sku_code", 50),
            Column("name", "VARCHAR(300)", "NOT NULL"),
            _varchar("category", 50),
            _varchar("brand", 50),
            _int("units_per_case", default=12),
            _decimal("weight_kg", 10, 2),
            _decimal("cost_per_case", 10, 2),
            _decimal("price_per_case", 10, 2),
            _varchar("value_segment", 30),
            _bool("is_active"),
            _int("supersedes_sku_id"),
        ],
        foreign_keys=[ForeignKey("supersedes_sku_id", "skus")],
    ),

    # ── DOMAIN D: ORDER (Demand Signal) ───────────────────────
    Table(
        name="channels",
        domain="DOMAIN D: ORDER (Demand Signal)",
        columns=[
            _id(), _code("channel_code", 20),
            Column("name", "VARCHAR(100)", "NOT NULL"),
            Column("channel_type", "VARCHAR(30)", "NOT NULL"),
            _bool("is_active"),
        ],
    ),
    Table(
        name="orders",
        columns=[
            _id(), _code("order_number", 50),
            _int("day", not_null=True),
            _int("source_id"),
            _int("retail_location_id"),
            _status("pending"),
            _int("total_cases"),
            _seq(),
        ],
    ),
    Table(
        name="order_lines",
        primary_key=["order_id", "line_number"],
        columns=[
            _int("order_id", not_null=True),
            _int("line_number", not_null=True),
            _int("sku_id", not_null=True),
            _decimal("quantity_cases", 12, 2, not_null=True),
            _decimal("unit_price", 10, 2),
            _status("open"),
        ],
        foreign_keys=[ForeignKey("order_id", "orders")],
    ),

    # ── DOMAIN E: FULFILL (Outbound) ─────────────────────────
    Table(
        name="distribution_centers",
        domain="DOMAIN E: FULFILL (Outbound)",
        columns=[
            _id(), _code("dc_code"),
            _name(),
            *_city_country(),
            *_latlon(),
            Column("type", "VARCHAR(30)", "NOT NULL"),
            _bool("is_active"),
        ],
    ),
    Table(
        name="retail_locations",
        columns=[
            _id(), _code("location_code", 50),
            _name(),
            *_city_country(),
            *_latlon(),
            _varchar("store_format", 30),
            _varchar("channel", 30),
            _bool("is_active"),
        ],
    ),
    Table(
        name="shipments",
        columns=[
            _id(), _code("shipment_number", 50),
            _int("ship_date", not_null=True),
            _int("arrival_date"),
            _int("origin_id"),
            _int("destination_id"),
            _status("planned"),
            _varchar("route_type", 30),
            _decimal("freight_cost", 12, 2),
            _decimal("total_weight_kg", 12, 2),
            _seq(),
        ],
    ),
    Table(
        name="shipment_lines",
        primary_key=["shipment_id", "line_number"],
        columns=[
            _int("shipment_id", not_null=True),
            _int("line_number", not_null=True),
            _int("sku_id", not_null=True),
            _decimal("quantity_cases", 12, 2, not_null=True),
            _decimal("weight_kg", 10, 2),
        ],
        foreign_keys=[ForeignKey("shipment_id", "shipments")],
    ),
    Table(
        name="inventory",
        columns=[
            _id(),
            _int("day", not_null=True),
            _varchar("location_type", 20),
            _int("location_id"),
            _int("sku_id", not_null=True),
            _decimal("quantity_cases", 12, 2, not_null=True, default=0),
        ],
    ),

    # ── DOMAIN E2: LOGISTICS (Transport Network) ─────────────
    Table(
        name="route_segments",
        domain="DOMAIN E2: LOGISTICS (Transport Network)",
        columns=[
            _id(), _code("segment_code", 50),
            Column("origin_type", "VARCHAR(20)", "NOT NULL"),
            _int("origin_id", not_null=True),
            Column("destination_type", "VARCHAR(20)", "NOT NULL"),
            _int("destination_id", not_null=True),
            Column("transport_mode", "VARCHAR(30)", "NOT NULL"),
            _decimal("distance_km", 10, 2),
            _decimal("transit_time_hours", 8, 2),
        ],
    ),

    # ── DOMAIN F: PLAN (Demand & Supply Planning) ────────────
    Table(
        name="demand_forecasts",
        domain="DOMAIN F: PLAN (Demand & Supply Planning)",
        columns=[
            _id(),
            Column("forecast_version", "VARCHAR(30)", "NOT NULL"),
            _int("sku_id", not_null=True),
            Column("location_type", "VARCHAR(20)", "NOT NULL"),
            _int("forecast_date", not_null=True),
            _decimal("forecast_quantity_cases", 12, 2, not_null=True),
            Column("forecast_type", "VARCHAR(30)", "NOT NULL"),
        ],
    ),

    # ── DOMAIN G: RETURN (Regenerate) ────────────────────────
    Table(
        name="returns",
        domain="DOMAIN G: RETURN (Regenerate)",
        columns=[
            _id(), _code("return_number"),
            _int("return_date", not_null=True),
            _int("source_id", not_null=True),
            _int("dc_id", not_null=True),
            _status("received"),
            _seq(),
        ],
    ),
    Table(
        name="return_lines",
        primary_key=["return_id", "line_number"],
        columns=[
            _int("return_id", not_null=True),
            _int("line_number", not_null=True),
            _int("sku_id", not_null=True),
            _decimal("quantity_cases", 12, 2, not_null=True),
            Column("condition", "VARCHAR(20)", "NOT NULL"),
        ],
        foreign_keys=[ForeignKey("return_id", "returns")],
    ),
    Table(
        name="disposition_logs",
        columns=[
            _int("return_id", not_null=True),
            _int("return_line_number", not_null=True),
            Column("disposition", "VARCHAR(20)", "NOT NULL"),
            _decimal("quantity_cases", 12, 2, not_null=True),
        ],
        foreign_keys=[ForeignKey("return_id", "returns")],
    ),

    # ── DOMAIN H: FINANCE ────────────────────────────────────
    Table(
        name="chart_of_accounts",
        domain="DOMAIN H: FINANCE",
        columns=[
            _id(),
            _code("account_code", 10),
            Column("account_name", "VARCHAR(200)", "NOT NULL"),
            Column("account_type", "VARCHAR(30)", "NOT NULL",
                   "asset, liability, revenue, expense"),
            _bool("is_active"),
        ],
    ),
    Table(
        name="gl_journal",
        columns=[
            _id(), _seq(),
            _int("entry_date", not_null=True),
            _int("posting_date", not_null=True),
            Column("account_code", "VARCHAR(10)", "NOT NULL"),
            _decimal("debit_amount", 14, 4, default=0),
            _decimal("credit_amount", 14, 4, default=0),
            Column("reference_type", "VARCHAR(30)", "NOT NULL"),
            _varchar("reference_id", 50),
            _varchar("node_id", 50),
            _varchar("product_id", 50),
            Column("description", "TEXT"),
            Column("is_reversal", "BOOLEAN", "DEFAULT false"),
        ],
    ),
    Table(
        name="ap_invoices",
        columns=[
            _id(), _seq(),
            _code("invoice_number"),
            _int("supplier_id"),
            _int("gr_id"),
            _int("invoice_date", not_null=True),
            _int("due_date", not_null=True),
            _decimal("total_amount", 14, 4, not_null=True),
            Column("currency", "VARCHAR(3)", "DEFAULT 'USD'"),
            _status("open"),
        ],
        foreign_keys=[
            ForeignKey("supplier_id", "suppliers"),
            ForeignKey("gr_id", "goods_receipts"),
        ],
    ),
    Table(
        name="ap_invoice_lines",
        primary_key=["invoice_id", "line_number"],
        columns=[
            _int("invoice_id", not_null=True),
            _int("line_number", not_null=True),
            _int("ingredient_id", not_null=True),
            _decimal("quantity_kg", 12, 4, not_null=True),
            _decimal("unit_cost", 10, 4, not_null=True),
            _decimal("line_amount", 14, 4, not_null=True),
        ],
        foreign_keys=[ForeignKey("invoice_id", "ap_invoices")],
    ),
    Table(
        name="ar_invoices",
        columns=[
            _id(), _seq(),
            _code("invoice_number"),
            _int("customer_location_id"),
            _int("shipment_id"),
            _int("invoice_date", not_null=True),
            _int("due_date", not_null=True),
            _decimal("total_amount", 14, 4, not_null=True),
            Column("currency", "VARCHAR(3)", "DEFAULT 'USD'"),
            _varchar("channel", 30),
            _status("open"),
        ],
        foreign_keys=[ForeignKey("shipment_id", "shipments")],
    ),
    Table(
        name="ar_invoice_lines",
        primary_key=["invoice_id", "line_number"],
        columns=[
            _int("invoice_id", not_null=True),
            _int("line_number", not_null=True),
            _int("sku_id", not_null=True),
            _decimal("quantity_cases", 12, 2, not_null=True),
            _decimal("unit_price", 10, 4, not_null=True),
            _decimal("line_amount", 14, 4, not_null=True),
        ],
        foreign_keys=[ForeignKey("invoice_id", "ar_invoices")],
    ),

    # ── DOMAIN H2: DATA QUALITY & VARIANCES ──────────────────
    Table(
        name="invoice_variances",
        domain="DOMAIN H2: DATA QUALITY & VARIANCES",
        columns=[
            _id(),
            _int("invoice_id", not_null=True),
            _int("line_number", not_null=True),
            Column("variance_type", "VARCHAR(10)", "NOT NULL",
                   "price or qty"),
            _decimal("expected_value", 14, 4, not_null=True),
            _decimal("actual_value", 14, 4, not_null=True),
            _decimal("variance_amount", 14, 4, not_null=True),
            _varchar("resolution_status", 20, "open"),
        ],
        foreign_keys=[ForeignKey("invoice_id", "ap_invoices")],
    ),
    Table(
        name="ap_payments",
        columns=[
            _id(), _seq(),
            _int("invoice_id", not_null=True),
            _int("payment_date", not_null=True),
            _decimal("amount", 14, 4, not_null=True),
            _decimal("discount_amount", 14, 4, default=0),
            _decimal("net_amount", 14, 4, not_null=True),
            Column("payment_method", "VARCHAR(20)", "DEFAULT 'EFT'"),
            _status("completed"),
        ],
        foreign_keys=[ForeignKey("invoice_id", "ap_invoices")],
    ),
    Table(
        name="ar_receipts",
        columns=[
            _id(), _seq(),
            _int("invoice_id", not_null=True),
            _int("receipt_date", not_null=True),
            _decimal("amount", 14, 4, not_null=True),
            _status("completed"),
        ],
        foreign_keys=[ForeignKey("invoice_id", "ar_invoices")],
    ),
]

# Table name → Table lookup
TABLE_MAP: dict[str, Table] = {t.name: t for t in TABLES}

# ---------------------------------------------------------------------------
# Index definitions (separate so tables stay clean)
# ---------------------------------------------------------------------------

INDEXES: list[tuple[str, str, list[str]]] = [
    # (index_name, table_name, columns)
    ("idx_orders_day", "orders", ["day"]),
    ("idx_orders_status", "orders", ["status"]),
    ("idx_shipments_ship_date", "shipments", ["ship_date"]),
    ("idx_shipments_route_type", "shipments", ["route_type"]),
    ("idx_shipments_status", "shipments", ["status"]),
    ("idx_inventory_day", "inventory", ["day"]),
    ("idx_inventory_location", "inventory", ["location_id"]),
    ("idx_batches_production_date", "batches", ["production_date"]),
    ("idx_batches_status", "batches", ["status"]),
    ("idx_gl_journal_entry_date", "gl_journal", ["entry_date"]),
    ("idx_gl_journal_account", "gl_journal", ["account_code"]),
    ("idx_gl_journal_seq", "gl_journal", ["transaction_sequence_id"]),
    ("idx_gl_journal_reference", "gl_journal", ["reference_id"]),
    ("idx_ap_invoices_date", "ap_invoices", ["invoice_date"]),
    ("idx_ar_invoices_date", "ar_invoices", ["invoice_date"]),
    ("idx_ar_invoices_channel", "ar_invoices", ["channel"]),
    ("idx_invoice_variances_invoice", "invoice_variances", ["invoice_id"]),
    ("idx_ap_payments_invoice", "ap_payments", ["invoice_id"]),
    ("idx_ap_payments_date", "ap_payments", ["payment_date"]),
    ("idx_ar_receipts_invoice", "ar_receipts", ["invoice_id"]),
    ("idx_ar_receipts_date", "ar_receipts", ["receipt_date"]),
    ("idx_purchase_orders_status", "purchase_orders", ["status"]),
    ("idx_goods_receipts_status", "goods_receipts", ["status"]),
    ("idx_work_orders_status", "work_orders", ["status"]),
    ("idx_returns_status", "returns", ["status"]),
]


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------

def generate_ddl() -> str:
    """Generate PostgreSQL DDL from the schema definitions."""
    lines: list[str] = [
        "-- " + "=" * 76,
        "-- Prism Consumer Goods (PCG) - ERP Schema",
        "-- " + "=" * 76,
        "-- Target: PostgreSQL",
        "-- Auto-generated by: scripts/erp/schema.py",
        "-- Load data with: bash data/output/erp/load_postgres.sh",
        "-- " + "=" * 76,
        "",
        "-- Drop all tables (reverse dependency order) for clean reload",
    ]

    # DROP TABLE statements in reverse order
    for table in reversed(TABLES):
        lines.append(f"DROP TABLE IF EXISTS {table.name} CASCADE;")
    lines.append("")

    # CREATE TABLE statements
    current_domain = ""
    for table in TABLES:
        if table.domain and table.domain != current_domain:
            current_domain = table.domain
            lines.append("-- " + "=" * 76)
            lines.append(f"-- {current_domain}")
            lines.append("-- " + "=" * 76)
            lines.append("")

        lines.append(f"CREATE TABLE {table.name} (")
        col_lines: list[str] = []

        for col in table.columns:
            parts = [f"    {col.name}", col.type]
            if table.has_serial_pk and col.name == "id":
                parts.append("PRIMARY KEY")
            elif col.constraints:
                parts.append(col.constraints)
            col_lines.append((" ".join(parts), col.comment))

        # Foreign keys
        for fk in table.foreign_keys:
            col_lines.append((
                f"    FOREIGN KEY ({fk.column}) REFERENCES {fk.ref_table}({fk.ref_column})",
                "",
            ))

        # Composite primary key
        if table.primary_key:
            col_lines.append((
                f"    PRIMARY KEY ({', '.join(table.primary_key)})",
                "",
            ))

        # Join with commas — comma goes BEFORE comment, not after
        for i, (sql, comment) in enumerate(col_lines):
            trailing = "," if i < len(col_lines) - 1 else ""
            if comment:
                lines.append(f"{sql}{trailing}  -- {comment}")
            else:
                lines.append(f"{sql}{trailing}")

        lines.append(");")
        lines.append("")

    # Indexes
    lines.append("-- " + "=" * 76)
    lines.append("-- Indexes")
    lines.append("-- " + "=" * 76)
    lines.append("")
    for idx_name, tbl_name, cols in INDEXES:
        col_str = ", ".join(cols)
        lines.append(f"CREATE INDEX {idx_name} ON {tbl_name}({col_str});")
    lines.append("")

    return "\n".join(lines)


def generate_duckdb_ddl() -> str:
    """Generate DuckDB-compatible DDL from the schema definitions."""
    lines: list[str] = [
        "-- " + "=" * 76,
        "-- Prism Consumer Goods (PCG) - ERP Schema",
        "-- " + "=" * 76,
        "-- Target: DuckDB",
        "-- Auto-generated by: scripts/erp/schema.py",
        "-- " + "=" * 76,
        "",
        "-- Drop all tables (reverse dependency order) for clean reload",
    ]

    for table in reversed(TABLES):
        lines.append(f"DROP TABLE IF EXISTS {table.name};")
    lines.append("")

    current_domain = ""
    for table in TABLES:
        if table.domain and table.domain != current_domain:
            current_domain = table.domain
            lines.append("-- " + "=" * 76)
            lines.append(f"-- {current_domain}")
            lines.append("-- " + "=" * 76)
            lines.append("")

        lines.append(generate_create_table_duckdb(table.name) + ";")
        lines.append("")

    lines.append("-- " + "=" * 76)
    lines.append("-- Indexes")
    lines.append("-- " + "=" * 76)
    lines.append("")
    for idx_name, tbl_name, cols in INDEXES:
        col_str = ", ".join(cols)
        lines.append(f"CREATE INDEX {idx_name} ON {tbl_name}({col_str});")
    lines.append("")

    return "\n".join(lines)


def _pg_to_duckdb(pg_type: str) -> str:
    """Convert a PostgreSQL type to DuckDB-compatible type."""
    import re

    upper = pg_type.upper()
    if upper == "SERIAL":
        return "INTEGER"
    if upper == "TEXT":
        return "VARCHAR"
    if re.match(r"VARCHAR\(\d+\)", upper):
        return "VARCHAR"
    return pg_type


def generate_create_table_duckdb(table_name: str, *, schema: str = "") -> str:
    """Generate a DuckDB CREATE TABLE statement for a single table.

    Includes column types, constraints (NOT NULL, DEFAULT), and primary keys.
    Foreign keys are omitted — DuckDB doesn't enforce them and they cause
    dependency-order issues during CREATE.

    Args:
        table_name: Schema table name (e.g. 'orders', 'suppliers').
        schema: Optional schema prefix (e.g. 'export_db').
    """
    table = TABLE_MAP.get(table_name)
    if table is None:
        raise KeyError(f"Unknown table: {table_name}")

    qualified = f"{schema}.{table_name}" if schema else table_name
    col_parts: list[str] = []

    for col in table.columns:
        ddb_type = _pg_to_duckdb(col.type)
        parts = [f"    {col.name}", ddb_type]
        if table.has_serial_pk and col.name == "id":
            parts.append("PRIMARY KEY")
        elif col.constraints:
            parts.append(col.constraints)
        col_parts.append(" ".join(parts))

    # Composite primary key
    if table.primary_key:
        col_parts.append(
            f"    PRIMARY KEY ({', '.join(table.primary_key)})"
        )

    body = ",\n".join(col_parts)
    return f"CREATE TABLE {qualified} (\n{body}\n)"


def get_column_names(table_name: str) -> list[str]:
    """Get ordered column names for a table (matches CSV header)."""
    table = TABLE_MAP.get(table_name)
    if table is None:
        raise KeyError(f"Unknown table: {table_name}")
    return table.column_names


def validate_csv_headers(table_name: str, csv_headers: list[str]) -> list[str]:
    """Validate CSV headers against schema. Returns list of errors (empty = OK)."""
    try:
        expected = get_column_names(table_name)
    except KeyError:
        return [f"Table '{table_name}' not in schema"]

    errors = []
    expected_set = set(expected)
    csv_set = set(csv_headers)

    missing = expected_set - csv_set
    extra = csv_set - expected_set

    if missing:
        errors.append(f"Columns in schema but not CSV: {missing}")
    if extra:
        errors.append(f"Columns in CSV but not schema: {extra}")
    if not errors and csv_headers != expected:
        errors.append(f"Column order mismatch: expected {expected}, got {csv_headers}")

    return errors
