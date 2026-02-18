"""Generate Neo4j-admin import header files.

These header CSVs define the node/relationship schema for `neo4j-admin import`.
They reference the same CSV data files used by Postgres COPY.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_neo4j_headers(output_dir: Path) -> None:
    """Write Neo4j import header files to output_dir/neo4j_headers/."""
    hdr_dir = output_dir / "neo4j_headers"
    hdr_dir.mkdir(parents=True, exist_ok=True)

    # ── Node headers ──────────────────────────────────────────
    _write_header(hdr_dir / "suppliers_header.csv", [
        "id:ID(Supplier)", "supplier_code", "name", "city", "country",
        "lat:double", "lon:double", "tier:int", "is_active:boolean",
    ])
    _write_header(hdr_dir / "plants_header.csv", [
        "id:ID(Plant)", "plant_code", "name", "city", "country",
        "lat:double", "lon:double", "capacity_tons_per_day:double", "is_active:boolean",
    ])
    _write_header(hdr_dir / "dcs_header.csv", [
        "id:ID(DC)", "dc_code", "name", "city", "country",
        "lat:double", "lon:double", "type", "is_active:boolean",
    ])
    _write_header(hdr_dir / "stores_header.csv", [
        "id:ID(Store)", "location_code", "name", "city", "country",
        "lat:double", "lon:double", "store_format", "channel", "is_active:boolean",
    ])
    _write_header(hdr_dir / "skus_header.csv", [
        "id:ID(SKU)", "sku_code", "name", "category", "brand",
        "units_per_case:int", "weight_kg:double", "cost_per_case:double",
        "price_per_case:double", "value_segment", "supersedes_sku_id:int",
        "is_active:boolean",
    ])
    _write_header(hdr_dir / "bulk_intermediates_header.csv", [
        "id:ID(BulkIntermediate)", "bulk_code", "name", "category",
        "bom_level:int", "weight_kg:double", "cost_per_kg:double",
        "unit_of_measure", "is_active:boolean",
    ])
    _write_header(hdr_dir / "ingredients_header.csv", [
        "id:ID(Ingredient)", "ingredient_code", "name", "category",
        "subcategory", "bom_level:int", "weight_kg:double", "cost_per_kg:double",
        "unit_of_measure", "is_active:boolean",
    ])
    _write_header(hdr_dir / "batches_header.csv", [
        "id:ID(Batch)", "batch_number", "wo_id:int", "plant_id:int",
        "formula_id:int", "product_id:int", "quantity_kg:double",
        "yield_percent:double", "production_date:int", "status",
        "product_type", "bom_level:int", "transaction_sequence_id:long",
    ])
    _write_header(hdr_dir / "shipments_header.csv", [
        "id:ID(Shipment)", "shipment_number", "ship_date:int", "arrival_date:int",
        "origin_id:int", "destination_id:int", "status", "route_type",
        "freight_cost:double", "total_weight_kg:double",
        "transaction_sequence_id:long",
    ])
    _write_header(hdr_dir / "orders_header.csv", [
        "id:ID(Order)", "order_number", "day:int", "source_id:int",
        "retail_location_id:int", "status", "total_cases:int",
        "transaction_sequence_id:long",
    ])
    _write_header(hdr_dir / "gl_entries_header.csv", [
        "id:ID(GLEntry)", "transaction_sequence_id:long", "entry_date:int",
        "posting_date:int", "account_code", "debit_amount:double",
        "credit_amount:double", "reference_type", "reference_id",
        "node_id", "product_id", "description", "is_reversal:boolean",
    ])
    _write_header(hdr_dir / "ap_invoices_header.csv", [
        "id:ID(APInvoice)", "transaction_sequence_id:long", "invoice_number",
        "supplier_id:int", "gr_id:int", "invoice_date:int", "due_date:int",
        "total_amount:double", "currency", "status",
    ])
    _write_header(hdr_dir / "ar_invoices_header.csv", [
        "id:ID(ARInvoice)", "transaction_sequence_id:long", "invoice_number",
        "customer_location_id:int", "shipment_id:int", "invoice_date:int",
        "due_date:int", "total_amount:double", "currency", "channel", "status",
    ])
    _write_header(hdr_dir / "chart_of_accounts_header.csv", [
        "id:ID(Account)", "account_code", "account_name", "account_type",
        "is_active:boolean",
    ])
    # Friction tables (v0.78.0)
    _write_header(hdr_dir / "invoice_variances_header.csv", [
        "id:ID(InvoiceVariance)", "invoice_id:int", "line_number:int",
        "variance_type", "expected_value:double", "actual_value:double",
        "variance_amount:double", "resolution_status",
    ])
    _write_header(hdr_dir / "ap_payments_header.csv", [
        "id:ID(APPayment)", "transaction_sequence_id:long", "invoice_id:int",
        "payment_date:int", "amount:double", "discount_amount:double",
        "net_amount:double", "payment_method", "status",
    ])
    _write_header(hdr_dir / "ar_receipts_header.csv", [
        "id:ID(ARReceipt)", "transaction_sequence_id:long", "invoice_id:int",
        "receipt_date:int", "amount:double", "status",
    ])

    # ── Relationship headers ──────────────────────────────────
    _write_header(hdr_dir / "rel_supplies_header.csv", [
        ":START_ID(Supplier)", ":END_ID(Ingredient)", "unit_cost:double",
        "lead_time_days:int", "min_order_qty:int",
    ])
    _write_header(hdr_dir / "rel_produced_at_header.csv", [
        ":START_ID(Batch)", ":END_ID(Plant)",
    ])
    _write_header(hdr_dir / "rel_batch_consumes_header.csv", [
        ":START_ID(Batch)", ":END_ID(Ingredient)", "quantity_kg:double",
    ])
    _write_header(hdr_dir / "rel_shipped_from_header.csv", [
        ":START_ID(Shipment)", "origin_id:int",
    ])
    _write_header(hdr_dir / "rel_shipped_to_header.csv", [
        ":START_ID(Shipment)", "destination_id:int",
    ])

    logger.info("Neo4j headers: %d files", len(list(hdr_dir.glob("*.csv"))))


def _write_header(path: Path, columns: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
