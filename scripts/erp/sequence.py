"""Deterministic transaction sequencing.

Assigns globally-unique, causally-ordered transaction_sequence_id values
to all transactional rows.

Formula: seq_id = day * 10_000_000 + category * 1_000_000 + row_counter

Categories encode causal ordering within a day:
  0 = Goods Receipts (inbound arrives first)
  1 = Production (uses received materials)
  2 = Shipment Dispatches
  3 = Shipment Arrivals
  4 = POS Sales (demand)
  5 = Returns
  6 = GL Journal
  7 = Invoices
"""

from __future__ import annotations

CATEGORY_SLOTS = {
    "goods_receipt": 0,
    "production": 1,
    "shipment_dispatch": 2,
    "shipment_arrival": 3,
    "sale": 4,
    "return": 5,
    "gl_journal": 6,
    "invoice": 7,
    "friction": 8,
    "payment": 9,
}

DAY_MULTIPLIER = 10_000_000
CAT_MULTIPLIER = 1_000_000


def make_seq_id(day: int, category: str, row_counter: int) -> int:
    """Compute a deterministic transaction sequence ID."""
    slot = CATEGORY_SLOTS[category]
    return day * DAY_MULTIPLIER + slot * CAT_MULTIPLIER + row_counter
