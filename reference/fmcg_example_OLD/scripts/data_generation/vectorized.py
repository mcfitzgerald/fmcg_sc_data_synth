"""
Vectorized Generators - NumPy-based generation for high-volume FMCG tables.

Uses NumPy structured arrays and vectorized operations to generate
high-volume tables (POS sales, order lines, shipments, legs) efficiently.

Key Techniques:
- Structured arrays: Contiguous memory, direct COPY output
- Vectorized choice: rng.choice(p=weights) for Zipf/Pareto
- Vectorized dates: numpy.datetime64 arithmetic
- Physics-based constraints: Temporal and Location-bound enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Iterator

import numpy as np
from numpy.random import Generator

from .realism_monitor import StochasticMode

# Import PromoCalendar for type hints (avoid circular import with TYPE_CHECKING)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .promo_calendar import PromoCalendar


# =============================================================================
# Structured Array Dtypes
# =============================================================================

POS_SALES_DTYPE = np.dtype([
    ("id", "i8"),
    ("retail_location_id", "i8"),
    ("sku_id", "i8"),
    ("sale_date", "datetime64[D]"),
    ("quantity_eaches", "i4"),
    ("quantity_cases", "f4"),
    ("revenue", "f8"),
    ("currency", "U3"),
    ("is_promotional", "?"),
    ("promo_id", "i8"),
    ("created_at", "datetime64[s]"),
])

ORDER_LINES_DTYPE = np.dtype([
    ("order_id", "i8"),
    ("line_number", "i4"),
    ("sku_id", "i8"),
    ("quantity_cases", "i4"),
    ("unit_price", "f4"),
    ("discount_percent", "f4"),
    ("status", "U12"),
    ("created_at", "datetime64[s]"),
])

SHIPMENT_LEGS_DTYPE = np.dtype([
    ("id", "i8"),
    ("shipment_id", "i8"),
    ("route_segment_id", "i8"),
    ("leg_sequence", "i4"),
    ("transport_mode", "U10"),
    ("carrier_id", "i8"),
    ("planned_departure", "datetime64[h]"),
    ("planned_arrival", "datetime64[h]"),
    ("actual_departure", "datetime64[h]"),
    ("actual_arrival", "datetime64[h]"),
    ("delay_hours", "f4"),
    ("status", "U12"),
])

SHIPMENTS_DTYPE = np.dtype([
    ("id", "i8"),
    ("shipment_number", "U32"),
    ("shipment_type", "U20"),
    ("origin_type", "U10"),
    ("origin_id", "i8"),
    ("destination_type", "U10"),
    ("destination_id", "i8"),
    ("order_id", "i8"),
    ("carrier_id", "i8"),
    ("route_id", "i8"),
    ("ship_date", "datetime64[D]"),
    ("expected_delivery_date", "datetime64[D]"),
    ("actual_delivery_date", "datetime64[D]"),
    ("status", "U20"),
    ("total_cases", "i4"),
    ("total_weight_kg", "f4"),
    ("total_pallets", "i4"),
    ("freight_cost", "f4"),
    ("currency", "U3"),
    ("tracking_number", "U20"),
    ("created_at", "datetime64[s]"),
    ("updated_at", "datetime64[s]"),
])

SHIPMENT_LINES_DTYPE = np.dtype([
    ("shipment_id", "i8"),
    ("line_number", "i4"),
    ("sku_id", "i8"),
    ("batch_id", "i8"),
    ("quantity_cases", "i4"),
    ("quantity_eaches", "i4"),
    ("batch_fraction", "f4"),
    ("weight_kg", "f4"),
    ("lot_number", "U20"),
    ("expiry_date", "datetime64[D]"),
    ("created_at", "datetime64[s]"),
])


# =============================================================================
# Distribution Helpers
# =============================================================================

def zipf_weights(n: int, alpha: float = 1.05) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


def lumpy_demand(rng: Generator, size: int, base_mean: float = 10.0, cv: float = 0.4) -> np.ndarray:
    variance = (cv * base_mean) ** 2
    if variance <= base_mean:
        return rng.poisson(base_mean, size=size)
    r = base_mean ** 2 / (variance - base_mean)
    p = r / (r + base_mean)
    return rng.negative_binomial(r, p, size=size)


def apply_promo_effects(
    quantities: np.ndarray,
    is_promo: np.ndarray,
    promo_week_mask: np.ndarray,
    lift_multiplier: float = 2.5,
    hangover_multiplier: float = 0.7,
) -> np.ndarray:
    """
    Apply promotional lift and post-promo hangover to demand.
    """
    result = quantities.astype(np.float32).copy()
    # Promo lift
    promo_mask = is_promo & (promo_week_mask == 1)
    result[promo_mask] *= lift_multiplier
    # Hangover effect
    hangover_mask = is_promo & (promo_week_mask == 2)
    result[hangover_mask] *= hangover_multiplier
    return np.maximum(result, 1).astype(np.int32)


# =============================================================================
# Base Generator Class
# =============================================================================

@dataclass
class VectorizedGenerator:
    seed: int = 42
    rng: Generator = field(init=False)
    stochastic_mode: StochasticMode = StochasticMode.NORMAL
    _next_id: int = 1

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self._next_id = 1


# =============================================================================
# POS Sales Generator
# =============================================================================

@dataclass
class POSSalesGenerator(VectorizedGenerator):
    sku_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    location_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    sku_weights: np.ndarray | None = None
    sku_prices: dict[int, float] = field(default_factory=dict)
    promo_calendar: "PromoCalendar | None" = None
    base_year: int = 2024
    weeks_per_year: int = 52
    zipf_alpha: float = 1.05
    base_demand_mean: float = 60.0
    demand_cv: float = 0.4

    def configure(self, sku_ids, location_ids, sku_weights=None, sku_prices=None, promo_calendar=None):
        self.sku_ids = np.asarray(sku_ids, dtype=np.int64)
        self.location_ids = np.asarray(location_ids, dtype=np.int64)
        self.sku_weights = np.asarray(sku_weights, dtype=np.float64) if sku_weights is not None else zipf_weights(len(self.sku_ids), self.zipf_alpha)
        self.sku_prices = sku_prices or {}
        self.promo_calendar = promo_calendar
        return self

    def generate_batch(self, batch_size: int) -> np.ndarray:
        batch = np.zeros(batch_size, dtype=POS_SALES_DTYPE)
        batch["id"] = np.arange(self._next_id, self._next_id + batch_size)
        self._next_id += batch_size
        batch["retail_location_id"] = self.rng.choice(self.location_ids, size=batch_size)
        sku_indices = self.rng.choice(len(self.sku_ids), size=batch_size, p=self.sku_weights)
        batch["sku_id"] = self.sku_ids[sku_indices]
        weeks = self.rng.integers(1, self.weeks_per_year + 1, size=batch_size)
        base_date = np.datetime64(f"{self.base_year}-01-01", "D")
        week_starts = base_date + (weeks - 1).astype("timedelta64[W]")
        day_offsets = self.rng.integers(0, 7, size=batch_size).astype("timedelta64[D]")
        batch["sale_date"] = week_starts + day_offsets
        base_quantities = lumpy_demand(self.rng, batch_size, self.base_demand_mean, self.demand_cv)
        if self.promo_calendar is not None:
            lifts, hangovers, is_promo, promo_ids = self.promo_calendar.get_effects_vectorized(weeks, batch["sku_id"], batch["retail_location_id"])
            quantity_eaches = np.maximum((base_quantities * lifts * hangovers).astype(np.int32), 1)
            batch["is_promotional"] = is_promo
            batch["promo_id"] = promo_ids
        else:
            quantity_eaches = base_quantities.astype(np.int32)
            batch["is_promotional"] = False
            batch["promo_id"] = 0
        batch["quantity_eaches"] = quantity_eaches
        batch["quantity_cases"] = (quantity_eaches / 12.0).astype(np.float32)
        prices = np.array([self.sku_prices.get(sku, 5.99) for sku in batch["sku_id"]], dtype=np.float32)
        prices = np.where(batch["is_promotional"], prices * 0.75, prices)
        batch["revenue"] = (quantity_eaches * prices).astype(np.float64)
        batch["currency"] = "USD"
        batch["created_at"] = np.datetime64("now", "s")
        return batch

    def generate_batches(self, total_rows: int, batch_size: int = 50000) -> Iterator[np.ndarray]:
        remaining = total_rows
        while remaining > 0:
            size = min(batch_size, remaining)
            yield self.generate_batch(size)
            remaining -= size


# =============================================================================
# Order Lines Generator
# =============================================================================

@dataclass
class OrderLinesGenerator(VectorizedGenerator):
    sku_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    sku_weights: np.ndarray | None = None
    sku_prices: dict[int, float] = field(default_factory=dict)
    order_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    zipf_alpha: float = 1.05
    quantity_mean: int = 12
    quantity_cv: float = 0.8
    promo_lift_multiplier: float = 3.0

    def configure(self, sku_ids, order_ids, sku_weights=None, sku_prices=None):
        self.sku_ids = np.asarray(sku_ids, dtype=np.int64)
        self.order_ids = np.asarray(order_ids, dtype=np.int64)
        self.sku_weights = np.asarray(sku_weights, dtype=np.float64) if sku_weights is not None else zipf_weights(len(self.sku_ids), self.zipf_alpha)
        self.sku_prices = sku_prices or {}
        return self

    def generate_for_orders(self, order_ids, lines_per_order, order_statuses=None, order_is_promo=None) -> np.ndarray:
        total_lines = int(lines_per_order.sum())
        batch = np.zeros(total_lines, dtype=ORDER_LINES_DTYPE)
        batch["order_id"] = np.repeat(order_ids, lines_per_order)
        line_nums = []
        for count in lines_per_order:
            line_nums.extend(range(1, int(count) + 1))
        batch["line_number"] = np.array(line_nums, dtype=np.int32)
        sku_indices = self.rng.choice(len(self.sku_ids), size=total_lines, p=self.sku_weights)
        batch["sku_id"] = self.sku_ids[sku_indices]
        quantities = lumpy_demand(self.rng, total_lines, float(self.quantity_mean), self.quantity_cv)
        if order_is_promo is not None:
            is_promo_expanded = np.repeat(order_is_promo, lines_per_order)
            quantities = np.where(is_promo_expanded, quantities * self.promo_lift_multiplier, quantities).astype(np.int64)
        batch["quantity_cases"] = np.maximum(quantities, 1).astype(np.int32)
        batch["unit_price"] = np.array([self.sku_prices.get(int(sku), 5.99) for sku in batch["sku_id"]], dtype=np.float32)
        base_discount = self.rng.uniform(0, 0.10, size=total_lines).astype(np.float32)
        if order_is_promo is not None:
            is_promo_expanded = np.repeat(order_is_promo, lines_per_order)
            base_discount = np.where(is_promo_expanded, self.rng.uniform(0.10, 0.25, size=total_lines), base_discount)
        batch["discount_percent"] = (base_discount * 100).astype(np.float32)
        if order_statuses is not None:
            expanded_statuses = np.repeat(order_statuses, lines_per_order)
            status_map = {"pending": "open", "confirmed": "open", "allocated": "allocated", "picking": "allocated", "shipped": "shipped", "delivered": "shipped", "cancelled": "cancelled"}
            batch["status"] = np.array([status_map.get(str(s), "open") for s in expanded_statuses])
        else:
            batch["status"] = "open"
        batch["created_at"] = np.datetime64("now", "s")
        return batch


# =============================================================================
# Shipments Generator
# =============================================================================

@dataclass
class ShipmentsGenerator(VectorizedGenerator):
    carrier_rates: dict[int, float] = field(default_factory=dict)
    distances: dict[tuple[str, int, str, int], float] = field(default_factory=dict)

    def configure(self, carrier_rates=None, distances=None):
        self.carrier_rates = carrier_rates or {}
        self.distances = distances or {}
        return self

    def _calculate_freight_cost(
        self,
        distances_km: np.ndarray,
        total_pallets: np.ndarray,
        shipment_types: np.ndarray,
        is_temperature_controlled: np.ndarray,
        lead_time_days: np.ndarray,
        carrier_is_preferred: np.ndarray,
    ) -> np.ndarray:
        """
        Pallet-tier freight cost model based on industry LTL/FTL economics.

        Pricing tiers (base cost per pallet, varies by distance):
        - LTL (1-6 pallets): $150-250/pallet - carrier consolidates, premium pricing
        - Volume LTL (7-12 pallets): $80-120/pallet - partial discount
        - Partial TL (13-20 pallets): $50-80/pallet - dedicated space
        - FTL (21+ pallets): $40-60/pallet - full truck efficiency

        Distance bands (multiplier on base):
        - Local (<200km): 0.6x
        - Regional (200-500km): 0.85x
        - National (500-1000km): 1.0x (base)
        - Long-haul (>1000km): 1.4x

        Additional factors:
        - Temperature control: +60%
        - Urgency (<3 days): +40%
        - Spot carrier (not preferred): +20%
        - Channel: plant_to_dc 0.7x, dc_to_dc 0.8x, dc_to_store 1.0x, DSD 1.5x

        Industry benchmarks (sources: Flock Freight, FreightRun, YK Freight):
        - LTL: $50-200/pallet average
        - FTL 53' trailer: 26 pallets max, $1,500-2,500/load
        - Target CTS: $1.00-3.00/case (BCG benchmark)
        """
        size = len(distances_km)

        # =================================================================
        # Step 1: Base cost per pallet by shipping tier
        # =================================================================
        # Pallet thresholds based on industry standards
        is_ltl = total_pallets <= 6
        is_volume_ltl = (total_pallets > 6) & (total_pallets <= 12)
        is_partial_tl = (total_pallets > 12) & (total_pallets <= 20)
        is_ftl = total_pallets > 20

        # Base cost per pallet (midpoint of industry ranges)
        base_per_pallet = np.select(
            [is_ltl, is_volume_ltl, is_partial_tl, is_ftl],
            [200.0, 100.0, 65.0, 50.0],  # $/pallet
            default=50.0,
        )

        # =================================================================
        # Step 2: Distance modifier
        # =================================================================
        distance_mult = np.where(
            distances_km < 200,
            0.60,  # Local delivery
            np.where(
                distances_km < 500,
                0.85,  # Regional
                np.where(
                    distances_km < 1000,
                    1.00,  # National (base)
                    1.40,  # Long-haul
                ),
            ),
        )

        # =================================================================
        # Step 3: Channel modifier (shipment type)
        # =================================================================
        channel_mult = np.select(
            [
                shipment_types == "plant_to_dc",
                shipment_types == "dc_to_dc",
                shipment_types == "dc_to_store",
                shipment_types == "direct_to_store",
            ],
            [0.70, 0.80, 1.00, 1.50],  # DSD has last-mile premium
            default=1.0,
        )

        # =================================================================
        # Step 4: Service modifiers
        # =================================================================
        # Temperature control (reefer vs dry van)
        temp_mult = np.where(is_temperature_controlled, 1.60, 1.00)

        # Urgency (expedited shipping)
        urgency_mult = np.where(
            lead_time_days < 3,
            1.40,  # Rush order
            np.where(lead_time_days < 5, 1.15, 1.00),
        )

        # Carrier type (contract vs spot market)
        carrier_mult = np.where(carrier_is_preferred, 1.00, 1.20)

        # =================================================================
        # Step 5: Calculate total freight cost
        # =================================================================
        # Cost = pallets × base_per_pallet × all_multipliers
        safe_pallets = np.maximum(total_pallets, 1)

        total_freight = (
            safe_pallets
            * base_per_pallet
            * distance_mult
            * channel_mult
            * temp_mult
            * urgency_mult
            * carrier_mult
        )

        # Add random noise (±15% for market volatility)
        noise = self.rng.uniform(0.85, 1.15, size=size)
        total_freight = total_freight * noise

        # Minimum freight charge ($75 for any shipment)
        total_freight = np.maximum(total_freight, 75.0)

        return total_freight.astype(np.float32)

    def generate_batch(
        self,
        origins,
        destinations,
        weights_kg,
        total_cases,
        shipment_types,
        carrier_ids,
        route_ids,
        base_dates,
        now,
        # CTS factor arrays (optional - uses defaults if not provided)
        is_temperature_controlled: np.ndarray | None = None,
        carrier_is_preferred: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate shipment batch with pallet-tier CTS calculation.

        Args:
            origins: List of (origin_type, origin_id) tuples
            destinations: List of (dest_type, dest_id) tuples
            weights_kg: Pre-calculated total weights per shipment
            total_cases: Pre-calculated total cases per shipment (from upstream physics)
            shipment_types: Array of shipment type strings
            carrier_ids: Array of carrier IDs
            route_ids: Array of route IDs
            base_dates: Array of ship dates
            now: Current datetime
            is_temperature_controlled: Whether origin DC has temperature control
            carrier_is_preferred: Whether the carrier is a preferred carrier
        """
        size = len(weights_kg)
        batch = np.zeros(size, dtype=SHIPMENTS_DTYPE)
        batch["id"] = np.arange(self._next_id, self._next_id + size)
        self._next_id += size
        year = base_dates[0].astype(object).year if size > 0 else 2024
        batch["shipment_number"] = np.array([f"SHIP-{year}-{i:08d}" for i in batch["id"]], dtype="U32")
        batch["shipment_type"] = shipment_types
        orig_types, orig_ids = zip(*origins)
        dest_types, dest_ids = zip(*destinations)
        batch["origin_type"] = np.array(orig_types)
        batch["origin_id"] = np.array(orig_ids)
        batch["destination_type"] = np.array(dest_types)
        batch["destination_id"] = np.array(dest_ids)
        batch["carrier_id"] = carrier_ids
        batch["route_id"] = route_ids
        batch["ship_date"] = base_dates
        is_store = np.isin(shipment_types, ["dc_to_store", "direct_to_store"])
        lead_days = np.where(is_store, self.rng.integers(1, 6, size=size), self.rng.integers(2, 15, size=size))
        batch["expected_delivery_date"] = batch["ship_date"] + lead_days.astype("timedelta64[D]")
        # Status distribution: 90% delivered for mature supply chain, 3% in_transit
        # This ensures realistic inventory turns (COGS/Inventory ratio ~6-14x)
        batch["status"] = self.rng.choice(["planned", "loading", "in_transit", "at_port", "delivered", "exception"], size=size, p=[0.01, 0.01, 0.03, 0.01, 0.90, 0.04])
        is_delivered = batch["status"] == "delivered"
        # Fix OTIF: bias toward on-time delivery (70% on-time/early, 30% late 1-2 days max)
        variance = self.rng.choice([-2, -1, 0, 0, 0, 1, 2], size=size)
        batch["actual_delivery_date"] = np.where(is_delivered, batch["expected_delivery_date"] + variance.astype("timedelta64[D]"), np.datetime64("NaT"))
        # Use pre-calculated total_cases from upstream (fixes mass balance)
        batch["total_cases"] = total_cases
        batch["total_weight_kg"] = weights_kg
        batch["total_pallets"] = np.maximum(1, batch["total_cases"] // 50)

        # Pallet-tier CTS calculation
        row_distances = np.array(
            [self.distances.get((ot, oid, dt, did), 500.0) for ot, oid, dt, did in zip(orig_types, orig_ids, dest_types, dest_ids)],
            dtype=np.float32,
        )

        # Use provided CTS factors or defaults
        if is_temperature_controlled is None:
            # Default: 20% of shipments are temperature controlled
            is_temperature_controlled = self.rng.random(size) < 0.20
        if carrier_is_preferred is None:
            # Default: 60% of carriers are preferred
            carrier_is_preferred = self.rng.random(size) < 0.60

        batch["freight_cost"] = self._calculate_freight_cost(
            distances_km=row_distances,
            total_pallets=batch["total_pallets"],
            shipment_types=shipment_types,
            is_temperature_controlled=is_temperature_controlled,
            lead_time_days=lead_days,
            carrier_is_preferred=carrier_is_preferred,
        )

        batch["currency"] = "USD"
        batch["tracking_number"] = np.char.add("TRK", self.rng.integers(1000000000, 9999999999, size=size).astype(str))
        batch["created_at"] = np.datetime64(now, "s")
        batch["updated_at"] = np.datetime64(now, "s")
        return batch


# =============================================================================
# Shipment Legs Generator
# =============================================================================

@dataclass
class ShipmentLegsGenerator(VectorizedGenerator):
    shipment_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    route_segment_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    carrier_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    delay_params: dict = field(default_factory=lambda: {"truck": {"lambda": 1.6}, "ocean": {"lambda": 8.0}, "air": {"lambda": 2.0}, "rail": {"lambda": 4.0}})
    disruption_params: dict = field(default_factory=lambda: {"gamma_shape": 2.0, "delay_multiplier": 4.0})
    base_year: int = 2024

    def configure(self, shipment_ids, route_segment_ids, carrier_ids):
        self.shipment_ids = np.asarray(shipment_ids, dtype=np.int64)
        self.route_segment_ids = np.asarray(route_segment_ids, dtype=np.int64)
        self.carrier_ids = np.asarray(carrier_ids, dtype=np.int64)
        return self

    def _generate_delays(self, size, mode):
        """Generate delays with ~85-90% on-time delivery (industry OTIF standard)."""
        params = self.delay_params.get(mode, self.delay_params["truck"])
        if self.stochastic_mode == StochasticMode.DISRUPTED:
            return self.rng.gamma(shape=self.disruption_params["gamma_shape"], scale=params["lambda"] * self.disruption_params["delay_multiplier"], size=size).astype(np.float32)
        # OTIF fix: 85% on-time (zero delay), 15% delayed (Poisson-distributed)
        on_time_rate = 0.85
        is_on_time = self.rng.random(size) < on_time_rate
        delays = np.zeros(size, dtype=np.float32)
        delayed_count = (~is_on_time).sum()
        if delayed_count > 0:
            delays[~is_on_time] = self.rng.poisson(params["lambda"], size=delayed_count).astype(np.float32)
        return delays

    def generate_for_shipments(self, shipment_ids, shipment_types, ship_dates, now=None) -> np.ndarray:
        if now is None: now = datetime.now()
        now_dt = np.datetime64(now, "s")
        n_shipments = len(shipment_ids)
        legs_count = np.ones(n_shipments, dtype=np.int32)
        mask_p2d = shipment_types == "plant_to_dc"; legs_count[mask_p2d] = self.rng.integers(1, 4, size=mask_p2d.sum())
        mask_d2d = shipment_types == "dc_to_dc"; legs_count[mask_d2d] = self.rng.integers(1, 3, size=mask_d2d.sum())
        mask_dir = shipment_types == "direct_to_store"; legs_count[mask_dir] = self.rng.integers(2, 5, size=mask_dir.sum())
        total_legs = int(legs_count.sum())
        batch = np.zeros(total_legs, dtype=SHIPMENT_LEGS_DTYPE)
        batch["id"] = np.arange(self._next_id, self._next_id + total_legs)
        self._next_id += total_legs
        batch["shipment_id"] = np.repeat(shipment_ids, legs_count)
        expanded_dates = np.repeat(ship_dates, legs_count)
        sequences = []
        for c in legs_count:
            sequences.extend(range(1, c + 1))
        batch["leg_sequence"] = np.array(sequences, dtype=np.int32)
        batch["route_segment_id"] = self.rng.choice(self.route_segment_ids, size=total_legs)
        batch["carrier_id"] = self.rng.choice(self.carrier_ids, size=total_legs)
        mode_choices = self.rng.choice(["truck", "ocean", "air", "rail"], size=total_legs)
        batch["transport_mode"] = mode_choices
        batch["planned_departure"] = expanded_dates.astype("datetime64[h]") + (self.rng.integers(6, 24, size=total_legs) + (batch["leg_sequence"] - 1) * 24).astype("timedelta64[h]")
        transit_times = {"truck": (24, 72), "ocean": (240, 480), "air": (12, 48), "rail": (48, 120)}
        planned_transit = np.array([self.rng.integers(*transit_times.get(m, (24, 72))) for m in mode_choices], dtype=np.int32)
        batch["planned_arrival"] = batch["planned_departure"] + planned_transit.astype("timedelta64[h]")
        delays = np.zeros(total_legs, dtype=np.float32)
        for mode in set(mode_choices):
            mask = mode_choices == mode
            delays[mask] = self._generate_delays(mask.sum(), mode)
        batch["delay_hours"] = delays
        batch["actual_departure"] = batch["planned_departure"] + (delays * 0.3).astype("timedelta64[h]")
        batch["actual_arrival"] = batch["planned_arrival"] + delays.astype("timedelta64[h]")
        batch["status"] = np.where(delays == 0, "on_time", np.where(delays < 24, "delayed", "severely_delayed"))
        return batch


# =============================================================================
# Shipment Lines Generator
# =============================================================================

@dataclass
class ShipmentLinesGenerator(VectorizedGenerator):
    sku_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    sku_weights: np.ndarray | None = None
    sku_weight_kg: dict[int, float] = field(default_factory=dict)
    batch_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    batch_output_cases: dict[int, int] = field(default_factory=dict)
    batch_numbers: dict[int, str] = field(default_factory=dict)
    batch_expiry: dict[int, date] = field(default_factory=dict)
    batch_production_dates: dict[int, date] = field(default_factory=dict)
    origin_skus: dict[int, list[int]] = field(default_factory=dict)
    zipf_alpha: float = 0.8
    base_year: int = 2024

    def configure(self, sku_ids, batch_ids, sku_weights=None, sku_weight_kg=None, batch_output_cases=None, batch_numbers=None, batch_expiry=None, batch_production_dates=None, origin_skus=None):
        self.sku_ids = np.asarray(sku_ids, dtype=np.int64)
        self.batch_ids = np.asarray(batch_ids, dtype=np.int64)
        self.sku_weights = np.asarray(sku_weights, dtype=np.float64) if sku_weights is not None else zipf_weights(len(self.sku_ids), self.zipf_alpha)
        self.sku_weight_kg = sku_weight_kg or {}
        self.batch_output_cases = batch_output_cases or {}
        self.batch_numbers = batch_numbers or {}
        self.batch_expiry = batch_expiry or {}
        self.batch_production_dates = batch_production_dates or {}
        self.origin_skus = origin_skus or {}
        return self

    def generate_for_shipments(self, shipment_ids, shipment_origins, total_cases, ship_dates, now=None) -> np.ndarray:
        if now is None: now = datetime.now()
        now_dt = np.datetime64(now, "s")
        n_shipments = len(shipment_ids)
        lines_per_shipment = np.where(total_cases < 100, self.rng.integers(1, 4, size=n_shipments), np.where(total_cases < 500, self.rng.integers(2, 6, size=n_shipments), self.rng.integers(3, 11, size=n_shipments))).astype(np.int32)
        total_lines = int(lines_per_shipment.sum())
        batch = np.zeros(total_lines, dtype=SHIPMENT_LINES_DTYPE)
        batch["shipment_id"] = np.repeat(shipment_ids, lines_per_shipment)
        expanded_origins = np.repeat(shipment_origins, lines_per_shipment)
        expanded_ship_dates = np.repeat(ship_dates, lines_per_shipment)
        line_nums = []
        for count in lines_per_shipment:
            line_nums.extend(range(1, int(count) + 1))
        batch["line_number"] = np.array(line_nums, dtype=np.int32)
        
        # Location physics
        unique_origins = np.unique(expanded_origins)
        batch_sku_ids = np.zeros(total_lines, dtype=np.int64)
        for oid in unique_origins:
            mask = expanded_origins == oid
            valid_skus = self.origin_skus.get(oid, self.sku_ids)
            if len(valid_skus) == 0:
                valid_skus = self.sku_ids
            batch_sku_ids[mask] = self.rng.choice(valid_skus, size=np.sum(mask))
        batch["sku_id"] = batch_sku_ids

        # Temporal physics
        sorted_indices = np.argsort([self.batch_production_dates.get(bid, date(2024,1,1)) for bid in self.batch_ids])
        sorted_ids = self.batch_ids[sorted_indices]
        sorted_dates = np.array([np.datetime64(self.batch_production_dates.get(bid, date(2024,1,1)), "D") for bid in sorted_ids])
        limit_indices = np.maximum(np.searchsorted(sorted_dates, expanded_ship_dates, side='right'), 1)
        batch["batch_id"] = sorted_ids[np.clip((self.rng.random(size=total_lines) * limit_indices).astype(np.int64), 0, len(sorted_ids) - 1)]

        expanded_total = np.repeat(total_cases, lines_per_shipment)
        expanded_line_counts = np.repeat(lines_per_shipment, lines_per_shipment)
        quantities = np.maximum((expanded_total // expanded_line_counts * self.rng.uniform(0.8, 1.2, size=total_lines)).astype(np.int32), 1)
        batch["quantity_cases"] = quantities
        batch["quantity_eaches"] = quantities * 24
        batch_output = np.array([self.batch_output_cases.get(int(bid), 1000) for bid in batch["batch_id"]], dtype=np.float32)
        batch["batch_fraction"] = np.minimum(1.0, quantities / batch_output).astype(np.float32)
        batch["weight_kg"] = (quantities * np.array([self.sku_weight_kg.get(int(sku), 5.0) for sku in batch["sku_id"]], dtype=np.float32)).astype(np.float32)
        batch["lot_number"] = np.array([self.batch_numbers.get(int(bid), f"LOT-{bid:08d}")[:20] for bid in batch["batch_id"]], dtype="U20")
        batch["expiry_date"] = np.array([np.datetime64(self.batch_expiry.get(int(bid), None) or (expanded_ship_dates[i] + np.timedelta64(180, "D")), "D") for i, bid in enumerate(batch["batch_id"])], dtype="datetime64[D]")
        batch["created_at"] = now_dt
        return batch


# =============================================================================
# Utility Functions
# =============================================================================

def structured_to_dicts(array: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for record in array:
        row = {}
        for name in array.dtype.names:
            val = record[name]
            if isinstance(val, (np.integer, np.int64, np.int32)): row[name] = int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)): row[name] = float(val)
            elif isinstance(val, np.bool_): row[name] = bool(val)
            elif isinstance(val, np.datetime64):
                if np.isnat(val): row[name] = None
                else:
                    ts = (val - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
                    if "D" in str(val.dtype): row[name] = date.fromtimestamp(ts)
                    else: row[name] = datetime.fromtimestamp(ts)
            elif isinstance(val, np.str_): row[name] = str(val) if val else None
            else: row[name] = val
        rows.append(row)
    return rows


def structured_to_copy_lines(array: np.ndarray, columns: list[str]) -> list[str]:
    lines = []
    for record in array:
        values = []
        for col in columns:
            val = record[col]
            if isinstance(val, np.bool_): values.append("t" if val else "f")
            elif isinstance(val, np.datetime64): values.append("\\N" if np.isnat(val) else str(val))
            elif isinstance(val, (np.integer, np.floating)): values.append("\\N" if (isinstance(val, np.floating) and np.isnan(val)) else str(val))
            elif val is None or (isinstance(val, np.str_) and not val): values.append("\\N")
            else: values.append(str(val).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n"))
        lines.append("\t".join(values))
    return lines


# Column lists for COPY output
POS_SALES_COLUMNS = ["id", "retail_location_id", "sku_id", "sale_date", "quantity_eaches", "quantity_cases", "revenue", "currency", "is_promotional", "promo_id", "created_at"]
ORDER_LINES_COLUMNS = ["order_id", "line_number", "sku_id", "quantity_cases", "unit_price", "discount_percent", "status", "created_at"]
SHIPMENTS_COLUMNS = ["id", "shipment_number", "shipment_type", "origin_type", "origin_id", "destination_type", "destination_id", "order_id", "carrier_id", "route_id", "ship_date", "expected_delivery_date", "actual_delivery_date", "status", "total_cases", "total_weight_kg", "total_pallets", "freight_cost", "currency", "tracking_number", "created_at", "updated_at"]
SHIPMENT_LEGS_COLUMNS = ["id", "shipment_id", "route_segment_id", "leg_sequence", "transport_mode", "carrier_id", "planned_departure", "planned_arrival", "actual_departure", "actual_arrival", "delay_hours", "status"]
SHIPMENT_LINES_COLUMNS = ["shipment_id", "line_number", "sku_id", "batch_id", "quantity_cases", "quantity_eaches", "batch_fraction", "weight_kg", "lot_number", "expiry_date", "created_at"]