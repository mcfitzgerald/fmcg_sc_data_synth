"""
Promo Calendar - Multi-promotion lift system for vectorized POS sales generation.

Enables all 100 promotions to affect POS demand with account-level targeting,
replacing the single Black Friday promo limitation.

Key Features:
- Pre-computed (week, account_id, sku_id) -> PromoEffect index
- Overlapping promos resolved with max-lift strategy
- Hangover effects when no follow-on promo exists
- Vectorized lookup for 500K rows in <1.5 seconds

Usage:
    from data_generation import PromoCalendar

    calendar = PromoCalendar.build(
        promotions=data["promotions"],
        promotion_skus=data["promotion_skus"],
        promotion_accounts=data["promotion_accounts"],
        retail_locations=data["retail_locations"],
    )

    lifts, hangovers, is_promo, promo_ids = calendar.get_effects_vectorized(
        weeks, sku_ids, location_ids
    )
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np


def date_to_week(d: date | str) -> int:
    """
    Convert a date to ISO week number (1-52/53).

    Args:
        d: Date object or ISO date string (YYYY-MM-DD)

    Returns:
        Week number (1-52, occasionally 53)
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return d.isocalendar()[1]


@dataclass(frozen=True, slots=True)
class PromoEffect:
    """
    Promotion effect for a (week, sku, account) combination.

    Represents either an active promo (with lift) or hangover period (with dip).
    """

    promo_id: int
    lift_multiplier: float
    hangover_multiplier: float
    discount_percent: float
    is_hangover: bool = False

    @property
    def effective_multiplier(self) -> float:
        """Get the multiplier to apply (lift or hangover)."""
        if self.is_hangover:
            return self.hangover_multiplier
        return self.lift_multiplier


@dataclass
class PromoCalendar:
    """
    Pre-computed promo calendar for O(1) vectorized lookups.

    Builds an index structure mapping (week, account_id, sku_id) to PromoEffect,
    resolving overlapping promotions by keeping the one with highest lift.
    """

    # Core index: week -> account_id -> sku_id -> PromoEffect
    _index: dict[int, dict[int, dict[int, PromoEffect]]] = field(
        default_factory=dict
    )

    # Lookup helper: retail_location_id -> retail_account_id
    _location_to_account: dict[int, int] = field(default_factory=dict)

    # Statistics for validation
    _promo_weeks: set[int] = field(default_factory=set)
    _promo_sku_ids: set[int] = field(default_factory=set)
    _promo_account_ids: set[int] = field(default_factory=set)
    _promo_count: int = 0

    @classmethod
    def build(
        cls,
        promotions: list[dict[str, Any]],
        promotion_skus: list[dict[str, Any]],
        promotion_accounts: list[dict[str, Any]],
        retail_locations: list[dict[str, Any]],
    ) -> "PromoCalendar":
        """
        Build promo calendar from promotion data.

        Args:
            promotions: List of promotion dicts with id, start_date, end_date,
                       lift_multiplier, hangover_multiplier, hangover_weeks
            promotion_skus: List of {promo_id, sku_id} link dicts
            promotion_accounts: List of {promo_id, retail_account_id} link dicts
            retail_locations: List of location dicts with id, retail_account_id

        Returns:
            Built PromoCalendar ready for vectorized lookups
        """
        # Step 1: Build helper indices
        promo_to_skus: dict[int, set[int]] = defaultdict(set)
        promo_to_accounts: dict[int, set[int]] = defaultdict(set)
        location_to_account: dict[int, int] = {}

        for ps in promotion_skus:
            promo_to_skus[ps["promo_id"]].add(ps["sku_id"])

        for pa in promotion_accounts:
            promo_to_accounts[pa["promo_id"]].add(pa["retail_account_id"])

        for loc in retail_locations:
            location_to_account[loc["id"]] = loc["retail_account_id"]

        # Step 2: Build week-indexed calendar
        # Using defaultdict for cleaner nested dict creation
        index: dict[int, dict[int, dict[int, PromoEffect]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        promo_weeks: set[int] = set()
        promo_sku_ids: set[int] = set()
        promo_account_ids: set[int] = set()

        for promo in promotions:
            promo_id = promo["id"]

            # Parse dates to weeks
            start_date = promo["start_date"]
            end_date = promo["end_date"]
            start_week = date_to_week(start_date)
            end_week = date_to_week(end_date)

            # Handle year boundary (if end_week < start_week, assume same year)
            if end_week < start_week:
                end_week = 52  # Clamp to year end

            # Handle None values explicitly (dict.get() returns None if key exists with None)
            lift = float(promo.get("lift_multiplier") or 1.5)
            hangover_mult = float(promo.get("hangover_multiplier") or 0.7)
            hangover_weeks_count = int(promo.get("hangover_weeks") or 1)
            discount = float(promo.get("discount_percent") or 15.0)

            sku_ids = promo_to_skus.get(promo_id, set())
            account_ids = promo_to_accounts.get(promo_id, set())

            if not sku_ids or not account_ids:
                continue  # Skip promos with no targeting

            # Track statistics
            promo_sku_ids.update(sku_ids)
            promo_account_ids.update(account_ids)

            # Add promo weeks (with lift)
            for week in range(start_week, end_week + 1):
                promo_weeks.add(week)
                for account_id in account_ids:
                    for sku_id in sku_ids:
                        effect = PromoEffect(
                            promo_id=promo_id,
                            lift_multiplier=lift,
                            hangover_multiplier=hangover_mult,
                            discount_percent=discount,
                            is_hangover=False,
                        )
                        _update_with_max_lift(
                            index[week][account_id], sku_id, effect
                        )

            # Add hangover weeks (with dip) - only if no active promo
            for hw in range(1, hangover_weeks_count + 1):
                hangover_week = end_week + hw
                if hangover_week > 52:
                    continue  # Don't wrap to next year

                for account_id in account_ids:
                    for sku_id in sku_ids:
                        # Only add hangover if no active promo exists for this combo
                        if sku_id not in index[hangover_week][account_id]:
                            effect = PromoEffect(
                                promo_id=promo_id,
                                lift_multiplier=1.0,
                                hangover_multiplier=hangover_mult,
                                discount_percent=0.0,
                                is_hangover=True,
                            )
                            index[hangover_week][account_id][sku_id] = effect

        # Convert defaultdicts to regular dicts for faster lookup
        final_index: dict[int, dict[int, dict[int, PromoEffect]]] = {
            week: {acct: dict(skus) for acct, skus in accounts.items()}
            for week, accounts in index.items()
        }

        return cls(
            _index=final_index,
            _location_to_account=location_to_account,
            _promo_weeks=promo_weeks,
            _promo_sku_ids=promo_sku_ids,
            _promo_account_ids=promo_account_ids,
            _promo_count=len(promotions),
        )

    def get_effect(
        self, week: int, sku_id: int, location_id: int
    ) -> PromoEffect | None:
        """
        Get promo effect for a single (week, sku, location) tuple.

        Args:
            week: Week number (1-52)
            sku_id: SKU ID
            location_id: Retail location ID (will be mapped to account)

        Returns:
            PromoEffect if promo active, None otherwise
        """
        if week not in self._index:
            return None

        account_id = self._location_to_account.get(location_id)
        if account_id is None:
            return None

        week_data = self._index[week]
        if account_id not in week_data:
            return None

        return week_data[account_id].get(sku_id)

    def get_effects_vectorized(
        self,
        weeks: np.ndarray,
        sku_ids: np.ndarray,
        location_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get promo effects for arrays of (week, sku, location) tuples.

        Optimized for batch processing of 500K+ rows.

        Args:
            weeks: (N,) int array - week numbers (1-52)
            sku_ids: (N,) int64 array - SKU IDs
            location_ids: (N,) int64 array - retail_location_ids

        Returns:
            Tuple of (lift_multipliers, hangover_multipliers, is_promotional, promo_ids)
            All arrays shape (N,)
        """
        n = len(weeks)

        # Initialize output arrays
        lifts = np.ones(n, dtype=np.float32)
        hangovers = np.ones(n, dtype=np.float32)
        is_promo = np.zeros(n, dtype=bool)
        promo_ids = np.zeros(n, dtype=np.int64)

        # Vectorized account lookup
        account_ids = np.array(
            [self._location_to_account.get(int(loc_id), 0) for loc_id in location_ids],
            dtype=np.int64,
        )

        # Get unique weeks that have promos (reduces dict lookups)
        unique_weeks = np.unique(weeks)
        active_weeks = [w for w in unique_weeks if int(w) in self._index]

        if not active_weeks:
            return lifts, hangovers, is_promo, promo_ids

        # Process by unique week (reduces dict lookups from N to ~52)
        for week in active_weeks:
            week_int = int(week)
            if week_int not in self._index:
                continue

            week_mask = weeks == week
            week_indices = np.where(week_mask)[0]

            week_accounts = account_ids[week_mask]
            week_skus = sku_ids[week_mask]

            week_data = self._index[week_int]

            # Process each row in this week
            for i, (acc, sku) in enumerate(zip(week_accounts, week_skus)):
                acc_int = int(acc)
                sku_int = int(sku)

                if acc_int not in week_data:
                    continue

                if sku_int not in week_data[acc_int]:
                    continue

                effect = week_data[acc_int][sku_int]
                idx = week_indices[i]

                if effect.is_hangover:
                    hangovers[idx] = effect.hangover_multiplier
                else:
                    lifts[idx] = effect.lift_multiplier
                    is_promo[idx] = True
                    promo_ids[idx] = effect.promo_id

        return lifts, hangovers, is_promo, promo_ids

    @property
    def stats(self) -> dict[str, int]:
        """Get calendar statistics for validation."""
        return {
            "promo_count": self._promo_count,
            "active_weeks": len(self._promo_weeks),
            "targeted_skus": len(self._promo_sku_ids),
            "targeted_accounts": len(self._promo_account_ids),
        }


def _update_with_max_lift(
    sku_dict: dict[int, PromoEffect],
    sku_id: int,
    new_effect: PromoEffect,
) -> None:
    """
    Update SKU dict with new effect, keeping max lift for overlaps.

    When multiple promotions target the same (week, account, sku),
    we keep the one with the highest lift multiplier.

    Args:
        sku_dict: Dict mapping sku_id -> PromoEffect
        sku_id: SKU ID to update
        new_effect: New promo effect to potentially insert
    """
    existing = sku_dict.get(sku_id)

    if existing is None:
        sku_dict[sku_id] = new_effect
        return

    # Keep the one with higher lift (for active promos)
    if not existing.is_hangover and not new_effect.is_hangover:
        if new_effect.lift_multiplier > existing.lift_multiplier:
            sku_dict[sku_id] = new_effect
    elif existing.is_hangover and not new_effect.is_hangover:
        # Active promo always beats hangover
        sku_dict[sku_id] = new_effect
    # Otherwise keep existing (hangover doesn't beat active, existing beats new hangover)
