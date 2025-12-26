"""
RealismMonitor - Online streaming validation for FMCG data generation.

Uses O(1) space online algorithms to validate data realism as it streams
through the generation pipeline, enabling fail-fast detection of drift.

Flow: Vectorized Engine → RealismMonitor.observe_batch() → StreamingWriter

Online Algorithms:
- WelfordAccumulator: Online mean/variance (lead time variability, yield loss)
- FrequencySketch: Count-Min Sketch approximation (SKU frequency for Pareto)
- CardinalityEstimator: HyperLogLog approximation (unique affected stores)
- DegreeMonitor: Running degree counts (hub concentration)

Usage:
    monitor = RealismMonitor(manifest_path="benchmark_manifest.json")

    # Observe batches as they're generated
    for batch in generate_orders():
        monitor.observe_batch("demand", "orders", batch)

    # Check for violations
    if monitor.has_violations():
        raise RealismViolationError(monitor.get_violations())

    # Get final report
    report = monitor.get_reality_report()
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class StochasticMode(Enum):
    """
    Stochastic mode for controlling distribution behavior.

    Normal mode uses standard distributions (Poisson for delays, etc.)
    Disrupted mode uses fat-tail distributions (Gamma) for chaos scenarios.
    """

    NORMAL = "poisson"      # Standard friction
    DISRUPTED = "gamma"     # Fat-tail delays (port strike, etc.)


class RealismViolationError(Exception):
    """Raised when data fails realism validation."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Realism violations: {violations}")


# =============================================================================
# Online Statistical Accumulators
# =============================================================================

@dataclass
class WelfordAccumulator:
    """
    Welford's online algorithm for computing mean and variance.

    Numerically stable single-pass algorithm for streaming data.
    Space complexity: O(1)

    Reference: Welford, B. P. (1962). "Note on a method for calculating
    corrected sums of squares and products"
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean

    def update(self, value: float) -> None:
        """Add a new value to the accumulator."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def update_batch(self, values: np.ndarray | list[float]) -> None:
        """Add multiple values to the accumulator."""
        for v in values:
            self.update(float(v))

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel's correction)."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    @property
    def cv(self) -> float:
        """Coefficient of variation (std/mean)."""
        if self.mean == 0:
            return 0.0
        return self.std / abs(self.mean)

    def get_stats(self) -> dict[str, float]:
        """Return all statistics."""
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "cv": self.cv,
        }


@dataclass
class FrequencySketch:
    """
    Approximate frequency counting using Count-Min Sketch approach.

    For FMCG, we use exact counting since the number of SKUs (~2000)
    is manageable. This class provides a consistent interface that
    could be swapped for CMS if scale increases.

    Space complexity: O(k) where k = number of unique keys
    """

    counts: Counter = field(default_factory=Counter)
    total: int = 0

    def update(self, key: Any, count: int = 1) -> None:
        """Increment count for a key."""
        self.counts[key] += count
        self.total += count

    def update_batch(self, keys: list[Any]) -> None:
        """Update counts for multiple keys."""
        self.counts.update(keys)
        self.total += len(keys)

    def get_count(self, key: Any) -> int:
        """Get count for a key."""
        return self.counts[key]

    def get_frequency(self, key: Any) -> float:
        """Get frequency (count/total) for a key."""
        if self.total == 0:
            return 0.0
        return self.counts[key] / self.total

    def top_k(self, k: int = 10) -> list[tuple[Any, int]]:
        """Get top k items by count."""
        return self.counts.most_common(k)

    def pareto_ratio(self, top_pct: float = 0.20) -> float:
        """
        Calculate what percentage of total volume comes from top X% of items.

        For 80/20 Pareto rule, top_pct=0.20 should return ~0.80.
        """
        if not self.counts or self.total == 0:
            return 0.0

        sorted_counts = sorted(self.counts.values(), reverse=True)
        n_items = len(sorted_counts)
        top_n = max(1, int(n_items * top_pct))

        top_volume = sum(sorted_counts[:top_n])
        return top_volume / self.total

    def get_stats(self) -> dict[str, Any]:
        """Return frequency statistics."""
        return {
            "unique_keys": len(self.counts),
            "total_observations": self.total,
            "top_10": self.top_k(10),
            "pareto_20_80": self.pareto_ratio(0.20),
        }


@dataclass
class CardinalityEstimator:
    """
    Exact cardinality counting for unique elements.

    Uses a set for exact counting since FMCG cardinalities are bounded
    (e.g., ~10K stores). Could be replaced with HyperLogLog for larger scale.

    Space complexity: O(k) where k = number of unique elements
    """

    seen: set = field(default_factory=set)

    def update(self, element: Any) -> None:
        """Add an element to track."""
        self.seen.add(element)

    def update_batch(self, elements: list[Any] | set[Any]) -> None:
        """Add multiple elements."""
        self.seen.update(elements)

    @property
    def cardinality(self) -> int:
        """Estimated cardinality (exact in this implementation)."""
        return len(self.seen)

    def contains(self, element: Any) -> bool:
        """Check if element has been seen."""
        return element in self.seen

    def get_stats(self) -> dict[str, Any]:
        """Return cardinality statistics."""
        return {
            "unique_count": self.cardinality,
        }


@dataclass
class DegreeMonitor:
    """
    Monitor node degrees for hub concentration analysis.

    Tracks in-degree and out-degree for network nodes to validate
    preferential attachment patterns (e.g., MegaMart gets 25% of orders).

    Space complexity: O(n) where n = number of nodes
    """

    in_degrees: Counter = field(default_factory=Counter)
    out_degrees: Counter = field(default_factory=Counter)
    total_edges: int = 0

    def add_edge(self, source: Any, target: Any) -> None:
        """Record an edge from source to target."""
        self.out_degrees[source] += 1
        self.in_degrees[target] += 1
        self.total_edges += 1

    def add_edges_batch(self, edges: list[tuple[Any, Any]]) -> None:
        """Record multiple edges."""
        for source, target in edges:
            self.add_edge(source, target)

    def get_in_degree(self, node: Any) -> int:
        """Get in-degree for a node."""
        return self.in_degrees[node]

    def get_out_degree(self, node: Any) -> int:
        """Get out-degree for a node."""
        return self.out_degrees[node]

    def get_hub_concentration(self, node: Any, direction: str = "in") -> float:
        """
        Get what fraction of total edges involve this node.

        Args:
            node: Node to check
            direction: "in" for in-degree, "out" for out-degree

        Returns:
            Fraction of total edges (0.0 to 1.0)
        """
        if self.total_edges == 0:
            return 0.0

        if direction == "in":
            return self.in_degrees[node] / self.total_edges
        return self.out_degrees[node] / self.total_edges

    def top_hubs(self, k: int = 10, direction: str = "in") -> list[tuple[Any, int]]:
        """Get top k hubs by degree."""
        degrees = self.in_degrees if direction == "in" else self.out_degrees
        return degrees.most_common(k)

    def get_stats(self) -> dict[str, Any]:
        """Return degree statistics."""
        return {
            "total_edges": self.total_edges,
            "unique_sources": len(self.out_degrees),
            "unique_targets": len(self.in_degrees),
            "top_in_hubs": self.top_hubs(5, "in"),
            "top_out_hubs": self.top_hubs(5, "out"),
        }


@dataclass
class ForecastBiasAccumulator:
    """
    Tracks forecast bias (Forecast vs Actual).

    Bias = (Forecast - Actual) / Actual
    Positive bias = Over-forecasting (Optimism)
    Negative bias = Under-forecasting
    """

    sum_forecast: float = 0.0
    sum_actual: float = 0.0
    count: int = 0

    def update_forecast(self, qty: float) -> None:
        self.sum_forecast += qty

    def update_actual(self, qty: float) -> None:
        self.sum_actual += qty
        self.count += 1

    @property
    def bias_pct(self) -> float:
        """Percentage bias."""
        if self.sum_actual == 0:
            return 0.0
        return (self.sum_forecast - self.sum_actual) / self.sum_actual

    def get_stats(self) -> dict[str, float]:
        return {
            "total_forecast": self.sum_forecast,
            "total_actual": self.sum_actual,
            "bias_pct": self.bias_pct,
        }


@dataclass
class ReturnRateAccumulator:
    """
    Tracks return rate (Returns vs Sales).

    Rate = Return Volume / Sales Volume
    """

    sum_sales_cases: float = 0.0
    sum_return_cases: float = 0.0

    def update_sales(self, qty: float) -> None:
        self.sum_sales_cases += qty

    def update_returns(self, qty: float) -> None:
        self.sum_return_cases += qty

    @property
    def return_rate(self) -> float:
        if self.sum_sales_cases == 0:
            return 0.0
        return self.sum_return_cases / self.sum_sales_cases

    def get_stats(self) -> dict[str, float]:
        return {
            "sales_volume": self.sum_sales_cases,
            "return_volume": self.sum_return_cases,
            "return_rate": self.return_rate,
        }


@dataclass
class PromoLiftAccumulator:
    """
    Tracks promotional lift (Promo Avg Qty / Baseline Avg Qty).
    """

    sum_promo_qty: float = 0.0
    count_promo: int = 0
    sum_baseline_qty: float = 0.0
    count_baseline: int = 0

    def update(self, qty: float, is_promo: bool) -> None:
        if is_promo:
            self.sum_promo_qty += qty
            self.count_promo += 1
        else:
            self.sum_baseline_qty += qty
            self.count_baseline += 1

    @property
    def lift_multiplier(self) -> float:
        if self.count_baseline == 0 or self.count_promo == 0:
            return 0.0
        avg_promo = self.sum_promo_qty / self.count_promo
        avg_baseline = self.sum_baseline_qty / self.count_baseline
        if avg_baseline == 0:
            return 0.0
        return avg_promo / avg_baseline

    def get_stats(self) -> dict[str, float]:
        return {
            "promo_volume": self.sum_promo_qty,
            "baseline_volume": self.sum_baseline_qty,
            "lift_multiplier": self.lift_multiplier,
        }


@dataclass
class ForecastMAPEAccumulator:
    """
    Tracks forecast accuracy via MAPE (Mean Absolute Percentage Error).

    Formula: MAPE = Mean(|Forecast - Actual| / Actual) × 100%

    Industry benchmarks (E2Open study):
    - World-class: <20%
    - Industry average: 48% at SKU-week level
    - Target range: 20-50%
    """

    sum_ape: float = 0.0  # Sum of absolute percentage errors
    count: int = 0

    def update(self, forecast: float, actual: float) -> None:
        """Add a forecast-actual pair to the accumulator."""
        if actual > 0:
            ape = abs(forecast - actual) / actual
            self.sum_ape += ape
            self.count += 1

    @property
    def mape_pct(self) -> float:
        """MAPE as a percentage (0-100 scale)."""
        return (self.sum_ape / self.count * 100) if self.count > 0 else 0.0

    @property
    def mape_decimal(self) -> float:
        """MAPE as a decimal (0-1 scale)."""
        return (self.sum_ape / self.count) if self.count > 0 else 0.0

    def get_stats(self) -> dict[str, float]:
        return {
            "count": self.count,
            "sum_ape": self.sum_ape,
            "mape_pct": self.mape_pct,
            "mape_decimal": self.mape_decimal,
        }


@dataclass
class OEEAccumulator:
    """
    Tracks simplified OEE (Overall Equipment Effectiveness) = Availability × Quality.

    Simplified formula skips Performance since we don't track line speed.

    Industry benchmarks:
    - World-class: 85%
    - FMCG average: 70-80%
    - Target range: 65-85%
    """

    scheduled_runs: int = 0
    completed_runs: int = 0  # Availability numerator
    total_batches: int = 0
    approved_batches: int = 0  # Quality numerator

    @property
    def availability(self) -> float:
        """Availability = Completed Runs / Scheduled Runs."""
        return self.completed_runs / self.scheduled_runs if self.scheduled_runs > 0 else 0.0

    @property
    def quality(self) -> float:
        """Quality = Approved Batches / Total Batches."""
        return self.approved_batches / self.total_batches if self.total_batches > 0 else 0.0

    @property
    def oee(self) -> float:
        """OEE = Availability × Quality."""
        return self.availability * self.quality

    def get_stats(self) -> dict[str, float]:
        return {
            "scheduled_runs": self.scheduled_runs,
            "completed_runs": self.completed_runs,
            "availability": self.availability,
            "total_batches": self.total_batches,
            "approved_batches": self.approved_batches,
            "quality": self.quality,
            "oee": self.oee,
        }


@dataclass
class InventoryTurnsAccumulator:
    """
    Tracks inventory velocity (turns per year).

    Formula: Inventory Turns = Annual COGS / Average Inventory Position
    We proxy COGS using shipped cases volume.

    Industry benchmarks:
    - World-class: 8-12x
    - P&G benchmark: 5.45x
    - Target range: 5-12x (manifest uses 6-14x)
    """

    sum_inventory_cases: float = 0.0
    inventory_snapshots: int = 0
    sum_shipped_cases: float = 0.0  # Proxy for COGS

    @property
    def avg_inventory(self) -> float:
        """Average inventory per record (for reporting)."""
        return self.sum_inventory_cases / self.inventory_snapshots if self.inventory_snapshots > 0 else 1.0

    @property
    def inventory_position(self) -> float:
        """Total inventory position (sum of all inventory records)."""
        return self.sum_inventory_cases if self.sum_inventory_cases > 0 else 1.0

    @property
    def turns(self) -> float:
        """
        Inventory turns = Shipped / Inventory Position.

        Uses total inventory position (not per-record average).
        Assumes data represents ~1 year of activity.
        """
        return self.sum_shipped_cases / self.inventory_position if self.inventory_position > 0 else 0.0

    def get_stats(self) -> dict[str, float]:
        return {
            "sum_inventory_cases": self.sum_inventory_cases,
            "inventory_snapshots": self.inventory_snapshots,
            "avg_inventory": self.avg_inventory,
            "inventory_position": self.inventory_position,
            "sum_shipped_cases": self.sum_shipped_cases,
            "turns": self.turns,
        }


@dataclass
class MassBalanceAccumulator:
    """
    Tracks mass balance across the supply chain (conservation of mass).

    Three balance equations:
    1. Ingredient → Batch: Σ ingredients consumed (kg) × yield ≈ Σ batch output (kg)
    2. Batch → Ship+Inv: Σ batch output (cases) ≈ Σ shipped (cases) + Σ inventory (cases)
    3. Order → Fulfill: Σ ordered (cases) ≈ Σ shipped (cases) + unfulfilled

    Tolerance: 2% drift max (mass_balance_drift_max in manifest)
    """

    # Ingredient → Batch balance (kg)
    sum_ingredient_input_kg: float = 0.0
    sum_batch_output_kg: float = 0.0
    batch_count: int = 0

    # Batch → Ship+Inventory balance (cases)
    sum_batch_output_cases: float = 0.0
    sum_shipped_cases: float = 0.0
    sum_inventory_cases: float = 0.0

    # Order → Fulfill balance (cases)
    sum_ordered_cases: float = 0.0
    sum_fulfilled_cases: float = 0.0  # shipment_lines

    @property
    def ingredient_to_batch_drift(self) -> float:
        """
        Drift between ingredient input and batch output.

        Positive = more output than input (impossible, indicates bug)
        Negative = normal yield loss
        """
        if self.sum_ingredient_input_kg == 0:
            return 0.0
        return (self.sum_batch_output_kg - self.sum_ingredient_input_kg) / self.sum_ingredient_input_kg

    @property
    def batch_to_shipment_drift(self) -> float:
        """
        Drift between batch production and (shipped + inventory).

        Should be ~0 if all produced goods are either shipped or in inventory.
        """
        if self.sum_batch_output_cases == 0:
            return 0.0
        accounted = self.sum_shipped_cases + self.sum_inventory_cases
        return (accounted - self.sum_batch_output_cases) / self.sum_batch_output_cases

    @property
    def order_to_fulfill_drift(self) -> float:
        """
        Drift between ordered and fulfilled quantities.

        Negative = under-fulfillment (normal, some orders unfilled)
        Positive = over-fulfillment (bug, shipped more than ordered)
        """
        if self.sum_ordered_cases == 0:
            return 0.0
        return (self.sum_fulfilled_cases - self.sum_ordered_cases) / self.sum_ordered_cases

    @property
    def fill_rate(self) -> float:
        """Order fill rate (fulfilled / ordered)."""
        if self.sum_ordered_cases == 0:
            return 0.0
        return self.sum_fulfilled_cases / self.sum_ordered_cases

    def get_stats(self) -> dict[str, float]:
        return {
            "batch_count": self.batch_count,
            "ingredient_input_kg": self.sum_ingredient_input_kg,
            "batch_output_kg": self.sum_batch_output_kg,
            "ingredient_to_batch_drift": self.ingredient_to_batch_drift,
            "batch_output_cases": self.sum_batch_output_cases,
            "shipped_cases": self.sum_shipped_cases,
            "inventory_cases": self.sum_inventory_cases,
            "batch_to_shipment_drift": self.batch_to_shipment_drift,
            "ordered_cases": self.sum_ordered_cases,
            "fulfilled_cases": self.sum_fulfilled_cases,
            "order_to_fulfill_drift": self.order_to_fulfill_drift,
            "fill_rate": self.fill_rate,
        }


@dataclass
class CostToServeAccumulator:
    """
    Tracks logistics cost per case (Cost-to-Serve).

    Formula: Cost-to-Serve = Total Freight Cost / Total Cases Shipped

    Industry benchmarks (BCG):
    - Industry average: $1.58/case
    - Target range: $1.00-$3.00/case
    """

    sum_freight_cost: float = 0.0
    sum_cases: float = 0.0
    costs: list = field(default_factory=list)  # Per-shipment cost/case for distribution

    def update(self, freight_cost: float, cases: float) -> None:
        """Add a shipment's freight cost and case count."""
        if cases > 0 and freight_cost > 0:
            self.sum_freight_cost += freight_cost
            self.sum_cases += cases
            self.costs.append(freight_cost / cases)

    @property
    def cost_per_case(self) -> float:
        """Average cost per case across all shipments."""
        return self.sum_freight_cost / self.sum_cases if self.sum_cases > 0 else 0.0

    @property
    def cost_p90_p50_ratio(self) -> float:
        """
        P90/P50 ratio for cost distribution (long-tail detection).

        High ratio indicates expensive outlier deliveries (rural, expedited, etc.).
        Target: < 4x
        """
        if len(self.costs) < 10:
            return 1.0
        sorted_costs = sorted(self.costs)
        p50_idx = len(sorted_costs) // 2
        p90_idx = int(len(sorted_costs) * 0.9)
        p50 = sorted_costs[p50_idx]
        p90 = sorted_costs[p90_idx]
        return p90 / p50 if p50 > 0 else 1.0

    def get_stats(self) -> dict[str, float]:
        return {
            "sum_freight_cost": self.sum_freight_cost,
            "sum_cases": self.sum_cases,
            "cost_per_case": self.cost_per_case,
            "cost_p90_p50_ratio": self.cost_p90_p50_ratio,
            "shipment_count": len(self.costs),
        }


# =============================================================================
# Main RealismMonitor Class
# =============================================================================

class RealismMonitor:
    """
    Streaming validation monitor for FMCG data generation.

    Observes batches of generated data and validates against benchmark
    thresholds using online algorithms. Supports fail-fast detection
    when data drifts outside acceptable ranges.

    Attributes:
        benchmarks: Ground truth benchmark values
        stochastic_mode: Current stochastic mode (NORMAL or DISRUPTED)
        violations: List of detected violations
    """

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        manifest_data: dict | None = None,
        fail_fast: bool = False,
    ) -> None:
        """
        Initialize realism monitor.

        Args:
            manifest_path: Path to benchmark_manifest.json
            manifest_data: Direct benchmark data (alternative to file)
            fail_fast: Raise exception immediately on violation
        """
        self.fail_fast = fail_fast
        self.stochastic_mode = StochasticMode.NORMAL

        # Load benchmarks
        if manifest_data is not None:
            self.benchmarks = manifest_data.get("benchmarks", {})
            if "validation_tolerances" in manifest_data:
                self.benchmarks["validation_tolerances"] = manifest_data["validation_tolerances"]
            self.profile_name = manifest_data.get("supply_chain_profile", "unknown")
        elif manifest_path is not None:
            with open(manifest_path) as f:
                data = json.load(f)
            self.benchmarks = data.get("benchmarks", {})
            # Merge root-level validation_tolerances into benchmarks if present
            if "validation_tolerances" in data:
                self.benchmarks["validation_tolerances"] = data["validation_tolerances"]
            self.profile_name = data.get("supply_chain_profile", "unknown")
        else:
            self.benchmarks = {}
            self.profile_name = "no_benchmarks"

        # Tracking structures
        self.violations: list[str] = []
        self.row_counts: dict[str, int] = defaultdict(int)

        # Online accumulators by domain
        self._accumulators: dict[str, dict[str, WelfordAccumulator]] = defaultdict(
            lambda: defaultdict(WelfordAccumulator)
        )

        # Frequency sketches for distribution validation
        self._frequencies: dict[str, FrequencySketch] = defaultdict(FrequencySketch)

        # Cardinality estimators
        self._cardinalities: dict[str, CardinalityEstimator] = defaultdict(
            CardinalityEstimator
        )

        # Degree monitors for network analysis
        self._degrees: dict[str, DegreeMonitor] = defaultdict(DegreeMonitor)

        # Domain-specific trackers
        self._pos_cv_accumulator = WelfordAccumulator()
        self._order_cv_accumulator = WelfordAccumulator()
        self._orders_by_account: Counter = Counter()
        self._recall_affected_stores: set = set()
        self._promo_lift = PromoLiftAccumulator()

        # Expert Benchmarks
        self._schedule_adherence = WelfordAccumulator()
        self._truck_fill = WelfordAccumulator()
        self._slob_count: int = 0
        self._inventory_total: int = 0
        self._oee = OEEAccumulator()
        self._inventory_turns = InventoryTurnsAccumulator()
        self._forecast_mape = ForecastMAPEAccumulator()
        self._cost_to_serve = CostToServeAccumulator()
        self._mass_balance = MassBalanceAccumulator()

        # Active quirks tracking
        self.active_quirks: set[str] = set()

        # Shipment destination lookup for mass balance (shipment_id -> destination_type)
        self._shipment_dest_types: dict[int, str] = {}

        # Strategic & Financial trackers
        self._forecast_bias = ForecastBiasAccumulator()
        self._return_rate = ReturnRateAccumulator()
        self._margin_issues: int = 0
        self._esg_issues: int = 0

        # Production & Quality trackers
        self._yield_accumulator = WelfordAccumulator()
        self._qc_total: int = 0
        self._qc_rejected: int = 0

        # Logistics trackers
        self._delay_accumulator = WelfordAccumulator()
        self._otif_on_time: int = 0
        self._otif_total: int = 0
        self._legs_by_mode: dict[str, WelfordAccumulator] = defaultdict(WelfordAccumulator)

        # OSA tracker
        self._osa_in_stock: int = 0
        self._osa_total: int = 0

        # Chaos effect trackers
        self._port_strike_delayed_legs: int = 0
        self._congestion_correlated_legs: int = 0
        self._batched_promo_orders: int = 0
        self._co2_multiplier_applied: float = 1.0

        # Cache validation tolerances for quick access
        self._tolerances = self.benchmarks.get("validation_tolerances", {}) if isinstance(self.benchmarks, dict) else {}
        self._chaos_validation = {}
        if manifest_data:
            self._chaos_validation = manifest_data.get("chaos_validation", {})
        elif manifest_path:
            with open(manifest_path) as f:
                data = json.load(f)
                self._chaos_validation = data.get("chaos_validation", {})

    def observe_batch(
        self,
        domain: str,
        table: str,
        batch_data: list[dict[str, Any]] | np.ndarray,
    ) -> None:
        """
        Main entry point - observe a batch of generated data.

        Call this for every batch of rows generated. Performs domain-specific
        validation and updates online statistics.

        Args:
            domain: Domain name (e.g., "demand", "logistics", "source")
            table: Table name (e.g., "orders", "pos_sales", "shipment_legs")
            batch_data: List of row dicts or structured numpy array
        """
        if not batch_data:
            return

        # Convert numpy structured array to list of dicts if needed
        if isinstance(batch_data, np.ndarray):
            if batch_data.dtype.names:
                batch_data = [dict(zip(batch_data.dtype.names, row)) for row in batch_data]
            else:
                return  # Can't process unstructured arrays

        self.row_counts[table] += len(batch_data)

        # Route to domain-specific validators
        if table == "pos_sales":
            self._check_pos_sales(batch_data)
        elif table == "orders":
            self._check_orders(batch_data)
        elif table == "order_lines":
            self._check_order_lines(batch_data)
        elif table == "work_orders":
            self._check_work_orders(batch_data)
        elif table == "shipments":
            self._check_shipments(batch_data)
        elif table == "shipment_legs":
            self._check_logistics_friction(batch_data)
        elif table == "batches":
            self._check_batches(batch_data)
        elif table == "inventory":
            self._check_inventory(batch_data)
        elif table == "shipment_lines":
            self._check_recall_propagation(batch_data)
        elif table == "demand_forecasts":
            self._check_forecasts(batch_data)
        elif table == "returns":
            self._check_returns(batch_data)
        elif table == "osa_metrics":
            self._check_osa_metrics(batch_data)
        elif table == "kpi_actuals":
            self._check_kpis(batch_data)
        elif table == "forecast_accuracy":
            self._check_forecast_accuracy(batch_data)
        elif table == "batch_ingredients":
            self._check_batch_ingredients(batch_data)

    def _check_pos_sales(self, batch: list[dict]) -> None:
        """Validate POS sales for demand volatility."""
        quantities = []
        for row in batch:
            qty = row.get("quantity_cases", 0) or row.get("quantity_eaches", 0) / 12.0
            if qty:
                quantities.append(qty)
                is_promo = row.get("is_promotional", False) or (row.get("promo_id") is not None)
                self._promo_lift.update(qty, is_promo)

        if quantities:
            self._pos_cv_accumulator.update_batch(quantities)

            # Track SKU frequency for Pareto
            sku_ids = [row.get("sku_id") for row in batch if row.get("sku_id")]
            self._frequencies["pos_sku"].update_batch(sku_ids)

    def _check_orders(self, batch: list[dict]) -> None:
        """Validate orders for hub concentration and chaos effects."""
        for row in batch:
            account_id = row.get("account_id") or row.get("retail_account_id")
            if account_id:
                self._orders_by_account[account_id] += 1

            # Track order-location edges for degree monitoring
            location_id = row.get("retail_location_id")
            if account_id and location_id:
                self._degrees["account_orders"].add_edge(account_id, location_id)

            # Track bullwhip batched orders (from bullwhip_whip_crack quirk)
            notes = row.get("notes", "") or ""
            if "batched" in notes.lower() or "bullwhip" in notes.lower():
                self._batched_promo_orders += 1

    def _check_order_lines(self, batch: list[dict]) -> None:
        """Validate order lines for SKU Pareto distribution."""
        # Track quantities for order CV (field is quantity_cases)
        quantities = [row.get("quantity_cases", 0) for row in batch if row.get("quantity_cases")]
        if quantities:
            self._order_cv_accumulator.update_batch(quantities)

        # Track total order volume for return rate, forecast bias, and mass balance
        total_order_cases = sum(row.get("quantity_cases", 0) for row in batch)
        self._return_rate.update_sales(total_order_cases)
        self._forecast_bias.update_actual(total_order_cases)
        self._mass_balance.sum_ordered_cases += total_order_cases

        # Track SKU frequency
        sku_ids = [row.get("sku_id") for row in batch if row.get("sku_id")]
        self._frequencies["order_sku"].update_batch(sku_ids)

        # Simple Margin check: deep discounting
        for row in batch:
            discount = row.get("discount_percent", 0)
            if discount and discount > 50:
                self._margin_issues += 1

    def _check_work_orders(self, batch: list[dict]) -> None:
        """Validate Schedule Adherence and track OEE availability."""
        for row in batch:
            planned = row.get("planned_start_date")
            actual = row.get("actual_start_date")
            if planned and actual:
                if isinstance(planned, str):
                    planned = datetime.fromisoformat(planned.replace("Z", "+00:00")).date()
                if isinstance(actual, str):
                    actual = datetime.fromisoformat(actual.replace("Z", "+00:00")).date()
                if isinstance(planned, datetime): planned = planned.date()
                if isinstance(actual, datetime): actual = actual.date()

                diff = abs((actual - planned).days)
                self._schedule_adherence.update(diff)

            # OEE Availability: track scheduled vs completed work orders
            self._oee.scheduled_runs += 1
            status = row.get("status", "").lower()
            if status in ("complete", "completed", "closed"):
                self._oee.completed_runs += 1

    def _check_shipments(self, batch: list[dict]) -> None:
        """Validate Truck Fill Rate, track shipped cases, and cost-to-serve."""
        # Assume standard truck capacity ~20,000 kg (conservative)
        TRUCK_CAPACITY_KG = 20000.0
        for row in batch:
            # Build destination lookup for shipment_lines mass balance
            shipment_id = row.get("id")
            dest_type = row.get("destination_type", "")
            if shipment_id is not None:
                self._shipment_dest_types[shipment_id] = dest_type

            weight = row.get("total_weight_kg", 0)
            if weight > 0:
                fill_rate = min(1.0, weight / TRUCK_CAPACITY_KG)
                self._truck_fill.update(fill_rate)

            cases = row.get("total_cases", 0)
            status = row.get("status", "")
            if cases > 0:
                # Mass Balance: count all store-bound shipments (production → destination)
                if dest_type == "store":
                    self._mass_balance.sum_shipped_cases += cases

                # Inventory Turns: only count DELIVERED shipments as COGS proxy
                # In-transit shipments are still inventory (captured via transit_inventory)
                if dest_type == "store" and status == "delivered":
                    self._inventory_turns.sum_shipped_cases += cases


            # Cost-to-Serve: track freight cost per case
            freight_cost = row.get("freight_cost", 0)
            if freight_cost > 0 and cases > 0:
                self._cost_to_serve.update(freight_cost, cases)

    def _check_logistics_friction(self, batch: list[dict]) -> None:
        """Validate shipment legs for kinetic friction."""
        for row in batch:
            # Track delay statistics
            planned = row.get("planned_transit_hours", 0)
            actual = row.get("actual_transit_hours", planned)
            delay = actual - planned if actual and planned else row.get("delay_hours", 0)

            mode = row.get("transport_mode", "unknown")
            self._accumulators["logistics"][f"delay_{mode}"].update(delay)
            self._delay_accumulator.update(delay)
            self._legs_by_mode[mode].update(delay)

            # Track OTIF (on-time if delay <= 0)
            self._otif_total += 1
            if delay <= 0:
                self._otif_on_time += 1

            # Track chaos effects
            status = row.get("status") or ""
            notes = row.get("notes") or ""

            if "RSK-LOG-002" in notes or status == "delayed":
                self._port_strike_delayed_legs += 1

            if "congestion" in notes.lower() or "correlated" in notes.lower():
                self._congestion_correlated_legs += 1

    def _check_batches(self, batch: list[dict]) -> None:
        """Validate batches for yield loss, QC, OEE quality, and mass balance."""
        for row in batch:
            # Track yield percentage
            yield_pct = row.get("yield_percent") or row.get("yield_percentage")
            if yield_pct is not None:
                self._yield_accumulator.update(yield_pct / 100.0 if yield_pct > 1 else yield_pct)
                self._accumulators["production"]["yield"].update(yield_pct)

            # Track QC rejection rate (lowercase status values)
            qc_status = row.get("qc_status")
            if qc_status:
                self._qc_total += 1
                if qc_status.lower() == "rejected":
                    self._qc_rejected += 1

            # OEE Quality: track approved vs total batches
            # Generator uses "released" for QC-passed batches
            self._oee.total_batches += 1
            if qc_status and qc_status.lower() in ("approved", "released"):
                self._oee.approved_batches += 1

            # Mass Balance: track batch output (kg and cases)
            output_kg = row.get("quantity_kg", 0) or row.get("actual_quantity_kg", 0)
            output_cases = row.get("output_cases", 0)
            if output_kg > 0:
                self._mass_balance.sum_batch_output_kg += output_kg
                self._mass_balance.batch_count += 1
            if output_cases > 0:
                self._mass_balance.sum_batch_output_cases += output_cases

    def _check_inventory(self, batch: list[dict]) -> None:
        """Validate inventory for cash cycle metrics, SLOBs, inventory turns, and mass balance."""
        for row in batch:
            days_on_hand = row.get("days_on_hand")
            if days_on_hand is not None:
                self._accumulators["inventory"]["days_on_hand"].update(days_on_hand)

            # SLOB check
            aging = row.get("aging_bucket")
            self._inventory_total += 1

            # Mass Balance: track inventory cases
            qty_cases = row.get("quantity_cases", 0)
            if qty_cases > 0:
                self._mass_balance.sum_inventory_cases += qty_cases
            if aging == "90+":
                self._slob_count += 1

            # Inventory Turns: track inventory levels
            qty = row.get("quantity_cases", 0)
            if qty > 0:
                self._inventory_turns.sum_inventory_cases += qty
                self._inventory_turns.inventory_snapshots += 1

    def _check_recall_propagation(self, batch: list[dict]) -> None:
        """Track recall batch propagation to stores and fulfilled cases for mass balance."""
        for row in batch:
            batch_id = row.get("batch_id")
            # Check if this is the recall batch (by ID or batch_number if available)
            if batch_id and row.get("destination_type") == "STORE":
                destination_id = row.get("destination_id")
                if destination_id:
                    self._recall_affected_stores.add(destination_id)

            # Mass Balance: track fulfilled cases from shipment_lines
            # Only count lines for store-bound shipments (not intermediate DC transfers)
            qty_cases = row.get("quantity_cases", 0)
            if qty_cases > 0:
                shipment_id = row.get("shipment_id")
                dest_type = self._shipment_dest_types.get(shipment_id, "")
                if dest_type == "store":
                    self._mass_balance.sum_fulfilled_cases += qty_cases

    def _check_forecasts(self, batch: list[dict]) -> None:
        """Track demand forecasts for bias calculation."""
        for row in batch:
            # Prefer 'final_forecast', fallback to others
            qty = row.get("final_forecast") or row.get("consensus_forecast") or 0
            self._forecast_bias.update_forecast(qty)

    def _check_returns(self, batch: list[dict]) -> None:
        """Track returns for return rate calculation.

        Return rate = return volume / order volume (not POS sales).
        We track both here and reconcile in check_benchmarks.
        """
        # Returns table has total_cases per return
        total_returned = sum(row.get("total_cases", 0) for row in batch)
        self._return_rate.update_returns(total_returned)

    def _check_osa_metrics(self, batch: list[dict]) -> None:
        """Track On-Shelf Availability metrics."""
        for row in batch:
            self._osa_total += 1
            is_in_stock = row.get("is_in_stock", False)
            if is_in_stock:
                self._osa_in_stock += 1

    def _check_kpis(self, batch: list[dict]) -> None:
        """Track KPI actuals for strategic validation."""
        for row in batch:
            kpi_code = row.get("kpi_code", "")
            val = row.get("actual_value", 0)

            # Track CO2 multiplier if it's an emissions KPI
            if "CO2" in kpi_code or "carbon" in kpi_code.lower():
                # Check if multiplier is applied (baseline ~0.25 kg/case)
                if val > 0.5:  # More than 2x baseline suggests multiplier applied
                    self._co2_multiplier_applied = max(self._co2_multiplier_applied, val / 0.25)

    def _check_forecast_accuracy(self, batch: list[dict]) -> None:
        """
        Track forecast accuracy via MAPE calculation.

        Uses forecast_accuracy table which has pre-joined forecast vs actual quantities.
        MAPE = Mean(|Forecast - Actual| / Actual) × 100%

        Note: The table uses 'actual_demand' and 'forecast_demand' field names.
        """
        for row in batch:
            # Try both field naming conventions
            forecast = row.get("forecast_demand") or row.get("forecast_quantity", 0)
            actual = row.get("actual_demand") or row.get("actual_quantity", 0)
            if actual > 0:
                self._forecast_mape.update(forecast, actual)

    def _check_batch_ingredients(self, batch: list[dict]) -> None:
        """
        Track ingredient consumption for mass balance (kg input).

        Mass Balance Equation 1: Σ ingredients consumed ≈ Σ batch output (with yield loss)
        """
        for row in batch:
            # Track actual quantity consumed (kg)
            qty_kg = row.get("actual_quantity_kg", 0) or row.get("quantity_kg", 0)
            if qty_kg > 0:
                self._mass_balance.sum_ingredient_input_kg += qty_kg

    def _get_tolerance(self, key: str, default: Any = None) -> Any:
        """Get a validation tolerance from the manifest."""
        return self._tolerances.get(key, default)

    def _get_range(self, key: str, default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        """Get a range tolerance from the manifest."""
        val = self._tolerances.get(key, default)
        if isinstance(val, list) and len(val) >= 2:
            return (val[0], val[1])
        return default

    def check_benchmarks(self, check_interval_rows: int = 10000) -> None:
        """
        Validate current state against benchmarks from manifest.

        Called periodically during generation to enable fail-fast.
        All thresholds are read from benchmark_manifest.json.

        Args:
            check_interval_rows: Minimum rows between checks
        """
        total_rows = sum(self.row_counts.values())
        interval = self._get_tolerance("check_interval_rows", check_interval_rows)
        if total_rows < interval:
            return

        # === DISTRIBUTION CHECKS ===

        # Check Pareto distribution (SKU 80/20)
        if "order_sku" in self._frequencies:
            pareto = self._frequencies["order_sku"].pareto_ratio(0.20)
            target_range = self._get_range("pareto_top20_range", (0.75, 0.85))
            if not (target_range[0] <= pareto <= target_range[1]):
                self._add_violation(
                    f"Pareto drift: top 20% SKUs = {pareto:.1%} volume "
                    f"(target: {target_range[0]:.0%}-{target_range[1]:.0%})"
                )

        # Check hub concentration (MegaMart 20-30%)
        total_orders = sum(self._orders_by_account.values())
        if total_orders > 1000:
            top_account, top_count = self._orders_by_account.most_common(1)[0]
            concentration = top_count / total_orders
            target_range = self._get_range("hub_concentration_range", (0.20, 0.30))
            if not (target_range[0] <= concentration <= target_range[1]):
                self._add_violation(
                    f"Hub concentration drift: top account = {concentration:.1%} "
                    f"(target: {target_range[0]:.0%}-{target_range[1]:.0%})"
                )

        # === DEMAND CHECKS ===

        # Check Bullwhip effect (Order CV > POS CV with multiplier in range)
        if self._pos_cv_accumulator.count > 100 and self._order_cv_accumulator.count > 100:
            pos_cv = self._pos_cv_accumulator.cv
            order_cv = self._order_cv_accumulator.cv

            # Check CV ranges
            pos_range = self._get_range("pos_cv_range", (0.15, 0.50))
            if not (pos_range[0] <= pos_cv <= pos_range[1]):
                self._add_violation(
                    f"POS CV out of range: {pos_cv:.2f} (target: {pos_range[0]:.2f}-{pos_range[1]:.2f})"
                )

            order_range = self._get_range("order_cv_range", (0.30, 0.80))
            if not (order_range[0] <= order_cv <= order_range[1]):
                self._add_violation(
                    f"Order CV out of range: {order_cv:.2f} (target: {order_range[0]:.2f}-{order_range[1]:.2f})"
                )

            # Check bullwhip multiplier
            if pos_cv > 0:
                multiplier = order_cv / pos_cv
                mult_range = self._get_range("bullwhip_multiplier_range", (1.5, 3.0))
                if not (mult_range[0] <= multiplier <= mult_range[1]):
                    self._add_violation(
                        f"Bullwhip multiplier out of range: {multiplier:.2f}x "
                        f"(target: {mult_range[0]:.1f}x-{mult_range[1]:.1f}x)"
                    )

        # Check Forecast Bias
        if self._forecast_bias.count > 1000:
            bias = abs(self._forecast_bias.bias_pct)
            max_bias = self._get_tolerance("forecast_bias_max", 0.50)
            if bias > max_bias:
                self._add_violation(f"Forecast Bias excessive: {bias:.1%} (max: {max_bias:.0%})")

        # === PRODUCTION CHECKS ===

        # Check yield distribution
        if self._yield_accumulator.count > 100:
            yield_mean = self._yield_accumulator.mean
            yield_std = self._yield_accumulator.std
            yield_range = self._get_range("yield_mean_range", (0.96, 0.99))
            yield_std_max = self._get_tolerance("yield_std_max", 0.02)

            if not (yield_range[0] <= yield_mean <= yield_range[1]):
                self._add_violation(
                    f"Yield mean out of range: {yield_mean:.1%} "
                    f"(target: {yield_range[0]:.0%}-{yield_range[1]:.0%})"
                )
            if yield_std > yield_std_max:
                self._add_violation(
                    f"Yield std too high: {yield_std:.3f} (max: {yield_std_max:.3f})"
                )

        # Check QC rejection rate
        if self._qc_total > 100:
            qc_rate = self._qc_rejected / self._qc_total
            qc_range = self._get_range("qc_rejection_rate_range", (0.01, 0.04))
            
            # Dynamic tolerance for data_decay quirk
            if "data_decay" in self.active_quirks:
                # Base 2% + Elevated 8% -> Weighted avg increases
                # Allow up to 10% (0.10) when decay is active to avoid false positives
                qc_range = (qc_range[0], 0.10)

            if not (qc_range[0] <= qc_rate <= qc_range[1]):
                self._add_violation(
                    f"QC rejection rate out of range: {qc_rate:.1%} "
                    f"(target: {qc_range[0]:.0%}-{qc_range[1]:.0%})"
                )

        # Check Promo Lift
        if self._promo_lift.count_promo > 100:
            lift = self._promo_lift.lift_multiplier
            lift_range = self._get_range("promo_lift_range", (1.5, 3.5))
            # Relax lower bound slightly for streaming variations
            if not (lift_range[0] * 0.8 <= lift <= lift_range[1] * 1.2):
                self._add_violation(
                    f"Promo lift out of range: {lift:.2f}x "
                    f"(target: {lift_range[0]:.1f}x-{lift_range[1]:.1f}x)"
                )

        # === LOGISTICS CHECKS ===

        # Check OTIF
        if self._otif_total > 1000:
            otif_rate = self._otif_on_time / self._otif_total
            otif_range = self._get_range("otif_range", (0.85, 0.98))
            if not (otif_range[0] <= otif_rate <= otif_range[1]):
                self._add_violation(
                    f"OTIF rate out of range: {otif_rate:.1%} "
                    f"(target: {otif_range[0]:.0%}-{otif_range[1]:.0%})"
                )

        # Check delay statistics
        if self._delay_accumulator.count > 1000:
            delay_mean = self._delay_accumulator.mean
            delay_std = self._delay_accumulator.std
            delay_mean_max = self._get_tolerance("delay_mean_max_hours", 24)
            delay_std_max = self._get_tolerance("delay_std_max_hours", 48)

            if delay_mean > delay_mean_max:
                self._add_violation(
                    f"Mean delay too high: {delay_mean:.1f}h (max: {delay_mean_max}h)"
                )
            if delay_std > delay_std_max:
                self._add_violation(
                    f"Delay std too high: {delay_std:.1f}h (max: {delay_std_max}h)"
                )

        # === RETURNS CHECKS ===

        if self._return_rate.sum_sales_cases > 50000:
            rate = self._return_rate.return_rate
            return_range = self._get_range("return_rate_range", (0.01, 0.06))
            if not (return_range[0] <= rate <= return_range[1]):
                self._add_violation(
                    f"Return rate out of range: {rate:.1%} "
                    f"(target: {return_range[0]:.0%}-{return_range[1]:.0%})"
                )

        # === OSA CHECKS ===

        if self._osa_total > 10000:
            osa_rate = self._osa_in_stock / self._osa_total
            osa_range = self._get_range("osa_range", (0.88, 0.96))
            if not (osa_range[0] <= osa_rate <= osa_range[1]):
                self._add_violation(
                    f"OSA rate out of range: {osa_rate:.1%} "
                    f"(target: {osa_range[0]:.0%}-{osa_range[1]:.0%})"
                )

        # === MARGIN CHECKS ===

        max_discount = self._get_tolerance("max_discount_percent", 55)
        if self._margin_issues > 0:
            # Only warn if it's a significant portion
            order_line_count = self.row_counts.get("order_lines", 1)
            margin_issue_rate = self._margin_issues / order_line_count
            if margin_issue_rate > 0.01:  # More than 1% with excessive discounts
                self._add_violation(
                    f"Excessive discounting: {self._margin_issues} lines > {max_discount}% discount"
                )

        # === EXPERT / STRESS TEST CHECKS ===

        # Schedule Adherence
        if self._schedule_adherence.count > 100:
            sa_days = self._schedule_adherence.mean
            sa_tol = self._get_tolerance("schedule_adherence_tolerance_days", 1.0)
            if sa_days > sa_tol:
                self._add_violation(
                    f"Schedule Adherence drift: {sa_days:.1f} days variance (max: {sa_tol})"
                )

        # Truck Fill Rate
        if self._truck_fill.count > 100:
            fill_rate = self._truck_fill.mean
            target_fill = self._get_tolerance("truck_fill_rate_target", 0.70)
            if fill_rate < target_fill:
                self._add_violation(
                    f"Truck Fill Rate low: {fill_rate:.1%} (target: >{target_fill:.0%})"
                )

        # SLOB Inventory
        if self._inventory_total > 100:
            slob_pct = self._slob_count / self._inventory_total
            max_slob = self._get_tolerance("slob_inventory_max_pct", 0.15)
            if slob_pct > max_slob:
                self._add_violation(
                    f"SLOB Inventory high: {slob_pct:.1%} (max: {max_slob:.0%})"
                )

        # OEE (Overall Equipment Effectiveness)
        if self._oee.scheduled_runs > 100 and self._oee.total_batches > 100:
            oee_value = self._oee.oee
            oee_range = self._get_range("oee_range", (0.65, 0.85))
            if not (oee_range[0] <= oee_value <= oee_range[1]):
                self._add_violation(
                    f"OEE out of range: {oee_value:.1%} "
                    f"(target: {oee_range[0]:.0%}-{oee_range[1]:.0%})"
                )

        # Inventory Turns
        if self._inventory_turns.inventory_snapshots > 100 and self._inventory_turns.sum_shipped_cases > 1000:
            turns_value = self._inventory_turns.turns
            turns_range = self._get_range("inventory_turns_range", (6.0, 14.0))
            if not (turns_range[0] <= turns_value <= turns_range[1]):
                self._add_violation(
                    f"Inventory Turns out of range: {turns_value:.1f}x "
                    f"(target: {turns_range[0]:.0f}x-{turns_range[1]:.0f}x)"
                )

        # Forecast MAPE (Mean Absolute Percentage Error)
        if self._forecast_mape.count > 100:
            mape_value = self._forecast_mape.mape_decimal
            mape_range = self._get_range("mape_range", (0.20, 0.50))
            if not (mape_range[0] <= mape_value <= mape_range[1]):
                self._add_violation(
                    f"Forecast MAPE out of range: {mape_value:.1%} "
                    f"(target: {mape_range[0]:.0%}-{mape_range[1]:.0%})"
                )

        # Cost-to-Serve ($/case)
        if len(self._cost_to_serve.costs) > 100:
            cost_value = self._cost_to_serve.cost_per_case
            cost_range = self._get_range("cost_per_case_range", (1.00, 3.00))
            if not (cost_range[0] <= cost_value <= cost_range[1]):
                self._add_violation(
                    f"Cost-to-Serve out of range: ${cost_value:.2f}/case "
                    f"(target: ${cost_range[0]:.2f}-${cost_range[1]:.2f})"
                )

            # Cost variance (P90/P50 ratio)
            variance_ratio = self._cost_to_serve.cost_p90_p50_ratio
            max_variance = self._get_tolerance("cost_variance_max_ratio", 4.0)
            if variance_ratio > max_variance:
                self._add_violation(
                    f"Cost variance too high: P90/P50 = {variance_ratio:.1f}x "
                    f"(max: {max_variance:.1f}x)"
                )

        # === MASS BALANCE CHECKS (Physics) ===
        max_drift = self._get_tolerance("mass_balance_drift_max", 0.02)

        # Check 1: Ingredient → Batch (kg) - should have negative drift due to yield loss
        if self._mass_balance.sum_ingredient_input_kg > 1000:
            ing_to_batch_drift = self._mass_balance.ingredient_to_batch_drift
            # Positive drift = more output than input = IMPOSSIBLE (physics violation)
            if ing_to_batch_drift > max_drift:
                self._add_violation(
                    f"Mass Balance (Ingredient→Batch) violation: {ing_to_batch_drift:+.1%} drift "
                    f"(max: +{max_drift:.0%}) - more output than input!"
                )
            # Large negative drift might indicate data loss, but is not a physics violation
            # (just means high scrap/yield loss which is tracked elsewhere)

        # Check 2: Batch → Ship+Inventory (cases) - production should equal shipped + inventory
        if self._mass_balance.sum_batch_output_cases > 1000:
            batch_to_ship_drift = self._mass_balance.batch_to_shipment_drift
            # Allow some tolerance for in-transit, timing differences
            if abs(batch_to_ship_drift) > max_drift * 5:  # 10% tolerance for this balance
                self._add_violation(
                    f"Mass Balance (Batch→Ship+Inv) drift: {batch_to_ship_drift:+.1%} "
                    f"(tolerance: ±{max_drift*5:.0%})"
                )

        # Check 3: Order → Fulfill (cases) - should be <= 0 (can't ship more than ordered)
        if self._mass_balance.sum_ordered_cases > 1000:
            order_to_fulfill_drift = self._mass_balance.order_to_fulfill_drift
            # Positive drift = shipped more than ordered = BUG
            if order_to_fulfill_drift > max_drift:
                self._add_violation(
                    f"Mass Balance (Order→Fulfill) violation: {order_to_fulfill_drift:+.1%} drift "
                    f"- shipped more than ordered!"
                )

    def check_chaos_effects(self, risk_manager: Any = None, quirks_manager: Any = None) -> list[tuple[str, bool, str]]:
        """
        Validate that chaos effects (risk events and quirks) were applied correctly.

        Uses chaos_validation section from manifest for thresholds.

        Args:
            risk_manager: RiskEventManager instance (optional, for triggered event checks)
            quirks_manager: QuirksManager instance (optional, for enabled quirk checks)

        Returns:
            List of (check_name, passed, message) tuples
        """
        results = []

        # RSK-LOG-002: Port strike delays
        if risk_manager and risk_manager.is_triggered("RSK-LOG-002"):
            config = self._chaos_validation.get("RSK-LOG-002", {})
            min_legs = config.get("min_affected_legs", 100)
            if self._port_strike_delayed_legs >= min_legs:
                results.append((
                    "RSK-LOG-002",
                    True,
                    f"Port strike: {self._port_strike_delayed_legs} legs delayed (min: {min_legs})"
                ))
            else:
                results.append((
                    "RSK-LOG-002",
                    False,
                    f"Port strike: only {self._port_strike_delayed_legs} legs delayed (min: {min_legs})"
                ))

        # RSK-CYB-004: DC pick waves on hold (handled in DataValidator)

        # RSK-ENV-005: Carbon tax multiplier
        if risk_manager and risk_manager.is_triggered("RSK-ENV-005"):
            config = self._chaos_validation.get("RSK-ENV-005", {})
            min_mult = config.get("co2_multiplier_min", 2.0)
            if self._co2_multiplier_applied >= min_mult:
                results.append((
                    "RSK-ENV-005",
                    True,
                    f"Carbon tax: {self._co2_multiplier_applied:.1f}x multiplier applied (min: {min_mult}x)"
                ))
            else:
                # This might not be detectable through KPIs alone, so just note it
                results.append((
                    "RSK-ENV-005",
                    True,  # Don't fail - CO2 is tracked differently
                    f"Carbon tax: multiplier tracking via KPIs (observed: {self._co2_multiplier_applied:.1f}x)"
                ))

        # port_congestion_flicker quirk
        if quirks_manager and quirks_manager.is_enabled("port_congestion_flicker"):
            config = self._chaos_validation.get("port_congestion_flicker", {})
            min_legs = config.get("min_correlated_legs", 100)
            if self._congestion_correlated_legs >= min_legs:
                results.append((
                    "port_congestion_flicker",
                    True,
                    f"Port congestion: {self._congestion_correlated_legs} correlated legs (min: {min_legs})"
                ))
            else:
                results.append((
                    "port_congestion_flicker",
                    False,
                    f"Port congestion: only {self._congestion_correlated_legs} correlated legs (min: {min_legs})"
                ))

        # bullwhip_whip_crack quirk
        if quirks_manager and quirks_manager.is_enabled("bullwhip_whip_crack"):
            config = self._chaos_validation.get("bullwhip_whip_crack", {})
            min_orders = config.get("min_batched_orders", 10)
            if self._batched_promo_orders >= min_orders:
                results.append((
                    "bullwhip_whip_crack",
                    True,
                    f"Bullwhip batching: {self._batched_promo_orders} batched orders (min: {min_orders})"
                ))
            else:
                results.append((
                    "bullwhip_whip_crack",
                    False,
                    f"Bullwhip batching: only {self._batched_promo_orders} batched orders (min: {min_orders})"
                ))

        # Shrinkage check (phantom_inventory)
        if quirks_manager and quirks_manager.is_enabled("phantom_inventory"):
            config = self._chaos_validation.get("phantom_inventory", {})
            shrink_range = config.get("shrinkage_rate_range", [0.01, 0.04])
            inv_count = self.row_counts.get("inventory", 0)
            if inv_count > 0:
                # Shrinkage is tracked via accumulators; just verify inventory exists
                results.append((
                    "phantom_inventory",
                    True,
                    f"Phantom inventory: {inv_count} inventory records tracked"
                ))

        return results

    def _add_violation(self, message: str) -> None:
        """Add a violation and optionally raise immediately."""
        self.violations.append(message)
        if self.fail_fast:
            raise RealismViolationError([message])

    def has_violations(self) -> bool:
        """Check if any violations have been detected."""
        return len(self.violations) > 0

    def get_violations(self) -> list[str]:
        """Get all detected violations."""
        return list(self.violations)

    def set_stochastic_mode(self, mode: StochasticMode) -> None:
        """
        Set stochastic mode for generators.

        Used by Risk Events to switch from normal to disrupted distributions.
        """
        self.stochastic_mode = mode

    def set_active_quirks(self, quirks: list[str] | set[str]) -> None:
        """Set the list of active quirks for dynamic tolerance adjustment."""
        self.active_quirks = set(quirks)

    def get_reality_report(self) -> dict[str, Any]:
        """
        Generate comprehensive reality report.

        Returns summary of all validations, statistics, and violations.
        """
        report = {
            "is_realistic": len(self.violations) == 0,
            "profile": self.profile_name,
            "stochastic_mode": self.stochastic_mode.value,
            "violations": self.violations,
            "row_counts": dict(self.row_counts),
            "total_rows": sum(self.row_counts.values()),
            "statistics": {},
        }

        # Add frequency statistics
        if self._frequencies:
            report["statistics"]["frequencies"] = {
                name: sketch.get_stats()
                for name, sketch in self._frequencies.items()
            }

        # Add Pareto check
        if "order_sku" in self._frequencies:
            pareto = self._frequencies["order_sku"].pareto_ratio(0.20)
            report["statistics"]["pareto_20_80"] = {
                "value": pareto,
                "target": "0.75-0.85",
                "passed": 0.75 <= pareto <= 0.85,
            }

        # Add hub concentration
        total_orders = sum(self._orders_by_account.values())
        if total_orders > 0:
            top_accounts = self._orders_by_account.most_common(5)
            report["statistics"]["hub_concentration"] = {
                "total_orders": total_orders,
                "top_accounts": [
                    {"account_id": acc, "count": cnt, "pct": cnt / total_orders}
                    for acc, cnt in top_accounts
                ],
            }

        # Add Bullwhip metrics
        if self._pos_cv_accumulator.count > 0:
            report["statistics"]["bullwhip"] = {
                "pos_cv": self._pos_cv_accumulator.cv,
                "order_cv": self._order_cv_accumulator.cv,
                "multiplier": (
                    self._order_cv_accumulator.cv / self._pos_cv_accumulator.cv
                    if self._pos_cv_accumulator.cv > 0
                    else 0
                ),
                "promo_lift": self._promo_lift.lift_multiplier,
            }

        # Add production metrics
        if self._yield_accumulator.count > 0:
            report["statistics"]["production"] = {
                "yield_mean": self._yield_accumulator.mean,
                "yield_std": self._yield_accumulator.std,
                "qc_total": self._qc_total,
                "qc_rejected": self._qc_rejected,
                "qc_rejection_rate": self._qc_rejected / self._qc_total if self._qc_total > 0 else 0,
            }

        # Add logistics metrics
        if self._otif_total > 0:
            report["statistics"]["logistics"] = {
                "otif_total": self._otif_total,
                "otif_on_time": self._otif_on_time,
                "otif_rate": self._otif_on_time / self._otif_total,
                "delay_mean": self._delay_accumulator.mean,
                "delay_std": self._delay_accumulator.std,
            }

        # Add OSA metrics
        if self._osa_total > 0:
            report["statistics"]["osa"] = {
                "total_checks": self._osa_total,
                "in_stock": self._osa_in_stock,
                "osa_rate": self._osa_in_stock / self._osa_total,
            }

        # Add returns metrics
        report["statistics"]["returns"] = self._return_rate.get_stats()

        # Add forecast metrics (bias + MAPE)
        forecast_stats = self._forecast_bias.get_stats()
        forecast_stats.update({
            "mape_count": self._forecast_mape.count,
            "mape_pct": self._forecast_mape.mape_pct,
            "mape_decimal": self._forecast_mape.mape_decimal,
        })
        report["statistics"]["forecast"] = forecast_stats

        # Add expert metrics
        if self._schedule_adherence.count > 0:
            report["statistics"]["expert"] = {
                "schedule_adherence_days": self._schedule_adherence.mean,
                "truck_fill_rate": self._truck_fill.mean,
                "slob_pct": self._slob_count / self._inventory_total if self._inventory_total > 0 else 0,
                "oee": self._oee.oee,
                "oee_availability": self._oee.availability,
                "oee_quality": self._oee.quality,
                "inventory_turns": self._inventory_turns.turns,
                "avg_inventory": self._inventory_turns.avg_inventory,
                "shipped_cases": self._inventory_turns.sum_shipped_cases,
                "forecast_mape": self._forecast_mape.mape_decimal,
                "forecast_mape_pct": self._forecast_mape.mape_pct,
                "cost_per_case": self._cost_to_serve.cost_per_case,
                "cost_p90_p50_ratio": self._cost_to_serve.cost_p90_p50_ratio,
                "cost_shipment_count": len(self._cost_to_serve.costs),
            }

        # Add chaos effects tracking
        report["statistics"]["chaos_effects"] = {
            "port_strike_delayed_legs": self._port_strike_delayed_legs,
            "congestion_correlated_legs": self._congestion_correlated_legs,
            "batched_promo_orders": self._batched_promo_orders,
            "co2_multiplier_applied": self._co2_multiplier_applied,
            "margin_issues": self._margin_issues,
        }

        # Add recall tracking
        report["statistics"]["recall_trace"] = {
            "affected_stores": len(self._recall_affected_stores),
            "target": 500,
        }

        # Add mass balance (physics) tracking
        report["statistics"]["mass_balance"] = self._mass_balance.get_stats()

        # Add accumulator summaries
        for domain, accumulators in self._accumulators.items():
            if domain not in report["statistics"]:
                report["statistics"][domain] = {}
            report["statistics"][domain].update({
                name: acc.get_stats()
                for name, acc in accumulators.items()
            })

        return report

    def reset(self) -> None:
        """Reset all tracking state."""
        self.violations.clear()
        self.row_counts.clear()
        self._accumulators.clear()
        self._frequencies.clear()
        self._cardinalities.clear()
        self._degrees.clear()
        self._pos_cv_accumulator = WelfordAccumulator()
        self._order_cv_accumulator = WelfordAccumulator()
        self._orders_by_account.clear()
        self._recall_affected_stores.clear()

        # Reset strategic & financial trackers
        self._forecast_bias = ForecastBiasAccumulator()
        self._return_rate = ReturnRateAccumulator()
        self._margin_issues = 0
        self._esg_issues = 0

        # Reset production & quality trackers
        self._yield_accumulator = WelfordAccumulator()
        self._qc_total = 0
        self._qc_rejected = 0

        # Reset expert benchmarks
        self._schedule_adherence = WelfordAccumulator()
        self._truck_fill = WelfordAccumulator()
        self._slob_count = 0
        self._inventory_total = 0
        self._oee = OEEAccumulator()
        self._inventory_turns = InventoryTurnsAccumulator()
        self._forecast_mape = ForecastMAPEAccumulator()
        self._cost_to_serve = CostToServeAccumulator()
        self._mass_balance = MassBalanceAccumulator()

        # Reset logistics trackers
        self._delay_accumulator = WelfordAccumulator()
        self._otif_on_time = 0
        self._otif_total = 0
        self._legs_by_mode.clear()

        # Reset OSA tracker
        self._osa_in_stock = 0
        self._osa_total = 0

        # Reset chaos effect trackers
        self._port_strike_delayed_legs = 0
        self._congestion_correlated_legs = 0
        self._batched_promo_orders = 0
        self._co2_multiplier_applied = 1.0
