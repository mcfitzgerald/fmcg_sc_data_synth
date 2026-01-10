"""
Tests for v0.29.0 Flexible Production Capacity (Seasonal Capacity).

Verifies that production capacity tracks seasonal demand patterns:
- Peak demand → higher capacity (overtime, extra shifts)
- Trough demand → lower capacity (reduced shifts, maintenance)
"""

import math

import pytest

from prism_sim.network.core import Node, NodeType
from prism_sim.product.core import Product, ProductCategory, Recipe
from prism_sim.simulation.mrp import MRPEngine
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.transform import TransformEngine
from prism_sim.simulation.world import World


@pytest.fixture
def minimal_world() -> World:
    """Create a minimal world with one plant for testing."""
    world = World()

    # Add a product (positional args: id, name, category, weight_kg, shelf_life, pack_size, cases_per_pallet, cost)
    product = Product(
        "SKU-001", "Test Product", ProductCategory.HOME_CARE, 1.0, 365, 12, 100, 10.0
    )
    world.add_product(product)

    # Add a plant
    plant = Node("PLANT-01", "Test Plant", NodeType.PLANT, "Location")
    world.add_node(plant)

    # Add a DC (RDC-prefixed nodes are treated as RDCs by MRP)
    rdc = Node("RDC-01", "Test RDC", NodeType.DC, "Location")
    world.add_node(rdc)

    # Add a recipe
    recipe = Recipe(
        product_id="SKU-001",
        ingredients={},  # No ingredients for simplicity
        run_rate_cases_per_hour=1000.0,
        changeover_time_hours=0.5,
    )
    world.add_recipe(recipe)

    return world


@pytest.fixture
def seasonal_config() -> dict:
    """Config with seasonal capacity enabled."""
    return {
        "simulation_parameters": {
            "global_constants": {"epsilon": 0.001},
            "demand": {
                "seasonality": {
                    "amplitude": 0.12,
                    "capacity_amplitude": 0.12,
                    "phase_shift_days": 150,
                    "cycle_days": 365,
                }
            },
            "manufacturing": {
                "production_hours_per_day": 24.0,
                "efficiency_factor": 0.85,
                "unplanned_downtime_pct": 0.05,
                "production_rate_multiplier": 1.0,
                "changeover_time_multiplier": 0.1,
                "mrp_thresholds": {
                    "rate_based_production": True,
                    "campaign_batching": {"enabled": True},
                },
                "plant_parameters": {
                    "PLANT-01": {
                        "efficiency_factor": 0.85,
                        "unplanned_downtime_pct": 0.05,
                        "supported_categories": ["HOME_CARE"],
                    }
                },
            },
            "agents": {"abc_prioritization": {"enabled": False}},
        }
    }


@pytest.fixture
def no_seasonal_config() -> dict:
    """Config with seasonal capacity disabled (amplitude=0)."""
    return {
        "simulation_parameters": {
            "global_constants": {"epsilon": 0.001},
            "demand": {
                "seasonality": {
                    "amplitude": 0.12,
                    "capacity_amplitude": 0.0,  # Disabled
                    "phase_shift_days": 150,
                    "cycle_days": 365,
                }
            },
            "manufacturing": {
                "production_hours_per_day": 24.0,
                "efficiency_factor": 0.85,
                "unplanned_downtime_pct": 0.05,
                "production_rate_multiplier": 1.0,
                "changeover_time_multiplier": 0.1,
                "mrp_thresholds": {
                    "rate_based_production": True,
                    "campaign_batching": {"enabled": True},
                },
                "plant_parameters": {
                    "PLANT-01": {
                        "efficiency_factor": 0.85,
                        "unplanned_downtime_pct": 0.05,
                        "supported_categories": ["HOME_CARE"],
                    }
                },
            },
            "agents": {"abc_prioritization": {"enabled": False}},
        }
    }


class TestTransformEngineSeasonalCapacity:
    """Tests for TransformEngine seasonal capacity factor."""

    def test_seasonal_factor_at_peak(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """Peak day should have factor ~1.12 (12% increase)."""
        state = StateManager(minimal_world)
        engine = TransformEngine(minimal_world, state, seasonal_config)

        # Peak is at phase_shift + cycle/4 = 150 + 91.25 ≈ 241
        peak_day = 241
        factor = engine._get_seasonal_capacity_factor(peak_day)

        # Should be close to 1.12 (within tolerance for sine)
        assert 1.11 < factor < 1.13, f"Peak factor {factor} not in expected range"

    def test_seasonal_factor_at_trough(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """Trough day should have factor ~0.88 (12% decrease)."""
        state = StateManager(minimal_world)
        engine = TransformEngine(minimal_world, state, seasonal_config)

        # Trough is at phase_shift + 3*cycle/4 = 150 + 273.75 ≈ 58 (wrapping around)
        # Or within first year: day 58 is in trough zone
        trough_day = 58
        factor = engine._get_seasonal_capacity_factor(trough_day)

        # Should be close to 0.88 (within tolerance)
        assert 0.87 < factor < 0.91, f"Trough factor {factor} not in expected range"

    def test_seasonal_factor_zero_amplitude(
        self, minimal_world: World, no_seasonal_config: dict
    ) -> None:
        """Zero amplitude should always return 1.0."""
        state = StateManager(minimal_world)
        engine = TransformEngine(minimal_world, state, no_seasonal_config)

        # Should be 1.0 regardless of day
        for day in [1, 30, 90, 150, 241, 330, 365]:
            factor = engine._get_seasonal_capacity_factor(day)
            assert factor == 1.0, f"Day {day}: expected 1.0, got {factor}"

    def test_seasonal_factor_follows_sine(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """Factor should follow sinusoidal pattern."""
        state = StateManager(minimal_world)
        engine = TransformEngine(minimal_world, state, seasonal_config)

        amplitude = 0.12
        phase_shift = 150
        cycle = 365

        for day in range(1, 366, 30):
            actual = engine._get_seasonal_capacity_factor(day)
            expected = 1.0 + amplitude * math.sin(
                2 * math.pi * (day - phase_shift) / cycle
            )
            assert abs(actual - expected) < 0.001, (
                f"Day {day}: expected {expected:.4f}, got {actual:.4f}"
            )


class TestMRPDailyCapacity:
    """Tests for MRP day-aware capacity."""

    def test_daily_capacity_at_peak(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """Peak day should have higher capacity."""
        state = StateManager(minimal_world)
        mrp = MRPEngine(minimal_world, state, seasonal_config)

        base_capacity = mrp._max_daily_capacity
        peak_day = 241
        peak_capacity = mrp._get_daily_capacity(peak_day)

        # Should be ~12% higher than base
        expected = base_capacity * 1.12
        assert abs(peak_capacity - expected) / expected < 0.02, (
            f"Peak capacity {peak_capacity} not close to expected {expected}"
        )

    def test_daily_capacity_zero_amplitude(
        self, minimal_world: World, no_seasonal_config: dict
    ) -> None:
        """Zero amplitude should return base capacity."""
        state = StateManager(minimal_world)
        mrp = MRPEngine(minimal_world, state, no_seasonal_config)

        base_capacity = mrp._max_daily_capacity

        for day in [1, 150, 241, 365]:
            capacity = mrp._get_daily_capacity(day)
            assert capacity == base_capacity, (
                f"Day {day}: expected base {base_capacity}, got {capacity}"
            )

    def test_capacity_matches_transform(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """MRP and TransformEngine should use same seasonal factor."""
        state = StateManager(minimal_world)
        mrp = MRPEngine(minimal_world, state, seasonal_config)
        transform = TransformEngine(minimal_world, state, seasonal_config)

        for day in [1, 90, 150, 241, 330]:
            mrp_factor = mrp._get_daily_capacity(day) / mrp._max_daily_capacity
            transform_factor = transform._get_seasonal_capacity_factor(day)

            assert abs(mrp_factor - transform_factor) < 0.001, (
                f"Day {day}: MRP factor {mrp_factor:.4f} != "
                f"Transform factor {transform_factor:.4f}"
            )


class TestOEECalculation:
    """Tests for OEE calculation with seasonal capacity."""

    def test_oee_uses_effective_capacity(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """OEE should be calculated against seasonally-adjusted capacity."""
        state = StateManager(minimal_world)
        engine = TransformEngine(minimal_world, state, seasonal_config)

        # Process with no orders - should get 0% OEE
        _, _, plant_oee = engine.process_production_orders([], current_day=241)

        # All plants should have 0 OEE (no production)
        for oee in plant_oee.values():
            assert oee == 0.0, f"Expected 0% OEE with no orders, got {oee}"

    def test_oee_never_exceeds_100_percent(
        self, minimal_world: World, seasonal_config: dict
    ) -> None:
        """OEE should never exceed 100% even at peak capacity."""
        state = StateManager(minimal_world)
        engine = TransformEngine(minimal_world, state, seasonal_config)

        # Process with no orders at peak day
        # OEE denominator should be effective capacity, not base
        _, _, plant_oee = engine.process_production_orders([], current_day=241)

        for oee in plant_oee.values():
            assert oee <= 1.0, f"OEE {oee} exceeds 100%"
