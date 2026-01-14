"""
Tests for Multi-Line Manufacturing (v0.32.0).

Verifies that plants operate with multiple parallel production lines
instead of a single aggregate capacity pool.
"""

import pytest

from prism_sim.network.core import Node, NodeType, ProductionOrder, ProductionOrderStatus
from prism_sim.product.core import Product, ProductCategory, Recipe
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.transform import TransformEngine
from prism_sim.simulation.world import World


@pytest.fixture
def multi_line_world() -> World:
    """Create a world with one plant and two products."""
    world = World()

    # Add products
    p1 = Product("SKU-A", "Product A", ProductCategory.HOME_CARE, 1.0, 365, 12, 100, 10.0)
    p2 = Product("SKU-B", "Product B", ProductCategory.HOME_CARE, 1.0, 365, 12, 100, 10.0)
    p3 = Product("SKU-C", "Product C", ProductCategory.HOME_CARE, 1.0, 365, 12, 100, 10.0)
    world.add_product(p1)
    world.add_product(p2)
    world.add_product(p3)

    # Add plant
    plant = Node("PLANT-01", "Test Plant", NodeType.PLANT, "Location")
    world.add_node(plant)

    # Add recipes
    # Rate: 1000 cases/hr. Changeover: 1 hr.
    r1 = Recipe("SKU-A", {}, 1000.0, 1.0)
    r2 = Recipe("SKU-B", {}, 1000.0, 1.0)
    r3 = Recipe("SKU-C", {}, 1000.0, 1.0)
    world.add_recipe(r1)
    world.add_recipe(r2)
    world.add_recipe(r3)

    return world


@pytest.fixture
def line_config() -> dict:
    """Config with 2 lines per plant."""
    return {
        "simulation_parameters": {
            "global_constants": {"epsilon": 0.001},
            "manufacturing": {
                "production_hours_per_day": 10.0,  # 10 hours per day per line
                "efficiency_factor": 1.0,
                "unplanned_downtime_pct": 0.0,
                "production_rate_multiplier": 1.0,
                "changeover_time_multiplier": 1.0, # 1.0 multiplier for clear math
                "default_num_lines": 2,
                "plant_parameters": {
                    "PLANT-01": {
                        "num_lines": 2
                    }
                },
            },
            "demand": {"seasonality": {"capacity_amplitude": 0.0}},
            "agents": {"abc_prioritization": {"enabled": False}},
        }
    }


class TestTransformEngineLines:
    """Tests for line-level production logic."""

    def test_initialization_creates_lines(self, multi_line_world: World, line_config: dict):
        """Verify that plant state initializes with correct number of lines."""
        state = StateManager(multi_line_world)
        engine = TransformEngine(multi_line_world, state, line_config)

        plant_state = engine._plant_states["PLANT-01"]
        assert len(plant_state.lines) == 2
        assert plant_state.lines[0].line_id == "PLANT-01-L1"
        assert plant_state.lines[1].line_id == "PLANT-01-L2"
        
        # Check capacity
        # 10 hours * 1.0 eff * 1.0 uptime = 10.0 hours
        assert plant_state.lines[0].max_capacity_hours == 10.0
        assert plant_state.max_capacity_hours == 20.0  # Sum of 2 lines

    def test_parallel_assignment(self, multi_line_world: World, line_config: dict):
        """Verify that different products are assigned to different lines (parallelism)."""
        state = StateManager(multi_line_world)
        engine = TransformEngine(multi_line_world, state, line_config)
        
        # Create 2 orders for different products
        # 5000 cases @ 1000/hr = 5 hours
        o1 = ProductionOrder("PO-1", "PLANT-01", "SKU-A", 5000.0, 1, 1)
        o2 = ProductionOrder("PO-2", "PLANT-01", "SKU-B", 5000.0, 1, 1)
        
        orders = [o1, o2]
        
        # Process
        updated_orders, batches, _ = engine.process_production_orders(orders, current_day=1)
        
        plant_state = engine._plant_states["PLANT-01"]
        lines = plant_state.lines
        
        # Expectation:
        # Order 1 goes to Line 1 (or 2). Line usage: 5 hours.
        # Order 2 goes to the OTHER line because it has more capacity (10h vs 5h)
        # Result: Both orders complete fully.
        
        assert o1.status == ProductionOrderStatus.COMPLETE
        assert o2.status == ProductionOrderStatus.COMPLETE
        assert len(batches) == 2
        
        # Verify lines were used in parallel
        # One line used for SKU-A, one for SKU-B
        products_on_lines = {l.last_product_id for l in lines}
        assert "SKU-A" in products_on_lines
        assert "SKU-B" in products_on_lines
        
        # Verify capacity usage
        # Each line should have ~5 hours remaining (10 - 5)
        for line in lines:
            assert abs(line.remaining_capacity_hours - 5.0) < 0.001

    def test_sticky_assignment(self, multi_line_world: World, line_config: dict):
        """Verify that same product sticks to the same line to avoid changeover."""
        state = StateManager(multi_line_world)
        engine = TransformEngine(multi_line_world, state, line_config)
        
        plant_state = engine._plant_states["PLANT-01"]
        
        # Pre-condition: Line 1 ran SKU-A yesterday
        plant_state.lines[0].last_product_id = "SKU-A"
        plant_state.lines[0].remaining_capacity_hours = 10.0 # Reset for day
        
        # Line 2 ran SKU-B
        plant_state.lines[1].last_product_id = "SKU-B"
        plant_state.lines[1].remaining_capacity_hours = 10.0
        
        # New order for SKU-A
        o1 = ProductionOrder("PO-1", "PLANT-01", "SKU-A", 2000.0, 1, 1) # 2 hours
        
        updated_orders, batches, _ = engine.process_production_orders([o1], current_day=1)
        
        # Should pick Line 1 (SKU-A) even if capacities are equal
        assert plant_state.lines[0].remaining_capacity_hours == 8.0 # 10 - 2
        assert plant_state.lines[1].remaining_capacity_hours == 10.0 # Untouched

    def test_changeover_isolation(self, multi_line_world: World, line_config: dict):
        """Verify changeover on one line doesn't affect the other."""
        state = StateManager(multi_line_world)
        engine = TransformEngine(multi_line_world, state, line_config)
        
        plant_state = engine._plant_states["PLANT-01"]
        
        # Pre-condition: Both lines ran SKU-A yesterday
        plant_state.lines[0].last_product_id = "SKU-A"
        plant_state.lines[1].last_product_id = "SKU-A"
        
        # Order for SKU-B (Changeover needed)
        # 1000 cases = 1 hr run + 1 hr changeover = 2 hrs total
        o1 = ProductionOrder("PO-1", "PLANT-01", "SKU-B", 1000.0, 1, 1)
        
        engine.process_production_orders([o1], current_day=1)
        
        # One line should have taken the hit (2 hours used)
        # The other should be full (10 hours)
        
        capacities = [l.remaining_capacity_hours for l in plant_state.lines]
        capacities.sort()
        
        assert abs(capacities[0] - 8.0) < 0.001 # Used 2 hours
        assert abs(capacities[1] - 10.0) < 0.001 # Used 0 hours

    def test_oee_calculation(self, multi_line_world: World, line_config: dict):
        """Verify OEE aggregates correctly across lines."""
        state = StateManager(multi_line_world)
        engine = TransformEngine(multi_line_world, state, line_config)
        
        # Order 1: SKU-A, 4000 cases (4 hrs). Line 1.
        o1 = ProductionOrder("PO-1", "PLANT-01", "SKU-A", 4000.0, 1, 1)
        
        # Order 2: SKU-B, 4000 cases (4 hrs). Line 2.
        o2 = ProductionOrder("PO-2", "PLANT-01", "SKU-B", 4000.0, 1, 1)
        
        _, _, plant_oee = engine.process_production_orders([o1, o2], current_day=1)
        
        # Line 1: 4h run / 4h scheduled = 100% Availability
        # Line 2: 4h run / 4h scheduled = 100% Availability
        # Plant OEE = 1.0 (assuming 100% yield/performance)
        # Note: If we include changeover, OEE drops.
        
        # Let's verify via the return value
        # OEE = (Avail * Perf * Qual)
        # Default yield is 98.5% -> 0.985
        
        expected_oee = 1.0 * 1.0 * 0.985
        assert abs(plant_oee["PLANT-01"] - expected_oee) < 0.001
        
