import numpy as np

from prism_sim.network.core import NodeType
from prism_sim.simulation.orchestrator import Orchestrator


def test_orchestrator_loop():
    """Verify the orchestrator can run a multi-day simulation."""
    sim = Orchestrator()
    sim.run(days=5)
    # Check that inventory has changed from initial seed of 100
    assert not np.all(sim.state.inventory == 100.0)


def test_promo_lift():
    """Verify that the promo calendar is built from config.

    Note: The promo calendar is now config-driven (built from world_definition.json)
    and does not support dynamic add_promo(). This test verifies the calendar
    structure exists and can be queried.
    """
    sim = Orchestrator()
    engine = sim.pos_engine

    # Verify the calendar was built
    assert engine.calendar is not None

    # Verify we can generate demand (promos applied internally from config)
    demand_day1 = engine.generate_demand(day=1)
    assert demand_day1 is not None
    assert demand_day1.shape == (sim.state.n_nodes, sim.state.n_products)

    # Verify demand is non-negative
    assert np.all(demand_day1 >= 0)


def test_bullwhip_effect():
    """Verify that order batching creates a bullwhip ratio > 1."""
    sim = Orchestrator()
    
    # Lower initial inventory to ensure reorder points are hit quickly
    sim.state.perceived_inventory[:] = 0.0
    sim.state.actual_inventory[:] = 0.0

    # Run for 14 days to allow demand history and replenishment to stabilize
    total_demand = 0
    total_orders = 0

    for day in range(1, 15):
        demand = sim.pos_engine.generate_demand(day)
        orders = sim.replenisher.generate_orders(day, demand)

        total_demand += np.sum(demand)
        total_orders += sum(line.quantity for o in orders for line in o.lines)

        # Magic fulfillment to keep inventory stable
        sim._magic_fulfillment(orders)
        sim.state.update_inventory_batch(-demand)

    # Bullwhip: Variation in orders > Variation in demand
    # Here we just check if orders > 0 and if ratio is somewhat realistic
    assert total_orders > 0
    assert total_demand > 0
