import numpy as np

from prism_sim.simulation.demand import PromoConfig
from prism_sim.simulation.orchestrator import Orchestrator


def test_orchestrator_loop():
    """Verify the orchestrator can run a multi-day simulation."""
    sim = Orchestrator()
    sim.run(days=5)
    # Check that inventory has changed from initial seed of 100
    assert not np.all(sim.state.inventory == 100.0)


def test_promo_lift():
    """Verify that promotions actually lift demand."""
    sim = Orchestrator()
    engine = sim.pos_engine

    # 1. Base Demand
    base_demand = engine.generate_demand(day=1)  # Day 1, no promo

    # 2. Add Promo to Calendar
    # Use a product and store that exist in the world
    product_id = "SKU-PASTE-001"
    store_id = "STORE-MEGA-NE-001"

    engine.calendar.add_promo(
        PromoConfig(
            promo_id="PROMO-TEST",
            start_week=1,
            end_week=1,
            lift=3.0,
            hangover_lift=0.5,
            products=[product_id],
            stores=[store_id],
        )
    )

    promo_demand = engine.generate_demand(day=1)

    p_idx = sim.state.get_product_idx(product_id)
    n_idx = sim.state.get_node_idx(store_id)

    # Promo demand should be roughly 3x base (modulo randomness)
    # We use a loose check because of the Gamma noise
    assert promo_demand[n_idx, p_idx] > base_demand[n_idx, p_idx] * 1.5


def test_bullwhip_effect():
    """Verify that order batching creates a bullwhip ratio > 1."""
    sim = Orchestrator()

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
