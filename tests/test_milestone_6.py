import numpy as np

from prism_sim.network.core import Shipment, ShipmentStatus
from prism_sim.simulation.monitor import RealismMonitor, WelfordAccumulator
from prism_sim.simulation.orchestrator import Orchestrator
from prism_sim.simulation.quirks import QuirkManager, ShrinkageEvent


def test_welford_accumulator():
    """Test O(1) stats calculation."""
    acc = WelfordAccumulator()
    data = [10, 20, 30, 40, 50]
    for x in data:
        acc.update(x)

    assert acc.mean == 30.0
    assert np.isclose(acc.variance, 250.0)
    assert np.isclose(acc.std_dev, np.sqrt(250.0))

def test_realism_monitor():
    """Test metric tracking."""
    monitor = RealismMonitor({"validation": {"oee_range": [0.65, 0.85]}})
    monitor.record_oee(0.75)
    monitor.record_truck_fill(0.90)

    report = monitor.get_report()
    assert report["oee"]["mean"] == 0.75
    assert report["oee"]["status"] == "OK"
    assert report["truck_fill"]["mean"] == 0.90

def test_quirks_integration():
    """Test that quirks are initialized and integrated."""
    sim = Orchestrator()

    # Enable quirks in config for testing
    sim.config["simulation_parameters"]["quirks"] = {
        "port_congestion": {
            "enabled": True,
            "ar_coefficient": 0.9,
            "noise_std_hours": 0.0,
            "cluster_threshold_hours": 100.0,
            "affected_ports": ["A", "B"]
        },
        "optimism_bias": {
            "enabled": True, "bias_pct": 0.50, "affected_age_months": 6
        },
        "phantom_inventory": {
            "enabled": True, "shrinkage_pct": 0.02, "detection_lag_days": 1
        }
    }
    # Re-init managers with new config
    sim.quirks = QuirkManager(config=sim.config)

    # 1. Optimism Bias Test
    raw_demand = sim.pos_engine.generate_demand(1)
    product_ids = [sim.state.product_idx_to_id[i] for i in range(sim.state.n_products)]

    # Register launches for all products to ensure they are "new"
    for p_id in product_ids:
        sim.quirks.optimism_bias.register_product_launch(p_id, 0)

    biased_demand = sim.quirks.apply_optimism_bias(raw_demand, product_ids, 1)

    assert np.allclose(biased_demand, raw_demand * 1.5)

    # 2. Port Congestion Test
    shipment = Shipment(
        id="TEST-001",
        source_id="A", target_id="B",
        creation_day=1, arrival_day=4,
        lines=[], status=ShipmentStatus.IN_TRANSIT
    )
    # Mock previous delay for port "A"
    sim.quirks.port_congestion._prev_delay_by_port["A"] = 10.0 # hours

    # apply_port_congestion modifies in place
    sim.quirks.apply_port_congestion([shipment])

    # Next delay = 0.9 * 10 + 0 = 9 hours = 0.375 days
    # Since it rounds up: ceil(9/24) = 1 day
    assert shipment.arrival_day == 5 # 4 + 1

def test_phantom_inventory():
    """Test dual inventory tracking."""
    sim = Orchestrator()
    # Force shrinkage settings
    sim.config["simulation_parameters"]["quirks"]["phantom_inventory"] = {
        "enabled": True, "shrinkage_pct": 0.02, "detection_lag_days": 1
    }

    # Init quirks
    sim.quirks = QuirkManager(config=sim.config)

    # Set initial inventory
    sim.state.perceived_inventory[:] = 100.0
    sim.state.actual_inventory[:] = 100.0

    # Let's force a shrinkage event manually to test logic
    n_idx, p_idx = 0, 0
    node_id = sim.state.node_idx_to_id[n_idx]
    prod_id = sim.state.product_idx_to_id[p_idx]

    # Simulate shrinkage on Day 1
    sim.state.actual_inventory[n_idx, p_idx] -= 5.0
    event = ShrinkageEvent(
        day_occurred=1,
        day_discovered=2,
        node_id=node_id,
        product_id=prod_id,
        quantity_lost=5.0
    )
    sim.quirks.phantom_inventory._pending_discoveries[2] = [event]

    assert sim.state.perceived_inventory[n_idx, p_idx] == 100.0
    assert sim.state.actual_inventory[n_idx, p_idx] == 95.0

    # Process discovery on day 2
    sim.quirks.process_discoveries(sim.state, 2)

    assert sim.state.perceived_inventory[n_idx, p_idx] == 95.0

def test_risk_events():
    """Test risk trigger logic."""
    sim = Orchestrator()
    sim.config["simulation_parameters"]["risk_events"] = {
        "enabled": True,
        "events": [
            {
                "code": "TEST-RISK",
                "type": "port_strike",
                "trigger_day": 5,
                "duration_days": 2,
                "parameters": {"delay_multiplier": 4.0}
            }
        ]
    }
    from prism_sim.simulation.risk_events import RiskEventManager
    sim.risks = RiskEventManager(sim.config["simulation_parameters"])

    # Run to day 4 -> No risk
    sim.risks.check_triggers(4)
    assert len(sim.risks.active_events) == 0
    assert sim.risks.get_logistics_delay_multiplier() == 1.0

    # Run to day 5 -> Risk triggered
    sim.risks.check_triggers(5)
    assert len(sim.risks.active_events) == 1
    assert sim.risks.active_events[0].event_code == "TEST-RISK"
    assert sim.risks.get_logistics_delay_multiplier() == 4.0

    # Run to day 7 -> Risk recovered (5 + 2 = 7)
    sim.risks.check_recovery(7)
    assert len(sim.risks.active_events) == 0

def test_mass_balance_conservation():
    """Verify no mass balance violations during normal operation."""
    sim = Orchestrator()
    # Disable quirks to simplify balance check (though auditor handles shrinkage now)
    sim.config["simulation_parameters"]["quirks"] = {"enabled": False}

    # Run simulation for a few days
    sim.run(days=5)

    # Check no violations accumulated
    assert len(sim.auditor.all_violations) == 0

def test_mass_balance_detects_leak():
    """Verify mass balance detects artificial inventory leak."""
    sim = Orchestrator()

    # Manually start a day
    sim.auditor.start_day(1)

    # Find a location/product with small inventory to ensure drift > 2%
    # Use a store (not a plant with 10M inventory) for reliable detection
    # First, find a store node index
    store_idx = None
    for node_id, idx in sim.state.node_id_to_idx.items():
        if node_id.startswith("STORE-"):
            store_idx = idx
            break

    if store_idx is None:
        # Fallback: just use first node but inject much larger amount
        store_idx = 0

    # Get opening inventory for this cell
    opening = sim.state.actual_inventory[store_idx, 0]

    # Inject leak > 2% of opening (or at least 100 units if opening is small)
    # This violates Conservation Law: Closing > Opening but no Inflow recorded
    leak_amount = max(opening * 0.10, 100.0)  # 10% leak or 100 units
    sim.state.actual_inventory[store_idx, 0] += leak_amount

    # End day
    sim.auditor.end_day()

    # Check
    violations = sim.auditor.check_mass_balance()

    assert len(violations) > 0, f"Expected violations: open={opening}, leak={leak_amount}"
    assert "Drift" in violations[0]

def test_full_step_integration():
    """Run a full step to ensure no crashes with new logic."""
    sim = Orchestrator()
    # Small run to ensure nothing blows up
    sim.run(days=2)
