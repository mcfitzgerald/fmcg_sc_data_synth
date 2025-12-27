from prism_sim.config.loader import load_manifest
from prism_sim.simulation.builder import WorldBuilder
from prism_sim.simulation.state import StateManager


def test_state_manager_initialization():
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    state = StateManager(world)

    # Check Dimensions
    n_nodes = len(world.nodes)
    n_products = len(world.products)

    assert state.inventory.shape == (n_nodes, n_products)
    assert state.n_nodes == n_nodes
    assert state.n_products == n_products


def test_state_operations():
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()
    state = StateManager(world)

    node_id = "RDC-MW"
    # Pick a valid product from the world keys to be safe, or use a known one from Deep NAM
    # SKU-ORAL-001 should exist as per generator logic (i=0)
    prod_id = "SKU-ORAL-001"

    # Initial state should be 0
    assert state.get_inventory(node_id, prod_id) == 0.0

    # Update Inventory
    state.update_inventory(node_id, prod_id, 100.0)
    assert state.get_inventory(node_id, prod_id) == 100.0
