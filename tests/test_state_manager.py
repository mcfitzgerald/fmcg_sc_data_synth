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

    node_id = "RDC-NAM-CHI"
    prod_id = "SKU-SOAP-001"

    # Initial state should be 0
    assert state.get_inventory(node_id, prod_id) == 0.0

    # Update
    state.update_inventory(node_id, prod_id, 500.0)
    assert state.get_inventory(node_id, prod_id) == 500.0

    # Verify numpy array directly
    n_idx = state.get_node_idx(node_id)
    p_idx = state.get_product_idx(prod_id)
    assert state.inventory[n_idx, p_idx] == 500.0
