import pytest

from prism_sim.agents.allocation import AllocationAgent
from prism_sim.network.core import (
    Link,
    Node,
    NodeType,
    Order,
    OrderLine,
    ShipmentStatus,
)
from prism_sim.product.core import Product, ProductCategory
from prism_sim.simulation.logistics import LogisticsEngine
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


@pytest.fixture
def minimal_world():
    world = World()

    # 1. Products
    # Soap: Heavy (10kg), Small (0.01 m3) -> Weighs out
    soap = Product(
        "SOAP-001", "Soap", ProductCategory.PERSONAL_WASH, 10.0, 20, 20, 25, 100
    )
    world.add_product(soap)

    # Tissue: Light (1kg), Big (0.1 m3) -> Cubes out
    tissue = Product(
        "TISS-001", "Tissue", ProductCategory.HOME_CARE, 1.0, 50, 50, 40, 50
    )
    world.add_product(tissue)

    # 2. Nodes
    dc = Node("DC-01", "Distribution Center", NodeType.DC, "LocA")
    store = Node("STORE-01", "Retail Store", NodeType.STORE, "LocB")
    world.add_node(dc)
    world.add_node(store)

    # 3. Link
    link = Link("L-01", "DC-01", "STORE-01", lead_time_days=2.0)
    world.add_link(link)

    return world


def test_allocation_shortage(minimal_world):
    state = StateManager(minimal_world)
    config = {
        "simulation_parameters": {
            "global_constants": {
                "epsilon": 0.001
            }
        }
    }
    allocator = AllocationAgent(state, config)

    # Setup Inventory: 100 Soap (set both perceived and actual)
    dc_idx = state.node_id_to_idx["DC-01"]
    soap_idx = state.product_id_to_idx["SOAP-001"]
    state.perceived_inventory[dc_idx, soap_idx] = 100.0
    state.actual_inventory[dc_idx, soap_idx] = 100.0

    # Setup Orders: 2 Stores ordering 80 each (Total 160)
    # Note: Using same store ID just for simplicity of order generation logic check
    orders = [
        Order("O1", "DC-01", "STORE-01", 1, [OrderLine("SOAP-001", 80.0)]),
        Order("O2", "DC-01", "STORE-01", 1, [OrderLine("SOAP-001", 80.0)]),
    ]

    # Run Allocation
    result = allocator.allocate_orders(orders)
    allocated = result.allocated_orders

    # Check Fair Share
    # Total Demand 160, Avail 100. Ratio = 0.625
    expected_qty = 80.0 * (100.0 / 160.0)  # 50.0

    assert len(allocated) == 2
    assert allocated[0].lines[0].quantity == pytest.approx(expected_qty)
    assert allocated[1].lines[0].quantity == pytest.approx(expected_qty)

    # Verify allocation_matrix tracks what was decremented
    assert result.allocation_matrix[dc_idx, soap_idx] == pytest.approx(100.0)

    # Check Inventory Decrement
    assert state.inventory[dc_idx, soap_idx] == pytest.approx(0.0)


def test_tetris_weight_out(minimal_world):
    state = StateManager(minimal_world)
    logistics = LogisticsEngine(minimal_world, state, {})

    # Order: 2500 units of Soap (10kg each) = 25,000 kg
    # Limit: 20,000 kg
    orders = [Order("O1", "DC-01", "STORE-01", 1, [OrderLine("SOAP-001", 2500.0)])]

    shipments = logistics.create_shipments(orders, current_day=1)

    # Expect 2 shipments
    # Shipment 1: 2000 units (20,000 kg)
    # Shipment 2: 500 units (5,000 kg)

    assert len(shipments) == 2
    s1 = shipments[0]
    s2 = shipments[1]

    assert s1.total_weight_kg == pytest.approx(20000.0)
    assert s1.lines[0].quantity == 2000

    assert s2.total_weight_kg == pytest.approx(5000.0)
    assert s2.lines[0].quantity == 500


def test_tetris_cube_out(minimal_world):
    state = StateManager(minimal_world)
    logistics = LogisticsEngine(minimal_world, state, {})

    # Order: 700 units of Tissue (0.1 m3 each) = 70 m3
    # Limit: 60 m3
    orders = [Order("O1", "DC-01", "STORE-01", 1, [OrderLine("TISS-001", 700.0)])]

    shipments = logistics.create_shipments(orders, current_day=1)

    # Expect 2 shipments
    # Shipment 1: 600 units (60 m3)
    # Shipment 2: 100 units (10 m3)

    assert len(shipments) == 2
    assert shipments[0].total_volume_m3 == pytest.approx(60.0, rel=1e-3)
    assert shipments[0].lines[0].quantity == 600


def test_transit_physics(minimal_world):
    state = StateManager(minimal_world)
    # v0.15.5: Use LTL mode for store deliveries to avoid FTL minimum constraints
    logistics_config = {
        "simulation_parameters": {
            "logistics": {
                "store_delivery_mode": "LTL",
                "ltl_min_cases": 1.0,
            }
        }
    }
    logistics = LogisticsEngine(minimal_world, state, logistics_config)

    # Link Lead Time is 2 days
    # Day 1: Create Shipment
    orders = [Order("O1", "DC-01", "STORE-01", 1, [OrderLine("SOAP-001", 10.0)])]
    shipments = logistics.create_shipments(orders, current_day=1)

    assert len(shipments) == 1
    assert shipments[0].arrival_day == 3  # 1 + 2

    # Day 1 Check
    active, arrived = logistics.update_shipments(shipments, current_day=1)
    assert len(active) == 1
    assert len(arrived) == 0

    # Day 2 Check
    active, arrived = logistics.update_shipments(shipments, current_day=2)
    assert len(active) == 1
    assert len(arrived) == 0

    # Day 3 Check (Arrival)
    active, arrived = logistics.update_shipments(shipments, current_day=3)
    assert len(active) == 0
    assert len(arrived) == 1
    assert arrived[0].status == ShipmentStatus.DELIVERED
