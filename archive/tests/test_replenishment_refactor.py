import pytest
import numpy as np
from typing import Any

from prism_sim.agents.replenishment import MinMaxReplenisher
from prism_sim.network.core import (
    Node,
    NodeType,
    Link,
    StoreFormat,
    CustomerChannel,
    OrderType,
)
from prism_sim.product.core import Product, ProductCategory
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World

@pytest.fixture
def repl_world() -> World:
    world = World()
    
    # 1. Product (Soap)
    soap = Product("SOAP-001", "Soap", ProductCategory.PERSONAL_WASH, 10.0, 0.01, 20, 25, 100)
    world.add_product(soap)

    # 2. Nodes
    # Plant -> RDC -> Customer DC -> Store
    
    # RDC (Manufacturer)
    rdc = Node("RDC-01", "Regional DC", NodeType.DC, "LocA")
    world.add_node(rdc)

    # Customer DC (e.g. Walmart DC)
    cust_dc = Node("DC-WAL-01", "Walmart DC", NodeType.DC, "LocB", 
                   channel=CustomerChannel.B2M_LARGE, 
                   store_format=StoreFormat.RETAILER_DC)
    world.add_node(cust_dc)

    # Store (e.g. Walmart Store)
    store = Node("STORE-WAL-01", "Walmart Store", NodeType.STORE, "LocC", 
                 channel=CustomerChannel.B2M_LARGE, 
                 store_format=StoreFormat.SUPERMARKET)
    world.add_node(store)

    # 3. Links
    # RDC -> Cust DC
    link1 = Link("L1", "RDC-01", "DC-WAL-01", lead_time_days=2.0)
    world.add_link(link1)

    # Cust DC -> Store
    link2 = Link("L2", "DC-WAL-01", "STORE-WAL-01", lead_time_days=1.0)
    world.add_link(link2)

    return world

@pytest.fixture
def repl_config() -> dict[str, Any]:
    return {
        "simulation_parameters": {
            "agents": {
                "replenishment": {
                    "target_days_supply": 10.0,
                    "reorder_point_days": 4.0,
                    "min_order_qty": 50.0,
                    "batch_size_cases": 100.0,
                    "min_history_days": 0, # Disable cold start check for test
                    "echelon_safety_multiplier": 1.0, # Simplify math
                    "store_batch_size_cases": 10.0, # Small batch for store
                    "order_cycle_days": 1, # Force daily ordering
                    "channel_profiles": {
                        "B2M_LARGE": {
                            "target_days": 21.0,
                            "reorder_point_days": 14.0,
                            "batch_size": 10.0, # Small batch to match exact calcs better
                            "smoothing_factor": 0.3
                        }
                    }
                }
            }
        }
    }

def test_store_replenishment_logic(repl_world: World, repl_config: dict[str, Any]) -> None:
    state = StateManager(repl_world)
    agent = MinMaxReplenisher(repl_world, state, repl_config)

    # Setup
    store_idx = state.node_id_to_idx["STORE-WAL-01"]
    soap_idx = state.product_id_to_idx["SOAP-001"]

    # 1. Demand Signal (Smoothed POS)
    # 7-day average of 10 units/day
    demand_signal = np.zeros((state.n_nodes, state.n_products))
    demand_signal[store_idx, soap_idx] = 10.0 
    
    # Mock inflow/POS demand in agent to avoid cold start blending issues
    agent.smoothed_demand = demand_signal.copy()
    agent.inflow_history = np.zeros((7, state.n_nodes, state.n_products)) + demand_signal # Fill history

    # FIX: Populate demand history for variance calculation
    agent.history_idx = 10
    agent.demand_history_buffer[:10] = demand_signal # Constant demand -> 0 variance

    # 2. Inventory Position
    # Target = 14 days (default channel policy for B2M_LARGE is likely overriding our config? 
    # Let's check agent's vectors after init)
    target_days = agent.target_days_vec[store_idx, 0] # Should be 10.0 from config override or 21.0 from default channel?
    # Actually DEFAULT_CHANNEL_POLICIES in code has B2M_LARGE target=21.0. 
    # But let's see what logic prevails.
    
    # If Avg Demand = 10.
    # Target Stock = 10 * TargetDays.
    
    # Set Inventory to 0 to force order
    state.inventory[store_idx, soap_idx] = 0.0

    # Debug: Check ingredient mask
    assert not agent.ingredient_mask[soap_idx], "Soap should not be an ingredient"

    # Run
    orders = agent.generate_orders(day=10, demand_signal=demand_signal)

    # Verify
    # Should order for Store -> DC-WAL-01
    store_orders = [o for o in orders if o.target_id == "STORE-WAL-01"]
    assert len(store_orders) == 1
    
    order = store_orders[0]
    assert order.source_id == "DC-WAL-01"
    assert len(order.lines) == 1
    assert order.lines[0].product_id == "SOAP-001"
    
    qty = order.lines[0].quantity
    # Expected: Target ~ 21 days * 10 = 210. 
    # IP = 0. Order = 210.
    # Batch size = 10 (store_batch_size_cases).
    # So 210 is valid.
    assert qty > 100.0


def test_customer_dc_echelon_logic(repl_world: World, repl_config: dict[str, Any]) -> None:
    # Test that Customer DC orders from RDC based on DOWNSTREAM demand (Store)
    state = StateManager(repl_world)
    agent = MinMaxReplenisher(repl_world, state, repl_config)

    dc_idx = state.node_id_to_idx["DC-WAL-01"]
    store_idx = state.node_id_to_idx["STORE-WAL-01"]
    soap_idx = state.product_id_to_idx["SOAP-001"]

    # 1. Setup Demand
    # Store has 10 units/day demand
    demand_signal = np.zeros((state.n_nodes, state.n_products))
    demand_signal[store_idx, soap_idx] = 10.0
    
    agent.smoothed_demand = demand_signal.copy() # Used for Echelon calculation

    # 2. Setup Inventory
    # Store is full (doesn't need order) -> IP = Target
    # DC is empty -> IP = 0
    state.inventory[store_idx, soap_idx] = 200.0 
    state.inventory[dc_idx, soap_idx] = 0.0

    # 3. Verify Echelon Matrix Setup
    # DC should have row in echelon matrix
    assert agent.echelon_matrix is not None
    assert dc_idx in agent.dc_idx_to_echelon_row
    
    # Run
    # DC should order because its LOCAL inventory (0) is less than Echelon Target needed to cover Store demand
    orders = agent.generate_orders(day=10, demand_signal=demand_signal)

    dc_orders = [o for o in orders if o.target_id == "DC-WAL-01"]
    assert len(dc_orders) == 1
    
    order = dc_orders[0]
    assert order.source_id == "RDC-01"
    
    # Verify logic:
    # Echelon Demand = 10 (Store) + 0 (DC) = 10.
    # DC Target Days = 21 (B2M_LARGE default).
    # Echelon Target = 10 * 21 * 1.0 (safety) = 210.
    # Local IP = 0.
    # Order = 210.
    assert order.lines[0].quantity == pytest.approx(210.0, abs=20.0) # approx due to batching

