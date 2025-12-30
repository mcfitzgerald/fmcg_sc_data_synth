from typing import Any

import numpy as np

from prism_sim.network.core import Node, NodeType, Order, OrderLine, OrderType
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


class MinMaxReplenisher:
    """
    A simple replenishment agent that implements a (s, S) policy.
    Triggers the Bullwhip effect via order batching.
    Now supports Order Types (Standard, Rush, Promotional).
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        params = (
            config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )

        # Policy Parameters (Config driven)
        self.target_days = float(
            params.get("target_days_supply", 21.0) # Updated default
        )
        self.reorder_point_days = float(
            params.get("reorder_point_days", 10.0)
        )
        self.min_order_qty = float(
            params.get("min_order_qty", 10.0)
        )
        self.batch_size = float(
            params.get("batch_size_cases", 100.0)
        )

        # Optimization: Cache Store->Supplier map
        self.store_supplier_map = self._build_supplier_map()

    def _build_supplier_map(self) -> dict[str, str]:
        """Builds a lookup map for Store -> Source ID."""
        mapping = {}
        for link in self.world.links.values():
            mapping[link.target_id] = link.source_id
        return mapping

    def generate_orders(self, day: int, demand_history: np.ndarray) -> list[Order]:
        """
        Generates replenishment orders for Retail Stores and downstream DCs.
        """
        orders = []
        week = (day // 7) + 1
        
        # Identify active promos for this week
        active_promos = []
        promotions = self.config.get("promotions", [])
        for p in promotions:
            # Promo orders usually placed slightly ahead? 
            # Assume orders placed during promo week are FOR the promo demand.
            if p["start_week"] <= week <= p["end_week"]:
                active_promos.append(p)

        # 1. Identify Target Nodes (Stores + downstream DCs that order)
        # We need to replenish any node that consumes demand (Stores) OR acts as a buffer
        # In current logic, POSEngine generates demand for Stores.
        # Downstream DCs (if any) should also reorder?
        # For simplicity, we iterate all nodes that have links pointing to them.
        # But we filter by Type=STORE or Type=DC (if not RDC/Plant).
        
        target_indices = []
        target_ids = []
        
        # Valid targets are nodes that are destinations in links
        valid_targets = set(self.store_supplier_map.keys())
        
        for n_id in valid_targets:
            node = self.world.nodes.get(n_id)
            if node and (node.type == NodeType.STORE or (node.type == NodeType.DC and "RDC" not in node.id)):
                 idx = self.state.node_id_to_idx.get(n_id)
                 if idx is not None:
                     target_indices.append(idx)
                     target_ids.append(n_id)

        if not target_indices:
            return []

        target_idx_arr = np.array(target_indices)

        # 2. Get Inventory
        current_inv = self.state.inventory[target_idx_arr, :]
        # Shape: [N_Targets, Products]

        # 3. Get Demand Estimate
        avg_demand = demand_history[target_idx_arr, :]
        avg_demand = np.maximum(avg_demand, 0.1)

        # 4. Calculate Targets
        # Could customize per channel here if needed
        target_stock = avg_demand * self.target_days
        reorder_point = avg_demand * self.reorder_point_days

        # 5. Determine Order Quantities
        needs_order = current_inv < reorder_point
        raw_qty = target_stock - current_inv
        order_qty = np.where(needs_order, raw_qty, 0.0)

        # 6. Apply Batching
        batched_qty = np.ceil(order_qty / self.batch_size) * self.batch_size

        # 7. Create Orders
        rows, cols = np.nonzero(batched_qty)
        
        # Prepare Order Data
        orders_by_target: dict[int, dict] = {} # target_idx -> {lines: [], type: ...}

        for r, c in zip(rows, cols, strict=True):
            qty = batched_qty[r, c]
            if qty <= 0: continue

            t_idx = int(target_indices[r])
            p_idx = int(c)
            
            if t_idx not in orders_by_target:
                orders_by_target[t_idx] = {
                    "lines": [], 
                    "days_supply_min": 999.0,
                    "promo_id": None
                }
            
            p_id = self.state.product_idx_to_id[p_idx]
            orders_by_target[t_idx]["lines"].append(OrderLine(p_id, qty))
            
            # Check days supply for Rush classification
            d_supply = current_inv[r, c] / avg_demand[r, c]
            if d_supply < orders_by_target[t_idx]["days_supply_min"]:
                orders_by_target[t_idx]["days_supply_min"] = d_supply
            
            # Check Promo
            # If any active promo applies to this product and target channel, link it
            target_node = self.world.nodes.get(target_ids[r])
            if target_node and not orders_by_target[t_idx]["promo_id"]:
                 for promo in active_promos:
                      # Check categories
                      cat_match = "all" in promo.get("affected_categories", ["all"]) or \
                                  self.world.products[p_id].category.name in promo.get("affected_categories", [])
                      # Check channels
                      chan_match = not promo.get("affected_channels") or \
                                   (target_node.channel and target_node.channel.name in promo.get("affected_channels"))
                      
                      if cat_match and chan_match:
                          orders_by_target[t_idx]["promo_id"] = promo["code"]
                          break

        # Generate Objects
        order_count = 0
        for t_idx, data in orders_by_target.items():
            target_id = self.state.node_idx_to_id[t_idx]
            source_id = self.store_supplier_map.get(target_id)
            if not source_id: continue
            
            # Determine Order Type
            o_type = OrderType.STANDARD
            priority = 5
            
            if data["promo_id"]:
                o_type = OrderType.PROMOTIONAL
                priority = 2
            elif data["days_supply_min"] < 2.0: # Critical low stock
                o_type = OrderType.RUSH
                priority = 1
            
            order_count += 1
            orders.append(Order(
                id=f"ORD-{day}-{target_id}-{order_count}",
                source_id=source_id,
                target_id=target_id,
                creation_day=day,
                lines=data["lines"],
                order_type=o_type,
                promo_id=data["promo_id"],
                priority=priority
            ))

        return orders