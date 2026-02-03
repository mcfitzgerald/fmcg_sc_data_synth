"""DRP-Lite: Simplified Distribution Requirements Planning.

[v0.48.0] Adds forward-looking inventory netting to replace reactive
campaign batching for B/C items. Projects inventory forward at RDC level,
nets against in-transit and in-production, and generates time-phased
production requirements.

This addresses two structural root causes of the ~85% fill rate ceiling:
1. Campaign batching creates feast-famine oscillation (14-21 day cycles)
2. Multi-echelon independence (no coordination between echelons)

DRP replaces "Is DOS < trigger? Produce horizon_days worth" with
"What is my projected shortage? Produce exactly what's needed, when needed."
"""

import logging
from typing import Any

import numpy as np

from prism_sim.network.core import NodeType, ProductionOrderStatus
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World

logger = logging.getLogger(__name__)


class DRPPlanner:
    """Simplified Distribution Requirements Planning.

    Projects inventory forward at RDC level, nets against scheduled
    receipts (in-transit + in-production), and generates time-phased
    daily production targets for B/C items.

    A-items continue to use MRP's net-requirement scheduling (MPS-style)
    which is already near-continuous.
    """

    def __init__(
        self,
        world: World,
        state: StateManager,
        config: dict[str, Any],
        pos_engine: Any,
    ) -> None:
        self.world = world
        self.state = state
        self.config = config
        self.pos_engine = pos_engine

        # Extract config
        sim_params = config.get("simulation_parameters", {})
        mfg_config = sim_params.get("manufacturing", {})
        mrp_thresholds = mfg_config.get("mrp_thresholds", {})
        cal_config = sim_params.get("calibration", {})

        self.production_lead_time = int(
            mfg_config.get("production_lead_time_days", 3)
        )

        # DRP planning horizon (days forward to project)
        # Use the B-item production horizon as default
        campaign = mrp_thresholds.get("campaign_batching", {})
        self.planning_horizon = int(
            campaign.get("production_horizon_days_b", 14)
        )

        # Safety stock days by ABC class
        trigger_config = cal_config.get("trigger_components", {})
        self.safety_days_a = float(trigger_config.get("safety_buffer_a", 10))
        self.safety_days_b = float(trigger_config.get("safety_buffer_b", 6))
        self.safety_days_c = float(trigger_config.get("safety_buffer_c", 3))

        # Death spiral floor: minimum production as fraction of expected
        self.floor_pct = float(
            mrp_thresholds.get("production_floor_pct", 0.5)
        )

        # Collect node indices
        self._rdc_ids: list[str] = []
        self._plant_ids: list[str] = []
        self._finished_product_ids: list[str] = []

        for node_id, node in world.nodes.items():
            if node.type == NodeType.DC and node_id.startswith("RDC-"):
                self._rdc_ids.append(node_id)
            elif node.type == NodeType.PLANT:
                self._plant_ids.append(node_id)

        for p_id, product in world.products.items():
            if product.category != ProductCategory.INGREDIENT:
                self._finished_product_ids.append(p_id)

        # Build expected demand vector (same as MRP uses)
        self.expected_daily_demand = np.zeros(
            state.n_products, dtype=np.float64
        )
        if pos_engine is not None:
            base_matrix = pos_engine.get_base_demand_matrix()
            if base_matrix is not None:
                self.expected_daily_demand = np.sum(base_matrix, axis=0)

        # Pre-compute product indices for finished goods
        self._finished_p_indices = np.array(
            [
                state.product_id_to_idx[p_id]
                for p_id in self._finished_product_ids
                if p_id in state.product_id_to_idx
            ],
            dtype=np.int32,
        )

    def plan_requirements(
        self,
        day: int,
        abc_class: np.ndarray,
        active_production_orders: list[Any],
    ) -> np.ndarray:
        """Calculate daily production targets via forward inventory netting.

        For each product:
        1. Get deterministic forecast for planning horizon
        2. Get current RDC+Plant inventory position
        3. Add scheduled arrivals (in-transit + in-production)
        4. Project forward: find when inventory hits safety stock
        5. Net requirement = safety target - projected inventory
        6. Spread over lead time for level-loaded production

        Args:
            day: Current simulation day.
            abc_class: Per-product ABC classification (0=A, 1=B, 2=C).
            active_production_orders: Currently active production orders.

        Returns:
            [n_products] array of daily production targets.
        """
        n_products = self.state.n_products
        daily_target = np.zeros(n_products, dtype=np.float64)

        # 1. Get deterministic forecast for planning horizon
        if self.pos_engine is not None:
            forecast_total = self.pos_engine.get_deterministic_forecast(
                day, self.planning_horizon
            )
            daily_forecast = forecast_total / max(self.planning_horizon, 1)
        else:
            daily_forecast = self.expected_daily_demand.copy()

        # 2. Current inventory position (RDC + Plant on-hand)
        current_inv = np.zeros(n_products, dtype=np.float64)
        for rdc_id in self._rdc_ids:
            rdc_idx = self.state.node_id_to_idx.get(rdc_id)
            if rdc_idx is not None:
                current_inv += np.maximum(
                    0, self.state.actual_inventory[rdc_idx, :]
                )
        for plant_id in self._plant_ids:
            plant_idx = self.state.node_id_to_idx.get(plant_id)
            if plant_idx is not None:
                current_inv += np.maximum(
                    0, self.state.actual_inventory[plant_idx, :]
                )

        # 3. Scheduled arrivals: in-transit to RDCs/Plants + in-production
        scheduled = np.zeros(n_products, dtype=np.float64)

        # In-transit to RDCs
        rdc_id_set = set(self._rdc_ids)
        for shipment in self.state.active_shipments:
            if shipment.target_id in rdc_id_set:
                for line in shipment.lines:
                    p_idx = self.state.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        scheduled[p_idx] += line.quantity

        # In-production (not yet complete)
        for po in active_production_orders:
            if po.status != ProductionOrderStatus.COMPLETE:
                p_idx = self.state.product_id_to_idx.get(po.product_id)
                if p_idx is not None:
                    remaining = po.quantity_cases - po.produced_quantity
                    scheduled[p_idx] += max(0.0, remaining)

        # 4. Safety stock by ABC class
        safety_stock = np.zeros(n_products, dtype=np.float64)
        safe_forecast = np.maximum(daily_forecast, 0.1)
        safety_stock[abc_class == 0] = (
            safe_forecast[abc_class == 0] * self.safety_days_a
        )
        safety_stock[abc_class == 1] = (
            safe_forecast[abc_class == 1] * self.safety_days_b
        )
        abc_c_class = 2
        safety_stock[abc_class >= abc_c_class] = (
            safe_forecast[abc_class >= abc_c_class] * self.safety_days_c
        )

        # 5. Project inventory forward over planning horizon
        # Find the projected inventory at the end of the horizon
        # after consuming forecast demand and receiving scheduled arrivals
        projected_inv = current_inv + scheduled - (
            daily_forecast * self.planning_horizon
        )

        # 6. Net requirement: bring projected inventory up to 2x safety stock
        # (safety stock acts as both buffer and replenishment target)
        replenish_target = safety_stock * 2.0
        net_req = np.maximum(0.0, replenish_target - projected_inv)

        # 7. Spread production over lead time (level-load)
        spread_days = max(self.production_lead_time, 1)
        daily_target = net_req / spread_days

        # 8. Inventory-conditional floor (anti-windup death spiral prevention)
        # v0.51.0: Only apply floor when projected inventory is below safety
        # stock. When inventory is adequate, DRP's own net-requirement logic
        # correctly computes zero/low production — the floor should not override.
        floor_active = projected_inv < safety_stock  # boolean mask
        demand_floor = self.expected_daily_demand * self.floor_pct
        daily_target = np.where(
            floor_active,
            np.maximum(daily_target, demand_floor),
            daily_target,  # no floor when inventory is adequate
        )

        # Zero out non-finished-goods (ingredients, etc.)
        ingredient_mask = np.ones(n_products, dtype=bool)
        ingredient_mask[self._finished_p_indices] = False
        daily_target[ingredient_mask] = 0.0

        logger.debug(
            "DRP day %d: total_target=%.0f, avg_forecast=%.0f, "
            "avg_inv=%.0f, avg_scheduled=%.0f",
            day,
            float(np.sum(daily_target)),
            float(np.sum(daily_forecast)),
            float(np.sum(current_inv)),
            float(np.sum(scheduled)),
        )

        return daily_target
