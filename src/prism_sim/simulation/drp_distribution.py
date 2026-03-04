"""DRP Distribution Engine — proactive RDC→DC distribution based on need.

v0.89.0: Replaces push-based allocation (``_push_excess_rdc_inventory``) with a
coordinated Distribution Requirements Planning pass.  For each RDC, the engine:

1. Computes per-DC need = ABC-differentiated target inventory - inventory position
2. Aggregates needs, applies fair-share allocation against RDC available stock
3. Creates DRP shipments, deducting RDC inventory with FIFO age reduction

ECOM-FC / DTC-FC nodes (DC-type, zero downstream stores) use their own
``base_demand_matrix`` row as the demand signal instead of aggregated store POS.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from prism_sim.network.core import (
    Link,
    OrderLine,
    Shipment,
    ShipmentStatus,
)
from prism_sim.simulation.logistics import LogisticsEngine

if TYPE_CHECKING:
    from prism_sim.simulation.demand import POSEngine
    from prism_sim.simulation.mrp import MRPEngine
    from prism_sim.simulation.state import StateManager
    from prism_sim.simulation.world import World

logger = logging.getLogger(__name__)


class DRPDistributionEngine:
    """Proactive RDC→DC distribution engine (replaces push at step 10a).

    Parameters
    ----------
    world : World
        Immutable world graph (nodes, links, products).
    state : StateManager
        Mutable state tensors (inventory, in-transit, age).
    config : dict
        Full simulation config.
    pos_engine : POSEngine
        For ``get_base_demand_matrix()`` (stable POS demand signal).
    mrp_engine : MRPEngine
        For ``_get_seasonal_factor()`` and ``abc_class``.
    rdc_downstream_dcs : dict[str, list[str]]
        Primary RDC → [DC] topology map (pre-built by orchestrator).
    dc_downstream_stores : dict[str, list[str]]
        DC → [Store] topology map (pre-built by orchestrator).
    dc_secondary_rdc : dict[str, str]
        DC → secondary RDC (multi-source DCs).
    rdc_secondary_dcs : dict[str, list[str]]
        RDC → [secondary DCs] (DCs using this RDC as secondary source).
    secondary_order_fraction : float
        Fraction of demand routed to secondary RDC (0.0-1.0).
    route_map : dict[tuple[str, str], Link]
        (source_id, target_id) → Link for lead-time lookup.
    """

    def __init__(  # noqa: PLR0913
        self,
        world: World,
        state: StateManager,
        config: dict[str, Any],
        pos_engine: POSEngine,
        mrp_engine: MRPEngine,
        rdc_downstream_dcs: dict[str, list[str]],
        dc_downstream_stores: dict[str, list[str]],
        dc_secondary_rdc: dict[str, str],
        rdc_secondary_dcs: dict[str, list[str]],
        secondary_order_fraction: float,
        route_map: dict[tuple[str, str], Link],
    ) -> None:
        self._world = world
        self._state = state
        self._pos_engine = pos_engine
        self._mrp_engine = mrp_engine
        self._rdc_downstream_dcs = rdc_downstream_dcs
        self._dc_downstream_stores = dc_downstream_stores
        self._dc_secondary_rdc = dc_secondary_rdc
        self._rdc_secondary_dcs = rdc_secondary_dcs
        self._sec_frac = secondary_order_fraction
        self._route_map = route_map

        # --- Config ---
        sim_params = config.get("simulation_parameters", {})
        replen_params = sim_params.get("agents", {}).get("replenishment", {})
        drp_cfg = replen_params.get("drp_distribution", {})
        self._rdc_safety_dos = float(drp_cfg.get("rdc_safety_dos", 3.0))
        self._min_ship_cases = float(drp_cfg.get("min_ship_cases", 10.0))
        self._default_lead_time = float(
            sim_params.get("logistics", {}).get("default_lead_time_days", 3.0)
        )

        n_p = state.n_products

        # --- ABC-differentiated DC target DOS vector (same as deployment) ---
        dc_buffer = float(replen_params.get("dc_buffer_days", 7.0))
        self._dc_target_dos_vec = np.full(n_p, dc_buffer * 2.5, dtype=np.float64)  # C
        abc = mrp_engine.abc_class
        self._dc_target_dos_vec[abc == 0] = dc_buffer * 1.5  # A → 10.5
        self._dc_target_dos_vec[abc == 1] = dc_buffer * 2.0  # B → 14.0

        # --- Pre-compute per-DC expected demand (base, no seasonality) ---
        base_demand = pos_engine.get_base_demand_matrix()  # [n_nodes, n_products]
        self._dc_expected_demand: dict[str, np.ndarray] = {}

        # Collect ALL DCs served by any RDC (primary + secondary)
        all_dc_ids: set[str] = set()
        for dcs in rdc_downstream_dcs.values():
            all_dc_ids.update(dcs)
        for dcs in rdc_secondary_dcs.values():
            all_dc_ids.update(dcs)

        for dc_id in all_dc_ids:
            stores = dc_downstream_stores.get(dc_id, [])
            if stores:
                # Normal DC: aggregate downstream store POS
                demand = np.zeros(n_p, dtype=np.float64)
                for store_id in stores:
                    s_idx = state.node_id_to_idx.get(store_id)
                    if s_idx is not None:
                        demand += base_demand[s_idx, :]
            else:
                # ECOM-FC / DTC-FC: use own base_demand row
                fc_idx = state.node_id_to_idx.get(dc_id)
                if fc_idx is not None:
                    demand = base_demand[fc_idx, :].copy()
                else:
                    demand = np.zeros(n_p, dtype=np.float64)
            self._dc_expected_demand[dc_id] = demand

        logger.info(
            "DRP Distribution Engine initialised: %d RDCs, %d DCs (incl ECOM/DTC FCs), "
            "rdc_safety_dos=%.1f, min_ship_cases=%.0f",
            len(rdc_downstream_dcs),
            len(all_dc_ids),
            self._rdc_safety_dos,
            self._min_ship_cases,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rdc_throughput_demand(self, rdc_id: str) -> np.ndarray:
        """Return base daily demand (no seasonal) this RDC must serve.

        Sum of ``dc_expected_demand`` for primary DCs (with secondary fraction
        deduction) plus secondary DCs (with secondary fraction applied).
        Used by deployment to right-size Plant->RDC flow.
        """
        n_p = self._state.n_products
        demand = np.zeros(n_p, dtype=np.float64)
        sec_frac = self._sec_frac

        for dc_id in self._rdc_downstream_dcs.get(rdc_id, []):
            d = self._dc_expected_demand.get(dc_id, np.zeros(n_p))
            if sec_frac > 0 and dc_id in self._dc_secondary_rdc:
                demand += d * (1.0 - sec_frac)
            else:
                demand += d

        for sec_dc_id in self._rdc_secondary_dcs.get(rdc_id, []):
            demand += (
                self._dc_expected_demand.get(sec_dc_id, np.zeros(n_p))
                * sec_frac
            )

        return demand

    def compute_and_execute(self, day: int) -> list[Shipment]:
        """Run DRP distribution for all RDCs on *day*.

        Returns list of DRP shipments (already deducted from RDC inventory,
        ready for ``add_shipments_batch``).
        """
        shipments: list[Shipment] = []
        state = self._state
        n_p = state.n_products
        seasonal = self._mrp_engine._get_seasonal_factor(day)
        in_transit = state.get_in_transit_by_target()  # [n_nodes, n_products]
        min_ship = self._min_ship_cases
        dc_target_dos = self._dc_target_dos_vec
        sec_frac = self._sec_frac
        shipment_counter = 0

        for rdc_id, primary_dcs in self._rdc_downstream_dcs.items():
            rdc_idx = state.node_id_to_idx.get(rdc_id)
            if rdc_idx is None:
                continue

            rdc_on_hand = state.actual_inventory[rdc_idx, :]

            # --- Aggregate DC needs ---
            total_need = np.zeros(n_p, dtype=np.float64)
            dc_needs: dict[str, np.ndarray] = {}

            # Primary DCs
            for dc_id in primary_dcs:
                need = self._compute_dc_need(
                    dc_id, seasonal, in_transit, dc_target_dos,
                    is_primary=True, sec_frac=sec_frac,
                )
                dc_needs[dc_id] = need
                total_need += need

            # Secondary DCs (this RDC is their secondary source)
            for sec_dc_id in self._rdc_secondary_dcs.get(rdc_id, []):
                need = self._compute_dc_need(
                    sec_dc_id, seasonal, in_transit, dc_target_dos,
                    is_primary=False, sec_frac=sec_frac,
                )
                dc_needs[sec_dc_id] = need
                total_need += need

            if float(np.sum(total_need)) < 1.0:
                continue

            # --- RDC available = on_hand - safety reserve ---
            rdc_demand = np.zeros(n_p, dtype=np.float64)
            for dc_id in primary_dcs:
                d = self._dc_expected_demand.get(dc_id, np.zeros(n_p))
                if sec_frac > 0 and dc_id in self._dc_secondary_rdc:
                    rdc_demand += d * (1.0 - sec_frac)
                else:
                    rdc_demand += d
            for sec_dc_id in self._rdc_secondary_dcs.get(rdc_id, []):
                rdc_demand += self._dc_expected_demand.get(
                    sec_dc_id, np.zeros(n_p)
                ) * sec_frac
            rdc_demand *= seasonal

            rdc_safety = self._rdc_safety_dos * rdc_demand
            available = np.maximum(0.0, rdc_on_hand - rdc_safety)

            # --- Fair-share fill ratio (per-product, vectorized) ---
            with np.errstate(divide="ignore", invalid="ignore"):
                fill_ratio = np.where(
                    total_need > 0,
                    np.minimum(available / total_need, 1.0),
                    0.0,
                )

            # --- Create shipments per DC ---
            for dc_id, need in dc_needs.items():
                ship_qty = need * fill_ratio
                ship_qty[ship_qty < min_ship] = 0.0

                total_cases = float(np.sum(ship_qty))
                if total_cases < 100:  # noqa: PLR2004
                    continue

                # Build order lines for non-zero products
                lines: list[OrderLine] = []
                for p_idx in range(n_p):
                    qty = float(ship_qty[p_idx])
                    if qty >= min_ship:
                        p_id = state.product_idx_to_id[p_idx]
                        lines.append(OrderLine(p_id, qty, product_idx=p_idx))

                if not lines:
                    continue

                # Lead time from route map
                link = self._route_map.get((rdc_id, dc_id))
                lead_time = link.lead_time_days if link else self._default_lead_time

                shipment_counter += 1
                rdc_src_idx = state.node_id_to_idx.get(rdc_id, -1)
                dc_tgt_idx = state.node_id_to_idx.get(dc_id, -1)
                shipment = Shipment(
                    id=f"DRP-{day:03d}-{rdc_id}-{shipment_counter:04d}",
                    source_id=rdc_id,
                    target_id=dc_id,
                    creation_day=day,
                    arrival_day=day + int(lead_time),
                    lines=lines,
                    status=ShipmentStatus.IN_TRANSIT,
                    total_cases=total_cases,
                    source_idx=rdc_src_idx,
                    target_idx=dc_tgt_idx,
                )

                # Deduct from RDC with FIFO age reduction
                for line in lines:
                    p_idx = (
                        line.product_idx if line.product_idx >= 0
                        else state.product_id_to_idx.get(line.product_id)
                    )
                    if rdc_idx is not None and p_idx is not None and p_idx >= 0:
                        old_qty = max(
                            0.0,
                            float(state.actual_inventory[rdc_idx, p_idx]),
                        )
                        if old_qty > 0:
                            frac = max(0.0, (old_qty - line.quantity) / old_qty)
                            state.inventory_age[rdc_idx, p_idx] *= frac
                    state.update_inventory(rdc_id, line.product_id, -line.quantity)

                # Build parallel arrays for vectorized consumers
                LogisticsEngine._populate_shipment_arrays(shipment)
                shipments.append(shipment)

        return shipments

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_dc_need(
        self,
        dc_id: str,
        seasonal: float,
        in_transit: np.ndarray,
        dc_target_dos: np.ndarray,
        *,
        is_primary: bool,
        sec_frac: float,
    ) -> np.ndarray:
        """Compute per-product need vector for a single DC.

        need = max(0, target_inventory - inventory_position)
        Scaled by secondary-source fraction when applicable.
        """
        state = self._state
        n_p = state.n_products
        demand = self._dc_expected_demand.get(dc_id, np.zeros(n_p)) * seasonal
        target = dc_target_dos * demand  # ABC-differentiated

        dc_idx = state.node_id_to_idx.get(dc_id)
        if dc_idx is None:
            return np.zeros(n_p, dtype=np.float64)

        ip = np.maximum(0.0, state.actual_inventory[dc_idx, :]) + in_transit[dc_idx, :]
        need = np.maximum(0.0, target - ip)

        # Secondary-source fraction scaling
        if sec_frac > 0 and dc_id in self._dc_secondary_rdc:
            if is_primary:
                need *= (1.0 - sec_frac)
            else:
                need *= sec_frac

        return need
