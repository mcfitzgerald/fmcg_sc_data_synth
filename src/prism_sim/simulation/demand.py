from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from prism_sim.network.core import NodeType
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


@dataclass(frozen=True)
class PromoEffect:
    """Represents the impact of a promotion on a specific week/store/SKU."""

    promo_id: str
    lift_multiplier: float
    hangover_multiplier: float
    is_hangover: bool = False


class PromoCalendar:
    """Manages promotional events and their lift/hangover effects."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.n_nodes = len(world.nodes)
        self.n_products = len(world.products)
        # week -> [Nodes, Products] multiplier tensor
        self.promo_index: dict[int, np.ndarray] = {}

        # Pre-calculate ID to index maps to match StateManager
        self.node_to_idx = {
            id: i for i, id in enumerate(sorted(self.world.nodes.keys()))
        }
        self.prod_to_idx = {
            id: i for i, id in enumerate(sorted(self.world.products.keys()))
        }

    def _get_week_array(self) -> np.ndarray:
        return np.ones((self.n_nodes, self.n_products), dtype=np.float32)

    def add_promo(
        self,
        promo_id: str,
        start_week: int,
        end_week: int,
        lift: float,
        hangover_lift: float,
        products: list[str],
        stores: list[str],
    ) -> None:
        """Adds a promotional event to the calendar."""
        for week in range(start_week, end_week + 1):
            if week not in self.promo_index:
                self.promo_index[week] = self._get_week_array()

            for s_id in stores:
                n_idx = self.node_to_idx.get(s_id)
                if n_idx is None:
                    continue
                for p_id in products:
                    p_idx = self.prod_to_idx.get(p_id)
                    if p_idx is None:
                        continue
                    # Apply max lift for overlapping promos
                    self.promo_index[week][n_idx, p_idx] = float(np.maximum(
                        self.promo_index[week][n_idx, p_idx], lift
                    ))

        # Apply hangover effect to the following week
        h_week = end_week + 1
        if h_week <= 52:
            if h_week not in self.promo_index:
                self.promo_index[h_week] = self._get_week_array()

            for s_id in stores:
                n_idx = self.node_to_idx.get(s_id)
                if n_idx is None:
                    continue
                for p_id in products:
                    p_idx = self.prod_to_idx.get(p_id)
                    if p_idx is None:
                        continue
                    # Only apply hangover if no active promo exists
                    if self.promo_index[h_week][n_idx, p_idx] == 1.0:
                        self.promo_index[h_week][n_idx, p_idx] = hangover_lift

    def get_weekly_multipliers(self, week: int, state: StateManager) -> np.ndarray:
        """Returns the demand multipliers for a given week."""
        if week in self.promo_index:
            return self.promo_index[week]
        return np.ones((state.n_nodes, state.n_products), dtype=np.float32)


class POSEngine:
    """Point-of-Sale Engine. Generates daily consumer demand."""

    def __init__(self, world: World, state: StateManager, config: dict[str, Any]) -> None:
        self.world = world
        self.state = state
        self.config = config
        self.calendar = PromoCalendar(world)

        # Base Demand (Cases per day) - Randomized for now per Store/SKU
        # Shape: [Nodes, Products]
        self.base_demand = np.zeros(
            (state.n_nodes, state.n_products), dtype=np.float32
        )
        self._init_base_demand()

    def _init_base_demand(self) -> None:
        """
        Initializes base demand for Retail Stores.
        RDCs and Suppliers have 0 consumer demand.
        """
        profiles = (
            self.config.get("simulation_parameters", {})
            .get("demand", {})
            .get("category_profiles", {})
        )

        for n_id, node in self.world.nodes.items():
            if node.type != NodeType.STORE:
                continue

            n_idx = self.state.node_id_to_idx[n_id]

            for p_id, _ in self.world.products.items():
                p_idx = self.state.product_id_to_idx[p_id]

                # Assign base demand based on category
                mean_demand = 0.0

                # Check category match (rough logic based on ID string)
                if "PASTE" in p_id:
                    mean_demand = float(profiles.get("ORAL_CARE", {}).get(
                        "base_daily_demand", 50.0
                    ))
                elif "SOAP" in p_id:
                    mean_demand = float(profiles.get("PERSONAL_WASH", {}).get(
                        "base_daily_demand", 30.0
                    ))
                elif "DET" in p_id:
                    mean_demand = float(profiles.get("HOME_CARE", {}).get(
                        "base_daily_demand", 20.0
                    ))
                elif "ING" in p_id:
                    mean_demand = float(profiles.get("INGREDIENT", {}).get(
                        "base_daily_demand", 0.0
                    ))

                self.base_demand[n_idx, p_idx] = mean_demand

    def generate_demand(self, day: int) -> np.ndarray:
        """Generates demand for a specific day."""
        week = (day // 7) + 1

        # 1. Seasonality (Sine wave peaking in summer/Q3)
        seasonality = 1.0 + 0.2 * np.sin(2 * np.pi * (day - 150) / 365)

        # 2. Promo Multipliers
        promo_mult = self.calendar.get_weekly_multipliers(week, self.state)

        # 3. Randomness (Gamma distribution to prevent negative demand)
        # CV = 0.3 approx
        rng = np.random.default_rng(day)
        noise = rng.gamma(shape=10.0, scale=0.1, size=self.base_demand.shape)

        # 4. Combine
        # Demand = Base * Seasonality * Promo * Noise
        demand = self.base_demand * seasonality * promo_mult * noise

        return cast(np.ndarray, demand.astype(np.float32))

