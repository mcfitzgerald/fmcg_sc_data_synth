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


@dataclass
class PromoConfig:
    """Configuration for a promotional event."""

    promo_id: str
    start_week: int
    end_week: int
    lift: float
    hangover_lift: float
    products: list[str]
    stores: list[str]


class PromoCalendar:
    """Manages promotional events and their lift/hangover effects."""

    def __init__(self, world: World, weeks_per_year: int = 52) -> None:
        self.world = world
        self.weeks_per_year = weeks_per_year
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

    def _apply_multiplier_to_cells(
        self,
        week: int,
        stores: list[str],
        products: list[str],
        multiplier: float,
        only_if_default: bool = False,
    ) -> None:
        """Apply a multiplier to specific store/product combinations for a week."""
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
                if only_if_default:
                    # Only apply if no active promo exists
                    if self.promo_index[week][n_idx, p_idx] == 1.0:
                        self.promo_index[week][n_idx, p_idx] = multiplier
                else:
                    # Apply max lift for overlapping promos
                    self.promo_index[week][n_idx, p_idx] = float(
                        np.maximum(self.promo_index[week][n_idx, p_idx], multiplier)
                    )

    def add_promo(self, config: PromoConfig) -> None:
        """Adds a promotional event to the calendar."""
        # Apply lift for promo weeks
        for week in range(config.start_week, config.end_week + 1):
            self._apply_multiplier_to_cells(
                week, config.stores, config.products, config.lift
            )

        # Apply hangover effect to the following week
        h_week = config.end_week + 1
        if h_week <= self.weeks_per_year:
            self._apply_multiplier_to_cells(
                h_week,
                config.stores,
                config.products,
                config.hangover_lift,
                only_if_default=True,
            )

    def get_weekly_multipliers(self, week: int, state: StateManager) -> np.ndarray:
        """Returns the demand multipliers for a given week."""
        if week in self.promo_index:
            return self.promo_index[week]
        return np.ones((state.n_nodes, state.n_products), dtype=np.float32)


class POSEngine:
    """Point-of-Sale Engine. Generates daily consumer demand."""

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        # Get calendar config
        calendar_config = config.get("simulation_parameters", {}).get("calendar", {})
        weeks_per_year = calendar_config.get("weeks_per_year", 52)
        self.calendar = PromoCalendar(world, weeks_per_year)

        # Base Demand (Cases per day) - Randomized for now per Store/SKU
        # Shape: [Nodes, Products]
        self.base_demand = np.zeros((state.n_nodes, state.n_products), dtype=np.float32)
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

            for p_id, product in self.world.products.items():
                p_idx = self.state.product_id_to_idx[p_id]

                # Assign base demand based on category
                mean_demand = 0.0

                # Use Category Enum name to look up profile
                # Enum name is usually UPPERCASE (e.g. ORAL_CARE)
                cat_name = product.category.name

                profile = profiles.get(cat_name, {})
                if not profile:
                    # Try fallback to string matching for legacy config if needed
                    # or just defaults
                    pass

                if product.category.name == "ORAL_CARE":
                    mean_demand = float(profile.get("base_daily_demand", 50.0))
                elif product.category.name == "PERSONAL_WASH":
                    mean_demand = float(profile.get("base_daily_demand", 30.0))
                elif product.category.name == "HOME_CARE":
                    mean_demand = float(profile.get("base_daily_demand", 20.0))
                elif product.category.name == "INGREDIENT":
                    mean_demand = float(profile.get("base_daily_demand", 0.0))
                else:
                    # Generic fallback if defined in config
                    mean_demand = float(profile.get("base_daily_demand", 0.0))

                self.base_demand[n_idx, p_idx] = mean_demand

    def generate_demand(self, day: int) -> np.ndarray:
        """Generates demand for a specific day."""
        week = (day // 7) + 1

        # Get demand config
        demand_config = self.config.get("simulation_parameters", {}).get("demand", {})
        season_config = demand_config.get("seasonality", {})
        noise_config = demand_config.get("noise", {})

        # 1. Seasonality (Sine wave peaking in summer/Q3)
        # Defaults: amplitude=0.2, phase=150, cycle=365
        amplitude = season_config.get("amplitude", 0.2)
        phase = season_config.get("phase_shift_days", 150)
        cycle = season_config.get("cycle_days", 365)

        seasonality = 1.0 + amplitude * np.sin(2 * np.pi * (day - phase) / cycle)

        # 2. Promo Multipliers
        promo_mult = self.calendar.get_weekly_multipliers(week, self.state)

        # 3. Randomness (Gamma distribution to prevent negative demand)
        # Defaults: shape=10.0, scale=0.1
        g_shape = noise_config.get("gamma_shape", 10.0)
        g_scale = noise_config.get("gamma_scale", 0.1)

        rng = np.random.default_rng(day)
        noise = rng.gamma(shape=g_shape, scale=g_scale, size=self.base_demand.shape)

        # 4. Combine
        # Demand = Base * Seasonality * Promo * Noise
        demand = self.base_demand * seasonality * promo_mult * noise

        return cast(np.ndarray, demand.astype(np.float32))

    def get_average_demand_estimate(self) -> float:
        """
        Estimate the network-wide average daily demand per product.
        Used for priming inventory.
        """
        # Simple mean of the base demand matrix
        # This ignores seasonality/promo but is better than 1.0
        # We only care about positive demand nodes (Stores)
        total_base = np.sum(self.base_demand)
        n_products = self.state.n_products
        n_stores = np.count_nonzero(np.sum(self.base_demand, axis=1))

        if n_stores > 0 and n_products > 0:
            # Average demand per store per product
            # Use total sum divided by (stores * products) to get "per SKU-Location" avg
            # But usually we want "per SKU" at a location.
            # Let's return the mean of non-zero entries to be safe for sparse data
            non_zero_mean = total_base / np.count_nonzero(self.base_demand)
            return float(non_zero_mean)
        return 1.0
