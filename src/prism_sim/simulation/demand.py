import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict
from prism_sim.simulation.world import World
from prism_sim.simulation.state import StateManager
from prism_sim.network.core import NodeType


@dataclass(frozen=True)
class PromoEffect:
    """
    Represents the impact of a promotion on a specific week/store/SKU.
    """

    promo_id: str
    lift_multiplier: float
    hangover_multiplier: float
    is_hangover: bool = False


class PromoCalendar:
    """
    Manages promotional events and their lift/hangover effects.
    """

    def __init__(self, world: World):
        self.world = world
        # Map: week_num -> node_id -> product_id -> PromoEffect
        self._calendar: Dict[int, Dict[str, Dict[str, PromoEffect]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._active_weeks: Set[int] = set()

    def add_promo(
        self,
        promo_id: str,
        start_week: int,
        end_week: int,
        lift: float,
        hangover_lift: float,
        products: List[str],
        stores: List[str],
    ):
        """
        Registers a promotion.
        """
        # 1. Active Period
        for week in range(start_week, end_week + 1):
            self._active_weeks.add(week)
            for store_id in stores:
                for prod_id in products:
                    # Logic: Max Lift wins if overlapping
                    existing = self._calendar[week][store_id].get(prod_id)
                    if (
                        existing
                        and existing.lift_multiplier > lift
                        and not existing.is_hangover
                    ):
                        continue

                    self._calendar[week][store_id][prod_id] = PromoEffect(
                        promo_id=promo_id,
                        lift_multiplier=lift,
                        hangover_multiplier=hangover_lift,
                        is_hangover=False,
                    )

        # 2. Hangover Period (1 week post-promo)
        hangover_week = end_week + 1
        if hangover_week <= 52:
            self._active_weeks.add(hangover_week)
            for store_id in stores:
                for prod_id in products:
                    # Logic: Active promo beats hangover
                    existing = self._calendar[hangover_week][store_id].get(prod_id)
                    if existing and not existing.is_hangover:
                        continue

                    self._calendar[hangover_week][store_id][prod_id] = PromoEffect(
                        promo_id=promo_id,
                        lift_multiplier=1.0,  # No lift during hangover
                        hangover_multiplier=hangover_lift,
                        is_hangover=True,
                    )

    def get_weekly_multipliers(self, week: int, state: StateManager) -> np.ndarray:
        """
        Returns a (Nodes, Products) tensor of demand multipliers for the given week.
        Default multiplier is 1.0.
        """
        # Initialize with 1.0
        multipliers = np.ones((state.n_nodes, state.n_products), dtype=np.float32)

        if week not in self._calendar:
            return multipliers

        week_data = self._calendar[week]

        # Iterate through sparse calendar entries and update dense tensor
        # This is efficient because promos are sparse compared to full NxM space
        for store_id, prod_map in week_data.items():
            if store_id not in state.node_id_to_idx:
                continue
            n_idx = state.node_id_to_idx[store_id]

            for prod_id, effect in prod_map.items():
                if prod_id not in state.product_id_to_idx:
                    continue
                p_idx = state.product_id_to_idx[prod_id]

                if effect.is_hangover:
                    multipliers[n_idx, p_idx] = effect.hangover_multiplier
                else:
                    multipliers[n_idx, p_idx] = effect.lift_multiplier

        return multipliers


class POSEngine:
    """
    Point-of-Sale Engine. Generates daily consumer demand.
    """

    def __init__(self, world: World, state: StateManager, config: Dict):
        self.world = world
        self.state = state
        self.config = config
        self.calendar = PromoCalendar(world)

        # Base Demand (Cases per day) - Randomized for now per Store/SKU
        # Shape: [Nodes, Products]
        self.base_demand = np.zeros((state.n_nodes, state.n_products), dtype=np.float32)
        self._init_base_demand()

    def _init_base_demand(self):
        """
        Initializes base demand for Retail Stores.
        RDCs and Suppliers have 0 consumer demand.
        """
        profiles = self.config.get("simulation_parameters", {}).get("demand", {}).get("category_profiles", {})
        
        for n_id, node in self.world.nodes.items():
            if node.type != NodeType.STORE:
                continue

            n_idx = self.state.node_id_to_idx[n_id]

            for p_id, product in self.world.products.items():
                p_idx = self.state.product_id_to_idx[p_id]

                # Assign base demand based on category
                mean_demand = 0.0
                
                # Check category match (rough logic based on ID string or category enum if available)
                # Ideally we use product.category, but for now matching ID strings as per previous logic
                if "PASTE" in p_id:
                     mean_demand = profiles.get("ORAL_CARE", {}).get("base_daily_demand", 50.0)
                elif "SOAP" in p_id:
                     mean_demand = profiles.get("PERSONAL_WASH", {}).get("base_daily_demand", 30.0)
                elif "DET" in p_id:
                     mean_demand = profiles.get("HOME_CARE", {}).get("base_daily_demand", 20.0)
                elif "ING" in p_id:
                     mean_demand = profiles.get("INGREDIENT", {}).get("base_daily_demand", 0.0)

                self.base_demand[n_idx, p_idx] = mean_demand

    def generate_demand(self, day: int) -> np.ndarray:
        """
        Generates demand for a specific day.
        Formula: Base * Seasonality * Promo * Randomness
        """
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

        return demand.astype(np.float32)
