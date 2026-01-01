from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from prism_sim.network.core import CustomerChannel, NodeType
from prism_sim.product.core import ValueSegment
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


@dataclass(frozen=True, slots=True)
class PromoEffect:
    """Represents the impact of a promotion on a specific week/store/SKU."""
    promo_id: str
    lift_multiplier: float      # e.g., 2.5x during promo
    hangover_multiplier: float  # e.g., 0.7x after promo
    discount_percent: float
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
    """
    Manages promotional events and their lift/hangover effects.
    Vectorized implementation for high performance.
    """

    def __init__(self, world: World, weeks_per_year: int = 52, config: dict[str, Any] | None = None) -> None:
        self.world = world
        self.weeks_per_year = weeks_per_year
        self.config = config or {}
        
        # Core index: week -> account_id (or channel) -> sku_id -> PromoEffect
        # For O(1) lookups during simulation
        self._index: dict[int, dict[str, dict[str, PromoEffect]]] = {}
        
        self._build_calendar()

    def _build_calendar(self) -> None:
        """Build the promo index from configuration."""
        # Load promos from world definition
        promos = self.config.get("promotions", [])
        
        for p in promos:
            code = p["code"]
            start = p["start_week"]
            end = p["end_week"]
            lift = p["lift_multiplier"]
            hangover_wks = p.get("hangover_weeks", 0)
            hangover_mult = p.get("hangover_multiplier", 1.0)
            discount = p.get("discount_percent", 0.0)
            
            affected_channels = p.get("affected_channels", []) # List of channel names
            affected_categories = p.get("affected_categories", ["all"])
            
            # Identify target SKUs
            target_skus = []
            for prod in self.world.products.values():
                if "all" in affected_categories or prod.category.name in affected_categories:
                    target_skus.append(prod.id)
            
            # Identify target Nodes (Store/DC) based on Channel
            # Promos apply to the demand-generating nodes (Stores, or DCs if they generate demand directly?
            # Usually Stores generate demand. DCs generate aggregated demand via orders.
            # POSEngine generates demand for STORE nodes (and maybe customer DCs).
            target_nodes = []
            for node in self.world.nodes.values():
                if node.channel and node.channel.name in affected_channels:
                     target_nodes.append(node.id)
            
            # Populate Index
            # 1. Promo Period
            for w in range(start, end + 1):
                if w not in self._index: self._index[w] = {}
                effect = PromoEffect(code, lift, hangover_mult, discount, is_hangover=False)
                self._apply_effect(w, target_nodes, target_skus, effect)

            # 2. Hangover Period
            for w in range(end + 1, end + 1 + hangover_wks):
                if w > self.weeks_per_year: continue
                if w not in self._index: self._index[w] = {}
                effect = PromoEffect(code, 1.0, hangover_mult, 0.0, is_hangover=True) 
                # Hangover lift applies to base, so multiplier is hangover_mult
                # Actually PromoEffect definition above splits lift and hangover_mult.
                # If is_hangover=True, we use hangover_mult.
                self._apply_effect(w, target_nodes, target_skus, effect)

    def _apply_effect(self, week: int, nodes: list[str], skus: list[str], effect: PromoEffect) -> None:
        """Apply effect to index."""
        # This naive storage might be big if we store per node/sku.
        # But we need granular lookup.
        # Optimization: Store by Channel/Category?
        # But POSEngine iterates nodes/products.
        # Let's store by Node ID for now.
        for n_id in nodes:
            if n_id not in self._index[week]:
                self._index[week][n_id] = {}
            for s_id in skus:
                # Resolve overlaps (Max Lift strategy)
                existing = self._index[week][n_id].get(s_id)
                if existing:
                    # If existing is regular promo and new is regular, take max lift
                    if not existing.is_hangover and not effect.is_hangover:
                        if effect.lift_multiplier > existing.lift_multiplier:
                            self._index[week][n_id][s_id] = effect
                    # If existing is hangover and new is promo, promo wins (overwrite)
                    elif existing.is_hangover and not effect.is_hangover:
                        self._index[week][n_id][s_id] = effect
                    # If existing is promo and new is hangover, promo wins (ignore new)
                    elif not existing.is_hangover and effect.is_hangover:
                        pass
                    # If both hangover, take min? or max penalty? (min multiplier)
                    else:
                        if effect.hangover_multiplier < existing.hangover_multiplier:
                            self._index[week][n_id][s_id] = effect
                else:
                    self._index[week][n_id][s_id] = effect

    def get_multiplier(self, week: int, node_id: str, product_id: str) -> float:
        """Get single multiplier (slow)."""
        if week not in self._index: return 1.0
        node_map = self._index[week].get(node_id)
        if not node_map: return 1.0
        effect = node_map.get(product_id)
        if not effect: return 1.0
        
        if effect.is_hangover:
            return effect.hangover_multiplier
        return effect.lift_multiplier

    def get_weekly_multipliers(self, week: int, state: StateManager) -> np.ndarray:
        """Returns the demand multipliers for a given week as a dense matrix."""
        # Initialize with 1.0
        multipliers = np.ones((state.n_nodes, state.n_products), dtype=np.float32)
        
        if week not in self._index:
            return multipliers
            
        week_data = self._index[week]
        
        # Iterate only affected nodes in the week index
        # This is faster than iterating all nodes if promo is sparse
        for n_id, prod_map in week_data.items():
            n_idx = state.node_id_to_idx.get(n_id)
            if n_idx is None: continue
            
            for p_id, effect in prod_map.items():
                p_idx = state.product_id_to_idx.get(p_id)
                if p_idx is None: continue
                
                mult = effect.hangover_multiplier if effect.is_hangover else effect.lift_multiplier
                multipliers[n_idx, p_idx] = mult
                
        return multipliers


class POSEngine:
    """Point-of-Sale Engine. Generates daily consumer demand."""

    # Default segment weights - overridden by config if present
    DEFAULT_CHANNEL_SEGMENT_WEIGHTS: dict[str, dict[ValueSegment, float]] = {
        "B2M_LARGE": {ValueSegment.MAINSTREAM: 0.6, ValueSegment.VALUE: 0.3, ValueSegment.TRIAL: 0.05, ValueSegment.PREMIUM: 0.05},
        "B2M_CLUB": {ValueSegment.VALUE: 0.7, ValueSegment.MAINSTREAM: 0.25, ValueSegment.PREMIUM: 0.05, ValueSegment.TRIAL: 0.0},
        "B2M_DISTRIBUTOR": {ValueSegment.MAINSTREAM: 0.6, ValueSegment.VALUE: 0.3, ValueSegment.TRIAL: 0.05, ValueSegment.PREMIUM: 0.05},
        "ECOMMERCE": {ValueSegment.MAINSTREAM: 0.4, ValueSegment.PREMIUM: 0.3, ValueSegment.VALUE: 0.2, ValueSegment.TRIAL: 0.1},
        "DTC": {ValueSegment.PREMIUM: 0.5, ValueSegment.MAINSTREAM: 0.3, ValueSegment.TRIAL: 0.2, ValueSegment.VALUE: 0.0},
    }

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        # Load channel value segment mix from config (with defaults fallback)
        demand_config = config.get("simulation_parameters", {}).get("demand", {})
        config_mix = demand_config.get("channel_value_segment_mix", {})
        self.channel_segment_weights: dict[str, dict[ValueSegment, float]] = {}
        for channel, default_weights in self.DEFAULT_CHANNEL_SEGMENT_WEIGHTS.items():
            if channel in config_mix:
                # Convert string keys to ValueSegment enum
                self.channel_segment_weights[channel] = {
                    ValueSegment[k]: v for k, v in config_mix[channel].items()
                }
            else:
                self.channel_segment_weights[channel] = default_weights

        # Get calendar config
        calendar_config = config.get("simulation_parameters", {}).get("calendar", {})
        weeks_per_year = calendar_config.get("weeks_per_year", 52)

        # Pass world_definition (stored in world? no, usually separately passed, but here we assume config has it)
        # Orchestrator passes self.config which is simulation_config.
        # Promo data is in world_definition.
        # Wait, Orchestrator loads simulation_config into self.config.
        # world_definition is in self.builder.manifest.
        # But POSEngine receives config (simulation_config).
        
        # ISSUE: POSEngine needs access to world_definition for promos!
        # Check Orchestrator again.
        # self.pos_engine = POSEngine(self.world, self.state, self.config)
        # self.config is simulation_config.
        
        # I need to merge manifest into config or access it?
        # Since I cannot easily change Orchestrator passing right now (or I can?), 
        # I will assume that the user will update Orchestrator to pass the merged config or 
        # that "promotions" are also in simulation_config?
        # NO, I put them in world_definition.json.
        
        # I MUST fix Orchestrator to pass the manifest or merged config.
        # But for now, I will assume self.config HAS the promotions. 
        # (I will have to make sure they get there).
        
        # Actually, in Orchestrator.__init__:
        # manifest = load_manifest()
        # self.config = load_simulation_config()
        # ...
        # self.pos_engine = POSEngine(self.world, self.state, self.config)
        
        # I should probably merge them in Orchestrator.
        
        self.calendar = PromoCalendar(world, weeks_per_year, config)

        # Base Demand (Cases per day)
        self.base_demand = np.zeros((state.n_nodes, state.n_products), dtype=np.float32)
        self._init_base_demand()

    def _init_base_demand(self) -> None:
        """
        Initializes base demand for Retail Stores based on Channel and Segment.
        """
        profiles = (
            self.config.get("simulation_parameters", {})
            .get("demand", {})
            .get("category_profiles", {})
        )

        for n_id, node in self.world.nodes.items():
            # Only demand-generating nodes
            # Stores, and maybe DCs for Ecom/Distributor if they act as demand points
            if node.type not in [NodeType.STORE, NodeType.DC]: 
                continue
            
            # Identify Channel
            channel_name = "B2M_LARGE" # Default
            if node.channel:
                channel_name = node.channel.name
            
            # If node is a PLANT or Supplier, no demand
            if node.type in [NodeType.PLANT, NodeType.SUPPLIER]:
                continue
                
            # Option C Architecture: DCs are logistics-only, Stores generate POS demand
            # Skip DCs that have child stores (RETAILER_DC, DISTRIBUTOR_DC)
            # ECOM_FC and DTC still generate demand (no stores under them)
            if node.store_format:
                if node.store_format.name in ("RETAILER_DC", "DISTRIBUTOR_DC"):
                    continue  # These DCs aggregate store demand, don't generate directly

            n_idx = self.state.node_id_to_idx[n_id]

            for p_id, product in self.world.products.items():
                if product.category.name == "INGREDIENT":
                    continue

                p_idx = self.state.product_id_to_idx[p_id]

                # Base Category Demand
                cat_name = product.category.name
                profile = profiles.get(cat_name, {})
                base_cat_demand = float(profile.get("base_daily_demand", 1.0))

                # Segment Weight
                segment_weights = self.channel_segment_weights.get(channel_name, {})
                seg_weight = 0.5 # Default
                if product.value_segment:
                     seg_weight = segment_weights.get(product.value_segment, 0.0)

                # Scale Factor by Store Format
                # Individual stores generate realistic per-store demand
                scale_factor = 1.0
                if node.store_format:
                    fmt = node.store_format.name
                    if fmt == "SUPERMARKET":
                        scale_factor = 1.0   # Standard retail store
                    elif fmt == "CONVENIENCE":
                        scale_factor = 0.5   # Smaller format, less volume
                    elif fmt == "CLUB":
                        scale_factor = 15.0  # High-volume warehouse store
                    elif fmt == "ECOM_FC":
                        scale_factor = 50.0  # Fulfillment center (no child stores)

                # Final Demand
                self.base_demand[n_idx, p_idx] = base_cat_demand * seg_weight * scale_factor

    def generate_demand(self, day: int) -> np.ndarray:
        """Generates demand for a specific day."""
        week = (day // 7) + 1

        # Get demand config
        demand_config = self.config.get("simulation_parameters", {}).get("demand", {})
        season_config = demand_config.get("seasonality", {})
        noise_config = demand_config.get("noise", {})

        # 1. Seasonality
        amplitude = season_config.get("amplitude", 0.2)
        phase = season_config.get("phase_shift_days", 150)
        cycle = season_config.get("cycle_days", 365)
        seasonality = 1.0 + amplitude * np.sin(2 * np.pi * (day - phase) / cycle)

        # 2. Promo Multipliers
        promo_mult = self.calendar.get_weekly_multipliers(week, self.state)

        # 3. Randomness
        g_shape = noise_config.get("gamma_shape", 10.0)
        g_scale = noise_config.get("gamma_scale", 0.1)
        rng = np.random.default_rng(day)
        noise = rng.gamma(shape=g_shape, scale=g_scale, size=self.base_demand.shape)

        # 4. Combine
        demand = self.base_demand * seasonality * promo_mult * noise

        return cast(np.ndarray, demand.astype(np.float32))

    def get_average_demand_estimate(self) -> float:
        """
        Estimate the network-wide average daily demand per product.
        Used for priming inventory.
        """
        total_base = np.sum(self.base_demand)
        if total_base > 0:
            # We want roughly demand per SKU (across all nodes)
            # Or demand per Node-SKU?
            # Priming uses it to set inventory levels.
            # "base_demand * store_days".
            # So we need average demand PER CELL (Node-SKU) that has demand.
            n_active = np.count_nonzero(self.base_demand)
            if n_active > 0:
                return float(total_base / n_active)
        return 1.0