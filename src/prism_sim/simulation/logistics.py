import math
from typing import Any

from prism_sim.network.core import (
    CustomerChannel,
    Link,
    Node,
    NodeType,
    Order,
    OrderLine,
    Shipment,
    ShipmentStatus,
)
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


class LogisticsEngine:
    """
    Handles the physical movement of goods (Levels 10-11).
    Implements 'Tetris' logic (Bin Packing), Lead Time delays, and Channel FTL rules.
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        log_config = config.get("simulation_parameters", {}).get("logistics", {})
        constraints = log_config.get("constraints", {})
        self.max_weight_kg = float(constraints.get("truck_max_weight_kg", 20000.0))
        self.max_volume_m3 = float(constraints.get("truck_max_volume_m3", 60.0))
        self.epsilon_weight = float(constraints.get("epsilon_weight_kg", 0.001))
        self.epsilon_volume = float(constraints.get("epsilon_volume_m3", 0.0001))
        
        # Channel Rules
        self.channel_rules = log_config.get("channel_rules", {})

        # LTL mode for store deliveries (v0.15.5)
        # Stores receive small orders - no FTL consolidation needed
        self.store_delivery_mode = log_config.get("store_delivery_mode", "FTL")
        self.ltl_min_cases = float(log_config.get("ltl_min_cases", 10.0))

        # Default FTL minimum for routes without channel-specific rules
        self.default_ftl_min_pallets = float(
            log_config.get("default_ftl_min_pallets", 5.0)
        )

        self.route_map: dict[tuple[str, str], Link] = {}
        self._build_route_map()

        # State for consolidation
        self.held_orders: list[Order] = []

        # Track FTL vs LTL shipments for metrics
        self.ftl_shipment_count = 0
        self.ltl_shipment_count = 0

    def _build_route_map(self) -> None:
        for link in self.world.links.values():
            self.route_map[(link.source_id, link.target_id)] = link

    def create_shipments(self, orders: list[Order], current_day: int) -> list[Shipment]:
        """
        Converts Allocations (Orders) into Shipments (Trucks).

        Applies Channel FTL rules and Weight/Cube constraints.
        v0.15.5: Added LTL mode for store deliveries to fix low truck fill rate.

        FTL (Full Truckload): Used for DC-to-DC shipments. Requires minimum
        pallets to consolidate. Orders held until threshold met.

        LTL (Less Than Truckload): Used for DC-to-Store shipments. Ships
        immediately without pallet minimum - stores receive smaller deliveries.
        """
        # 1. Combine new orders with held orders
        active_orders = self.held_orders + orders
        self.held_orders = []  # Reset, will repopulate with remaining

        # 2. Group by Route (Source -> Target)
        orders_by_route: dict[tuple[str, str], list[Order]] = {}
        for order in active_orders:
            route = (order.source_id, order.target_id)
            if route not in orders_by_route:
                orders_by_route[route] = []
            orders_by_route[route].append(order)

        new_shipments: list[Shipment] = []
        shipment_counter = 0

        # 3. Process each route
        for route, route_orders in orders_by_route.items():
            source_id, target_id = route
            target_node = self.world.nodes.get(target_id)

            # v0.15.5: Determine if this is an LTL (store) or FTL (DC) shipment
            is_store_delivery = (
                target_node is not None
                and target_node.type == NodeType.STORE
                and self.store_delivery_mode == "LTL"
            )

            # Determine Channel Rules
            channel = target_node.channel if target_node else None
            rules = self._get_channel_rules(channel)
            # Use channel-specific minimum, or default FTL minimum for non-channel routes
            min_pallets = rules.get(
                "min_order_pallets", self.default_ftl_min_pallets
            )

            # Calculate total pallets/cases for this route
            total_pallets = sum(self._calculate_pallets(o) for o in route_orders)
            total_cases = sum(
                sum(line.quantity for line in o.lines) for o in route_orders
            )

            # Check shipment mode constraints
            if is_store_delivery:
                # LTL mode: Ship if above minimum cases (no pallet constraint)
                if total_cases < self.ltl_min_cases:
                    # Too small even for LTL, hold for consolidation
                    self.held_orders.extend(route_orders)
                    continue
                self.ltl_shipment_count += 1
            else:
                # FTL mode: Require minimum pallets
                if total_pallets < min_pallets and min_pallets > 0:
                    # Not enough volume, hold ALL orders for this route
                    self.held_orders.extend(route_orders)
                    continue
                self.ftl_shipment_count += 1

            # If we proceed, we pack shipments
            # We decompose orders into lines for packing "Tetris" style
            lines_for_packing: list[OrderLine] = []
            for order in route_orders:
                lines_for_packing.extend(order.lines)

            # Find Lead Time
            link = self.route_map.get(route)
            lead_time = link.lead_time_days if link else 1.0
            arrival_day = current_day + int(lead_time)

            # Bin Packing Loop
            current_shipment = self._new_shipment(
                source_id, target_id, current_day, arrival_day, shipment_counter
            )
            shipment_counter += 1

            for line in lines_for_packing:
                product = self.world.products.get(line.product_id)
                if not product:
                    continue

                remaining_qty = line.quantity

                while remaining_qty > 1e-9:
                    # Check space
                    weight_space = self.max_weight_kg - current_shipment.total_weight_kg
                    vol_space = self.max_volume_m3 - current_shipment.total_volume_m3

                    if (
                        weight_space <= self.epsilon_weight
                        or vol_space <= self.epsilon_volume
                    ):
                        # Full, close and start new
                        new_shipments.append(current_shipment)
                        current_shipment = self._new_shipment(
                            source_id,
                            target_id,
                            current_day,
                            arrival_day,
                            shipment_counter,
                        )
                        shipment_counter += 1
                        weight_space = self.max_weight_kg
                        vol_space = self.max_volume_m3

                    # How much fits?
                    unit_weight = max(product.weight_kg, self.epsilon_weight)
                    unit_vol = max(product.volume_m3, self.epsilon_volume)

                    max_by_weight = weight_space / unit_weight
                    max_by_vol = vol_space / unit_vol

                    fit_qty = min(remaining_qty, max_by_weight, max_by_vol)

                    if fit_qty < 1e-9:
                        if not current_shipment.lines:
                            # Item exceeds empty truck dimensions?
                            # Forcing 1 unit if it's just a float issue, 
                            # but if it's physically too big, error.
                            # Assuming standard pallets fit.
                            raise ValueError(
                                f"Product {product.id} exceeds truck capacity"
                            )
                        else:
                            # Full
                            new_shipments.append(current_shipment)
                            current_shipment = self._new_shipment(
                                source_id,
                                target_id,
                                current_day,
                                arrival_day,
                                shipment_counter,
                            )
                            shipment_counter += 1
                            continue

                    # Add to shipment
                    current_shipment.lines.append(OrderLine(product.id, fit_qty))
                    current_shipment.total_weight_kg += fit_qty * unit_weight
                    current_shipment.total_volume_m3 += fit_qty * unit_vol
                    remaining_qty -= fit_qty

            # Add final shipment
            if current_shipment.lines:
                # Add Emissions calculation
                dist = link.distance_km if link else 0.0
                current_shipment.emissions_kg = self._calculate_emissions(current_shipment, dist)
                new_shipments.append(current_shipment)

        return new_shipments

    def _calculate_emissions(self, shipment: Shipment, distance_km: float) -> float:
        """
        Calculate Scope 3 emissions for a shipment.
        Factor: 0.1 kg CO2 per ton-km.
        """
        weight_tons = shipment.total_weight_kg / 1000.0
        base_emissions = weight_tons * distance_km * 0.1
        
        # Emissions can be affected by risk events (carbon tax spike)
        # Note: We don't have direct access to risks here easily without passing it, 
        # but we can check config for a static multiplier or assume it's applied later.
        # For now, return base.
        return base_emissions

    def update_shipments(
        self, shipments: list[Shipment], current_day: int
    ) -> tuple[list[Shipment], list[Shipment]]:
        """
        Advances shipment state.
        Returns (active_shipments, arrived_shipments)
        """
        active = []
        arrived = []
        for s in shipments:
            if s.arrival_day <= current_day:
                s.status = ShipmentStatus.DELIVERED
                arrived.append(s)
            else:
                active.append(s)
        return active, arrived

    def _new_shipment(
        self, source: str, target: str, day: int, arrival: int, counter: int
    ) -> Shipment:
        return Shipment(
            id=f"SHP-{day}-{source}-{target}-{counter}",
            source_id=source,
            target_id=target,
            creation_day=day,
            arrival_day=arrival,
            lines=[],
            status=ShipmentStatus.IN_TRANSIT,
        )

    def _get_channel_rules(self, channel: CustomerChannel | None) -> dict[str, Any]:
        """Get logistics rules for a channel."""
        if not channel:
            return {}
        return self.channel_rules.get(channel.value, self.channel_rules.get(channel.name, {}))

    def _calculate_pallets(self, order: Order) -> float:
        """Calculate total pallets for an order."""
        pallets = 0.0
        for line in order.lines:
            product = self.world.products.get(line.product_id)
            if product and product.cases_per_pallet > 0:
                pallets += line.quantity / product.cases_per_pallet
        return pallets
    
    def get_staged_inventory(self) -> dict[str, float]:
        """
        Return the amount of inventory currently held in staging (waiting for FTL).
        Used for Mass Balance checks.
        Returns: product_id -> total_quantity
        """
        staged: dict[str, float] = {}
        for order in self.held_orders:
            for line in order.lines:
                staged[line.product_id] = staged.get(line.product_id, 0.0) + line.quantity
        return staged