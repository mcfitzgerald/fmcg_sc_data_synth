from typing import List, Dict, Tuple
import math
from prism_sim.simulation.world import World
from prism_sim.network.core import Order, OrderLine, Shipment, ShipmentStatus, Link
from prism_sim.simulation.state import StateManager


class LogisticsEngine:
    """
    Handles the physical movement of goods (Levels 10-11).
    Implements 'Tetris' logic (Bin Packing) and Lead Time delays.
    """

    def __init__(self, world: World, state: StateManager, config: Dict):
        self.world = world
        self.state = state
        self.config = config
        
        constraints = config.get("simulation_parameters", {}).get("logistics", {}).get("constraints", {})
        self.max_weight_kg = constraints.get("truck_max_weight_kg", 20000.0)
        self.max_volume_m3 = constraints.get("truck_max_volume_m3", 60.0)
        
        self.route_map: Dict[Tuple[str, str], Link] = {}
        self._build_route_map()

    def _build_route_map(self):
        for link in self.world.links.values():
            self.route_map[(link.source_id, link.target_id)] = link

    def create_shipments(self, orders: List[Order], current_day: int) -> List[Shipment]:
        """
        Converts Allocations (Orders) into Shipments (Trucks).
        Applies Weight/Cube constraints.
        """
        # Group by Route
        lines_by_route: Dict[Tuple[str, str], List[OrderLine]] = {}

        # We decompose Orders into Lines for packing
        for order in orders:
            route = (order.source_id, order.target_id)
            if route not in lines_by_route:
                lines_by_route[route] = []
            lines_by_route[route].extend(order.lines)

        new_shipments = []
        shipment_counter = 0

        for route, lines in lines_by_route.items():
            source_id, target_id = route

            # Find Lead Time
            link = self.route_map.get(route)
            lead_time = link.lead_time_days if link else 1  # Default 1 if no link
            arrival_day = current_day + int(lead_time)

            # Bin Packing
            current_shipment = self._new_shipment(
                source_id, target_id, current_day, arrival_day, shipment_counter
            )
            shipment_counter += 1

            for line in lines:
                product = self.world.products.get(line.product_id)
                if not product:
                    continue  # Skip unknown products

                remaining_qty = line.quantity

                while remaining_qty > 0:
                    # Check space
                    weight_space = self.max_weight_kg - current_shipment.total_weight_kg
                    vol_space = self.max_volume_m3 - current_shipment.total_volume_m3

                    if weight_space <= 0 or vol_space <= 0:
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
                    # Avoid div by zero
                    unit_weight = max(product.weight_kg, 0.001)
                    unit_vol = max(product.volume_m3, 0.0001)

                    max_by_weight = weight_space / unit_weight
                    max_by_vol = vol_space / unit_vol

                    # We deal with whole cases
                    fit_qty = math.floor(min(remaining_qty, max_by_weight, max_by_vol))

                    if fit_qty <= 0:
                        # Item too big for empty truck? Force 1 if empty, else new truck
                        if not current_shipment.lines:
                            fit_qty = 1  # Force fit one unit
                        else:
                            # New truck
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

            # Add final shipment if not empty
            if current_shipment.lines:
                new_shipments.append(current_shipment)

        return new_shipments

    def update_shipments(
        self, shipments: List[Shipment], current_day: int
    ) -> Tuple[List[Shipment], List[Shipment]]:
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

    def _new_shipment(self, source, target, day, arrival, counter):
        return Shipment(
            id=f"SHP-{day}-{source}-{target}-{counter}",
            source_id=source,
            target_id=target,
            creation_day=day,
            arrival_day=arrival,
            lines=[],
            status=ShipmentStatus.IN_TRANSIT,
        )
