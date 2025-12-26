"""
Level 10-11 Generator: Fulfillment (Shipments).

Level 10 Tables:
- pick_wave_orders (~180,000)
- shipments (~180,000)
- shipment_legs (~360,000)

Level 11 Tables:
- shipment_lines (~540,000)

Named entities:
- DC-NAM-CHI-001: Bottleneck DC (40% of NAM volume flows through Chicago)
"""

import random
import time
from datetime import date, datetime, timedelta

import numpy as np

from .base import BaseLevelGenerator
from ..lookup_builder import LookupBuilder
from ..vectorized import (
    zipf_weights, 
    ShipmentsGenerator,
    ShipmentLegsGenerator,
    ShipmentLinesGenerator, 
    structured_to_dicts,
    SHIPMENTS_COLUMNS,
    SHIPMENT_LEGS_COLUMNS
)


class Level10Generator(BaseLevelGenerator):
    """
    Generate Level 10 shipment data.

    Level 10 contains pick_wave_orders, shipments, and shipment_legs.
    """

    LEVEL = 10

    def generate(self) -> None:
        """Generate all Level 10 tables."""
        print("  Level 10: Shipments and legs...")
        level_start = time.time()
        now = datetime.now()

        self._generate_pick_wave_orders(now)
        self._generate_shipments(now)
        self._generate_shipment_legs(now)
        self._apply_chaos_logistics()

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(
            f"    Generated: {len(self.data['pick_wave_orders'])} pick_wave_orders, "
            f"{len(self.data['shipments'])} shipments, "
            f"{len(self.data['shipment_legs'])} legs ({level_elapsed:.1f}s)"
        )

    def _generate_pick_wave_orders(self, now: datetime) -> None:
        """Generate pick_wave_orders table (~180K)."""
        print("    Generating pick_wave_orders...")
        completed_statuses = {"shipped", "delivered"}
        order_status_by_id = {o["id"]: o["status"] for o in self.data["orders"]}

        pickable_orders = [
            oid for oid in self.ctx.order_ids.values()
            if order_status_by_id.get(oid) in completed_statuses
        ]

        pwo_count = 0
        for wave in self.data["pick_waves"]:
            wave_id = wave["id"]
            wave_status = wave["status"]
            target_orders = wave.get("total_orders", random.randint(10, 50))

            sample_size = min(target_orders, len(pickable_orders))
            if sample_size == 0:
                continue

            wave_orders = random.sample(pickable_orders, sample_size)

            for seq, order_id in enumerate(wave_orders, 1):
                if wave_status == "completed":
                    status = "staged"
                elif wave_status in ("loaded", "staged"):
                    status = "staged"
                elif wave_status in ("packing", "picking"):
                    status = random.choice(["picking", "picked", "packed"])
                else:
                    status = "pending"

                self.data["pick_wave_orders"].append(
                    {
                        "wave_id": wave_id,
                        "order_id": order_id,
                        "pick_sequence": seq,
                        "status": status,
                        "created_at": now,
                    }
                )
                pwo_count += 1

        print(f"      Generated {pwo_count:,} pick_wave_orders")

    def _get_sku_weights(self) -> dict:
        """Helper to get SKU weights for bin packing."""
        sku_weight_kg = {}
        for sku in self.data["skus"]:
            size_str = sku.get("size", "6oz")
            try:
                size_val = float(size_str.replace("oz", "").replace("ml", "").replace("g", ""))
                if "oz" in size_str:
                    sku_weight_kg[sku["id"]] = round(size_val * 0.028 * 24, 2)
                elif "ml" in size_str:
                    sku_weight_kg[sku["id"]] = round(size_val * 0.001 * 24, 2)
                else:
                    sku_weight_kg[sku["id"]] = 5.0
            except ValueError:
                sku_weight_kg[sku["id"]] = 5.0
        return sku_weight_kg

    def _generate_shipments(self, now: datetime) -> None:
        """
        Generate shipments table with SKU-Level Mass Balance and Emergent Logistics.
        Uses Demand-Pull distribution to ensure production matches SKU popularity.
        Vectorized for performance and physics accuracy.
        """
        print("    Generating shipments (Vectorized Physics)...")
        plant_ids = list(self.ctx.plant_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())
        carrier_ids = list(self.ctx.carrier_ids.values()) if self.ctx.carrier_ids else [1]
        route_ids = [r["id"] for r in self.data["routes"]] if self.data["routes"] else [1]
        location_ids = [loc["id"] for loc in self.data["retail_locations"]]

        chicago_dc_id = self.ctx.dc_ids.get("DC-NAM-CHI-001", dc_ids[0] if dc_ids else 1)
        sku_weights = self._get_sku_weights()

        # 1. Aggregate Demand per SKU
        demand_by_sku = {}
        for ol in self.data.get("order_lines", []):
            sid = ol.get("sku_id")
            demand_by_sku[sid] = demand_by_sku.get(sid, 0) + ol.get("quantity_cases", 0)

        # 2. Demand-Pull Supply Distribution
        supply_by_sku = {}
        formula_to_skus = LookupBuilder.build(self.data["skus"], key_field="formula_id")
        
        # Physics: Map which SKUs are available at which plants
        plant_to_skus = {pid: set() for pid in plant_ids}
        # production_lines have plant_id
        line_to_plant = {line["id"]: line["plant_id"] for line in self.data["production_lines"]}
        # work_orders have production_line_id and batch_id
        for wo in self.data.get("work_orders", []):
            pid = line_to_plant.get(wo.get("production_line_id"))
            if not pid: continue
            # Find formula from skus? Or work_order has product?
            # Let's assume plant can make all SKUs for now if we can't find direct link,
            # but ideally we use the formula link.
            # In Level 5, work_orders might have formula_id or product_id.
            # Actually, batches have formula_id.
            pass

        # Simpler plant_to_skus: Use the batches produced at plants
        batch_to_plant = {}
        for wo in self.data.get("work_orders", []):
            batch_id = wo.get("batch_id")
            if batch_id:
                batch_to_plant[batch_id] = line_to_plant.get(wo.get("production_line_id"))

        for b in self.data.get("batches", []):
            bid = b["id"]
            formula_id = b.get("formula_id")
            skus_for_formula = formula_to_skus.get(formula_id, [])
            if not skus_for_formula:
                continue
            
            pid = batch_to_plant.get(bid)
            if pid:
                for s in skus_for_formula:
                    plant_to_skus[pid].add(s["id"])

            total_cases = b.get("output_cases", 0)
            sku_ids = [s["id"] for s in skus_for_formula]
            formula_demand = sum(demand_by_sku.get(sid, 0) for sid in sku_ids)
            
            if formula_demand > 0:
                for sid in sku_ids:
                    share = demand_by_sku.get(sid, 0) / formula_demand
                    supply_by_sku[sid] = supply_by_sku.get(sid, 0) + int(total_cases * share)
            else:
                cases_per_sku = total_cases // len(sku_ids)
                for sid in sku_ids:
                    supply_by_sku[sid] = supply_by_sku.get(sid, 0) + cases_per_sku

        # 3. Calculate Shippable and Inventory
        shippable_to_store_by_sku = {}
        inventory_by_sku = {}
        all_skus = set(supply_by_sku.keys()) | set(demand_by_sku.keys())
        total_production_weight = 0

        for sid in all_skus:
            supply = supply_by_sku.get(sid, 0)
            demand = demand_by_sku.get(sid, 0)
            ss_pct = random.uniform(0.12, 0.20) if random.random() < 0.15 else random.uniform(0.05, 0.10)
            available = max(0, supply - int(supply * ss_pct))
            shippable = min(available, demand)
            shippable_to_store_by_sku[sid] = shippable
            inventory_by_sku[sid] = supply - shippable
            total_production_weight += supply * sku_weights.get(sid, 10.0)

        self.inventory_by_sku = inventory_by_sku
        self.ctx.plant_to_skus = {k: list(v) for k, v in plant_to_skus.items()}

        # 4. Emergent Logistics
        TRUCK_CAPACITY_KG = 20000
        TARGET_FILL_RATE = random.uniform(0.85, 0.95)
        num_shipments = max(100, int(total_production_weight / (TRUCK_CAPACITY_KG * TARGET_FILL_RATE)))
        
        print(f"    [Physics] Derived {num_shipments:,} shipments from {total_production_weight/1000:,.1f} tons")

        # Prepare for ShipmentsGenerator
        shipment_types_pool = ["plant_to_dc", "dc_to_dc", "dc_to_store", "direct_to_store"]
        shipment_type_weights = [0.30, 0.10, 0.55, 0.05]

        chosen_types = random.choices(shipment_types_pool, weights=shipment_type_weights, k=num_shipments)

        # Build lookups for CTS factors
        dc_temp_by_id = {
            dc["id"]: dc.get("is_temperature_controlled", random.random() < 0.25)
            for dc in self.data["distribution_centers"]
        }
        carrier_preferred = {
            c["id"]: c.get("is_preferred", random.random() < 0.6)
            for c in self.data.get("carriers", [])
        }

        origins = []
        destinations = []
        for stype in chosen_types:
            if stype == "plant_to_dc":
                origins.append(("plant", random.choice(plant_ids)))
                destinations.append(("dc", chicago_dc_id if random.random() < 0.40 else random.choice(dc_ids)))
            elif stype == "dc_to_dc":
                orig_id = random.choice(dc_ids)
                origins.append(("dc", orig_id))
                other_dcs = [d for d in dc_ids if d != orig_id]
                destinations.append(("dc", random.choice(other_dcs) if other_dcs else orig_id))
            elif stype == "dc_to_store":
                origins.append(("dc", chicago_dc_id if random.random() < 0.40 else random.choice(dc_ids)))
                destinations.append(("store", random.choice(location_ids)))
            else: # direct_to_store
                origins.append(("plant", random.choice(plant_ids)))
                destinations.append(("store", random.choice(location_ids)))

        # Vectorized generation
        gen = ShipmentsGenerator(seed=self.ctx.seed)
        
        # Prepare distances from route_segments
        distances = {}
        for seg in self.data.get("route_segments", []):
            key = (seg["origin_type"], seg["origin_id"], seg["destination_type"], seg["destination_id"])
            distances[key] = float(seg.get("distance_km", 500.0))
            
        # Carrier rates: need to join carrier_rates -> carrier_contracts -> carrier_id
        contract_to_carrier = {c["id"]: c["carrier_id"] for c in self.data.get("carrier_contracts", [])}
        carrier_rates = {}
        for r in self.data.get("carrier_rates", []):
            cid = contract_to_carrier.get(r["contract_id"])
            if cid:
                # Use rate_per_case as a proxy for per-km rate if missing, 
                # or just a reasonable default if multiple rates exist.
                carrier_rates[cid] = float(r.get("rate_per_case", 1.50)) / 100.0 + 1.0 # arbitrary scaling
        
        gen.configure(carrier_rates=carrier_rates, distances=distances)

        # Mass Balance Fix: Only store-bound shipments count toward shipped cases
        # Internal moves (plant_to_dc, dc_to_dc) are intermediate and don't count
        chosen_types_arr = np.array(chosen_types)
        store_bound_mask = np.isin(chosen_types_arr, ["dc_to_store", "direct_to_store"])
        num_store_shipments = store_bound_mask.sum()

        # Calculate total shippable cases from upstream physics
        total_shippable_cases = sum(shippable_to_store_by_sku.values())
        avg_kg_per_case = sum(sku_weights.values()) / len(sku_weights) if sku_weights else 5.0

        # Allocate cases only to store-bound shipments (fixes mass balance)
        cases_per_shipment = np.zeros(num_shipments, dtype=np.int32)
        if num_store_shipments > 0:
            avg_cases_per_store_shipment = total_shippable_cases / num_store_shipments
            store_cases = (avg_cases_per_store_shipment * np.random.uniform(0.6, 1.4, size=num_store_shipments)).astype(np.int32)
            store_cases = np.maximum(store_cases, 1)
            cases_per_shipment[store_bound_mask] = store_cases

        # Internal shipments get nominal case counts for truck fill calculation (not counted in mass balance)
        internal_mask = ~store_bound_mask
        if internal_mask.sum() > 0:
            # These move goods between facilities but don't count as "shipped to stores"
            avg_internal_cases = 500  # Typical internal move size
            cases_per_shipment[internal_mask] = (avg_internal_cases * np.random.uniform(0.5, 1.5, size=internal_mask.sum())).astype(np.int32)
            cases_per_shipment[internal_mask] = np.maximum(cases_per_shipment[internal_mask], 1)

        # Derive weights FROM cases using actual SKU weight
        weights_kg = (cases_per_shipment * avg_kg_per_case).astype(np.float32)

        store_cases_total = cases_per_shipment[store_bound_mask].sum()
        print(f"    [Physics] Store shipments: {num_store_shipments:,}, cases: {store_cases_total:,} (target: {total_shippable_cases:,})")
        print(f"    [Physics] Avg {avg_kg_per_case:.2f} kg/case")

        ship_dates = np.array([
            np.datetime64(date(self.ctx.base_year, 1, 1)) + np.timedelta64(random.randint(0, 364), "D")
            for _ in range(num_shipments)
        ])

        # Build CTS factor arrays
        chosen_carrier_ids = np.array([random.choice(carrier_ids) for _ in range(num_shipments)])

        # Temperature control from origin DC
        is_temperature_controlled = np.array([
            dc_temp_by_id.get(orig_id, False) if orig_type == "dc" else False
            for orig_type, orig_id in origins
        ])

        # Carrier preferred status
        carrier_is_preferred = np.array([
            carrier_preferred.get(cid, random.random() < 0.6)
            for cid in chosen_carrier_ids
        ])

        shipments_array = gen.generate_batch(
            origins=origins,
            destinations=destinations,
            weights_kg=weights_kg,
            total_cases=cases_per_shipment,
            shipment_types=np.array(chosen_types),
            carrier_ids=chosen_carrier_ids,
            route_ids=np.array([random.choice(route_ids) for _ in range(num_shipments)]),
            base_dates=ship_dates,
            now=now,
            is_temperature_controlled=is_temperature_controlled,
            carrier_is_preferred=carrier_is_preferred,
        )
        
        self.data["shipments"] = structured_to_dicts(shipments_array)
        for s in self.data["shipments"]:
            self.ctx.shipment_ids[s["shipment_number"]] = s["id"]

        print(f"      Generated {len(self.data['shipments']):,} shipments")
        self._generate_inventory(now)

    def _generate_inventory(self, now: datetime) -> None:
        """
        Generate inventory table based on calculated SKU-level remaining production.
        Ensures strict mass balance: Inventory = Production - Shipped (to stores).

        Tags inventory as 'safety_stock' or 'cycle_stock' based on demand coverage:
        - Safety stock: first 14 days of demand coverage
        - Cycle stock: remainder above safety stock threshold
        """
        inventory_by_sku = getattr(self, "inventory_by_sku", {})
        total_inv_cases = sum(inventory_by_sku.values())
        print(f"    Generating inventory (SKU-Level: {total_inv_cases:,} cases)...")

        batches_idx = LookupBuilder.build_unique(self.data["batches"], "id")
        dc_ids = list(self.ctx.dc_ids.values())
        batches_by_formula = LookupBuilder.build(self.data["batches"], key_field="formula_id")
        sku_formula_map = {sku["id"]: sku.get("formula_id") for sku in self.data["skus"]}

        # Calculate demand-based safety stock thresholds per SKU
        demand_by_sku = {}
        for ol in self.data.get("order_lines", []):
            sid = ol.get("sku_id")
            demand_by_sku[sid] = demand_by_sku.get(sid, 0) + ol.get("quantity_cases", 0)

        SAFETY_STOCK_DAYS = 14  # 2 weeks coverage
        safety_stock_threshold = {}
        for sid in inventory_by_sku:
            daily_demand = demand_by_sku.get(sid, 0) / 365
            safety_stock_threshold[sid] = int(daily_demand * SAFETY_STOCK_DAYS)

        inv_id = 1
        total_actual_cases = 0
        safety_stock_cases = 0
        cycle_stock_cases = 0
        eligible_skus = [sid for sid, qty in inventory_by_sku.items() if qty > 0]

        # Track cumulative inventory generated per SKU for tagging
        cumulative_by_sku = {sid: 0 for sid in eligible_skus}

        for sid in eligible_skus:
            formula_id = sku_formula_map.get(sid)
            compatible_batches = [b["id"] for b in batches_by_formula.get(formula_id, [])]
            if not compatible_batches:
                continue

            remaining_qty = inventory_by_sku[sid]
            threshold = safety_stock_threshold.get(sid, 0)

            # Distribute across random DCs
            target_dcs = random.sample(dc_ids, min(random.randint(1, 3), len(dc_ids)))

            for dc_id in target_dcs:
                if remaining_qty <= 0:
                    break

                # Use a small number of lots to prevent record explosion
                num_lots = random.randint(1, min(2, len(compatible_batches)))
                selected_batches = random.sample(compatible_batches, num_lots)

                for batch_id in selected_batches:
                    if remaining_qty <= 0:
                        break

                    batch = batches_idx.get(batch_id)
                    # Don't make individual records too large, but ensure we use all remaining_qty
                    qty_cases = min(remaining_qty, random.randint(200, 2000))
                    if batch_id == selected_batches[-1] and dc_id == target_dcs[-1]:
                        qty_cases = remaining_qty  # Ensure full depletion

                    # Determine inventory type based on cumulative position vs safety stock
                    cumulative_before = cumulative_by_sku[sid]
                    cumulative_after = cumulative_before + qty_cases

                    if cumulative_before < threshold:
                        # Some or all of this record is safety stock
                        if cumulative_after <= threshold:
                            inventory_type = "safety_stock"
                            safety_stock_cases += qty_cases
                        else:
                            # Split: first part is safety stock, rest is cycle stock
                            # For simplicity, tag entire record based on midpoint
                            midpoint = cumulative_before + qty_cases // 2
                            if midpoint < threshold:
                                inventory_type = "safety_stock"
                                safety_stock_cases += qty_cases
                            else:
                                inventory_type = "cycle_stock"
                                cycle_stock_cases += qty_cases
                    else:
                        inventory_type = "cycle_stock"
                        cycle_stock_cases += qty_cases

                    cumulative_by_sku[sid] = cumulative_after
                    remaining_qty -= qty_cases
                    total_actual_cases += qty_cases

                    self.ctx.inventory_ids[(dc_id, sid, batch_id)] = inv_id
                    self.data["inventory"].append(
                        {
                            "id": inv_id,
                            "location_type": "dc",
                            "location_id": dc_id,
                            "sku_id": sid,
                            "batch_id": batch_id,
                            "quantity_cases": qty_cases,
                            "quantity_eaches": qty_cases * 12,
                            "lot_number": batch.get("batch_number", "LOT-000"),
                            "expiry_date": batch.get("expiry_date", date(2025, 12, 31)),
                            "last_movement_date": batch.get("production_date", date(2024, 1, 1)) + timedelta(days=random.randint(1, 14)),
                            "aging_bucket": random.choice(["0-30", "31-60", "61-90", "90+"]),
                            "inventory_type": inventory_type,  # safety_stock or cycle_stock
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    inv_id += 1

        # Apply phantom inventory quirk
        if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("phantom_inventory"):
            self.data["inventory"] = self.ctx.quirks_manager.apply_phantom_inventory(
                self.data["inventory"], reference_date=now
            )

        print(f"      Generated {inv_id - 1:,} inventory records ({total_actual_cases:,} cases)")
        print(f"        Safety stock: {safety_stock_cases:,} cases, Cycle stock: {cycle_stock_cases:,} cases")



    def _generate_shipment_legs(self, now: datetime) -> None:
        """Generate shipment_legs table (~360K) using vectorized generator."""
        print("    Generating shipment_legs (Vectorized)...")
        segment_ids = list(self.ctx.route_segment_ids.values())
        carrier_ids = list(self.ctx.carrier_ids.values()) if self.ctx.carrier_ids else [1]

        # Extract data for vectorized generator
        shipment_ids = np.array([s["id"] for s in self.data["shipments"]], dtype=np.int64)
        shipment_types = np.array([s["shipment_type"] for s in self.data["shipments"]])
        ship_dates = np.array([np.datetime64(s["ship_date"], "D") for s in self.data["shipments"]])

        generator = ShipmentLegsGenerator(seed=self.ctx.seed, base_year=self.ctx.base_year)
        generator.configure(
            shipment_ids=shipment_ids,
            route_segment_ids=np.array(segment_ids, dtype=np.int64),
            carrier_ids=np.array(carrier_ids, dtype=np.int64)
        )

        legs_array = generator.generate_for_shipments(
            shipment_ids=shipment_ids,
            shipment_types=shipment_types,
            ship_dates=ship_dates,
            now=now
        )

        self.data["shipment_legs"] = structured_to_dicts(legs_array)
        print(f"      Generated {len(self.data['shipment_legs']):,} shipment_legs")

    def _apply_chaos_logistics(self) -> None:
        """Apply chaos injection to logistics (port strike, congestion)."""
        # RSK-LOG-002: Port strike delays
        if self.ctx.risk_manager:
            port_overrides = self.ctx.risk_manager.get_port_strike_overrides()
            if port_overrides:
                affected_port_codes = port_overrides.get("affected_ports", ["USLAX"])
                delay_multiplier = port_overrides.get("delay_multiplier", 4.0)

                affected_port_ids = {
                    self.ctx.port_ids.get(code)
                    for code in affected_port_codes
                    if code in self.ctx.port_ids
                }

                segment_lookup = {seg["id"]: seg for seg in self.data["route_segments"]}

                strike_count = 0
                for leg in self.data["shipment_legs"]:
                    seg_id = leg.get("route_segment_id")
                    if seg_id and seg_id in segment_lookup:
                        seg = segment_lookup[seg_id]
                        origin_is_port = (
                            seg.get("origin_type") == "port"
                            and seg.get("origin_id") in affected_port_ids
                        )
                        dest_is_port = (
                            seg.get("destination_type") == "port"
                            and seg.get("destination_id") in affected_port_ids
                        )
                        if origin_is_port or dest_is_port:
                            current_transit = leg.get("actual_transit_hours", 24)
                            extra_delay = current_transit * (delay_multiplier - 1)
                            leg["actual_transit_hours"] = round(current_transit + extra_delay, 2)
                            leg["status"] = "delayed"
                            leg["notes"] = f"RISK EVENT RSK-LOG-002: Port strike delay (+{extra_delay:.1f}h)"
                            strike_count += 1

                if strike_count > 0:
                    print(f"    [Risk] RSK-LOG-002: {strike_count} shipment legs delayed by port strike")

        # Port congestion flicker quirk
        if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("port_congestion_flicker"):
            segment_lookup = {seg["id"]: seg for seg in self.data["route_segments"]}
            self.data["shipment_legs"] = self.ctx.quirks_manager.apply_port_congestion(
                self.data["shipment_legs"],
                segment_lookup=segment_lookup,
                port_ids=self.ctx.port_ids,
            )
            congestion_count = sum(
                1 for leg in self.data["shipment_legs"] if leg.get("congestion_affected")
            )
            if congestion_count > 0:
                print(f"    [Quirk] Port congestion flicker: {congestion_count} legs with correlated delays")


class Level11Generator(BaseLevelGenerator):
    """
    Generate Level 11 shipment lines.

    Level 11 contains shipment_lines with batch tracking.
    """

    LEVEL = 11

    def generate(self) -> None:
        """Generate all Level 11 tables."""
        print("  Level 11: Shipment lines...")
        level_start = time.time()
        now = datetime.now()

        self._generate_shipment_lines(now)
        self._generate_transit_inventory(now)

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(f"    Generated: {len(self.data['shipment_lines'])} shipment_lines ({level_elapsed:.1f}s)")

    def _generate_shipment_lines(self, now: datetime) -> None:
        """
        Generate shipment_lines table (~1M) using vectorized generator.
        Enforces Temporal Physics (Ship Date >= Prod Date) and Location Physics.
        """
        print("    Generating shipment_lines (Vectorized Physics)...")

        # Prepare SKU data
        sku_ids = list(self.ctx.sku_ids.values())
        sku_weight_kg = {}
        for sku in self.data["skus"]:
            size_str = sku.get("size", "6oz")
            try:
                size_val = float(size_str.replace("oz", "").replace("ml", "").replace("g", ""))
                if "oz" in size_str:
                    sku_weight_kg[sku["id"]] = round(size_val * 0.028 * 24, 2)
                elif "ml" in size_str:
                    sku_weight_kg[sku["id"]] = round(size_val * 0.001 * 24, 2)
                else:
                    sku_weight_kg[sku["id"]] = 5.0
            except ValueError:
                sku_weight_kg[sku["id"]] = 5.0

        # Prepare batch data for Temporal Physics
        batch_ids = [b["id"] for b in self.data["batches"]]
        batch_output_cases = {b["id"]: b.get("output_cases", 1000) for b in self.data["batches"]}
        batch_numbers = {b["id"]: b.get("batch_number", f"LOT-{b['id']:08d}") for b in self.data["batches"]}
        batch_expiry = {b["id"]: b.get("expiry_date") for b in self.data["batches"]}
        batch_production_dates = {b["id"]: b.get("production_date") for b in self.data["batches"]}

        # Prepare location data for Location-Bound Physics
        # We need plant_to_skus from Level 10
        # Actually, let's just use the saved plant_to_skus if origin is plant
        # For DCs, we can assume all SKUs for now or filter by DC inventory
        origin_skus = getattr(self.ctx, "plant_to_skus", {}) # Assuming we might have it in context? 
        # Wait, I saved it to self.plant_to_skus in Level10Generator, but Level11 is a different object.
        # I should save it to self.ctx.
        
        # Let's check Level10Generator again. I saved it to self.plant_to_skus.
        # I'll modify Level10Generator to save it to self.ctx.plant_to_skus.
        
        # For now, I'll rebuild it or assume it's in ctx.
        if not hasattr(self.ctx, "plant_to_skus"):
            # Rebuild if missing (shouldn't happen if Level 10 ran)
            pass

        # Extract shipment arrays
        shipments = self.data["shipments"]
        shipment_ids = np.array([s["id"] for s in shipments], dtype=np.int64)
        shipment_origins = np.array([s["origin_id"] for s in shipments], dtype=np.int64)
        total_cases = np.array([s.get("total_cases", 100) for s in shipments], dtype=np.int32)
        ship_dates = np.array([
            np.datetime64(s.get("ship_date", date(self.ctx.base_year, 6, 15)), "D")
            for s in shipments
        ], dtype="datetime64[D]")

        # Create and configure vectorized generator
        generator = ShipmentLinesGenerator(seed=self.ctx.seed, base_year=self.ctx.base_year)
        generator.configure(
            sku_ids=sku_ids,
            batch_ids=batch_ids,
            sku_weight_kg=sku_weight_kg,
            batch_output_cases=batch_output_cases,
            batch_numbers=batch_numbers,
            batch_expiry=batch_expiry,
            batch_production_dates=batch_production_dates,
            origin_skus=getattr(self.ctx, "plant_to_skus", {})
        )

        # Generate all shipment lines in one vectorized call
        lines_array = generator.generate_for_shipments(
            shipment_ids=shipment_ids,
            shipment_origins=shipment_origins,
            total_cases=total_cases,
            ship_dates=ship_dates,
            now=now,
        )

        # Convert to list of dicts
        self.data["shipment_lines"] = structured_to_dicts(lines_array)

        print(f"      Generated {len(self.data['shipment_lines']):,} shipment_lines")

    def _generate_transit_inventory(self, now: datetime) -> None:
        """
        Generate inventory records for in_transit shipments.

        Uses location_type='in_transit' with shipment_id as location_id.
        This enables the inventory waterfall view to show goods in transit.
        """
        print("    Generating transit inventory...")

        # Get in_transit shipments
        in_transit_shipments = [s for s in self.data["shipments"] if s["status"] == "in_transit"]

        if not in_transit_shipments:
            print("      No in_transit shipments found")
            return

        # Build shipment_id -> shipment_lines lookup
        lines_by_shipment = {}
        for line in self.data["shipment_lines"]:
            sid = line["shipment_id"]
            if sid not in lines_by_shipment:
                lines_by_shipment[sid] = []
            lines_by_shipment[sid].append(line)

        # Get next inventory ID (continue from existing inventory)
        next_inv_id = len(self.data["inventory"]) + 1
        transit_count = 0

        for shipment in in_transit_shipments:
            ship_id = shipment["id"]
            ship_date = shipment.get("ship_date", date(2024, 1, 1))

            for line in lines_by_shipment.get(ship_id, []):
                self.data["inventory"].append({
                    "id": next_inv_id,
                    "location_type": "in_transit",
                    "location_id": ship_id,  # Polymorphic: points to shipment
                    "sku_id": line["sku_id"],
                    "batch_id": line["batch_id"],
                    "quantity_cases": line["quantity_cases"],
                    "quantity_eaches": line.get("quantity_eaches", line["quantity_cases"] * 12),
                    "lot_number": line.get("lot_number"),
                    "expiry_date": line.get("expiry_date"),
                    "last_movement_date": ship_date if isinstance(ship_date, date) else date(2024, 1, 1),
                    "aging_bucket": "0-30",  # Transit is always fresh
                    "created_at": now,
                    "updated_at": now,
                })
                next_inv_id += 1
                transit_count += 1

        print(f"      Generated {transit_count:,} transit inventory records")
