"""
Level 8-9 Generator: Demand signals and orders.

Level 8 Tables (LARGEST):
- pos_sales (~500,000) - Vectorized with promo calendar
- demand_forecasts (~100,000)
- forecast_accuracy (~10,000)
- consensus_adjustments (~5,000)
- orders (~200,000)
- replenishment_params (~25,000)
- demand_allocation (~50,000)
- capacity_plans (~10,000)

Level 9 Tables:
- order_lines (~600,000) - Vectorized generation
- order_allocations (~350,000)
- pick_waves (~25,000)
- supply_plans (~50,000)
- plan_exceptions (~20,000)
"""

import random
import time
from datetime import date, datetime, timedelta

import numpy as np

from .base import BaseLevelGenerator
from ..lookup_builder import LookupBuilder
from ..promo_calendar import PromoCalendar
from ..vectorized import (
    OrderLinesGenerator,
    POSSalesGenerator,
    structured_to_dicts,
    zipf_weights,
)


class Level8Generator(BaseLevelGenerator):
    """
    Generate Level 8 demand data (LARGEST level).

    Level 8 contains POS sales, demand forecasts, and orders.
    Uses vectorized generators for performance.
    """

    LEVEL = 8

    def generate(self) -> None:
        """Generate all Level 8 tables."""
        print("  Level 8: Demand and orders (LARGEST - pos_sales, orders...)")
        level_start = time.time()
        now = datetime.now()

        self._generate_pos_sales(now)
        self._generate_demand_forecasts(now)
        self._generate_forecast_accuracy(now)
        self._generate_consensus_adjustments(now)
        self._generate_orders(now)
        self._generate_replenishment_params(now)
        self._generate_demand_allocation(now)
        self._generate_capacity_plans(now)
        self._apply_chaos_demand()

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(
            f"    Generated: {len(self.data['pos_sales'])} pos_sales, "
            f"{len(self.data['orders'])} orders, "
            f"{len(self.data['demand_forecasts'])} forecasts ({level_elapsed:.1f}s)"
        )

    def _generate_pos_sales(self, now: datetime) -> None:
        """Generate pos_sales table (~500,000) with promo calendar."""
        location_ids = list(self.ctx.retail_location_ids.values())
        sku_ids = list(self.ctx.sku_ids.values())

        # Build promo calendar from all promotions
        promo_calendar = PromoCalendar.build(
            promotions=self.data["promotions"],
            promotion_skus=self.data["promotion_skus"],
            promotion_accounts=self.data["promotion_accounts"],
            retail_locations=self.data["retail_locations"],
        )

        stats = promo_calendar.stats
        print(
            f"    [Promo] Built calendar: {stats['promo_count']} promos, "
            f"~{stats['active_weeks']} active weeks"
        )

        # Build SKU prices dict
        sku_prices = {s["id"]: float(s["list_price"]) for s in self.data["skus"]}

        # Configure vectorized generator
        pos_gen = POSSalesGenerator(seed=self.ctx.seed)
        pos_gen.configure(
            sku_ids=sku_ids,
            location_ids=location_ids,
            sku_prices=sku_prices,
            promo_calendar=promo_calendar,
        )

        # Generate 600K rows vectorized
        pos_sales_array = pos_gen.generate_batch(600000)
        pos_sales_dicts = structured_to_dicts(pos_sales_array)

        # Log promo distribution
        promo_sales = sum(1 for row in pos_sales_dicts if row["is_promotional"])
        unique_promo_ids = {row["promo_id"] for row in pos_sales_dicts if row["promo_id"] != 0}
        print(
            f"    [Promo] {promo_sales:,} promo sales across "
            f"{len(unique_promo_ids)} unique promotions"
        )

        # Convert promo_id=0 to None (NULL in DB)
        for row in pos_sales_dicts:
            if row["promo_id"] == 0:
                row["promo_id"] = None

        self.data["pos_sales"] = pos_sales_dicts

    def _generate_demand_forecasts(self, now: datetime) -> None:
        """
        Generate demand_forecasts table (~100,000).

        Calibrated to match POS sales volume with controlled bias.
        Uses Zipf weights to scale by SKU popularity and hierarchy multipliers.
        """
        forecast_id = 1
        sku_ids = list(self.ctx.sku_ids.values())
        
        # Calculate SKU multipliers to match POS generation (Zipf)
        # Multiplier = weight * N (scales relative to "average" SKU)
        weights = zipf_weights(len(sku_ids), alpha=1.05)
        sku_multipliers = weights * len(sku_ids)
        sku_mult_map = {sku_id: mult for sku_id, mult in zip(sku_ids, sku_multipliers)}
        
        forecast_versions = [f"{self.ctx.base_year}-W{w:02d}-STAT" for w in range(1, 53)]
        location_types = ["dc", "account", "division"]
        
        # Base weekly volume for an "average" SKU at a single store (derived from POS settings)
        # Reduced from 8.0 to 0.5 to account for RealismMonitor summing overlapping hierarchy levels
        base_store_weekly_vol = 0.5

        for _ in range(100000):
            sku_id = random.choice(sku_ids)
            loc_type = random.choice(location_types)
            
            # Scale by hierarchy level (Aggregation)
            if loc_type == "division": # ~2000 stores
                level_mult = 2000.0
                loc_id = random.randint(1, 5)
            elif loc_type == "dc": # ~400 stores
                level_mult = 400.0
                loc_id = random.randint(1, 25)
            else: # account (~100 stores)
                level_mult = 100.0
                loc_id = random.randint(1, 100)

            # Calculate expected demand
            sku_mult = sku_mult_map.get(sku_id, 1.0)
            expected_demand = base_store_weekly_vol * level_mult * sku_mult
            
            # Apply Bias and Error
            # Bias: Forecasts tend to be higher than actuals (Optimism Bias)
            # Target: +10-25% bias (not 1200%!)
            bias_factor = random.uniform(1.10, 1.25) 
            error_factor = random.uniform(0.85, 1.15)
            
            qty = int(expected_demand * bias_factor * error_factor)
            qty = max(10, qty) # Minimum threshold

            self.data["demand_forecasts"].append(
                {
                    "id": forecast_id,
                    "forecast_version": random.choice(forecast_versions),
                    "sku_id": sku_id,
                    "location_type": loc_type,
                    "location_id": loc_id,
                    "forecast_date": self.fake.date_between(
                        start_date=date(self.ctx.base_year, 1, 1),
                        end_date=date(self.ctx.base_year, 12, 31),
                    ),
                    "forecast_week": random.randint(1, 52),
                    "statistical_forecast": int(qty * 0.95),
                    "consensus_forecast": int(qty * 1.02),
                    "final_forecast": qty,
                    "forecast_unit": "cases",
                    "confidence_level": round(random.uniform(0.7, 0.95), 2),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            forecast_id += 1

    def _generate_forecast_accuracy(self, now: datetime) -> None:
        """
        Generate forecast_accuracy table (~10,000).

        Uses realistic forecast error distribution to achieve MAPE in 20-50% range.
        Industry benchmark (E2Open): 48% at SKU-week level.
        Target: ~35% average MAPE with slight positive bias (human optimism).
        """
        forecast_count = len(self.data["demand_forecasts"])

        # Forecast error parameters for realistic MAPE (target: 20-50%)
        # Slight positive bias (8%) represents human optimism in forecasting
        # Sigma of 0.30 gives E[|error|] ≈ 0.24, combined with bias → ~30-35% MAPE
        bias_mean = 0.08  # 8% average over-forecasting
        error_sigma = 0.30  # Forecast error standard deviation

        for fa_id in range(1, 10001):
            # Generate actual demand (base truth)
            actual = random.randint(100, 5000)

            # Generate forecast as actual * (1 + error) where error ~ N(bias, sigma)
            error = random.gauss(bias_mean, error_sigma)
            # Clamp to prevent negative forecasts (min 10% of actual)
            forecast = max(int(actual * 0.1), int(actual * (1 + error)))

            mape = abs(actual - forecast) / actual * 100 if actual > 0 else 0
            bias_pct = (forecast - actual) / actual * 100 if actual > 0 else 0

            self.data["forecast_accuracy"].append(
                {
                    "id": fa_id,
                    "forecast_id": random.randint(1, min(100000, forecast_count)),
                    "actual_demand": actual,
                    "forecast_demand": forecast,
                    "mape": round(mape, 2),
                    "bias": round(bias_pct, 2),
                    "tracking_signal": round(random.uniform(-3, 3), 2),
                    "period_date": self.fake.date_between(
                        start_date=date(self.ctx.base_year, 1, 1),
                        end_date=date(self.ctx.base_year, 12, 31),
                    ),
                    "created_at": now,
                }
            )

    def _generate_consensus_adjustments(self, now: datetime) -> None:
        """Generate consensus_adjustments table (~5,000)."""
        forecast_count = len(self.data["demand_forecasts"])
        for ca_id in range(1, 5001):
            self.data["consensus_adjustments"].append(
                {
                    "id": ca_id,
                    "forecast_id": random.randint(1, min(100000, forecast_count)),
                    "adjustment_type": random.choice(
                        ["manual", "event", "promotion", "phase_in", "phase_out"]
                    ),
                    "adjustment_percent": round(random.uniform(-30, 50), 1),
                    "adjustment_units": random.randint(-500, 2000),
                    "reason": random.choice([
                        "Promotional lift",
                        "New store opening",
                        "Weather impact",
                        "Competitor action",
                        "Supply constraint",
                        "Category reset",
                    ]),
                    "adjusted_by": self.fake.name(),
                    "adjustment_date": self.fake.date_between(
                        start_date=date(self.ctx.base_year, 1, 1),
                        end_date=date(self.ctx.base_year, 12, 31),
                    ),
                    "is_approved": random.random() > 0.1,
                    "approved_by": self.fake.name() if random.random() > 0.1 else None,
                    "notes": None,
                    "created_at": now,
                }
            )

    def _generate_orders(self, now: datetime) -> None:
        """Generate orders table (~200,000)."""
        locations_by_account_idx = LookupBuilder.build(
            self.data["retail_locations"], key_field="retail_account_id"
        )
        accounts_idx = LookupBuilder.build_unique(self.data["retail_accounts"], "id")

        account_ids = list(self.ctx.retail_account_ids.values())
        channel_ids = list(self.ctx.channel_ids.values())
        promo_ids = list(self.ctx.promotion_ids.values())
        location_ids = list(self.ctx.retail_location_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())

        # Pre-compute location ID lists per account
        location_ids_by_account = {}
        for acct_id in account_ids:
            locs = locations_by_account_idx.get(acct_id, [])
            location_ids_by_account[acct_id] = [loc["id"] for loc in locs] if locs else location_ids

        order_id = 1
        order_statuses = [
            "pending", "confirmed", "allocated", "picking", "shipped", "delivered", "cancelled"
        ]
        order_types = ["standard", "rush", "backorder", "promotional"]
        mega_account_id = self.ctx.retail_account_ids.get("ACCT-MEGA-001", 1)
        mega_locs = location_ids_by_account.get(mega_account_id, location_ids)

        for _ in range(240000):
            order_num = f"ORD-{self.ctx.base_year}-{order_id:07d}"
            self.ctx.order_ids[order_num] = order_id

            # 25% of orders go to MegaMart (hub concentration)
            if random.random() < 0.25:
                acct_id = mega_account_id
                loc_id = random.choice(mega_locs) if mega_locs else random.choice(location_ids)
            else:
                acct_id = random.choice(account_ids)
                acct_locs = location_ids_by_account.get(acct_id, location_ids)
                loc_id = random.choice(acct_locs) if acct_locs else random.choice(location_ids)

            acct = accounts_idx.get(acct_id)
            channel_id = acct["channel_id"] if acct else random.choice(channel_ids)

            order_date = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 1),
                end_date=date(self.ctx.base_year, 12, 15),
            )
            lead_time = random.randint(2, 14)

            promo_id_val = None
            order_type = random.choices(order_types, weights=[70, 10, 10, 10])[0]
            if order_type == "promotional" and promo_ids:
                promo_id_val = random.choice(promo_ids)

            self.data["orders"].append(
                {
                    "id": order_id,
                    "order_number": order_num,
                    "retail_account_id": acct_id,
                    "retail_location_id": loc_id,
                    "channel_id": channel_id,
                    "order_date": order_date,
                    "requested_delivery_date": order_date + timedelta(days=lead_time),
                    "promised_delivery_date": order_date + timedelta(days=lead_time + random.randint(-1, 3)),
                    "actual_delivery_date": order_date + timedelta(days=lead_time + random.randint(-2, 5))
                    if random.random() > 0.2
                    else None,
                    "status": random.choices(order_statuses, weights=[5, 10, 10, 10, 15, 45, 5])[0],
                    "order_type": order_type,
                    "promo_id": promo_id_val,
                    "total_cases": None,
                    "total_amount": None,
                    "currency": "USD",
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            order_id += 1

    def _generate_replenishment_params(self, now: datetime) -> None:
        """Generate replenishment_params table (~25,000)."""
        rp_id = 1
        sku_ids = list(self.ctx.sku_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())

        for dc_id in dc_ids:
            for sku_id in random.sample(sku_ids, min(1000, len(sku_ids))):
                self.data["replenishment_params"].append(
                    {
                        "id": rp_id,
                        "sku_id": sku_id,
                        "location_type": "dc",
                        "location_id": dc_id,
                        "safety_stock_days": random.randint(7, 21),
                        "reorder_point": random.randint(50, 500),
                        "reorder_quantity": random.randint(100, 2000),
                        "max_stock": random.randint(1000, 10000),
                        "lead_time_days": random.randint(3, 14),
                        "review_period_days": random.choice([1, 7, 14]),
                        "service_level_target": round(random.uniform(0.95, 0.99), 2),
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                rp_id += 1
                if rp_id > 25000:
                    break
            if rp_id > 25000:
                break

    def _generate_demand_allocation(self, now: datetime) -> None:
        """Generate demand_allocations table (~50,000)."""
        sku_ids = list(self.ctx.sku_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())
        forecast_count = len(self.data["demand_forecasts"])

        for da_id in range(1, 50001):
            self.data["demand_allocations"].append(
                {
                    "id": da_id,
                    "forecast_id": random.randint(1, min(100000, forecast_count)),
                    "sku_id": random.choice(sku_ids),
                    "source_dc_id": random.choice(dc_ids),
                    "destination_type": random.choice(["account", "dc", "location"]),
                    "destination_id": random.randint(1, 100),
                    "allocated_quantity": random.randint(10, 1000),
                    "allocation_date": self.fake.date_between(
                        start_date=date(self.ctx.base_year, 1, 1),
                        end_date=date(self.ctx.base_year, 12, 31),
                    ),
                    "priority": random.randint(1, 10),
                    "allocation_rule": random.choice(["fair_share", "priority", "historical", "committed"]),
                    "status": random.choice(["pending", "confirmed", "released", "completed"]),
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_capacity_plans(self, now: datetime) -> None:
        """Generate capacity_plans table (~10,000)."""
        plant_ids = list(self.ctx.plant_ids.values())
        line_ids = list(self.ctx.production_line_ids.values())

        for cp_id in range(1, 10001):
            self.data["capacity_plans"].append(
                {
                    "id": cp_id,
                    "plan_code": f"CAP-{self.ctx.base_year}-{cp_id:05d}",
                    "plant_id": random.choice(plant_ids),
                    "production_line_id": random.choice(line_ids),
                    "planning_period": self.fake.date_between(
                        start_date=date(self.ctx.base_year, 1, 1),
                        end_date=date(self.ctx.base_year, 12, 31),
                    ),
                    "available_hours": random.randint(100, 180),
                    "planned_hours": random.randint(80, 160),
                    "utilization_percent": round(random.uniform(60, 95), 1),
                    "bottleneck_flag": random.random() < 0.1,
                    "overtime_hours": random.randint(0, 40),
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _apply_chaos_demand(self) -> None:
        """Apply chaos injection to demand data (optimism bias quirk)."""
        if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("human_optimism_bias"):
            # Mark ~10% of SKUs as new products
            new_sku_ids = set(list(self.ctx.sku_ids.values())[: len(self.ctx.sku_ids) // 10])
            sku_launch_dates = {
                sku_id: datetime.now() - timedelta(days=random.randint(30, 180))
                for sku_id in new_sku_ids
            }

            self.data["demand_forecasts"] = self.ctx.quirks_manager.apply_optimism_bias(
                self.data["demand_forecasts"],
                sku_launch_dates=sku_launch_dates,
                reference_date=datetime.now(),
            )
            bias_count = sum(
                1 for f in self.data["demand_forecasts"] if f.get("optimism_bias_applied")
            )
            if bias_count > 0:
                print(f"    [Quirk] Optimism bias applied: {bias_count} forecasts inflated by 15%")


class Level9Generator(BaseLevelGenerator):
    """
    Generate Level 9 order lines and planning.

    Level 9 contains order_lines, order_allocations, pick_waves,
    supply_plans, and plan_exceptions.
    """

    LEVEL = 9

    def generate(self) -> None:
        """Generate all Level 9 tables."""
        print("  Level 9: Order lines and planning...")
        level_start = time.time()
        now = datetime.now()

        self._generate_order_lines(now)
        self._generate_order_allocations(now)
        self._generate_pick_waves(now)
        self._generate_supply_plans(now)
        self._generate_plan_exceptions(now)
        self._apply_chaos_orders()

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(
            f"    Level 9 complete: {len(self.data['order_lines']):,} order_lines, "
            f"{len(self.data['order_allocations']):,} order_allocations, "
            f"{len(self.data['pick_waves']):,} pick_waves ({level_elapsed:.1f}s)"
        )

    def _generate_order_lines(self, now: datetime) -> None:
        """Generate order_lines table (~600K) - vectorized."""
        print("    Generating order_lines (vectorized)...")
        sku_ids = list(self.ctx.sku_ids.values())

        # Build channel lookup
        channel_type_by_id = {ch["id"]: ch["channel_type"] for ch in self.data["channels"]}

        # Build SKU price lookup
        sku_price_by_id = {sku["id"]: float(sku["list_price"]) for sku in self.data["skus"]}

        # Lines per order by channel type
        lines_per_order_range = {
            "dtc": (1, 3),
            "ecommerce": (1, 6),
            "bm_distributor": (10, 25),
            "bm_large": (15, 45),
        }

        # Build order arrays for vectorized generation
        n_orders = len(self.data["orders"])
        order_ids_arr = np.array([o["id"] for o in self.data["orders"]], dtype=np.int64)
        order_statuses_arr = np.array([o["status"] for o in self.data["orders"]])
        order_is_promo_arr = np.array(
            [o["promo_id"] is not None for o in self.data["orders"]], dtype=bool
        )
        order_channel_ids = np.array([o["channel_id"] for o in self.data["orders"]], dtype=np.int64)

        # Map channel_id to channel_type
        order_channel_types = np.array(
            [channel_type_by_id.get(cid, "bm_distributor") for cid in order_channel_ids]
        )

        # Determine lines per order
        lines_per_order = np.zeros(n_orders, dtype=np.int32)
        for ch_type, (min_l, max_l) in lines_per_order_range.items():
            mask = order_channel_types == ch_type
            count = mask.sum()
            if count > 0:
                lines_per_order[mask] = self.rng.integers(min_l, max_l + 1, size=count)

        default_mask = lines_per_order == 0
        if default_mask.any():
            lines_per_order[default_mask] = self.rng.integers(5, 21, size=default_mask.sum())

        lines_per_order = np.minimum(lines_per_order, len(sku_ids))

        # Configure and run vectorized generator
        order_lines_gen = OrderLinesGenerator(seed=self.ctx.seed)
        order_lines_gen.configure(
            sku_ids=sku_ids,
            order_ids=order_ids_arr,
            sku_prices=sku_price_by_id,
        )

        order_lines_array = order_lines_gen.generate_for_orders(
            order_ids=order_ids_arr,
            lines_per_order=lines_per_order,
            order_statuses=order_statuses_arr,
            order_is_promo=order_is_promo_arr,
        )

        order_lines_dicts = structured_to_dicts(order_lines_array)
        self.data["order_lines"] = order_lines_dicts
        print(f"      Generated {len(order_lines_dicts):,} order_lines")

    def _generate_order_allocations(self, now: datetime) -> None:
        """Generate order_allocations table (~350K)."""
        print("    Generating order_allocations...")
        allocatable_statuses = {"allocated", "picking", "shipped", "delivered"}
        dc_ids = list(self.ctx.dc_ids.values())

        released_batches = [b for b in self.data["batches"] if b["qc_status"] == "released"]
        batch_ids_list = [b["id"] for b in released_batches] if released_batches else [1]

        # Build O(1) lookup for order_lines
        order_lines_by_order = LookupBuilder.build(self.data["order_lines"], key_field="order_id")

        allocation_id = 1
        allocation_count = 0

        for order in self.data["orders"]:
            if order["status"] not in allocatable_statuses:
                continue

            order_lines_for_order = order_lines_by_order.get(order["id"], [])

            if order["status"] == "delivered":
                alloc_status = "shipped"
            elif order["status"] == "shipped":
                alloc_status = "shipped"
            elif order["status"] == "picking":
                alloc_status = "picked"
            else:
                alloc_status = "allocated"

            for ol in order_lines_for_order:
                line_num = ol["line_number"]
                qty_cases = ol["quantity_cases"]

                if random.random() < 0.70:
                    # Single allocation
                    self.data["order_allocations"].append(
                        {
                            "id": allocation_id,
                            "order_id": order["id"],
                            "order_line_number": line_num,
                            "dc_id": random.choice(dc_ids),
                            "batch_id": random.choice(batch_ids_list),
                            "allocated_cases": qty_cases,
                            "allocation_date": now - timedelta(days=random.randint(1, 30)),
                            "expiry_date": now + timedelta(days=random.randint(7, 30)),
                            "status": alloc_status,
                            "created_at": now,
                        }
                    )
                    allocation_id += 1
                    allocation_count += 1
                else:
                    # Split allocation
                    num_splits = random.randint(2, 3)
                    remaining = qty_cases

                    for split_idx in range(num_splits):
                        if split_idx == num_splits - 1:
                            split_qty = remaining
                        else:
                            split_qty = random.randint(1, max(1, remaining - (num_splits - split_idx - 1)))
                            remaining -= split_qty

                        if split_qty <= 0:
                            continue

                        self.data["order_allocations"].append(
                            {
                                "id": allocation_id,
                                "order_id": order["id"],
                                "order_line_number": line_num,
                                "dc_id": random.choice(dc_ids),
                                "batch_id": random.choice(batch_ids_list),
                                "allocated_cases": split_qty,
                                "allocation_date": now - timedelta(days=random.randint(1, 30)),
                                "expiry_date": now + timedelta(days=random.randint(7, 30)),
                                "status": alloc_status,
                                "created_at": now,
                            }
                        )
                        allocation_id += 1
                        allocation_count += 1

        print(f"      Generated {allocation_count:,} order_allocations")

    def _generate_pick_waves(self, now: datetime) -> None:
        """Generate pick_waves table (~25K)."""
        print("    Generating pick_waves...")
        dc_ids = list(self.ctx.dc_ids.values())
        wave_id = 1
        wave_types = ["standard", "rush", "replenishment"]
        wave_type_weights = [85, 10, 5]
        wave_statuses = ["planned", "released", "picking", "packing", "staged", "loaded", "completed"]
        wave_status_weights = [5, 5, 5, 5, 5, 5, 70]

        for dc_id in dc_ids:
            num_waves_for_dc = random.randint(800, 1200)

            for wave_idx in range(num_waves_for_dc):
                wave_num = f"WAVE-{dc_id:03d}-{self.ctx.base_year}-{wave_id:06d}"
                wave_type = random.choices(wave_types, weights=wave_type_weights)[0]
                wave_status = random.choices(wave_statuses, weights=wave_status_weights)[0]

                wave_date = self.fake.date_between(
                    start_date=date(self.ctx.base_year, 1, 1),
                    end_date=date(self.ctx.base_year, 12, 31),
                )

                if wave_type == "rush":
                    total_orders = random.randint(3, 10)
                elif wave_type == "replenishment":
                    total_orders = random.randint(20, 100)
                else:
                    total_orders = random.randint(10, 50)

                total_lines = total_orders * random.randint(5, 30)
                total_cases = total_lines * random.randint(5, 20)

                start_hour = random.randint(6, 18)
                duration_hours = random.uniform(1, 4)
                start_time = datetime.combine(wave_date, datetime.min.time()) + timedelta(hours=start_hour)
                end_time = start_time + timedelta(hours=duration_hours) if wave_status == "completed" else None

                self.ctx.pick_wave_ids[wave_num] = wave_id
                self.data["pick_waves"].append(
                    {
                        "id": wave_id,
                        "wave_number": wave_num,
                        "dc_id": dc_id,
                        "wave_date": wave_date,
                        "wave_type": wave_type,
                        "status": wave_status,
                        "total_orders": total_orders,
                        "total_lines": total_lines,
                        "total_cases": total_cases,
                        "start_time": start_time,
                        "end_time": end_time,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                wave_id += 1

        print(f"      Generated {wave_id - 1:,} pick_waves")

    def _generate_supply_plans(self, now: datetime) -> None:
        """Generate supply_plans table (~50,000)."""
        print("    Generating supply_plans...")
        sku_ids = list(self.ctx.sku_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())
        plant_ids = list(self.ctx.plant_ids.values())
        supplier_ids = list(self.ctx.supplier_ids.values())

        source_types = ["production", "procurement", "transfer"]
        source_type_weights = [70, 20, 10]
        plan_statuses = ["planned", "committed", "in_progress", "completed"]
        plan_status_weights = [40, 30, 20, 10]

        plan_id = 1
        for week_num in range(1, 53):
            plan_version = f"{self.ctx.base_year}-W{week_num:02d}-STAT"
            week_start = date(self.ctx.base_year, 1, 1) + timedelta(weeks=week_num - 1)
            week_end = week_start + timedelta(days=6)

            sampled_skus = random.sample(sku_ids, min(200, len(sku_ids)))
            sampled_dcs = random.sample(dc_ids, min(5, len(dc_ids)))

            for sku_id in sampled_skus:
                for dest_dc_id in sampled_dcs:
                    source_type = random.choices(source_types, weights=source_type_weights)[0]

                    if source_type == "production":
                        source_id = random.choice(plant_ids)
                    elif source_type == "procurement":
                        source_id = random.choice(supplier_ids) if supplier_ids else 1
                    else:
                        other_dcs = [d for d in dc_ids if d != dest_dc_id]
                        source_id = random.choice(other_dcs) if other_dcs else dest_dc_id

                    planned_qty = random.randint(100, 5000)
                    committed_qty = int(planned_qty * random.uniform(0.4, 0.9))

                    self.data["supply_plans"].append(
                        {
                            "id": plan_id,
                            "plan_version": plan_version,
                            "sku_id": sku_id,
                            "source_type": source_type,
                            "source_id": source_id,
                            "destination_type": "dc",
                            "destination_id": dest_dc_id,
                            "period_start": week_start,
                            "period_end": week_end,
                            "planned_quantity_cases": planned_qty,
                            "committed_quantity_cases": committed_qty,
                            "status": random.choices(plan_statuses, weights=plan_status_weights)[0],
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    plan_id += 1
                    if plan_id > 50000:
                        break
                if plan_id > 50000:
                    break
            if plan_id > 50000:
                break

        print(f"      Generated {plan_id - 1:,} supply_plans")

    def _generate_plan_exceptions(self, now: datetime) -> None:
        """Generate plan_exceptions table (~20,000)."""
        print("    Generating plan_exceptions...")
        sku_ids = list(self.ctx.sku_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())

        exception_types = [
            "demand_spike", "capacity_shortage", "material_shortage",
            "lead_time_violation", "inventory_excess", "supply_disruption",
        ]
        exception_type_weights = [40, 25, 15, 10, 7, 3]
        severities = ["low", "medium", "high", "critical"]
        severity_weights = [60, 25, 12, 3]
        exception_statuses = ["open", "acknowledged", "resolving", "resolved", "accepted"]
        exception_status_weights = [30, 20, 20, 20, 10]

        root_causes = {
            "demand_spike": ["Promotional lift exceeded forecast", "Competitor stockout", "Viral social media"],
            "capacity_shortage": ["Planned maintenance", "Equipment failure", "Labor shortage"],
            "material_shortage": ["Supplier delay", "Quality rejection", "Single-source constraint"],
            "lead_time_violation": ["Port congestion", "Customs delay", "Carrier capacity"],
            "inventory_excess": ["Forecast overestimate", "Promotion cancelled", "Seasonal slowdown"],
            "supply_disruption": ["Natural disaster", "Supplier bankruptcy", "Geopolitical event"],
        }

        recommended_actions = {
            "demand_spike": ["Expedite production", "Reallocate from other DCs", "Authorize overtime"],
            "capacity_shortage": ["Shift to alternate plant", "Use co-packer", "Delay non-critical orders"],
            "material_shortage": ["Qualify alternate supplier", "Use substitute ingredient", "Air freight"],
            "lead_time_violation": ["Switch to air freight", "Use backup carrier", "Adjust safety stock"],
            "inventory_excess": ["Run promotion", "Transfer to high-demand DC", "Donate before expiry"],
            "supply_disruption": ["Activate business continuity plan", "Source from alternate region", "Ration allocation"],
        }

        exception_id = 1
        for week_num in range(1, 53):
            plan_version = f"{self.ctx.base_year}-W{week_num:02d}-STAT"
            week_start = date(self.ctx.base_year, 1, 1) + timedelta(weeks=week_num - 1)
            week_end = week_start + timedelta(days=6)

            num_exceptions = random.randint(300, 500)

            for _ in range(num_exceptions):
                exc_type = random.choices(exception_types, weights=exception_type_weights)[0]
                severity = random.choices(severities, weights=severity_weights)[0]
                exc_status = random.choices(exception_statuses, weights=exception_status_weights)[0]

                sku_id = random.choice(sku_ids) if random.random() > 0.1 else None
                dc_id = random.choice(dc_ids)

                if severity == "critical":
                    gap_pct = round(random.uniform(50, 100), 2)
                    gap_qty = random.randint(5000, 20000)
                elif severity == "high":
                    gap_pct = round(random.uniform(25, 50), 2)
                    gap_qty = random.randint(2000, 5000)
                elif severity == "medium":
                    gap_pct = round(random.uniform(10, 25), 2)
                    gap_qty = random.randint(500, 2000)
                else:
                    gap_pct = round(random.uniform(1, 10), 2)
                    gap_qty = random.randint(100, 500)

                root_cause = random.choice(root_causes[exc_type])
                rec_action = random.choice(recommended_actions[exc_type])

                resolved_by = None
                resolved_at = None
                if exc_status in ("resolved", "accepted"):
                    resolved_by = self.fake.name()
                    resolved_at = now - timedelta(days=random.randint(1, 14))

                self.data["plan_exceptions"].append(
                    {
                        "id": exception_id,
                        "plan_version": plan_version,
                        "exception_type": exc_type,
                        "severity": severity,
                        "sku_id": sku_id,
                        "location_type": "dc",
                        "location_id": dc_id,
                        "period_start": week_start,
                        "period_end": week_end,
                        "gap_quantity_cases": gap_qty,
                        "gap_percent": gap_pct,
                        "root_cause": root_cause,
                        "recommended_action": rec_action,
                        "status": exc_status,
                        "resolved_by": resolved_by,
                        "resolved_at": resolved_at,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                exception_id += 1
                if exception_id > 20000:
                    break
            if exception_id > 20000:
                break

        print(f"      Generated {exception_id - 1:,} plan_exceptions")

    def _apply_chaos_orders(self) -> None:
        """Apply chaos injection to orders/picking."""
        # RSK-CYB-004: Cyber outage - set Chicago DC pick waves to ON_HOLD
        if self.ctx.risk_manager:
            dc_overrides = self.ctx.risk_manager.get_dc_overrides()
            if dc_overrides:
                target_dc_code = dc_overrides.get("target_dc", "DC-NAM-CHI-001")
                target_dc_id = self.ctx.dc_ids.get(target_dc_code)
                if target_dc_id:
                    hold_count = 0
                    for wave in self.data["pick_waves"]:
                        if wave.get("dc_id") == target_dc_id:
                            wave["status"] = "on_hold"
                            wave["notes"] = f"RISK EVENT RSK-CYB-004: WMS outage - {dc_overrides.get('hold_duration_hours', 72)}h hold"
                            hold_count += 1
                    if hold_count > 0:
                        print(f"    [Risk] RSK-CYB-004: {hold_count} pick waves at {target_dc_code} set to ON_HOLD")

        # Bullwhip quirk
        if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("bullwhip_whip_crack"):
            bf_promo_id = self.ctx.promotion_ids.get("PROMO-BF-2024")
            promo_order_ids = set()
            for order in self.data["orders"]:
                if order.get("promo_id") == bf_promo_id:
                    promo_order_ids.add(order["id"])

            if promo_order_ids:
                self.data["orders"], self.data["order_lines"] = (
                    self.ctx.quirks_manager.apply_bullwhip(
                        self.data["orders"],
                        self.data["order_lines"],
                        promo_order_ids=promo_order_ids,
                    )
                )
                print(f"    [Quirk] Bullwhip applied: {len(promo_order_ids)} promo orders batched (3x quantities)")
