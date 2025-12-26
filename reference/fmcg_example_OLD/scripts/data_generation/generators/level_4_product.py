"""
Level 4 Generator: SKU explosion and promotions.

Tables generated:
- skus (~2000: product x packaging x region explosion)
- sku_costs (~2000: one per SKU)
- sku_substitutes (~500: equivalency groups)
- promotions (~250: seasonal, FSI, TPR, BOGO mix)
- promotion_skus (link promotions to SKUs)
- promotion_accounts (link promotions to accounts)

Named entities:
- PROMO-BF-2024: Black Friday 2024 (3x demand, bullwhip)
"""

import random
from datetime import date, datetime, timedelta

from .base import BaseLevelGenerator
from ..constants import PACKAGING_TYPES, PRODUCTS


class Level4Generator(BaseLevelGenerator):
    """
    Generate Level 4 product and promotion data.

    Level 4 contains SKU explosion and promotion configuration.
    """

    LEVEL = 4

    def generate(self) -> None:
        """Generate all Level 4 tables."""
        print("  Level 4: SKUs and promotions...")
        now = datetime.now()

        self._generate_skus(now)
        self._generate_sku_costs(now)
        self._generate_sku_substitutes(now)
        self._generate_promotions(now)
        self._generate_promotion_skus(now)
        self._generate_promotion_accounts(now)

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['skus'])} SKUs, "
            f"{len(self.data['promotions'])} promotions, "
            f"{len(self.data['promotion_skus'])} promo-SKU links"
        )

    def _generate_skus(self, now: datetime) -> None:
        """Generate skus table (~2000: product x packaging x region explosion)."""
        sku_id = 1
        regions = ["NAM", "LATAM", "APAC", "EUR", "AFR-EUR", "GLOBAL"]
        languages = {
            "NAM": "en", "LATAM": "es", "APAC": "zh",
            "EUR": "en", "AFR-EUR": "fr", "GLOBAL": "en",
        }

        # SKU matrix: product x packaging x region (subset for realistic count)
        product_packaging = {
            "PROD-PW": ["PKG-TUBE-50", "PKG-TUBE-100", "PKG-TUBE-150", "PKG-TUBE-200", "PKG-TRIAL-30"],
            "PROD-CW": ["PKG-BOTL-250", "PKG-BOTL-500", "PKG-BOTL-750", "PKG-BOTL-1L", "PKG-REFILL-1L"],
            "PROD-AP": ["PKG-PUMP-250", "PKG-PUMP-500", "PKG-PUMP-750", "PKG-BOTL-250", "PKG-SACHET-10"],
        }

        for prod_code, pkg_codes in product_packaging.items():
            prod_id = self.ctx.product_ids[prod_code]
            prod_info = next(p for p in PRODUCTS if p["code"] == prod_code)
            brand = prod_info["brand"]

            # Get formulas for this product
            prod_formulas = [
                fid
                for fc, fid in self.ctx.formula_ids.items()
                if fc.split("-")[1] == prod_code[-2:]
            ]

            for pkg_code in pkg_codes:
                pkg_id = self.ctx.packaging_type_ids[pkg_code]
                pkg_info = next(p for p in PACKAGING_TYPES if p["code"] == pkg_code)

                for region in regions:
                    # Skip some region/product combos for realistic ~2000 SKUs
                    if region in ["AFR-EUR"] and prod_code == "PROD-AP":
                        continue  # Body wash not sold in Africa

                    lang = languages[region]
                    formula_id = random.choice(prod_formulas) if prod_formulas else None

                    code = f"SKU-{brand[:2].upper()}-{pkg_info['container'][:2].upper()}-{pkg_info['size']}{pkg_info['unit'][:1]}-{region}"
                    if code in self.ctx.sku_ids:
                        code = f"{code}-{sku_id}"

                    self.ctx.sku_ids[code] = sku_id

                    # Price based on size
                    base_price = pkg_info["size"] * 0.02 + random.uniform(1, 5)
                    shelf_life = 730 if prod_code == "PROD-PW" else 365

                    self.data["skus"].append(
                        {
                            "id": sku_id,
                            "sku_code": code,
                            "name": f"{brand} {pkg_info['name']} ({region})",
                            "product_id": prod_id,
                            "packaging_id": pkg_id,
                            "formula_id": formula_id,
                            "region": region,
                            "language": lang,
                            "upc": f"0{random.randint(10000000000, 99999999999)}",
                            "ean": f"{random.randint(1000000000000, 9999999999999)}",
                            "list_price": round(base_price, 2),
                            "currency": "USD",
                            "shelf_life_days": shelf_life,
                            "min_order_qty": pkg_info["per_case"],
                            "is_active": True,
                            "launch_date": date(2022, 1, 1),
                            "discontinue_date": None,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    sku_id += 1

        # Generate more SKUs to reach ~2000 (variants, promotional packs)
        while sku_id <= 2000:
            prod_code = random.choice(list(product_packaging.keys()))
            prod_id = self.ctx.product_ids[prod_code]
            prod_info = next(p for p in PRODUCTS if p["code"] == prod_code)
            brand = prod_info["brand"]
            pkg_code = random.choice(product_packaging[prod_code])
            pkg_id = self.ctx.packaging_type_ids[pkg_code]
            pkg_info = next(p for p in PACKAGING_TYPES if p["code"] == pkg_code)
            region = random.choice(regions)
            prod_formulas = [
                fid
                for fc, fid in self.ctx.formula_ids.items()
                if fc.split("-")[1] == prod_code[-2:]
            ]

            variant = random.choice(["PROMO", "MULTI", "VALUE", "LTD", "CLUB"])
            code = f"SKU-{brand[:2].upper()}-{variant}-{sku_id:04d}"
            self.ctx.sku_ids[code] = sku_id

            self.data["skus"].append(
                {
                    "id": sku_id,
                    "sku_code": code,
                    "name": f"{brand} {variant.title()} Pack {pkg_info['name']}",
                    "product_id": prod_id,
                    "packaging_id": pkg_id,
                    "formula_id": random.choice(prod_formulas) if prod_formulas else None,
                    "region": region,
                    "language": languages[region],
                    "upc": f"0{random.randint(10000000000, 99999999999)}",
                    "ean": f"{random.randint(1000000000000, 9999999999999)}",
                    "list_price": round(random.uniform(3, 25), 2),
                    "currency": "USD",
                    "shelf_life_days": random.choice([365, 730]),
                    "min_order_qty": pkg_info["per_case"],
                    "is_active": True,
                    "launch_date": self.fake.date_between(start_date="-2y", end_date="today"),
                    "discontinue_date": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            sku_id += 1

    def _generate_sku_costs(self, now: datetime) -> None:
        """Generate sku_costs table (~2000: one per SKU)."""
        cost_types = ["material", "labor", "overhead", "packaging", "freight"]
        for sku in self.data["skus"]:
            base_cost = sku["list_price"] * 0.45  # ~45% COGS
            for cost_type in cost_types:
                pct = {
                    "material": 0.50,
                    "labor": 0.15,
                    "overhead": 0.15,
                    "packaging": 0.12,
                    "freight": 0.08,
                }[cost_type]
                self.data["sku_costs"].append(
                    {
                        "sku_id": sku["id"],
                        "cost_type": cost_type,
                        "cost_amount": round(base_cost * pct, 4),
                        "currency": "USD",
                        "effective_from": date(2024, 1, 1),
                        "effective_to": None,
                        "created_at": now,
                        "updated_at": now,
                    }
                )

    def _generate_sku_substitutes(self, now: datetime) -> None:
        """Generate sku_substitutes table (~500: equivalency groups)."""
        sku_list = list(self.data["skus"])
        for _ in range(500):
            # Pick two random SKUs from same product
            prod_id = random.choice(list(self.ctx.product_ids.values()))
            prod_skus = [s for s in sku_list if s["product_id"] == prod_id]
            if len(prod_skus) < 2:
                continue
            sku1, sku2 = random.sample(prod_skus, 2)
            self.data["sku_substitutes"].append(
                {
                    "sku_id": sku1["id"],
                    "substitute_sku_id": sku2["id"],
                    "priority": random.randint(1, 5),
                    "substitution_ratio": round(random.uniform(0.8, 1.2), 2),
                    "is_bidirectional": random.random() < 0.7,
                    "effective_from": date(2024, 1, 1),
                    "effective_to": None,
                    "created_at": now,
                }
            )

    def _generate_promotions(self, now: datetime) -> None:
        """Generate promotions table (~250 realistic mix)."""
        promo_id = 1
        account_ids_list = list(self.ctx.retail_account_ids.values())

        # 1. SEASONAL EVENTS (12 promos) - Major retail calendar events
        seasonal_events = [
            ("New Year Reset", date(2024, 1, 1), 21, 2.5),
            ("Valentine's Day", date(2024, 2, 5), 14, 2.0),
            ("Easter/Spring", date(2024, 3, 25), 14, 2.2),
            ("Memorial Day", date(2024, 5, 20), 14, 2.0),
            ("July 4th Summer", date(2024, 6, 24), 21, 2.3),
            ("Back to School", date(2024, 8, 5), 28, 2.5),
            ("Labor Day", date(2024, 8, 26), 14, 2.0),
            ("Halloween", date(2024, 10, 14), 21, 2.2),
            ("Thanksgiving", date(2024, 11, 18), 14, 2.8),
            ("Black Friday", date(2024, 11, 25), 7, 3.5),  # PROMO-BF-2024
            ("Christmas", date(2024, 12, 2), 21, 2.5),
            ("Year End Clearance", date(2024, 12, 26), 7, 2.0),
        ]

        for event_name, start, duration, lift in seasonal_events:
            code = "PROMO-BF-2024" if "Black Friday" in event_name else f"PROMO-SEASONAL-{promo_id:03d}"
            self.ctx.promotion_ids[code] = promo_id

            end = start + timedelta(days=duration)
            num_skus = random.randint(300, 600)

            self.data["promotions"].append(
                {
                    "id": promo_id,
                    "promo_code": code,
                    "name": f"{event_name} {self.ctx.base_year}",
                    "promo_type": "seasonal",
                    "start_date": start,
                    "end_date": end,
                    "lift_multiplier": round(lift + random.uniform(-0.2, 0.2), 2),
                    "hangover_multiplier": round(random.uniform(0.55, 0.70), 2),
                    "hangover_weeks": 2,
                    "discount_percent": round(random.uniform(15, 30), 1),
                    "trade_spend_budget": round(random.uniform(500000, 2000000), 2),
                    "status": "completed" if end < date.today() else "active",
                    "notes": f"Seasonal event: {event_name}",
                    "created_at": now,
                    "updated_at": now,
                    "_promo_class": "seasonal",
                    "_target_sku_count": num_skus,
                }
            )
            promo_id += 1

        # 2. NATIONAL FSI/CIRCULAR PROMOS (52 promos) - Weekly circular features
        for week_num in range(1, 53):
            week_start = date(2024, 1, 1) + timedelta(weeks=week_num - 1)
            code = f"PROMO-FSI-W{week_num:02d}"
            self.ctx.promotion_ids[code] = promo_id
            num_skus = random.randint(150, 400)

            self.data["promotions"].append(
                {
                    "id": promo_id,
                    "promo_code": code,
                    "name": f"Weekly Circular W{week_num}",
                    "promo_type": "feature",
                    "start_date": week_start,
                    "end_date": week_start + timedelta(days=6),
                    "lift_multiplier": round(random.uniform(1.5, 2.0), 2),
                    "hangover_multiplier": round(random.uniform(0.75, 0.85), 2),
                    "hangover_weeks": 1,
                    "discount_percent": round(random.uniform(15, 25), 1),
                    "trade_spend_budget": round(random.uniform(100000, 300000), 2),
                    "status": "completed" if week_start < date.today() else "active",
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                    "_promo_class": "fsi",
                    "_target_sku_count": num_skus,
                }
            )
            promo_id += 1

        # 3. ACCOUNT-SPECIFIC TPRs (150 promos) - 3 TPRs per account
        for acct_id in account_ids_list:
            for tpr_num in range(3):
                code = f"PROMO-TPR-{promo_id:03d}"
                self.ctx.promotion_ids[code] = promo_id

                quarter_start = date(2024, 1 + tpr_num * 4, 1)
                start = self.fake.date_between(
                    start_date=quarter_start,
                    end_date=quarter_start + timedelta(days=100),
                )
                duration = random.choice([7, 14])
                end = start + timedelta(days=duration)
                num_skus = random.randint(20, 50)

                self.data["promotions"].append(
                    {
                        "id": promo_id,
                        "promo_code": code,
                        "name": f"TPR Q{tpr_num + 1} Account {acct_id}",
                        "promo_type": "tpr",
                        "start_date": start,
                        "end_date": end,
                        "lift_multiplier": round(random.uniform(2.0, 3.0), 2),
                        "hangover_multiplier": round(random.uniform(0.60, 0.75), 2),
                        "hangover_weeks": 1,
                        "discount_percent": round(random.uniform(20, 40), 1),
                        "trade_spend_budget": round(random.uniform(20000, 100000), 2),
                        "status": "completed" if end < date.today() else "active",
                        "notes": None,
                        "created_at": now,
                        "updated_at": now,
                        "_promo_class": "tpr",
                        "_target_account_id": acct_id,
                        "_target_sku_count": num_skus,
                    }
                )
                promo_id += 1

        # 4. BOGO/MULTI-BUY PROMOS (40 promos)
        for _ in range(40):
            code = f"PROMO-BOGO-{promo_id:03d}"
            self.ctx.promotion_ids[code] = promo_id

            start = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 1),
                end_date=date(self.ctx.base_year, 12, 1),
            )
            duration = random.choice([7, 14])
            end = start + timedelta(days=duration)
            num_skus = random.randint(10, 30)
            num_accounts = random.randint(30, 50)

            self.data["promotions"].append(
                {
                    "id": promo_id,
                    "promo_code": code,
                    "name": f"Buy More Save More #{promo_id}",
                    "promo_type": "bogo",
                    "start_date": start,
                    "end_date": end,
                    "lift_multiplier": round(random.uniform(2.5, 4.0), 2),
                    "hangover_multiplier": round(random.uniform(0.50, 0.65), 2),
                    "hangover_weeks": 2,
                    "discount_percent": round(random.uniform(25, 50), 1),
                    "trade_spend_budget": round(random.uniform(50000, 200000), 2),
                    "status": "completed" if end < date.today() else "active",
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                    "_promo_class": "bogo",
                    "_target_sku_count": num_skus,
                    "_target_account_count": num_accounts,
                }
            )
            promo_id += 1

    def _generate_promotion_skus(self, now: datetime) -> None:
        """Generate promotion_skus table (link promotions to SKUs)."""
        sku_ids_list = list(self.ctx.sku_ids.values())

        for promo in self.data["promotions"]:
            num_skus = promo.get("_target_sku_count", random.randint(20, 100))
            promo_skus = random.sample(sku_ids_list, min(num_skus, len(sku_ids_list)))
            for sku_id in promo_skus:
                self.data["promotion_skus"].append(
                    {
                        "promo_id": promo["id"],
                        "sku_id": sku_id,
                        "specific_discount_percent": round(random.uniform(10, 40), 1)
                        if random.random() < 0.3
                        else None,
                        "specific_lift_multiplier": round(random.uniform(1.2, 2.0), 2)
                        if random.random() < 0.2
                        else None,
                        "created_at": now,
                    }
                )

    def _generate_promotion_accounts(self, now: datetime) -> None:
        """Generate promotion_accounts table (link promotions to accounts)."""
        account_ids_list = list(self.ctx.retail_account_ids.values())

        for promo in self.data["promotions"]:
            promo_class = promo.get("_promo_class", "default")

            if promo_class == "tpr":
                # TPRs target single account
                target_acct = promo.get("_target_account_id")
                promo_accounts = [target_acct] if target_acct else [random.choice(account_ids_list)]
            elif promo_class in ("seasonal", "fsi"):
                # Seasonal and FSI are national - all accounts
                promo_accounts = account_ids_list
            elif promo_class == "bogo":
                # BOGO targets most accounts
                num_accounts = promo.get("_target_account_count", random.randint(30, 50))
                promo_accounts = random.sample(account_ids_list, min(num_accounts, len(account_ids_list)))
            else:
                # Default: 10-50 accounts
                num_accounts = random.randint(10, 50)
                promo_accounts = random.sample(account_ids_list, min(num_accounts, len(account_ids_list)))

            for acct_id in promo_accounts:
                self.data["promotion_accounts"].append(
                    {
                        "promo_id": promo["id"],
                        "retail_account_id": acct_id,
                        "trade_spend_allocation": round(promo["trade_spend_budget"] / len(promo_accounts), 2),
                        "created_at": now,
                    }
                )

        # Clean up internal markers before output
        for promo in self.data["promotions"]:
            promo.pop("_promo_class", None)
            promo.pop("_target_sku_count", None)
            promo.pop("_target_account_id", None)
            promo.pop("_target_account_count", None)
