"""
Level 1-2 Generator: Source data (Suppliers, Formulas).

Level 1 Tables:
- suppliers (200: 40 T1, 80 T2, 80 T3)
- plants (7)
- production_lines (35: ~5 per plant)
- carrier_contracts (100)
- route_segments (200)

Level 2 Tables:
- supplier_ingredients (~400 with preferential attachment)
- certifications (~500)
- formulas (~150)
- formula_ingredients (~1500)
- carrier_rates (~1000)
- routes (~50)
- route_segment_assignments (~150)
"""

import random
from datetime import date, datetime, timedelta

from .base import BaseLevelGenerator
from ..constants import DCS, INGREDIENTS, PLANTS
from ..helpers import barabasi_albert_attachment


class Level1Generator(BaseLevelGenerator):
    """
    Generate Level 1 master data.

    Level 1 contains master data with Level 0 dependencies:
    suppliers, plants, production_lines, carrier_contracts, route_segments.
    """

    LEVEL = 1

    def generate(self) -> None:
        """Generate all Level 1 tables."""
        print("  Level 1: Master data (suppliers, plants, production_lines...)")
        now = datetime.now()

        self._generate_suppliers(now)
        self._generate_plants(now)
        self._generate_production_lines(now)
        self._generate_carrier_contracts(now)
        self._generate_route_segments(now)

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['suppliers'])} suppliers, "
            f"{len(self.data['plants'])} plants, "
            f"{len(self.data['production_lines'])} lines, "
            f"{len(self.data['route_segments'])} segments"
        )

    def _generate_suppliers(self, now: datetime) -> None:
        """Generate suppliers table (200: 40 T1, 80 T2, 80 T3)."""
        supplier_id = 1
        regions = ["NAM", "LATAM", "APAC", "EUR", "AFR-EUR"]
        countries_by_region = {
            "NAM": ["USA", "Canada", "Mexico"],
            "LATAM": ["Brazil", "Argentina", "Colombia", "Chile"],
            "APAC": ["China", "India", "Japan", "South Korea", "Thailand", "Malaysia"],
            "EUR": ["Germany", "France", "UK", "Poland", "Italy", "Spain"],
            "AFR-EUR": ["Turkey", "UAE", "South Africa", "Egypt", "Saudi Arabia"],
        }

        # Named entity: Single-source Palm Oil supplier
        self.ctx.supplier_ids["SUP-PALM-MY-001"] = supplier_id
        self.data["suppliers"].append(
            {
                "id": supplier_id,
                "supplier_code": "SUP-PALM-MY-001",
                "name": "Malaysian Palm Plantations",
                "tier": 1,
                "country": "Malaysia",
                "city": "Kuala Lumpur",
                "region": "APAC",
                "contact_email": "supply@mpp.com.my",
                "contact_phone": "+60-3-1234-5678",
                "payment_terms_days": 60,
                "currency": "USD",
                "qualification_status": "qualified",
                "qualification_date": date(2020, 1, 15),
                "risk_score": 0.75,  # High risk - single source
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
        )
        supplier_id += 1

        # Generate tier distribution: 39 more T1, 80 T2, 80 T3
        tier_counts = [(1, 39), (2, 80), (3, 80)]
        for tier, count in tier_counts:
            for _ in range(count):
                region = random.choice(regions)
                country = random.choice(countries_by_region[region])
                code = f"SUP-{tier}_{supplier_id:03d}"
                self.ctx.supplier_ids[code] = supplier_id
                self.data["suppliers"].append(
                    {
                        "id": supplier_id,
                        "supplier_code": code,
                        "name": self.fake.company(),
                        "tier": tier,
                        "country": country,
                        "city": self.fake.city(),
                        "region": region,
                        "contact_email": self.fake.email(),
                        "contact_phone": self.fake.phone_number()[:20],
                        "payment_terms_days": [30, 45, 60][tier - 1],
                        "currency": "USD",
                        "qualification_status": random.choices(
                            ["qualified", "probation", "pending"], weights=[85, 10, 5]
                        )[0],
                        "qualification_date": self.fake.date_between(
                            start_date="-3y", end_date="today"
                        ),
                        "risk_score": round(
                            random.uniform(0.1, 0.6) + (tier - 1) * 0.15, 2
                        ),
                        "is_active": True,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                supplier_id += 1

    def _generate_plants(self, now: datetime) -> None:
        """Generate plants table."""
        for i, plt in enumerate(PLANTS, 1):
            div_id = self.ctx.division_ids[plt["division"]]
            self.ctx.plant_ids[plt["code"]] = i
            self.data["plants"].append(
                {
                    "id": i,
                    "plant_code": plt["code"],
                    "name": plt["name"],
                    "division_id": div_id,
                    "country": plt["country"],
                    "city": plt["city"],
                    "address": f"{random.randint(100, 9999)} Industrial Parkway",
                    "latitude": plt["lat"],
                    "longitude": plt["lon"],
                    "timezone": self.fake.timezone(),
                    "capacity_tons_per_day": plt["capacity_tons"],
                    "operating_hours_per_day": 16,
                    "operating_days_per_week": 6,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_production_lines(self, now: datetime) -> None:
        """Generate production_lines table (5 per plant = 35 total)."""
        line_id = 1
        line_types = ["mixing", "filling", "packaging", "labeling", "quality"]
        for plant_code, plant_id in self.ctx.plant_ids.items():
            for lt in line_types:
                code = f"LINE-{plant_code[-5:]}-{lt[:3].upper()}"
                self.ctx.production_line_ids[code] = line_id
                capacity = random.randint(500, 2000)
                self.data["production_lines"].append(
                    {
                        "id": line_id,
                        "line_code": code,
                        "name": f"{lt.title()} Line {line_id}",
                        "plant_id": plant_id,
                        "line_type": lt,
                        "product_family": None,  # Flexible
                        "capacity_units_per_hour": capacity,
                        "setup_time_minutes": random.randint(15, 45),
                        "changeover_time_minutes": random.randint(30, 90),
                        "oee_target": round(random.uniform(0.80, 0.90), 4),
                        "is_active": True,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                line_id += 1

    def _generate_carrier_contracts(self, now: datetime) -> None:
        """Generate carrier_contracts table (~100)."""
        contract_id = 1
        contract_types = ["annual", "spot", "dedicated", "volume_commitment"]
        carrier_ids = list(self.ctx.carrier_ids.values())

        for _ in range(100):
            carrier_id = random.choice(carrier_ids)
            start = self.fake.date_between(start_date="-1y", end_date="+3m")
            end = start + timedelta(days=random.choice([90, 180, 365]))
            contract_num = f"CTR-{self.ctx.base_year}-{contract_id:04d}"
            self.ctx.carrier_contract_ids[contract_num] = contract_id
            self.data["carrier_contracts"].append(
                {
                    "id": contract_id,
                    "contract_number": contract_num,
                    "carrier_id": carrier_id,
                    "contract_type": random.choice(contract_types),
                    "effective_from": start,
                    "effective_to": end,
                    "min_volume_commitment": random.randint(1000, 50000)
                    if random.random() > 0.3
                    else None,
                    "volume_unit": "cases",
                    "status": "active" if end > date.today() else "expired",
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            contract_id += 1

    def _generate_route_segments(self, now: datetime) -> None:
        """Generate route_segments table (~200)."""
        segment_id = 1

        # Named entity: Seasonal Shanghai->LA lane
        self.ctx.route_segment_ids["LANE-SH-LA-001"] = segment_id
        sha_port_id = self.ctx.port_ids["CNSHA"]
        lax_port_id = self.ctx.port_ids["USLAX"]
        self.data["route_segments"].append(
            {
                "id": segment_id,
                "segment_code": "LANE-SH-LA-001",
                "origin_type": "port",
                "origin_id": sha_port_id,
                "destination_type": "port",
                "destination_id": lax_port_id,
                "transport_mode": "ocean",
                "distance_km": 10500,
                "distance_miles": 6524,
                "transit_time_hours": 336,  # 14 days
                "is_seasonal": True,
                "seasonal_months": [3, 4, 5, 6, 7, 8, 9, 10, 11],  # Mar-Nov full capacity
                "capacity_reduction_percent": 50,  # 50% reduction Jan-Feb
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
        )
        segment_id += 1

        # Generate generic segments (DCs not yet created, will be linked in Level 3)
        modes = ["truck", "rail", "ocean", "air", "intermodal", "last_mile"]
        for _ in range(199):  # 199 more to get ~200 total
            mode = random.choices(modes, weights=[40, 15, 20, 5, 15, 5])[0]
            code = f"SEG-{mode[:3].upper()}-{segment_id:04d}"
            self.ctx.route_segment_ids[code] = segment_id

            # Pick random origin/dest types
            origin_type = random.choice(["plant", "dc", "port"])
            dest_type = random.choice(["dc", "port"])

            # Use placeholder IDs (will be properly linked in Level 2/3)
            origin_id = random.randint(1, 30)
            dest_id = random.randint(1, 30)

            distance = (
                random.randint(50, 15000)
                if mode == "ocean"
                else random.randint(50, 2000)
            )
            transit = distance / (
                800
                if mode == "air"
                else 80
                if mode == "truck"
                else 40
                if mode == "ocean"
                else 60
            )

            self.data["route_segments"].append(
                {
                    "id": segment_id,
                    "segment_code": code,
                    "origin_type": origin_type,
                    "origin_id": origin_id,
                    "destination_type": dest_type,
                    "destination_id": dest_id,
                    "transport_mode": mode,
                    "distance_km": round(distance, 2),
                    "distance_miles": round(distance * 0.621371, 2),
                    "transit_time_hours": round(transit, 2),
                    "is_seasonal": random.random() < 0.1,
                    "seasonal_months": None,
                    "capacity_reduction_percent": None,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            segment_id += 1


class Level2Generator(BaseLevelGenerator):
    """
    Generate Level 2 relationships and formulas.

    Level 2 contains relationship tables:
    supplier_ingredients, certifications, formulas, formula_ingredients,
    carrier_rates, routes, route_segment_assignments.
    """

    LEVEL = 2

    def generate(self) -> None:
        """Generate all Level 2 tables."""
        print("  Level 2: Relationships (supplier_ingredients, formulas...)")
        now = datetime.now()

        self._generate_supplier_ingredients(now)
        self._generate_certifications(now)
        self._generate_formulas(now)
        self._generate_formula_ingredients(now)
        self._generate_carrier_rates(now)
        self._generate_routes(now)
        self._generate_route_segment_assignments(now)

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['supplier_ingredients'])} supplier_ingredients, "
            f"{len(self.data['formulas'])} formulas, {len(self.data['carrier_rates'])} rates"
        )

    def _generate_supplier_ingredients(self, now: datetime) -> None:
        """Generate supplier_ingredients table (~400 with preferential attachment)."""
        si_id = 1
        ingredient_ids = list(self.ctx.ingredient_ids.values())
        supplier_ids = list(self.ctx.supplier_ids.values())

        # Track supplier degrees for preferential attachment
        supplier_degrees = {sid: 0 for sid in supplier_ids}

        # Named SPOFs: Palm Oil and Sorbitol single-source
        palm_ing_id = self.ctx.ingredient_ids["ING-PALM-001"]
        sorb_ing_id = self.ctx.ingredient_ids["ING-SORB-001"]
        palm_sup_id = self.ctx.supplier_ids["SUP-PALM-MY-001"]

        # Palm Oil - ONLY from SUP-PALM-MY-001 (SPOF)
        # Apply supplier opacity degradation if RSK-SUP-003 triggered
        supplier_overrides = {}
        if self.ctx.risk_manager:
            supplier_overrides = self.ctx.risk_manager.get_supplier_overrides()
        palm_otd = supplier_overrides.get("degraded_otd_rate", 0.50)
        if supplier_overrides:
            print(
                f"    [Risk] RSK-SUP-003: SUP-PALM-MY-001 OTD degraded to {palm_otd:.0%}"
            )

        self.ctx.supplier_ingredient_ids[(palm_sup_id, palm_ing_id)] = si_id
        self.data["supplier_ingredients"].append(
            {
                "id": si_id,
                "supplier_id": palm_sup_id,
                "ingredient_id": palm_ing_id,
                "supplier_part_number": "MPP-PALM-001",
                "unit_cost": 1.25,
                "currency": "USD",
                "lead_time_days": 90,  # Long lead time (problem ingredient)
                "min_order_qty": 10000,
                "order_multiple": 1000,
                "is_preferred": True,
                "is_approved": True,
                "approval_date": date(2020, 2, 1),
                "contract_start_date": date(2024, 1, 1),
                "contract_end_date": date(2025, 12, 31),
                "on_time_delivery_rate": palm_otd,
                "quality_acceptance_rate": 0.97,
                "created_at": now,
                "updated_at": now,
            }
        )
        supplier_degrees[palm_sup_id] += 1
        si_id += 1

        # Sorbitol - single source (pick a random T1 supplier)
        sorb_supplier_id = random.choice(
            [sid for sid in supplier_ids if sid != palm_sup_id]
        )
        self.ctx.supplier_ingredient_ids[(sorb_supplier_id, sorb_ing_id)] = si_id
        self.data["supplier_ingredients"].append(
            {
                "id": si_id,
                "supplier_id": sorb_supplier_id,
                "ingredient_id": sorb_ing_id,
                "supplier_part_number": f"SORB-{sorb_supplier_id:03d}",
                "unit_cost": 2.50,
                "currency": "USD",
                "lead_time_days": 21,
                "min_order_qty": 5000,
                "order_multiple": 500,
                "is_preferred": True,
                "is_approved": True,
                "approval_date": date(2021, 6, 15),
                "contract_start_date": date(2024, 1, 1),
                "contract_end_date": date(2025, 12, 31),
                "on_time_delivery_rate": 0.94,
                "quality_acceptance_rate": 0.99,
                "created_at": now,
                "updated_at": now,
            }
        )
        supplier_degrees[sorb_supplier_id] += 1
        si_id += 1

        # Generate ~400 more supplier-ingredient links using preferential attachment
        for ing_id in ingredient_ids:
            if ing_id in [palm_ing_id, sorb_ing_id]:
                continue  # Already handled

            # How many suppliers for this ingredient? (1-5, weighted toward 2-3)
            num_suppliers = random.choices([1, 2, 3, 4, 5], weights=[5, 35, 40, 15, 5])[0]

            # Select suppliers using preferential attachment
            selected = barabasi_albert_attachment(
                [supplier_degrees.get(sid, 0) for sid in supplier_ids],
                m=num_suppliers,
                rng=self.rng,
            )

            for idx in selected:
                sup_id = supplier_ids[idx]
                if (sup_id, ing_id) in self.ctx.supplier_ingredient_ids:
                    continue

                self.ctx.supplier_ingredient_ids[(sup_id, ing_id)] = si_id
                is_preferred = (
                    len([s for s in selected if supplier_ids[s] == sup_id]) == 1
                    and idx == selected[0]
                )
                self.data["supplier_ingredients"].append(
                    {
                        "id": si_id,
                        "supplier_id": sup_id,
                        "ingredient_id": ing_id,
                        "supplier_part_number": f"ING-{sup_id:03d}-{ing_id:03d}",
                        "unit_cost": round(random.uniform(0.5, 15.0), 4),
                        "currency": "USD",
                        "lead_time_days": random.randint(7, 45),
                        "min_order_qty": random.choice([100, 500, 1000, 5000]),
                        "order_multiple": random.choice([50, 100, 500]),
                        "is_preferred": is_preferred,
                        "is_approved": True,
                        "approval_date": self.fake.date_between(
                            start_date="-2y", end_date="today"
                        ),
                        "contract_start_date": date(2024, 1, 1),
                        "contract_end_date": date(2025, 12, 31),
                        "on_time_delivery_rate": round(random.uniform(0.85, 0.99), 4),
                        "quality_acceptance_rate": round(random.uniform(0.95, 0.999), 4),
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                supplier_degrees[sup_id] += 1
                si_id += 1

    def _generate_certifications(self, now: datetime) -> None:
        """Generate certifications table (~500)."""
        cert_id = 1
        cert_types = [
            "ISO9001", "ISO14001", "GMP", "Halal", "Kosher", "RSPO", "FSC", "FSSC22000"
        ]
        cert_bodies = [
            "SGS", "Bureau Veritas", "TÜV", "DNV", "Lloyd's Register", "Intertek"
        ]

        supplier_ids = list(self.ctx.supplier_ids.values())
        for sup_id in supplier_ids:
            # Each supplier gets 2-4 certifications
            num_certs = random.randint(2, 4)
            chosen_certs = random.sample(cert_types, num_certs)
            for cert_type in chosen_certs:
                issue = self.fake.date_between(start_date="-2y", end_date="-6m")
                expiry = issue + timedelta(days=random.choice([365, 730, 1095]))
                cert_code = f"CERT-{sup_id:03d}-{cert_type}"
                self.ctx.certification_ids[cert_code] = cert_id
                self.data["certifications"].append(
                    {
                        "id": cert_id,
                        "supplier_id": sup_id,
                        "certification_type": cert_type,
                        "certification_body": random.choice(cert_bodies),
                        "certificate_number": f"{cert_type}-{self.fake.bothify('???###')}",
                        "issue_date": issue,
                        "expiry_date": expiry,
                        "scope": f"Manufacturing and supply of {random.choice(['chemicals', 'ingredients', 'raw materials'])}",
                        "is_valid": expiry > date.today(),
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                cert_id += 1

    def _generate_formulas(self, now: datetime) -> None:
        """Generate formulas table (~150: 50 per product family x 3 variants)."""
        formula_id = 1
        product_formulas = {
            "PROD-PW": {"category": "oral_care", "batch_size": 500, "mix_time": 45, "cure_time": 2},
            "PROD-CW": {"category": "home_care", "batch_size": 1000, "mix_time": 30, "cure_time": 0},
            "PROD-AP": {"category": "personal_care", "batch_size": 750, "mix_time": 40, "cure_time": 1},
        }

        for prod_code, prod_info in product_formulas.items():
            prod_id = self.ctx.product_ids[prod_code]
            variants = (
                ["Original", "Fresh", "Sensitive", "Whitening", "Premium"]
                if prod_code == "PROD-PW"
                else ["Original", "Lemon", "Antibacterial", "Eco", "Concentrated"]
                if prod_code == "PROD-CW"
                else ["Original", "Moisturizing", "Energizing", "Sensitive", "Luxury"]
            )

            for variant in variants:
                for version in [1, 2, 3]:  # 3 versions each
                    code = f"FRM-{prod_code[-2:]}-{variant[:3].upper()}-V{version}"
                    self.ctx.formula_ids[code] = formula_id
                    self.data["formulas"].append(
                        {
                            "id": formula_id,
                            "formula_code": code,
                            "name": f"{prod_code.replace('PROD-', '')} {variant} Formula V{version}",
                            "product_id": prod_id,
                            "version": version,
                            "status": "approved" if version < 3 else "draft",
                            "batch_size_kg": prod_info["batch_size"],
                            "yield_percent": round(random.uniform(96, 99), 2),
                            "mix_time_minutes": prod_info["mix_time"] + random.randint(-5, 5),
                            "cure_time_hours": prod_info["cure_time"],
                            "effective_from": date(2024, 1, 1) if version == 1 else None,
                            "effective_to": None,
                            "approved_by": "R&D Director" if version < 3 else None,
                            "approved_date": date(2024, 1, 15) if version < 3 else None,
                            "notes": None,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    formula_id += 1

    def _generate_formula_ingredients(self, now: datetime) -> None:
        """Generate formula_ingredients table (~1500: ~10 per formula)."""
        # Map ingredient categories to products
        prod_ingredients = {
            "PROD-PW": [
                "abrasive", "humectant", "flavor", "active", "thickener",
                "preservative", "colorant", "ph_adjuster", "surfactant", "solvent",
            ],
            "PROD-CW": [
                "surfactant", "fragrance", "preservative", "colorant",
                "ph_adjuster", "thickener", "solvent", "antibacterial",
            ],
            "PROD-AP": [
                "surfactant", "moisturizer", "fragrance", "preservative",
                "colorant", "ph_adjuster", "emulsifier", "thickener", "antioxidant", "solvent",
            ],
        }

        # Group ingredients by category
        ing_by_category = {}
        for ing in INGREDIENTS:
            cat = ing["category"]
            if cat not in ing_by_category:
                ing_by_category[cat] = []
            ing_by_category[cat].append(self.ctx.ingredient_ids[ing["code"]])

        for formula_code, formula_id_val in self.ctx.formula_ids.items():
            # Determine product
            prod_code = "PROD-" + formula_code.split("-")[1]
            categories = prod_ingredients.get(
                prod_code, ["surfactant", "preservative", "solvent"]
            )

            sequence = 1
            batch_size = next(
                f["batch_size_kg"]
                for f in self.data["formulas"]
                if f["id"] == formula_id_val
            )
            remaining_pct = 100.0

            for cat in categories:
                if cat not in ing_by_category or not ing_by_category[cat]:
                    continue

                ing_id = random.choice(ing_by_category[cat])

                # Water is typically 60-80% of personal care products
                if cat == "solvent":
                    pct = min(remaining_pct, random.uniform(40, 70))
                else:
                    pct = min(remaining_pct, random.uniform(1, 15))

                remaining_pct -= pct
                qty = batch_size * pct / 100

                self.data["formula_ingredients"].append(
                    {
                        "formula_id": formula_id_val,
                        "ingredient_id": ing_id,
                        "sequence": sequence,
                        "quantity_kg": round(qty, 4),
                        "quantity_percent": round(pct, 2),
                        "is_active": True,
                        "tolerance_percent": 2.0,
                        "notes": None,
                        "created_at": now,
                    }
                )
                sequence += 1

    def _generate_carrier_rates(self, now: datetime) -> None:
        """Generate carrier_rates table (~1000)."""
        rate_id = 1
        contract_ids = list(self.ctx.carrier_contract_ids.values())
        transport_modes = [
            "ftl", "ltl", "rail", "ocean_fcl", "ocean_lcl", "air", "parcel", "intermodal"
        ]

        for _ in range(1000):
            contract_id = random.choice(contract_ids)
            mode = random.choice(transport_modes)
            rate_code = f"RATE-{rate_id:05d}"
            self.ctx.carrier_rate_ids[rate_code] = rate_id

            # Weight breaks
            weight_min = random.choice([0, 100, 500, 1000, 5000])
            weight_max = weight_min + random.choice([100, 500, 1000, 5000, 10000])

            self.data["carrier_rates"].append(
                {
                    "id": rate_id,
                    "contract_id": contract_id,
                    "origin_type": random.choice(["plant", "dc", "city"]),
                    "origin_code": f"LOC-{random.randint(1, 100):03d}",
                    "destination_type": random.choice(["dc", "city", "region"]),
                    "destination_code": f"LOC-{random.randint(1, 100):03d}",
                    "transport_mode": mode,
                    "weight_break_min_kg": weight_min,
                    "weight_break_max_kg": weight_max,
                    "rate_per_kg": round(random.uniform(0.05, 0.50), 4)
                    if mode != "air"
                    else round(random.uniform(1.0, 5.0), 4),
                    "rate_per_case": round(random.uniform(0.5, 5.0), 4),
                    "rate_per_pallet": round(random.uniform(20, 100), 2),
                    "rate_per_shipment": round(random.uniform(100, 2000), 2)
                    if mode in ["ftl", "ocean_fcl"]
                    else None,
                    "fuel_surcharge_percent": round(random.uniform(5, 20), 2),
                    "currency": "USD",
                    "transit_days": random.randint(1, 30),
                    "effective_from": date(2024, 1, 1),
                    "effective_to": date(2024, 12, 31),
                    "created_at": now,
                }
            )
            rate_id += 1

    def _generate_routes(self, now: datetime) -> None:
        """Generate routes table (~50)."""
        route_id = 1
        for _ in range(50):
            code = f"ROUTE-{route_id:03d}"
            self.ctx.route_ids[code] = route_id
            self.data["routes"].append(
                {
                    "id": route_id,
                    "route_code": code,
                    "name": f"Route {route_id}",
                    "origin_type": random.choice(["plant", "dc"]),
                    "origin_id": random.randint(1, 25),
                    "destination_type": random.choice(["dc", "port"]),
                    "destination_id": random.randint(1, 25),
                    "total_distance_km": round(random.uniform(100, 5000), 2),
                    "total_transit_hours": round(random.uniform(4, 120), 2),
                    "total_segments": random.randint(1, 4),
                    "is_preferred": random.random() < 0.2,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            route_id += 1

    def _generate_route_segment_assignments(self, now: datetime) -> None:
        """Generate route_segment_assignments table (~150)."""
        segment_ids = list(self.ctx.route_segment_ids.values())
        for route_code, route_id_val in self.ctx.route_ids.items():
            num_segments = self.data["routes"][route_id_val - 1]["total_segments"]
            chosen_segments = random.sample(
                segment_ids, min(num_segments, len(segment_ids))
            )
            for seq, seg_id in enumerate(chosen_segments, 1):
                self.data["route_segment_assignments"].append(
                    {
                        "route_id": route_id_val,
                        "segment_id": seg_id,
                        "sequence": seq,
                        "created_at": now,
                    }
                )
