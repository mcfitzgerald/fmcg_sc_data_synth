"""
Level 0 Generator: Reference/Master data with no FK dependencies.

Tables generated:
- divisions
- channels
- products
- packaging_types
- ports
- carriers
- emission_factors
- kpi_thresholds
- business_rules
- ingredients

This is the foundation level - all other levels depend on Level 0 data.
"""

import random
from datetime import date, datetime

from .base import BaseLevelGenerator, GeneratorContext
from ..constants import (
    BUSINESS_RULES,
    CARRIERS,
    CHANNELS,
    DIVISIONS,
    INGREDIENTS,
    KPI_THRESHOLDS,
    PACKAGING_TYPES,
    PORTS,
    PRODUCTS,
)


class Level0Generator(BaseLevelGenerator):
    """
    Generate Level 0 reference data.

    Level 0 contains foundational master data with no foreign key dependencies.
    All other generation levels reference this data.
    """

    LEVEL = 0

    def generate(self) -> None:
        """
        Generate all Level 0 tables.

        Tables: divisions, channels, products, packaging_types, ports,
                carriers, emission_factors, kpi_thresholds, business_rules,
                ingredients
        """
        print("  Level 0: Reference data (divisions, channels, products...)")
        now = datetime.now()

        self._generate_divisions(now)
        self._generate_channels(now)
        self._generate_products(now)
        self._generate_packaging_types(now)
        self._generate_ports(now)
        self._generate_carriers(now)
        self._generate_emission_factors(now)
        self._generate_kpi_thresholds(now)
        self._generate_business_rules(now)
        self._generate_ingredients(now)

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['divisions'])} divisions, "
            f"{len(self.data['channels'])} channels, "
            f"{len(self.data['products'])} products, "
            f"{len(self.data['ingredients'])} ingredients"
        )

    def _generate_divisions(self, now: datetime) -> None:
        """Generate divisions table."""
        for i, div in enumerate(DIVISIONS, 1):
            self.ctx.division_ids[div["code"]] = i
            self.data["divisions"].append(
                {
                    "id": i,
                    "division_code": div["code"],
                    "name": div["name"],
                    "headquarters_city": div["hq_city"],
                    "headquarters_country": div["hq_country"],
                    "president": div["president"],
                    "revenue_target": div["revenue_target"],
                    "currency": "USD",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_channels(self, now: datetime) -> None:
        """Generate channels table."""
        for i, ch in enumerate(CHANNELS, 1):
            self.ctx.channel_ids[ch["code"]] = i
            self.data["channels"].append(
                {
                    "id": i,
                    "channel_code": ch["code"],
                    "name": ch["name"],
                    "channel_type": ch["channel_type"],
                    "volume_percent": ch["volume_pct"],
                    "margin_percent": ch["margin_pct"],
                    "payment_terms_days": ch["payment_days"],
                    "description": f"{ch['name']} sales channel",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_products(self, now: datetime) -> None:
        """Generate products table."""
        for i, prod in enumerate(PRODUCTS, 1):
            self.ctx.product_ids[prod["code"]] = i
            self.data["products"].append(
                {
                    "id": i,
                    "product_code": prod["code"],
                    "name": prod["name"],
                    "brand": prod["brand"],
                    "category": prod["category"],
                    "subcategory": prod["subcategory"],
                    "description": f"{prod['brand']} - Premium {prod['subcategory'].replace('_', ' ')}",
                    "launch_date": date.fromisoformat(prod["launch_date"]),
                    "discontinue_date": None,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_packaging_types(self, now: datetime) -> None:
        """Generate packaging_types table."""
        for i, pkg in enumerate(PACKAGING_TYPES, 1):
            self.ctx.packaging_type_ids[pkg["code"]] = i
            case_weight = (
                pkg["size"] * pkg["per_case"] / 1000
            ) * 1.1  # ~10% packaging weight
            self.data["packaging_types"].append(
                {
                    "id": i,
                    "packaging_code": pkg["code"],
                    "name": pkg["name"],
                    "container_type": pkg["container"],
                    "size_value": pkg["size"],
                    "size_unit": pkg["unit"],
                    "material": pkg["material"],
                    "is_recyclable": pkg["recyclable"],
                    "units_per_case": pkg["per_case"],
                    "case_weight_kg": round(case_weight, 3),
                    "case_dimensions_cm": "40x30x25",
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_ports(self, now: datetime) -> None:
        """Generate ports table."""
        for i, port in enumerate(PORTS, 1):
            self.ctx.port_ids[port["code"]] = i
            teu = (
                random.randint(500_000, 20_000_000)
                if port["port_type"] == "ocean"
                else None
            )
            self.data["ports"].append(
                {
                    "id": i,
                    "port_code": port["code"],
                    "name": port["name"],
                    "port_type": port["port_type"],
                    "country": port["country"],
                    "city": port["city"],
                    "latitude": port["lat"],
                    "longitude": port["lon"],
                    "timezone": self.fake.timezone(),
                    "handling_capacity_teu": teu,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_carriers(self, now: datetime) -> None:
        """Generate carriers table."""
        for i, car in enumerate(CARRIERS, 1):
            self.ctx.carrier_ids[car["code"]] = i
            self.data["carriers"].append(
                {
                    "id": i,
                    "carrier_code": car["code"],
                    "name": car["name"],
                    "carrier_type": car["carrier_type"],
                    "scac_code": car["scac"],
                    "headquarters_country": car["country"],
                    "service_regions": None,  # Array type
                    "sustainability_rating": car["sustainability"],
                    "on_time_delivery_rate": car["otd"],
                    "damage_rate": round(random.uniform(0.001, 0.02), 4),
                    "is_preferred": car["otd"] >= 0.95,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_emission_factors(self, now: datetime) -> None:
        """Generate emission_factors table with carbon tax overrides."""
        emission_id = 1
        modes = [
            ("truck", "diesel", 0.062, 0.0000894),
            ("truck", "electric", 0.020, 0.0000290),
            ("rail", "diesel", 0.022, 0.0000318),
            ("rail", "electric", 0.008, 0.0000116),
            ("ocean", "hfo", 0.008, 0.0000116),
            ("ocean", "lng", 0.006, 0.0000087),
            ("air", "jet_fuel", 0.500, 0.000725),
            ("last_mile", "diesel", 0.150, 0.000217),
            ("last_mile", "electric", 0.050, 0.0000725),
            ("intermodal", "diesel", 0.035, 0.0000507),
        ]

        # Apply carbon tax spike if RSK-ENV-005 triggered
        co2_multiplier = 1.0
        if self.ctx.risk_manager:
            emission_overrides = self.ctx.risk_manager.get_emission_overrides()
            co2_multiplier = emission_overrides.get("co2_multiplier", 1.0)
            if co2_multiplier > 1.0:
                print(
                    f"    [Risk] RSK-ENV-005: Carbon tax spike - {co2_multiplier}x CO2 emission multiplier"
                )

        for mode, fuel, co2_km, co2_ton_km in modes:
            self.ctx.emission_factor_ids[f"{mode}_{fuel}"] = emission_id

            self.data["emission_factors"].append(
                {
                    "id": emission_id,
                    "transport_mode": mode,
                    "fuel_type": fuel,
                    "carrier_id": None,
                    "co2_kg_per_km": co2_km * co2_multiplier,
                    "co2_kg_per_ton_km": co2_ton_km * co2_multiplier,
                    "source": "GLEC Framework 2.0"
                    if co2_multiplier == 1.0
                    else "GLEC + Carbon Tax",
                    "effective_from": date(2024, 1, 1),
                    "effective_to": None,
                    "created_at": now,
                }
            )
            emission_id += 1

    def _generate_kpi_thresholds(self, now: datetime) -> None:
        """Generate kpi_thresholds table."""
        for i, kpi in enumerate(KPI_THRESHOLDS, 1):
            self.ctx.kpi_threshold_ids[kpi["code"]] = i
            self.data["kpi_thresholds"].append(
                {
                    "id": i,
                    "kpi_code": kpi["code"],
                    "kpi_name": kpi["name"],
                    "kpi_category": kpi["category"],
                    "desmet_dimension": kpi["desmet"],
                    "unit": kpi["unit"],
                    "direction": kpi["direction"],
                    "target_value": kpi["target"],
                    "warning_threshold": kpi["warning"],
                    "critical_threshold": kpi["critical"],
                    "industry_benchmark": kpi["target"] * 0.95,
                    "scope": "global",
                    "effective_from": date(2024, 1, 1),
                    "effective_to": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_business_rules(self, now: datetime) -> None:
        """Generate business_rules table."""
        for i, rule in enumerate(BUSINESS_RULES, 1):
            self.ctx.business_rule_ids[rule["code"]] = i
            self.data["business_rules"].append(
                {
                    "id": i,
                    "rule_code": rule["code"],
                    "rule_name": rule["name"],
                    "rule_category": rule["category"],
                    "rule_type": rule["rule_type"],
                    "scope": "global",
                    "scope_id": None,
                    "condition_expression": rule["condition"],
                    "action_expression": rule["action"],
                    "priority": rule["priority"],
                    "is_active": True,
                    "effective_from": date(2024, 1, 1),
                    "effective_to": None,
                    "created_by": "System",
                    "approved_by": "Admin",
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_ingredients(self, now: datetime) -> None:
        """Generate ingredients table."""
        for i, ing in enumerate(INGREDIENTS, 1):
            self.ctx.ingredient_ids[ing["code"]] = i
            temp_min = 2 if ing["storage"] == "refrigerated" else 15
            temp_max = 8 if ing["storage"] == "refrigerated" else 30
            self.data["ingredients"].append(
                {
                    "id": i,
                    "ingredient_code": ing["code"],
                    "name": ing["name"],
                    "cas_number": ing["cas"],
                    "category": ing["category"],
                    "purity_percent": round(random.uniform(95, 99.9), 2),
                    "storage_temp_min_c": temp_min,
                    "storage_temp_max_c": temp_max,
                    "storage_conditions": ing["storage"],
                    "shelf_life_days": ing["shelf_days"],
                    "hazmat_class": ing["hazmat"],
                    "unit_of_measure": "kg",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
