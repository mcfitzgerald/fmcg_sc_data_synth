"""
Level 3 Generator: Network/Locations data.

Tables generated:
- distribution_centers (25 from DCS constant)
- retail_accounts (~100 with hub concentration)
- retail_locations (~10,000 with preferential attachment)

Named entities:
- DC-NAM-CHI-001: Bottleneck DC Chicago (40% NAM volume)
- ACCT-MEGA-001: MegaMart (4,500 stores, 25% of orders)
"""

import random
from datetime import datetime

from .base import BaseLevelGenerator
from ..constants import DCS
from ..helpers import barabasi_albert_attachment


class Level3Generator(BaseLevelGenerator):
    """
    Generate Level 3 location data.

    Level 3 contains distribution centers, retail accounts, and retail locations.
    Uses preferential attachment for realistic hub concentration.
    """

    LEVEL = 3

    def generate(self) -> None:
        """Generate all Level 3 tables."""
        print("  Level 3: Locations (retail_accounts, retail_locations, DCs...)")
        now = datetime.now()

        self._generate_distribution_centers(now)
        self._generate_retail_accounts(now)
        self._generate_retail_locations(now)

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['distribution_centers'])} DCs, "
            f"{len(self.data['retail_accounts'])} accounts, "
            f"{len(self.data['retail_locations'])} locations"
        )

    def _generate_distribution_centers(self, now: datetime) -> None:
        """Generate distribution_centers table (25 from DCS constant)."""
        for i, dc in enumerate(DCS, 1):
            div_id = self.ctx.division_ids[dc["division"]]
            self.ctx.dc_ids[dc["code"]] = i
            self.data["distribution_centers"].append(
                {
                    "id": i,
                    "dc_code": dc["code"],
                    "name": dc["name"],
                    "division_id": div_id,
                    "dc_type": dc["dc_type"],
                    "country": dc["country"],
                    "city": dc["city"],
                    "address": f"{random.randint(1, 9999)} Logistics Way",
                    "latitude": dc["lat"],
                    "longitude": dc["lon"],
                    "capacity_cases": dc["capacity_cases"],
                    "capacity_pallets": dc["capacity_cases"] // 50,
                    "operating_hours": "06:00-22:00",
                    "is_temperature_controlled": random.random() < 0.3,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def _generate_retail_accounts(self, now: datetime) -> None:
        """Generate retail_accounts table (~100 with hub concentration)."""
        account_id = 1
        account_types = {
            "bm_large": ["megamart", "valueclub", "urbanessential"],
            "bm_distributor": ["regional_grocer", "indie_retail"],
            "ecommerce": ["digital_first", "omni_retailer"],
            "dtc": ["prism_direct"],
        }

        # Named entity: MegaMart (ACCT-MEGA-001) - hot node with 4,500 stores
        mega_div_id = self.ctx.division_ids["NAM"]
        mega_channel_id = self.ctx.channel_ids["B2M-LARGE"]
        self.ctx.retail_account_ids["ACCT-MEGA-001"] = account_id
        self.data["retail_accounts"].append(
            {
                "id": account_id,
                "account_code": "ACCT-MEGA-001",
                "name": "MegaMart Corporation",
                "account_type": "megamart",
                "channel_id": mega_channel_id,
                "division_id": mega_div_id,
                "parent_account_id": None,
                "headquarters_country": "USA",
                "headquarters_city": "Bentonville",
                "store_count": 4500,
                "annual_volume_cases": 50_000_000,
                "payment_terms_days": 60,
                "credit_limit": 100_000_000,
                "is_strategic": True,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
        )
        account_id += 1

        # Generate ~99 more accounts across divisions and channels
        for channel_code, channel_id in self.ctx.channel_ids.items():
            types = account_types.get(
                channel_code.replace("B2M-", "").lower(), ["regional_grocer"]
            )
            num_accounts = (
                25
                if channel_code == "B2M-LARGE"
                else 30
                if channel_code == "B2M-DIST"
                else 20
                if channel_code == "ECOM"
                else 10
            )

            for _ in range(num_accounts):
                if account_id > 100:
                    break

                div_code = random.choice(list(self.ctx.division_ids.keys()))
                div_id = self.ctx.division_ids[div_code]
                acct_type = random.choice(types)
                code = f"ACCT-{acct_type[:4].upper()}-{account_id:03d}"

                store_count = (
                    random.randint(5, 500)
                    if acct_type not in ["megamart", "valueclub"]
                    else random.randint(500, 2000)
                )
                volume = store_count * random.randint(5000, 20000)

                self.ctx.retail_account_ids[code] = account_id
                self.data["retail_accounts"].append(
                    {
                        "id": account_id,
                        "account_code": code,
                        "name": self.fake.company()
                        + " "
                        + random.choice(["Stores", "Retail", "Markets", "Grocers", ""]),
                        "account_type": acct_type,
                        "channel_id": channel_id,
                        "division_id": div_id,
                        "parent_account_id": None,
                        "headquarters_country": random.choice(
                            ["USA", "UK", "Germany", "France", "Brazil", "China", "India"]
                        ),
                        "headquarters_city": self.fake.city(),
                        "store_count": store_count,
                        "annual_volume_cases": volume,
                        "payment_terms_days": random.choice([30, 45, 60]),
                        "credit_limit": volume * 2,
                        "is_strategic": random.random() < 0.15,
                        "is_active": True,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                account_id += 1

    def _generate_retail_locations(self, now: datetime) -> None:
        """Generate retail_locations table (~10,000 with preferential attachment)."""
        location_id = 1
        store_formats = ["hypermarket", "supermarket", "convenience", "pharmacy", "drugstore"]
        dc_ids_list = list(self.ctx.dc_ids.values())
        account_ids_list = list(self.ctx.retail_account_ids.values())

        # Track account degrees for preferential attachment
        account_degrees = {aid: 0 for aid in account_ids_list}

        # MegaMart gets 4,500 stores (named entity requirement)
        mega_account_id = self.ctx.retail_account_ids["ACCT-MEGA-001"]
        us_cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
            "Seattle", "Denver", "Boston", "Detroit", "Nashville",
        ]

        for i in range(4500):
            code = f"LOC-MEGA-{location_id:05d}"
            self.ctx.retail_location_ids[code] = location_id
            city = random.choice(us_cities)
            # Assign to nearest NAM DC
            nam_dcs = [
                dc_id
                for dc_code, dc_id in self.ctx.dc_ids.items()
                if dc_code.startswith("DC-NAM")
            ]
            primary_dc = random.choice(nam_dcs) if nam_dcs else random.choice(dc_ids_list)

            self.data["retail_locations"].append(
                {
                    "id": location_id,
                    "location_code": code,
                    "name": f"MegaMart #{location_id}",
                    "retail_account_id": mega_account_id,
                    "store_format": random.choices(store_formats[:3], weights=[40, 50, 10])[0],
                    "country": "USA",
                    "city": city,
                    "address": f"{random.randint(1, 9999)} {self.fake.street_name()}",
                    "postal_code": self.fake.zipcode(),
                    "latitude": round(random.uniform(25, 48), 6),
                    "longitude": round(random.uniform(-122, -71), 6),
                    "timezone": "America/New_York",
                    "square_meters": random.randint(2000, 15000),
                    "weekly_traffic": random.randint(5000, 50000),
                    "primary_dc_id": primary_dc,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            account_degrees[mega_account_id] += 1
            location_id += 1

        # Generate ~5,500 more stores for other accounts (preferential attachment)
        remaining = 10000 - 4500
        for _ in range(remaining):
            # Preferential attachment to accounts
            selected_idx = barabasi_albert_attachment(
                [account_degrees.get(aid, 0) for aid in account_ids_list],
                m=1,
                rng=self.rng,
            )[0]
            acct_id = account_ids_list[selected_idx]

            # Get account's division
            acct = next(a for a in self.data["retail_accounts"] if a["id"] == acct_id)
            div_id = acct["division_id"]

            # Find DCs in same division
            div_dcs = [
                dc["id"]
                for dc in self.data["distribution_centers"]
                if dc["division_id"] == div_id
            ]
            primary_dc = random.choice(div_dcs) if div_dcs else random.choice(dc_ids_list)

            code = f"LOC-{acct_id:03d}-{location_id:05d}"
            self.ctx.retail_location_ids[code] = location_id

            country = acct.get("headquarters_country", "USA")
            self.data["retail_locations"].append(
                {
                    "id": location_id,
                    "location_code": code,
                    "name": f"Store #{location_id}",
                    "retail_account_id": acct_id,
                    "store_format": random.choice(store_formats),
                    "country": country,
                    "city": self.fake.city(),
                    "address": f"{random.randint(1, 9999)} {self.fake.street_name()}",
                    "postal_code": self.fake.zipcode() if country == "USA" else self.fake.postcode(),
                    "latitude": round(random.uniform(-40, 60), 6),
                    "longitude": round(random.uniform(-120, 140), 6),
                    "timezone": self.fake.timezone(),
                    "square_meters": random.randint(200, 5000),
                    "weekly_traffic": random.randint(500, 20000),
                    "primary_dc_id": primary_dc,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            account_degrees[acct_id] = account_degrees.get(acct_id, 0) + 1
            location_id += 1
