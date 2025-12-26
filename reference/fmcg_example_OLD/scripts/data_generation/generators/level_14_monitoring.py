"""
Level 14 Generator: Monitoring and KPIs (Leaf level).

Tables generated:
- kpi_actuals (~50K rows) - Weekly KPI measurements
- osa_metrics (~500K rows) - On-shelf availability tracking
- risk_events (~500 rows) - Supply chain disruptions
- audit_logs (~50K rows) - Change tracking for key tables

This is the final generation level - it has no downstream dependencies.
"""

import random
import time
from datetime import date, datetime, timedelta

from .base import BaseLevelGenerator, GeneratorContext


class Level14Generator(BaseLevelGenerator):
    """
    Generate Level 14 monitoring data.

    Level 14 is the leaf level, generating operational metrics and audit data.
    Key patterns:
    - KPI actuals: Weekly measurements against thresholds, variance calculation
    - OSA metrics: On-shelf availability with 92-95% target, lower during promos
    - Risk events: Supply chain disruptions with severity scoring
    - Audit log: Change tracking for key tables
    """

    LEVEL = 14

    def generate(self) -> None:
        """
        Generate all Level 14 tables.

        Tables: kpi_actuals, osa_metrics, risk_events, audit_logs
        """
        print("  Level 14: Monitoring (KPIs, OSA, risk_events...)")
        level_start = time.time()
        now = datetime.now()

        self._generate_kpi_actuals(now)
        self._generate_osa_metrics(now)
        self._generate_risk_events(now)
        self._generate_audit_log(now)

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        self._report_level_stats(level_elapsed)

    def _report_level_stats(self, elapsed: float) -> None:
        """Report generation statistics for this level."""
        tables = ["kpi_actuals", "osa_metrics", "risk_events", "audit_logs"]
        total_rows = sum(len(self.data.get(t, [])) for t in tables)
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
        print(f"    Level 14 completed: {total_rows:,} rows in {elapsed:.2f}s ({rows_per_sec:,.0f} rows/sec)")

    def _generate_kpi_actuals(self, now: datetime) -> None:
        """Generate kpi_actuals table - weekly KPI measurements."""
        print("    Generating kpi_actuals...")

        kpi_statuses = ["green", "yellow", "red"]
        trends = ["improving", "stable", "declining"]

        kpi_id_counter = 1
        for week_num in range(1, 53):
            measurement_date = date(self.ctx.base_year, 1, 1) + timedelta(
                weeks=week_num - 1
            )

            for kpi in self.data["kpi_thresholds"]:
                kpi_id = kpi["id"]
                target = float(kpi.get("target_value", 95))
                min_val = float(kpi.get("min_threshold", target * 0.8))
                max_val = float(kpi.get("max_threshold", target * 1.2))

                # Generate realistic actual values with some noise
                # Most weeks hit target, some miss
                if random.random() < 0.85:
                    # Within target range
                    actual = round(target + random.uniform(-5, 5), 2)
                else:
                    # Miss (either high or low)
                    if random.random() < 0.5:
                        actual = round(target - random.uniform(8, 20), 2)
                    else:
                        actual = round(target + random.uniform(8, 20), 2)

                # Variance
                variance = round(actual - target, 2)
                variance_pct = (
                    round((actual - target) / target * 100, 2) if target != 0 else 0
                )

                # Status based on variance
                if abs(variance_pct) <= 5:
                    status = "green"
                elif abs(variance_pct) <= 15:
                    status = "yellow"
                else:
                    status = "red"

                # Trend (simplified - could be computed from history)
                trend = random.choices(trends, weights=[30, 50, 20])[0]

                self.data["kpi_actuals"].append(
                    {
                        "id": kpi_id_counter,
                        "kpi_id": kpi_id,
                        "measurement_date": measurement_date,
                        "measurement_week": week_num,
                        "measurement_month": measurement_date.month,
                        "scope_type": "global",
                        "scope_id": None,
                        "actual_value": actual,
                        "target_value": target,
                        "variance_percent": variance_pct,
                        "status": status,
                        "trend": trend,
                        "created_at": now,
                    }
                )
                kpi_id_counter += 1

        print(f"      Generated {kpi_id_counter - 1:,} kpi_actuals")

    def _generate_osa_metrics(self, now: datetime) -> None:
        """Generate osa_metrics table - on-shelf availability tracking."""
        print("    Generating osa_metrics...")

        oos_reasons = [
            "dc_stockout",
            "delivery_miss",
            "shelf_gap",
            "planogram_issue",
            "demand_spike",
            "unknown",
        ]
        oos_reason_weights = [30, 25, 15, 10, 15, 5]

        # Named entities for special weeks
        promo_bf_week = 47  # Black Friday

        osa_id = 1

        # Get location and SKU IDs
        location_ids = [loc["id"] for loc in self.data.get("retail_locations", [])]
        sku_ids = list(self.ctx.sku_ids.values())

        if not location_ids or not sku_ids:
            print("      Warning: No locations or SKUs available for OSA metrics")
            return

        # Sample locations and SKUs for OSA (not all combinations - would be billions)
        sampled_locations = random.sample(location_ids, min(500, len(location_ids)))
        sampled_skus = random.sample(sku_ids, min(200, len(sku_ids)))

        for week_num in range(1, 53):
            measurement_date = date(self.ctx.base_year, 1, 1) + timedelta(
                weeks=week_num - 1
            )

            # OSA target varies by week (lower during promos)
            if week_num in (promo_bf_week, promo_bf_week + 1):
                base_osa_rate = 0.88  # Lower during/after Black Friday
            else:
                base_osa_rate = 0.94  # Normal weeks

            for loc_id in sampled_locations:
                # Generate measurements for a subset of SKUs per location
                sku_sample = random.sample(sampled_skus, min(20, len(sampled_skus)))

                for sku_id in sku_sample:
                    is_in_stock = random.random() < base_osa_rate

                    shelf_capacity = random.randint(10, 50)
                    if is_in_stock:
                        shelf_quantity = random.randint(3, shelf_capacity)
                        oos_reason = None
                    else:
                        shelf_quantity = 0
                        oos_reason = random.choices(
                            oos_reasons, weights=oos_reason_weights
                        )[0]

                    days_of_stock = round(
                        shelf_quantity / max(1, random.uniform(0.5, 3)), 2
                    )

                    self.data["osa_metrics"].append(
                        {
                            "id": osa_id,
                            "retail_location_id": loc_id,
                            "sku_id": sku_id,
                            "measurement_date": measurement_date,
                            "measurement_time": None,  # Could add time granularity
                            "is_in_stock": is_in_stock,
                            "shelf_capacity": shelf_capacity,
                            "shelf_quantity": shelf_quantity,
                            "days_of_stock": days_of_stock,
                            "out_of_stock_reason": oos_reason,
                            "created_at": now,
                        }
                    )
                    osa_id += 1

        print(f"      Generated {osa_id - 1:,} osa_metrics")

    def _generate_risk_events(self, now: datetime) -> None:
        """Generate risk_events table - supply chain disruptions."""
        print("    Generating risk_events...")

        event_types = [
            "supplier_disruption",
            "quality_hold",
            "logistics_delay",
            "demand_shock",
            "capacity_constraint",
            "natural_disaster",
            "geopolitical",
            "recall",
        ]
        event_type_weights = [25, 20, 20, 15, 10, 5, 3, 2]

        severities = ["low", "medium", "high", "critical"]
        severity_weights = [40, 35, 20, 5]

        risk_statuses = [
            "identified",
            "assessing",
            "mitigating",
            "monitoring",
            "resolved",
            "accepted",
        ]
        risk_status_weights = [15, 10, 15, 20, 35, 5]

        # Get entity pools
        supplier_ids = list(self.ctx.supplier_ids.values())
        plant_ids = list(self.ctx.plant_ids.values())
        dc_ids = list(self.ctx.dc_ids.values())
        sku_ids = list(self.ctx.sku_ids.values())

        # Affected entity types mapping
        entity_types_by_event = {
            "supplier_disruption": ("supplier", supplier_ids),
            "quality_hold": ("plant", plant_ids),
            "logistics_delay": ("dc", dc_ids),
            "demand_shock": ("sku", sku_ids),
            "capacity_constraint": ("plant", plant_ids),
            "natural_disaster": ("dc", dc_ids),
            "geopolitical": ("supplier", supplier_ids),
            "recall": ("sku", sku_ids),
        }

        # Descriptions by type
        descriptions = {
            "supplier_disruption": "Supply interruption from key supplier",
            "quality_hold": "Quality issue identified in production batch",
            "logistics_delay": "Transport delays affecting DC operations",
            "demand_shock": "Unexpected demand surge for product line",
            "capacity_constraint": "Production capacity limitation",
            "natural_disaster": "Weather event affecting regional operations",
            "geopolitical": "Trade policy change affecting imports",
            "recall": "Product recall initiated for safety concern",
        }

        root_causes = {
            "supplier_disruption": [
                "Financial instability",
                "Labor strike",
                "Equipment failure",
            ],
            "quality_hold": [
                "Contamination detected",
                "Specification deviation",
                "Supplier quality issue",
            ],
            "logistics_delay": [
                "Port congestion",
                "Carrier shortage",
                "Weather disruption",
            ],
            "demand_shock": [
                "Competitor stockout",
                "Viral marketing",
                "Seasonal spike",
            ],
            "capacity_constraint": [
                "Maintenance outage",
                "Labor shortage",
                "Raw material shortage",
            ],
            "natural_disaster": ["Hurricane", "Flooding", "Winter storm"],
            "geopolitical": ["Tariff change", "Export restriction", "Sanctions"],
            "recall": ["Consumer complaint", "Regulatory finding", "Internal QA"],
        }

        for risk_id in range(1, 501):
            event_type = random.choices(event_types, weights=event_type_weights)[0]
            severity = random.choices(severities, weights=severity_weights)[0]
            status = random.choices(risk_statuses, weights=risk_status_weights)[0]

            # Affected entity
            entity_type, entity_pool = entity_types_by_event[event_type]
            affected_id = random.choice(entity_pool) if entity_pool else None

            # Probability and impact
            probability = round(random.uniform(0.1, 0.9), 2)
            impact_score = random.randint(1, 10)

            identified_date = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 1),
                end_date=date(self.ctx.base_year, 12, 31),
            )

            target_resolution = identified_date + timedelta(days=random.randint(7, 90))
            actual_resolution = None
            if status == "resolved":
                actual_resolution = identified_date + timedelta(
                    days=random.randint(5, 60)
                )

            self.data["risk_events"].append(
                {
                    "id": risk_id,
                    "event_code": f"RISK-{self.ctx.base_year}-{risk_id:05d}",
                    "event_type": event_type,
                    "severity": severity,
                    "probability": probability,
                    "impact_score": impact_score,
                    "affected_entity_type": entity_type,
                    "affected_entity_id": affected_id,
                    "description": descriptions[event_type],
                    "root_cause": random.choice(root_causes[event_type]),
                    "mitigation_plan": f"Execute contingency plan for {event_type}",
                    "status": status,
                    "identified_date": identified_date,
                    "target_resolution_date": target_resolution,
                    "actual_resolution_date": actual_resolution,
                    "owner": self.fake.name(),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        print("      Generated 500 risk_events")

    def _generate_audit_log(self, now: datetime) -> None:
        """Generate audit_log table - change tracking for key tables."""
        print("    Generating audit_log...")

        # Tables to audit
        audit_tables = [
            ("orders", self.data.get("orders", [])[:5000]),
            ("shipments", self.data.get("shipments", [])[:5000]),
            ("batches", self.data.get("batches", [])[:2000]),
            ("returns", self.data.get("returns", [])),
            ("inventory", self.data.get("inventory", [])[:5000]),
        ]

        actions = ["INSERT", "UPDATE", "DELETE"]
        action_weights = [50, 45, 5]

        audit_id = 1
        for table_name, records in audit_tables:
            if not records:
                continue

            # Generate ~10K audit entries per major table
            sample_size = min(10000, len(records) * 3)
            for _ in range(sample_size):
                record = random.choice(records)
                record_id = record.get("id", random.randint(1, 100000))
                action = random.choices(actions, weights=action_weights)[0]

                # Simplified old/new values
                old_values = None
                new_values = None

                if action == "INSERT":
                    new_values = {"status": record.get("status", "pending")}
                elif action == "UPDATE":
                    old_values = {"status": "pending"}
                    new_values = {"status": record.get("status", "completed")}
                else:  # DELETE
                    old_values = {"id": record_id}

                changed_at = self.fake.date_time_between(
                    start_date=datetime(self.ctx.base_year, 1, 1), end_date=now
                )

                self.data["audit_logs"].append(
                    {
                        "id": audit_id,
                        "table_name": table_name,
                        "record_id": record_id,
                        "action": action,
                        "old_values": str(old_values)
                        if old_values
                        else None,  # JSONB as string for COPY
                        "new_values": str(new_values) if new_values else None,
                        "changed_fields": ["status"] if action == "UPDATE" else None,
                        "changed_by": self.fake.user_name(),
                        "changed_at": changed_at,
                        "ip_address": self.fake.ipv4(),
                        "user_agent": None,
                    }
                )
                audit_id += 1

        print(f"      Generated {audit_id - 1:,} audit_logs entries")
