"""
Level 12-13 Generator: Returns and Disposition.

Level 12 Tables:
- rma_authorizations (~10,000)
- returns (~10,000)
- return_lines (~30,000)

Level 13 Tables:
- disposition_logs (~30,000)

Key patterns:
- 3-5% return rate on orders
- Reason codes: damaged, expired, quality_defect, overstock, recall, wrong_shipment, customer_return
- Returns link to original orders and ship to regional DCs
- Return lines have condition assessment (sellable, damaged, expired, contaminated)
- Disposition weighted by condition: sellable→restock, damaged→liquidate/scrap
"""

import random
import time
from datetime import date, datetime, timedelta

from .base import BaseLevelGenerator


class Level12Generator(BaseLevelGenerator):
    """
    Generate Level 12 returns data.

    Level 12 contains RMA authorizations, returns, and return lines.
    """

    LEVEL = 12

    def generate(self) -> None:
        """Generate all Level 12 tables."""
        print("  Level 12: Returns...")
        level_start = time.time()
        now = datetime.now()

        self._generate_rma_authorizations(now)
        self._generate_returns(now)
        self._generate_return_lines(now)

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(
            f"    Generated: {len(self.data['rma_authorizations'])} RMAs, "
            f"{len(self.data['returns'])} returns, "
            f"{len(self.data['return_lines'])} return_lines ({level_elapsed:.1f}s)"
        )

    def _generate_rma_authorizations(self, now: datetime) -> None:
        """Generate rma_authorizations table (~10,000)."""
        print("    Generating rma_authorizations...")
        account_ids = list(self.ctx.retail_account_ids.values())

        reason_codes = [
            "damaged", "expired", "quality_defect", "overstock",
            "recall", "wrong_shipment", "customer_return",
        ]
        reason_weights = [25, 15, 15, 20, 5, 10, 10]

        rma_statuses = ["requested", "approved", "rejected", "expired"]
        rma_status_weights = [10, 75, 10, 5]

        rma_id = 1
        for _ in range(10000):
            rma_num = f"RMA-{self.ctx.base_year}-{rma_id:06d}"

            account_id = random.choice(account_ids)
            reason = random.choices(reason_codes, weights=reason_weights)[0]
            status = random.choices(rma_statuses, weights=rma_status_weights)[0]

            request_date = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 1),
                end_date=date(self.ctx.base_year, 12, 31),
            )

            approved_by = None
            approval_date = None
            expiry_date = None
            if status in ("approved", "expired"):
                approved_by = self.fake.name()
                approval_date = request_date + timedelta(days=random.randint(1, 3))
                expiry_date = approval_date + timedelta(days=30)

            self.ctx.rma_ids[rma_num] = rma_id
            self.data["rma_authorizations"].append(
                {
                    "id": rma_id,
                    "rma_number": rma_num,
                    "retail_account_id": account_id,
                    "request_date": request_date,
                    "reason_code": reason,
                    "status": status,
                    "approved_by": approved_by,
                    "approval_date": approval_date,
                    "expiry_date": expiry_date,
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            rma_id += 1

        print(f"      Generated {rma_id - 1:,} rma_authorizations")

    def _generate_returns(self, now: datetime) -> None:
        """Generate returns table (~10,000)."""
        print("    Generating returns...")
        dc_ids = list(self.ctx.dc_ids.values())
        location_ids = [loc["id"] for loc in self.data["retail_locations"]]

        delivered_orders = [o for o in self.data["orders"] if o["status"] == "delivered"]
        approved_rmas = [r for r in self.data["rma_authorizations"] if r["status"] == "approved"]

        return_statuses = ["pending", "in_transit", "received", "inspecting", "processed", "closed"]
        return_status_weights = [5, 5, 10, 10, 20, 50]

        return_id = 1
        for _ in range(10000):
            return_num = f"RET-{self.ctx.base_year}-{return_id:06d}"

            # 80% of returns have an RMA
            rma_id_ref = None
            if random.random() < 0.80 and approved_rmas:
                rma = random.choice(approved_rmas)
                rma_id_ref = rma["id"]

            # 90% of returns link to an order
            order_id = None
            if random.random() < 0.90 and delivered_orders:
                order = random.choice(delivered_orders)
                order_id = order["id"]

            location_id = random.choice(location_ids) if location_ids else None
            dc_id = random.choice(dc_ids)

            return_date = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 15),
                end_date=date(self.ctx.base_year, 12, 31),
            )

            status = random.choices(return_statuses, weights=return_status_weights)[0]

            received_date = None
            if status in ("received", "inspecting", "processed", "closed"):
                received_date = return_date + timedelta(days=random.randint(2, 7))

            total_cases = random.randint(5, 200)
            credit_amount = round(total_cases * random.uniform(20, 80), 2)

            self.ctx.return_ids[return_num] = return_id
            self.data["returns"].append(
                {
                    "id": return_id,
                    "return_number": return_num,
                    "rma_id": rma_id_ref,
                    "order_id": order_id,
                    "retail_location_id": location_id,
                    "dc_id": dc_id,
                    "return_date": return_date,
                    "received_date": received_date,
                    "status": status,
                    "total_cases": total_cases,
                    "credit_amount": credit_amount,
                    "currency": "USD",
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            return_id += 1

        print(f"      Generated {return_id - 1:,} returns")

    def _generate_return_lines(self, now: datetime) -> None:
        """Generate return_lines table (~30,000: ~3 lines per return)."""
        print("    Generating return_lines...")
        sku_ids = list(self.ctx.sku_ids.values())
        batch_ids = [b["id"] for b in self.data["batches"]]
        batch_lookup = {b["id"]: b for b in self.data["batches"]}

        conditions = ["sellable", "damaged", "expired", "contaminated", "unknown"]
        condition_weights = [40, 25, 20, 10, 5]

        line_count = 0
        for ret in self.data["returns"]:
            ret_id = ret["id"]
            total_cases = ret["total_cases"]

            # 1-5 lines per return
            num_lines = random.randint(1, 5)
            remaining = total_cases

            for line_num in range(1, num_lines + 1):
                sku_id = random.choice(sku_ids)
                batch_id = random.choice(batch_ids) if batch_ids else None
                batch = batch_lookup.get(batch_id, {}) if batch_id else {}

                # Cases for this line
                if line_num == num_lines:
                    qty = max(1, remaining)
                else:
                    qty = random.randint(1, max(1, remaining - (num_lines - line_num)))
                    remaining -= qty

                condition = random.choices(conditions, weights=condition_weights)[0]

                # Inspection notes based on condition
                notes = None
                if condition == "damaged":
                    notes = random.choice([
                        "Crushed packaging", "Water damage", "Torn labels", "Dented cans"
                    ])
                elif condition == "expired":
                    notes = "Past expiry date on receipt"
                elif condition == "contaminated":
                    notes = random.choice([
                        "Foreign object", "Odor detected", "Color change", "Recall batch"
                    ])

                self.data["return_lines"].append(
                    {
                        "return_id": ret_id,
                        "line_number": line_num,
                        "sku_id": sku_id,
                        "batch_id": batch_id,
                        "lot_number": batch.get("batch_number"),
                        "quantity_cases": qty,
                        "condition": condition,
                        "inspection_notes": notes,
                        "created_at": now,
                    }
                )
                line_count += 1

        print(f"      Generated {line_count:,} return_lines")


class Level13Generator(BaseLevelGenerator):
    """
    Generate Level 13 disposition data.

    Level 13 contains disposition_logs.
    """

    LEVEL = 13

    def generate(self) -> None:
        """Generate all Level 13 tables."""
        print("  Level 13: Disposition logs...")
        level_start = time.time()
        now = datetime.now()

        self._generate_disposition_logs(now)

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(f"    Generated: {len(self.data['disposition_logs'])} disposition_logs ({level_elapsed:.1f}s)")

    def _generate_disposition_logs(self, now: datetime) -> None:
        """Generate disposition_logs table (~30,000: one per return line)."""
        print("    Generating disposition_logs...")
        dc_ids = list(self.ctx.dc_ids.values())

        # Disposition types with weights matching FMCG industry patterns
        dispositions = ["restock", "scrap", "donate", "rework", "liquidate", "quarantine"]
        disposition_weights = [55, 10, 5, 5, 20, 5]

        # Destination types by disposition
        destination_types = {
            "restock": "dc",
            "scrap": "scrap_vendor",
            "donate": "charity",
            "rework": "plant",
            "liquidate": "liquidator",
            "quarantine": "dc",
        }

        disp_id = 1
        for ret_line in self.data["return_lines"]:
            return_id = ret_line["return_id"]
            line_number = ret_line["line_number"]
            qty_cases = ret_line["quantity_cases"]
            condition = ret_line["condition"]

            # Adjust disposition weights based on condition
            if condition == "sellable":
                weights = [80, 2, 2, 3, 10, 3]
            elif condition == "damaged":
                weights = [10, 30, 5, 10, 40, 5]
            elif condition == "expired":
                weights = [0, 50, 30, 0, 15, 5]
            elif condition == "contaminated":
                weights = [0, 60, 0, 0, 0, 40]
            else:
                weights = disposition_weights

            disposition = random.choices(dispositions, weights=weights)[0]
            dest_type = destination_types[disposition]

            # Destination ID
            if dest_type == "dc":
                dest_id = random.choice(dc_ids)
            else:
                dest_id = random.randint(1, 10)  # External vendor IDs

            # Recovery value / disposal cost
            unit_value = random.uniform(15, 50)
            if disposition == "restock":
                recovery_value = round(qty_cases * unit_value, 2)
                disposal_cost = 0
            elif disposition == "liquidate":
                recovery_value = round(qty_cases * unit_value * 0.3, 2)  # 30% of value
                disposal_cost = 0
            elif disposition == "rework":
                recovery_value = round(qty_cases * unit_value * 0.7, 2)
                disposal_cost = round(qty_cases * 2, 2)  # $2/case rework cost
            elif disposition == "donate":
                recovery_value = round(qty_cases * unit_value * 0.1, 2)  # Tax benefit
                disposal_cost = round(qty_cases * 1, 2)  # Handling cost
            else:  # scrap or quarantine
                recovery_value = 0
                disposal_cost = round(qty_cases * random.uniform(3, 8), 2)

            processed_by = self.fake.name()
            processed_at = now - timedelta(days=random.randint(1, 60))

            self.data["disposition_logs"].append(
                {
                    "id": disp_id,
                    "return_id": return_id,
                    "return_line_number": line_number,
                    "disposition": disposition,
                    "quantity_cases": qty_cases,
                    "destination_location_type": dest_type,
                    "destination_location_id": dest_id,
                    "recovery_value": recovery_value,
                    "disposal_cost": disposal_cost,
                    "processed_by": processed_by,
                    "processed_at": processed_at,
                    "notes": None,
                }
            )
            disp_id += 1

        print(f"      Generated {disp_id - 1:,} disposition_logs")
