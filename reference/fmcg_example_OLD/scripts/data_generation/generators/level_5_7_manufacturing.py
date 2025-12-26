"""
Level 5-7 Generator: Manufacturing (POs, Batches, Inventory).

Level 5 Tables:
- purchase_orders (~25,000)
- work_orders (~50,000)
- goods_receipts (~20,000)

Level 6 Tables:
- purchase_order_lines (~75,000)
- goods_receipt_lines (~60,000)
- batches (~48,000)
- batch_cost_ledger (~48,000)

Level 7 Tables:
- batch_ingredients (~480,000 = ~48k batches × ~10 ingredients each)
- inventory (~50,000)

Named entities:
- B-2024-RECALL-001: Contaminated Sorbitol batch for recall tracing
"""

import random
import time
from datetime import date, datetime, timedelta

from .base import BaseLevelGenerator
from ..lookup_builder import LookupBuilder


class Level5Generator(BaseLevelGenerator):
    """
    Generate Level 5 procurement and work orders.

    Level 5 contains purchase_orders, work_orders, goods_receipts.
    """

    LEVEL = 5

    def generate(self) -> None:
        """Generate all Level 5 tables."""
        print("  Level 5: Purchase orders, work orders, goods receipts...")
        now = datetime.now()

        self._generate_purchase_orders(now)
        self._generate_work_orders(now)
        self._generate_goods_receipts(now)

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['purchase_orders'])} POs, "
            f"{len(self.data['work_orders'])} WOs, "
            f"{len(self.data['goods_receipts'])} GRs"
        )

    def _generate_purchase_orders(self, now: datetime) -> None:
        """Generate purchase_orders table (~25,000)."""
        po_id = 1
        supplier_ids = list(self.ctx.supplier_ids.values())
        plant_ids = list(self.ctx.plant_ids.values())
        po_statuses = ["draft", "submitted", "approved", "in_transit", "received", "closed", "cancelled"]
        po_status_weights = [5, 5, 10, 15, 20, 40, 5]

        for _ in range(25000):
            po_num = f"PO-{self.ctx.base_year}-{po_id:06d}"
            self.ctx.purchase_order_ids[po_num] = po_id
            supplier_id = random.choice(supplier_ids)
            plant_id = random.choice(plant_ids)

            order_date = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 1),
                end_date=date(self.ctx.base_year, 12, 31),
            )
            lead_days = random.randint(7, 45)
            expected_delivery = order_date + timedelta(days=lead_days)

            status = random.choices(po_statuses, weights=po_status_weights)[0]
            actual_delivery = None
            if status in ["received", "closed"]:
                delay = random.randint(-3, 7)
                actual_delivery = expected_delivery + timedelta(days=delay)

            self.data["purchase_orders"].append(
                {
                    "id": po_id,
                    "po_number": po_num,
                    "supplier_id": supplier_id,
                    "plant_id": plant_id,
                    "order_date": order_date,
                    "expected_delivery_date": expected_delivery,
                    "actual_delivery_date": actual_delivery,
                    "status": status,
                    "currency": "USD",
                    "total_amount": None,  # Calculated from lines
                    "payment_terms_days": random.choice([30, 45, 60]),
                    "incoterms": random.choice(["FOB", "CIF", "DDP", "EXW"]),
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            po_id += 1

    def _generate_work_orders(self, now: datetime) -> None:
        """Generate work_orders table (~50,000)."""
        wo_id = 1
        plant_ids = list(self.ctx.plant_ids.values())
        line_ids = list(self.ctx.production_line_ids.values())
        formula_ids = list(self.ctx.formula_ids.values())
        wo_statuses = ["draft", "released", "in_progress", "complete", "closed", "cancelled"]
        # Weights tuned for OEE target 65-85%: 85% complete+closed for ~70-75% OEE
        wo_status_weights = [3, 5, 5, 35, 50, 2]
        wo_types = ["production", "rework", "trial", "sample"]
        wo_type_weights = [85, 8, 5, 2]

        for _ in range(50000):
            wo_num = f"WO-{self.ctx.base_year}-{wo_id:06d}"
            self.ctx.work_order_ids[wo_num] = wo_id
            plant_id = random.choice(plant_ids)
            line_id = random.choice(line_ids)
            formula_id = random.choice(formula_ids)

            planned_start = self.fake.date_between(
                start_date=date(self.ctx.base_year, 1, 1),
                end_date=date(self.ctx.base_year, 12, 31),
            )
            planned_end = planned_start + timedelta(days=random.randint(1, 5))
            status = random.choices(wo_statuses, weights=wo_status_weights)[0]

            actual_start = None
            actual_end = None
            if status in ["in_progress", "complete", "closed"]:
                actual_start = planned_start + timedelta(days=random.randint(-1, 2))
            if status in ["complete", "closed"]:
                actual_end = actual_start + timedelta(days=random.randint(1, 4))

            planned_qty = random.randint(500, 5000)
            actual_qty = (
                int(planned_qty * random.uniform(0.95, 1.02))
                if status in ["complete", "closed"]
                else None
            )

            self.data["work_orders"].append(
                {
                    "id": wo_id,
                    "wo_number": wo_num,
                    "plant_id": plant_id,
                    "production_line_id": line_id,
                    "formula_id": formula_id,
                    "wo_type": random.choices(wo_types, weights=wo_type_weights)[0],
                    "priority": random.randint(1, 5),
                    "planned_start_date": planned_start,
                    "planned_end_date": planned_end,
                    "actual_start_date": actual_start,
                    "actual_end_date": actual_end,
                    "planned_quantity_kg": planned_qty,
                    "actual_quantity_kg": actual_qty,
                    "status": status,
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            wo_id += 1

    def _generate_goods_receipts(self, now: datetime) -> None:
        """Generate goods_receipts table (~20,000)."""
        gr_id = 1
        received_pos = [
            po
            for po in self.data["purchase_orders"]
            if po["status"] in ["received", "closed"]
        ]
        gr_statuses = ["pending_inspection", "approved", "rejected", "partial"]

        for po in received_pos[:20000]:
            gr_num = f"GR-{self.ctx.base_year}-{gr_id:06d}"
            self.ctx.goods_receipt_ids[gr_num] = gr_id

            receipt_date = po.get("actual_delivery_date") or (
                po["expected_delivery_date"] + timedelta(days=random.randint(0, 5))
            )

            self.data["goods_receipts"].append(
                {
                    "id": gr_id,
                    "gr_number": gr_num,
                    "po_id": po["id"],
                    "receipt_date": receipt_date,
                    "inspection_date": receipt_date + timedelta(days=random.randint(0, 2)),
                    "inspector": self.fake.name(),
                    "status": random.choices(
                        gr_statuses, weights=[10, 80, 5, 5]
                    )[0],
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            gr_id += 1


class Level6Generator(BaseLevelGenerator):
    """
    Generate Level 6 line items and batches.

    Level 6 contains purchase_order_lines, goods_receipt_lines, batches, batch_cost_ledger.
    """

    LEVEL = 6

    def generate(self) -> None:
        """Generate all Level 6 tables."""
        print("  Level 6: PO lines, batches, cost ledger...")
        now = datetime.now()

        self._generate_purchase_order_lines(now)
        self._generate_goods_receipt_lines(now)
        self._generate_batches(now)
        self._generate_batch_cost_ledger(now)
        self._apply_chaos_manufacturing()

        self.ctx.generated_levels.add(self.LEVEL)
        print(
            f"    Generated: {len(self.data['purchase_order_lines'])} PO lines, "
            f"{len(self.data['batches'])} batches"
        )

    def _generate_purchase_order_lines(self, now: datetime) -> None:
        """Generate purchase_order_lines table (~75,000: ~3 per PO)."""
        ingredient_ids = list(self.ctx.ingredient_ids.values())
        supplier_ing_idx = LookupBuilder.build(
            self.data["supplier_ingredients"], key_field="supplier_id"
        )

        for po in self.data["purchase_orders"]:
            num_lines = random.randint(2, 5)
            # Get ingredients this supplier can provide
            sup_ings = supplier_ing_idx.get(po["supplier_id"], [])
            available_ings = [si["ingredient_id"] for si in sup_ings]
            if not available_ings:
                available_ings = ingredient_ids

            for line_num in range(1, num_lines + 1):
                ing_id = random.choice(available_ings)
                qty = random.randint(100, 10000)
                unit_cost = round(random.uniform(0.5, 15.0), 4)

                self.data["purchase_order_lines"].append(
                    {
                        "po_id": po["id"],
                        "line_number": line_num,
                        "ingredient_id": ing_id,
                        "quantity_kg": qty,
                        "unit_cost": unit_cost,
                        "line_amount": round(qty * unit_cost, 2),
                        "currency": "USD",
                        "received_quantity_kg": qty if po["status"] in ["received", "closed"] else 0,
                        "status": "received" if po["status"] in ["received", "closed"] else "open",
                        "created_at": now,
                    }
                )

    def _generate_goods_receipt_lines(self, now: datetime) -> None:
        """Generate goods_receipt_lines table (~60,000: ~3 per GR)."""
        po_lines_idx = LookupBuilder.build(
            self.data["purchase_order_lines"], key_field="po_id"
        )

        for gr in self.data["goods_receipts"]:
            po_lines = po_lines_idx.get(gr["po_id"], [])

            for pol in po_lines:
                inspected_qty = pol["received_quantity_kg"]
                accepted_qty = int(inspected_qty * random.uniform(0.95, 1.0))
                rejected_qty = inspected_qty - accepted_qty

                self.data["goods_receipt_lines"].append(
                    {
                        "gr_id": gr["id"],
                        "po_line_number": pol["line_number"],
                        "ingredient_id": pol["ingredient_id"],
                        "inspected_quantity_kg": inspected_qty,
                        "accepted_quantity_kg": accepted_qty,
                        "rejected_quantity_kg": rejected_qty,
                        "lot_number": f"LOT-{self.fake.bothify('???###')}",
                        "expiry_date": gr["receipt_date"] + timedelta(days=random.randint(180, 730)),
                        "storage_location": f"WH-{random.randint(1, 5):02d}-{random.choice('ABCD')}-{random.randint(1, 50):02d}",
                        "quality_grade": random.choices(["A", "B", "C"], weights=[80, 15, 5])[0],
                        "notes": None,
                        "created_at": now,
                    }
                )

    def _generate_batches(self, now: datetime) -> None:
        """Generate batches table (~48,000)."""
        batch_id = 1
        completed_wos = [
            wo
            for wo in self.data["work_orders"]
            if wo["status"] in ["complete", "closed"]
        ]
        qc_statuses = ["pending", "in_progress", "released", "hold", "rejected"]
        qc_weights = [5, 5, 80, 7, 3]

        # Named entity: Contaminated Sorbitol batch
        # RSK-BIO-001 risk event: When triggered, batch should be REJECTED
        if completed_wos:
            wo = completed_wos[0]
            sorb_ing_id = self.ctx.ingredient_ids.get("ING-SORB-001", 1)
            self.ctx.batch_ids["B-2024-RECALL-001"] = batch_id

            # Check if RSK-BIO-001 (contamination) risk event is triggered
            recall_status = "hold"
            if self.ctx.risk_manager and self.ctx.risk_manager.is_triggered("RSK-BIO-001"):
                recall_status = "REJECTED"

            self.data["batches"].append(
                {
                    "id": batch_id,
                    "batch_number": "B-2024-RECALL-001",
                    "work_order_id": wo["id"],
                    "formula_id": wo["formula_id"],
                    "plant_id": wo["plant_id"],
                    "production_date": wo.get("actual_start_date") or wo["planned_start_date"],
                    "expiry_date": (wo.get("actual_start_date") or wo["planned_start_date"]) + timedelta(days=365),
                    "quantity_kg": wo.get("actual_quantity_kg") or wo["planned_quantity_kg"],
                    "output_cases": (wo.get("actual_quantity_kg") or wo["planned_quantity_kg"]) // 10,
                    "yield_percent": round(random.uniform(96, 99), 2),
                    "qc_status": recall_status,
                    "qc_inspector": "QC Dept",
                    "qc_date": wo.get("actual_end_date"),
                    "notes": "RECALL: Contaminated sorbitol detected - isolate and quarantine",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            batch_id += 1

        # Generate remaining batches
        for wo in completed_wos[:48000]:
            batch_num = f"B-{self.ctx.base_year}-{batch_id:06d}"
            self.ctx.batch_ids[batch_num] = batch_id

            prod_date = wo.get("actual_start_date") or wo["planned_start_date"]
            qty_kg = wo.get("actual_quantity_kg") or wo["planned_quantity_kg"]

            self.data["batches"].append(
                {
                    "id": batch_id,
                    "batch_number": batch_num,
                    "work_order_id": wo["id"],
                    "formula_id": wo["formula_id"],
                    "plant_id": wo["plant_id"],
                    "production_date": prod_date,
                    "expiry_date": prod_date + timedelta(days=random.choice([365, 545, 730])),
                    "quantity_kg": qty_kg,
                    "output_cases": qty_kg // 10,
                    "yield_percent": round(random.uniform(94, 99), 2),
                    "qc_status": random.choices(qc_statuses, weights=qc_weights)[0],
                    "qc_inspector": self.fake.name() if random.random() > 0.2 else None,
                    "qc_date": wo.get("actual_end_date"),
                    "notes": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            batch_id += 1

    def _apply_chaos_manufacturing(self) -> None:
        """Apply chaos injection to manufacturing (data decay)."""
        if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("data_decay"):
            reference_date = datetime.now()
            self.data["batches"] = self.ctx.quirks_manager.apply_data_decay(
                self.data["batches"],
                reference_date=reference_date,
            )
            decay_count = sum(
                1 for b in self.data["batches"] if b.get("data_decay_affected")
            )
            if decay_count > 0:
                print(f"    [Quirk] Data decay applied: {decay_count} batches rejected")

    def _generate_batch_cost_ledger(self, now: datetime) -> None:
        """Generate batch_cost_ledger table (~48,000: one per batch)."""
        cost_types = ["material", "labor", "overhead", "quality", "packaging"]

        for batch in self.data["batches"]:
            total_cost = batch["quantity_kg"] * random.uniform(2, 10)
            for cost_type in cost_types:
                pct = {
                    "material": 0.55,
                    "labor": 0.15,
                    "overhead": 0.12,
                    "quality": 0.08,
                    "packaging": 0.10,
                }[cost_type]
                self.data["batch_cost_ledger"].append(
                    {
                        "batch_id": batch["id"],
                        "cost_type": cost_type,
                        "cost_amount": round(total_cost * pct, 2),
                        "currency": "USD",
                        "posting_date": batch["production_date"],
                        "created_at": now,
                    }
                )


class Level7Generator(BaseLevelGenerator):
    """
    Generate Level 7 batch consumption and inventory.

    Level 7 contains batch_ingredients and inventory.
    """

    LEVEL = 7

    def generate(self) -> None:
        """Generate all Level 7 tables."""
        print("  Level 7: Batch ingredients and inventory...")
        level_start = time.time()
        now = datetime.now()

        self._generate_batch_ingredients(now)
        self._generate_inventory(now)
        self._apply_chaos_inventory()

        self.ctx.generated_levels.add(self.LEVEL)
        level_elapsed = time.time() - level_start
        print(
            f"    Generated: {len(self.data['batch_ingredients'])} batch_ingredients, "
            f"{len(self.data['inventory'])} inventory records ({level_elapsed:.1f}s)"
        )

    def _generate_batch_ingredients(self, now: datetime) -> None:
        """
        Generate batch_ingredients table (~480,000 = ~48k batches × ~10 ingredients).

        Mass Balance Physics:
        - Ingredient input (kg) × yield = Batch output (kg)
        - Therefore: input = output / yield > output (yield loss)
        - Formula yield_percent (96-99%) determines the loss factor
        """
        # Build O(1) lookup indices
        formulas_idx = LookupBuilder.build_unique(self.data["formulas"], "id")
        formula_ings_idx = LookupBuilder.build(
            self.data["formula_ingredients"], key_field="formula_id"
        )

        for batch in self.data["batches"]:
            formula_ings = formula_ings_idx.get(batch["formula_id"], [])
            formula = formulas_idx.get(batch["formula_id"])
            if not formula:
                continue

            # Apply yield loss: we need MORE input to get the desired output
            # If yield = 97%, then input = output / 0.97 = output × 1.031
            yield_percent = formula.get("yield_percent", 97.0)
            yield_factor = 100.0 / yield_percent  # e.g., 100/97 = 1.031
            scale_factor = batch["quantity_kg"] / formula["batch_size_kg"] * yield_factor

            for fi in formula_ings:
                planned_qty = fi["quantity_kg"] * scale_factor
                # Actual varies slightly from planned (measurement variance)
                actual_qty = round(planned_qty * random.uniform(0.98, 1.02), 4)
                # Scrap is material lost during processing (subset of actual)
                scrap = round(actual_qty * random.uniform(0.005, 0.02), 4)

                self.data["batch_ingredients"].append(
                    {
                        "batch_id": batch["id"],
                        "ingredient_id": fi["ingredient_id"],
                        "sequence": fi["sequence"],
                        "planned_quantity_kg": round(planned_qty, 4),
                        "actual_quantity_kg": actual_qty,
                        "scrap_quantity_kg": scrap,
                        "lot_number": f"LOT-{self.fake.bothify('???###')}",
                        "notes": None,
                        "created_at": now,
                    }
                )

    def _generate_inventory(self, now: datetime) -> None:
        """
        Generate inventory table.
        
        NOTE: Inventory generation has been moved to Level 10 (Fulfillment)
        to ensure strict Mass Balance (Physics) consistency.
        Inventory = Production - Shipments.
        """
        # Placeholder - generation moved to Level 10
        pass

    def _apply_chaos_inventory(self) -> None:
        """Apply chaos injection to inventory (phantom inventory quirk)."""
        if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("phantom_inventory"):
            reference_date = datetime.now()
            self.data["inventory"] = self.ctx.quirks_manager.apply_phantom_inventory(
                self.data["inventory"],
                reference_date=reference_date,
            )
            shrinkage_count = sum(
                1 for inv in self.data["inventory"] if inv.get("has_shrinkage")
            )
            if shrinkage_count > 0:
                print(
                    f"    [Quirk] Phantom inventory applied: {shrinkage_count} records with shrinkage"
                )
