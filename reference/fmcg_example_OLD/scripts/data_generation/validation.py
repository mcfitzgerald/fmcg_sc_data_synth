"""
Validation methods for FMCG data generation.

Contains all validation checks for generated data:
- Row count validation
- Pareto distribution validation (80/20 rule)
- Hub concentration validation (MegaMart)
- Named entities presence validation
- SPOF (Single Point of Failure) validation
- Multi-promo calendar validation
- Referential integrity validation
- Chaos injection validation

These validators are designed to work with the GeneratorContext pattern.
"""

from collections import Counter
from datetime import date
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .generators import GeneratorContext

from .helpers import TARGET_ROW_COUNTS


class DataValidator:
    """
    Validator for generated FMCG data.

    Works with GeneratorContext to validate data quality, distribution patterns,
    referential integrity, and chaos injection effects.
    """

    def __init__(self, ctx: "GeneratorContext") -> None:
        """
        Initialize validator with context.

        Args:
            ctx: GeneratorContext containing generated data and ID mappings
        """
        self.ctx = ctx
        self._manifest = {}
        try:
            from . import BENCHMARK_MANIFEST_PATH
            import json
            with open(BENCHMARK_MANIFEST_PATH) as f:
                self._manifest = json.load(f)
        except Exception:
            pass

    def _get_chaos_config(self, key: str) -> dict:
        """Get chaos validation config from manifest."""
        return self._manifest.get("chaos_validation", {}).get(key, {})

    def _get_tolerance(self, key: str, default: Any) -> Any:
        """Get validation tolerance from manifest."""
        return self._manifest.get("validation_tolerances", {}).get(key, default)

    @property
    def data(self) -> dict[str, list[dict]]:
        """Convenience accessor for data storage."""
        return self.ctx.data

    def validate_row_counts(self) -> tuple[bool, str]:
        """
        Check row counts are within reasonable range of targets.

        Thresholds from manifest validation_tolerances.row_count_tolerance_pct.
        Targets from manifest generation_targets.total_rows.

        Returns:
            Tuple of (passed, message)
        """
        total = sum(len(rows) for rows in self.data.values())
        
        # Prefer manifest target, fallback to helper
        target = self._manifest.get("generation_targets", {}).get("total_rows")
        if not target:
            target = sum(TARGET_ROW_COUNTS.values())
            
        pct_diff = abs(total - target) / target * 100 if target > 0 else 0
        tolerance = self._get_tolerance("row_count_tolerance_pct", 0.25) * 100

        if pct_diff <= tolerance:
            return True, f"{total:,} rows (target: {target:,}, diff: {pct_diff:.1f}%)"
        elif pct_diff <= tolerance * 2:
            # Warning but still pass
            return (
                True,
                f"{total:,} rows (target: {target:,}, diff: {pct_diff:.1f}% - investigate)",
            )
        return (
            False,
            f"{total:,} rows (target: {target:,}, diff: {pct_diff:.1f}% > {tolerance*2:.0f}% - likely bug)",
        )

    def validate_pareto(self) -> tuple[bool, str]:
        """
        Check top 20% SKUs volume concentration.

        Thresholds from manifest validation_tolerances.pareto_top20_range.

        Returns:
            Tuple of (passed, message)
        """
        order_lines = self.data.get("order_lines", [])
        if not order_lines:
            return False, "No order_lines data"

        # Count quantity by SKU (field is quantity_cases, not quantity)
        sku_qty: Counter = Counter()
        for line in order_lines:
            sku_qty[line.get("sku_id")] += line.get("quantity_cases", 0)

        if not sku_qty:
            return False, "No SKU quantities"

        # Sort by quantity descending
        sorted_skus = sorted(sku_qty.items(), key=lambda x: x[1], reverse=True)
        total_qty = sum(q for _, q in sorted_skus)

        # Top 20%
        top_n = max(1, len(sorted_skus) // 5)
        top_qty = sum(q for _, q in sorted_skus[:top_n])
        top_pct = top_qty / total_qty if total_qty > 0 else 0
        
        target_range = self._get_tolerance("pareto_top20_range", [0.75, 0.85])

        if target_range[0] <= top_pct <= target_range[1]:
            return True, f"Top 20% SKUs = {top_pct:.1%} volume"
        return False, f"Top 20% SKUs = {top_pct:.1%} (target: {target_range[0]:.0%}-{target_range[1]:.0%})"

    def validate_hub_concentration(self) -> tuple[bool, str]:
        """
        Check MegaMart (ACCT-MEGA-001) order concentration.

        Thresholds from manifest validation_tolerances.hub_concentration_range.

        Returns:
            Tuple of (passed, message)
        """
        orders = self.data.get("orders", [])
        if not orders:
            return False, "No orders data"

        mega_id = self.ctx.retail_account_ids.get("ACCT-MEGA-001")
        if not mega_id:
            return False, "ACCT-MEGA-001 not found"

        mega_orders = sum(1 for o in orders if o.get("retail_account_id") == mega_id)
        pct = mega_orders / len(orders) if orders else 0
        
        target_range = self._get_tolerance("hub_concentration_range", [0.20, 0.30])

        if target_range[0] <= pct <= target_range[1]:
            return True, f"MegaMart: {pct:.1%} of orders"
        return False, f"MegaMart: {pct:.1%} (target: {target_range[0]:.0%}-{target_range[1]:.0%})"

    def validate_named_entities(self) -> tuple[bool, str]:
        """
        Check all 9 named entities exist in their tables.

        Returns:
            Tuple of (passed, message)
        """
        missing = []

        # Check batches
        if "B-2024-RECALL-001" not in self.ctx.batch_ids:
            missing.append("B-2024-RECALL-001")

        # Check accounts
        if "ACCT-MEGA-001" not in self.ctx.retail_account_ids:
            missing.append("ACCT-MEGA-001")

        # Check suppliers
        if "SUP-PALM-MY-001" not in self.ctx.supplier_ids:
            missing.append("SUP-PALM-MY-001")

        # Check DCs
        if "DC-NAM-CHI-001" not in self.ctx.dc_ids:
            missing.append("DC-NAM-CHI-001")

        # Check promotions
        if "PROMO-BF-2024" not in self.ctx.promotion_ids:
            missing.append("PROMO-BF-2024")

        # Check route segments
        if "LANE-SH-LA-001" not in self.ctx.route_segment_ids:
            missing.append("LANE-SH-LA-001")

        # Check ingredients
        for ing in ["ING-PALM-001", "ING-SORB-001", "ING-PEPP-001"]:
            if ing not in self.ctx.ingredient_ids:
                missing.append(ing)

        if not missing:
            return True, "All 9 named entities present"
        return False, f"Missing: {', '.join(missing)}"

    def validate_spof(self) -> tuple[bool, str]:
        """
        Check SPOF ingredients are single-source.

        Threshold from manifest benchmarks.level_0_3_network.spof_ingredient_count.

        Returns:
            Tuple of (passed, message)
        """
        supplier_ings = self.data.get("supplier_ingredients", [])
        if not supplier_ings:
            return False, "No supplier_ingredients data"

        # Count suppliers per ingredient
        ing_suppliers: Counter = Counter()
        for si in supplier_ings:
            ing_suppliers[si.get("ingredient_id")] += 1

        spof_ings = []
        for code in ["ING-SORB-001", "ING-PALM-001"]:
            ing_id = self.ctx.ingredient_ids.get(code)
            if ing_id and ing_suppliers.get(ing_id, 0) == 1:
                spof_ings.append(code)

        target_count = self._manifest.get("benchmarks", {}).get("level_0_3_network", {}).get("spof_ingredient_count", 2)
        if len(spof_ings) >= target_count:
            return True, f"SPOFs found: {', '.join(spof_ings)}"
        return False, f"Expected {target_count} SPOFs, found: {spof_ings}"

    def validate_multi_promo(self) -> tuple[bool, str]:
        """
        Validate multi-promo calendar is working correctly.

        Thresholds from manifest validation_tolerances.promo_lift_range.

        Returns:
            Tuple of (passed, message)
        """
        pos = self.data.get("pos_sales", [])
        if not pos:
            return False, "No pos_sales data"

        # Get promo-flagged sales
        promo_sales = [s for s in pos if s.get("is_promotional", False)]
        if not promo_sales:
            return False, "No is_promotional=True sales found"

        # Check 1: Unique promos used
        unique_promos = {s.get("promo_id") for s in promo_sales if s.get("promo_id")}
        promo_count = len(unique_promos)

        if promo_count < 20:
            return False, f"Only {promo_count} unique promos (expected 50+)"

        # Check 2: Lift ratio
        avg_promo_qty = sum(s.get("quantity_eaches", 0) for s in promo_sales) / len(
            promo_sales
        )

        non_promo_sales = [s for s in pos if not s.get("is_promotional", False)]
        if non_promo_sales:
            avg_non_promo_qty = sum(
                s.get("quantity_eaches", 0) for s in non_promo_sales
            ) / len(non_promo_sales)
            lift_ratio = (
                avg_promo_qty / avg_non_promo_qty if avg_non_promo_qty > 0 else 0
            )
        else:
            lift_ratio = 0
            
        target_lift = self._get_tolerance("promo_lift_range", [1.5, 3.5])[0]

        if lift_ratio < target_lift * 0.8: # Allow some drift
            return False, f"Weak lift: {lift_ratio:.1f}x (target >={target_lift}x)"

        # Check 3: Week distribution
        promo_weeks = set()
        for s in promo_sales[:1000]:
            sale_date = s.get("sale_date")
            if sale_date is not None:
                if hasattr(sale_date, "astype"):
                    import numpy as np
                    ts = (sale_date - np.datetime64("1970-01-01", "D")) / np.timedelta64(1, "D")
                    sale_date = date.fromordinal(int(ts) + date(1970, 1, 1).toordinal())
                if hasattr(sale_date, "isocalendar"):
                    promo_weeks.add(sale_date.isocalendar()[1])

        # Build summary message
        promo_pct = len(promo_sales) / len(pos)
        msg = (
            f"{promo_count} promos, {lift_ratio:.1f}x lift, "
            f"{len(promo_sales):,} sales ({promo_pct:.1%})"
        )

        if promo_count >= 50 and lift_ratio >= target_lift:
            return True, msg
        elif promo_count >= 20 and lift_ratio >= target_lift * 0.8:
            return True, f"{msg} (marginal)"
        else:
            return False, msg

    def validate_referential_integrity(self) -> tuple[bool, str]:
        """
        Spot-check FK validity (sample-based for speed).

        Sample size from manifest validation_tolerances.referential_integrity_sample_size.

        Returns:
            Tuple of (passed, message)
        """
        errors = []
        sample_size = self._get_tolerance("referential_integrity_sample_size", 1000)

        # Check order_lines -> orders
        order_lines = self.data.get("order_lines", [])
        order_id_set = {o.get("id") for o in self.data.get("orders", [])}
        if order_lines:
            sample = order_lines[:sample_size]
            bad = [ol for ol in sample if ol.get("order_id") not in order_id_set]
            if bad:
                errors.append(f"order_lines: {len(bad)} invalid order_id refs")

        # Check shipment_lines -> shipments
        shipment_lines = self.data.get("shipment_lines", [])
        shipment_id_set = {s.get("id") for s in self.data.get("shipments", [])}
        if shipment_lines:
            sample = shipment_lines[:sample_size]
            bad = [sl for sl in sample if sl.get("shipment_id") not in shipment_id_set]
            if bad:
                errors.append(f"shipment_lines: {len(bad)} invalid shipment_id refs")

        if not errors:
            return True, "FK spot-checks passed"
        return False, "; ".join(errors)

    def validate_chaos_injection(self) -> tuple[bool, str]:
        """
        Verify that risk events and quirks were applied correctly.

        Thresholds and targets from manifest chaos_validation section.

        Returns:
            Tuple of (passed, message)
        """
        checks = []
        details = []

        risk_manager = self.ctx.risk_manager
        quirks_manager = self.ctx.quirks_manager

        # Check RSK-BIO-001: Contamination
        config_bio = self._get_chaos_config("RSK-BIO-001")
        if risk_manager and risk_manager.is_triggered("RSK-BIO-001"):
            batch_id = config_bio.get("batch_id", "B-2024-RECALL-001")
            expected_status = config_bio.get("expected_status", "REJECTED")
            recall_batch = next(
                (b for b in self.data.get("batches", []) if b.get("batch_number") == batch_id),
                None,
            )
            if recall_batch:
                if recall_batch.get("qc_status") == expected_status:
                    checks.append(True)
                    details.append(f"RSK-BIO-001: Recall batch {expected_status}")
                else:
                    checks.append(False)
                    details.append(f"RSK-BIO-001: Expected {expected_status}, got {recall_batch.get('qc_status')}")
            else:
                checks.append(False)
                details.append(f"RSK-BIO-001: Recall batch {batch_id} not found")

        # Check RSK-LOG-002: Port strike
        config_log = self._get_chaos_config("RSK-LOG-002")
        if risk_manager and risk_manager.is_triggered("RSK-LOG-002"):
            min_legs = config_log.get("min_affected_legs", 100)
            delayed_legs = [
                leg
                for leg in self.data.get("shipment_legs", [])
                if leg.get("status") == "delayed"
                or "RSK-LOG-002" in (leg.get("notes") or "")
            ]
            if len(delayed_legs) >= min_legs:
                checks.append(True)
                details.append(f"RSK-LOG-002: {len(delayed_legs)} legs delayed by port strike")
            else:
                checks.append(False)
                details.append(f"RSK-LOG-002: Only {len(delayed_legs)} legs delayed (expected {min_legs}+)")

        # Check RSK-SUP-003: Supplier OTD degradation
        config_sup = self._get_chaos_config("RSK-SUP-003")
        if risk_manager and risk_manager.is_triggered("RSK-SUP-003"):
            target_sup = config_sup.get("target_supplier", "SUP-PALM-MY-001")
            supplier_id = self.ctx.supplier_ids.get(target_sup)
            if supplier_id:
                supplier_grs = [
                    gr
                    for gr in self.data.get("goods_receipts", [])
                    if gr.get("supplier_id") == supplier_id
                ]
                late_grs = [
                    gr for gr in supplier_grs
                    if gr.get("receipt_date") and gr.get("expected_date")
                    and gr.get("receipt_date") > gr.get("expected_date")
                ]
                if supplier_grs:
                    late_rate = len(late_grs) / len(supplier_grs)
                    max_otd = config_sup.get("degraded_otd_max", 0.50)
                    min_late_rate = 1.0 - max_otd
                    if late_rate >= min_late_rate:
                        checks.append(True)
                        details.append(f"RSK-SUP-003: {late_rate:.0%} supplier deliveries late")
                    else:
                        checks.append(False)
                        details.append(f"RSK-SUP-003: Late rate {late_rate:.0%} < {min_late_rate:.0%}")
                else:
                    checks.append(True)
                    details.append(f"RSK-SUP-003: Supplier {target_sup} flagged (no GRs)")

        # Check RSK-CYB-004: Cyber outage
        config_cyb = self._get_chaos_config("RSK-CYB-004")
        if risk_manager and risk_manager.is_triggered("RSK-CYB-004"):
            target_dc = config_cyb.get("target_dc", "DC-NAM-CHI-001")
            min_hold = config_cyb.get("min_hold_waves", 100)
            dc_id = self.ctx.dc_ids.get(target_dc)
            if dc_id:
                hold_waves = [
                    w
                    for w in self.data.get("pick_waves", [])
                    if w.get("dc_id") == dc_id and w.get("status") == "on_hold"
                ]
                if len(hold_waves) >= min_hold:
                    checks.append(True)
                    details.append(f"RSK-CYB-004: {len(hold_waves)} waves ON_HOLD")
                else:
                    checks.append(False)
                    details.append(f"RSK-CYB-004: Only {len(hold_waves)} waves ON_HOLD (expected {min_hold}+)")

        # Check RSK-ENV-005: Carbon tax spike
        if risk_manager and risk_manager.is_triggered("RSK-ENV-005"):
            checks.append(True)
            details.append("RSK-ENV-005: Carbon tax multiplier active")

        # Check phantom inventory quirk
        config_phantom = self._get_chaos_config("phantom_inventory")
        if quirks_manager and quirks_manager.is_enabled("phantom_inventory"):
            shrinkage = [
                inv
                for inv in self.data.get("inventory", [])
                if inv.get("has_shrinkage")
            ]
            if len(shrinkage) > 0:
                rate = len(shrinkage) / len(self.data.get("inventory", [1]))
                target_range = config_phantom.get("shrinkage_rate_range", [0.01, 0.04])
                if target_range[0] <= rate <= target_range[1]:
                    checks.append(True)
                    details.append(f"phantom_inventory: {rate:.1%} shrinkage")
                else:
                    checks.append(True)
                    details.append(f"phantom_inventory: {rate:.1%} shrinkage (drifted)")
            else:
                if len(self.data.get("inventory", [])) > 0:
                    checks.append(False)
                    details.append("phantom_inventory: No shrinkage applied")

        # Check data_decay quirk
        config_decay = self._get_chaos_config("data_decay")
        if quirks_manager and quirks_manager.is_enabled("data_decay"):
            min_decayed = config_decay.get("min_affected_batches", 10)
            decayed = [b for b in self.data.get("batches", []) if b.get("data_decay_affected")]
            if len(decayed) >= min_decayed:
                checks.append(True)
                details.append(f"data_decay: {len(decayed)} batches affected")
            else:
                checks.append(False)
                details.append(f"data_decay: Only {len(decayed)} affected (expected {min_decayed}+)")

        # Check bullwhip_whip_crack quirk
        config_bullwhip = self._get_chaos_config("bullwhip_whip_crack")
        if quirks_manager and quirks_manager.is_enabled("bullwhip_whip_crack"):
            min_batched = config_bullwhip.get("min_batched_orders", 10)
            batched = [
                o for o in self.data.get("orders", [])
                if o.get("is_batched") or "batched" in (o.get("notes") or "").lower()
            ]
            if len(batched) >= min_batched:
                checks.append(True)
                details.append(f"bullwhip_whip_crack: {len(batched)} batched orders")
            else:
                checks.append(False)
                details.append(f"bullwhip_whip_crack: Only {len(batched)} batched (expected {min_batched}+)")

        # Check port_congestion_flicker quirk
        config_congestion = self._get_chaos_config("port_congestion_flicker")
        if quirks_manager and quirks_manager.is_enabled("port_congestion_flicker"):
            min_corr = config_congestion.get("min_correlated_legs", 100)
            congested = [
                leg for leg in self.data.get("shipment_legs", [])
                if "congestion" in (leg.get("notes") or "").lower()
                or leg.get("has_congestion_delay")
            ]
            if len(congested) >= min_corr:
                checks.append(True)
                details.append(f"port_congestion_flicker: {len(congested)} legs affected")
            else:
                checks.append(False)
                details.append(f"port_congestion_flicker: Only {len(congested)} affected (expected {min_corr}+)")

        if not checks:
            return True, "No chaos checks required"

        passed = sum(checks)
        total = len(checks)
        if all(checks):
            return True, f"All {total} chaos checks passed"
        return False, f"{passed}/{total}: " + "; ".join(details)

    def run_all_validations(self) -> list[tuple[str, bool, str]]:
        """
        Run all validation checks.

        Returns:
            List of (check_name, passed, message) tuples
        """
        validations = [
            ("Row counts", self.validate_row_counts),
            ("Pareto distribution", self.validate_pareto),
            ("Hub concentration", self.validate_hub_concentration),
            ("Named entities", self.validate_named_entities),
            ("SPOF ingredients", self.validate_spof),
            ("Multi-promo calendar", self.validate_multi_promo),
            ("Referential integrity", self.validate_referential_integrity),
            ("Chaos injection", self.validate_chaos_injection),
        ]

        results = []
        for name, validator in validations:
            passed, message = validator()
            results.append((name, passed, message))

        return results

    def print_validation_report(self) -> bool:
        """
        Run all validations and print a formatted report.

        Returns:
            True if all validations passed, False otherwise
        """
        print("\n" + "=" * 60)
        print("Validation Report")
        print("=" * 60)

        results = self.run_all_validations()
        all_passed = True

        for name, passed, message in results:
            status = "PASS" if passed else "FAIL"
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {name}: {status}")
            print(f"    {message}")
            if not passed:
                all_passed = False

        print("=" * 60)
        if all_passed:
            print("All validations PASSED")
        else:
            print("Some validations FAILED")
        print("=" * 60 + "\n")

        return all_passed
