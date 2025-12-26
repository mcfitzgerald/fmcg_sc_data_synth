#!/usr/bin/env python3
"""
Generate synthetic FMCG supply chain data for fictional company Prism Consumer Goods.

Target data volumes (~11.6M rows across 67 tables):

Note: Row counts reflect realistic B2B CPG model (Manufacturer->Retailer pattern).
Order lines use industry benchmarks: avg 16 lines/order for big box retailers.
See: https://impactwms.com/2020/09/01/know-your-order-profile

DOMAIN A - SOURCE (~200K rows):
- 50 ingredients (ING-xxx codes, CAS numbers)
- 200 suppliers (tiered: 40 T1, 80 T2, 80 T3)
- 130 supplier-ingredient links with lead times, MOQs
- 500 certifications (ISO, GMP, Halal, Kosher, RSPO)
- 25,000 purchase orders
- 75,000 purchase order lines
- 20,000 goods receipts
- 60,000 goods receipt lines

DOMAIN B - TRANSFORM (~900K rows):
- 7 plants (Tennessee, Texas, Brazil, China, India, Poland, Turkey)
- 35 production lines (~5 per plant)
- 45 formulas (recipes)
- 450 formula ingredients (avg 10 ingredients per formula)
- 50,000 work orders
- 150,000 work order materials
- 48,000 batches (production lots)
- 150,000 batch ingredients
- 48,000 batch cost ledger entries

DOMAIN C - PRODUCT (~10K rows):
- 3 products (PrismWhite, ClearWave, AquaPure)
- 20 packaging types
- 2,000 SKUs (product x packaging x size x region explosion)
- 2,000 SKU costs
- 500 SKU substitute links

DOMAIN D - ORDER (~7M rows):
- 4 channels (B&M Large, B&M Dist, Ecom, DTC)
- 100 promotions (with temporal effectivity and hangover)
- 200,000 orders
- 3,200,000 order lines (realistic B2B: 10-40 lines for big box)
- 3,700,000 order allocations

DOMAIN E - FULFILL (~2M rows):
- 5 divisions (NAM, LATAM, APAC, EUR, AFR-EUR)
- 25 distribution centers
- 20 ports
- 86 retail accounts (archetypes)
- 10,000 retail locations (stores)
- 180,000 shipments
- 1,000,000 shipment lines (with batch_fraction for splitting)
- 23,000 inventory records
- 25,000 pick waves
- 727,000 pick wave orders

DOMAIN E2 - LOGISTICS (~260K rows):
- 20 carriers
- 100 carrier contracts (temporal effectivity)
- 1,000 carrier rates
- 200 route segments (atomic legs)
- 50 routes (composed, multi-leg)
- 500 route segment assignments
- 261,000 shipment legs (multi-leg routing)

DOMAIN E3 - ESG (~200K rows):
- 100 emission factors
- 180,000 shipment emissions
- 200 supplier ESG scores
- 50 sustainability targets
- 100 modal shift opportunities

DOMAIN F - PLAN (~900K rows):
- 500,000 POS sales (52 weeks x ~10K stores x ~10 SKUs per store)
- 100,000 demand forecasts
- 10,000 forecast accuracy records
- 5,000 consensus adjustments
- 25,000 replenishment params
- 50,000 demand allocations
- 10,000 capacity plans
- 50,000 supply plans
- 20,000 plan exceptions

DOMAIN G - RETURN (~80K rows):
- 10,000 RMA authorizations
- 10,000 returns
- 30,000 return lines
- 30,000 disposition logs

DOMAIN H - ORCHESTRATE (~570K rows):
- 20 KPI thresholds (Desmet Triangle)
- 1,000 KPI actuals (weekly measurements)
- 520,000 OSA metrics (on-shelf availability)
- 100 business rules
- 500 risk events
- 46,000 audit log entries

NAMED ENTITIES (Deterministic Testing - Section 4.8):
- B-2024-RECALL-001: Contaminated Sorbitol batch -> 500 stores
- ACCT-MEGA-001: MegaMart (4,500 stores, 25% of orders)
- SUP-PALM-MY-001: Single-source Palm Oil supplier (SPOF)
- DC-NAM-CHI-001: Bottleneck DC Chicago (2,000 stores, 40% NAM volume)
- PROMO-BF-2024: Black Friday 2024 (3x demand, bullwhip)
- LANE-SH-LA-001: Seasonal Shanghai->LA (50% capacity Jan-Feb)
- ING-PALM-001: Palm Oil (60-120 day lead time, 50% OT)
- ING-SORB-001: Sorbitol (single supplier, all toothpaste)
- ING-PEPP-001: Peppermint Oil (Q2-Q3 only, 3x price Q1)

DATA GENERATION PRINCIPLES (Section 4):
- Barabasi-Albert preferential attachment for network topology
- Zipf distribution for SKU popularity (80/20 Pareto rule)
- Lumpy demand with promo spikes and post-promo hangover
- Temporal flickering for seasonal routes and carrier contracts
- FMCG benchmarks: 8-12 inventory turns, 95%+ OTIF, 92-95% OSA

Implementation Status: Modular generators v0.9.39
"""

import argparse
import random
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from faker import Faker

# Import from modular data_generation package
from data_generation import (
    BENCHMARK_MANIFEST_PATH,
    # Level Generators (0-14)
    GeneratorContext,
    Level0Generator,
    Level1Generator,
    Level2Generator,
    Level3Generator,
    Level4Generator,
    Level5Generator,
    Level6Generator,
    Level7Generator,
    Level8Generator,
    Level9Generator,
    Level10Generator,
    Level11Generator,
    Level12Generator,
    Level13Generator,
    Level14Generator,
    # Validation
    DataValidator,
    # Performance modules
    PooledFaker,
    QuirksManager,
    RealismMonitor,
    RiskEventManager,
    StaticDataPool,
    # Helpers
    TARGET_ROW_COUNTS,
    create_named_entities,
)

# Output path
OUTPUT_PATH = Path(__file__).parent.parent / "postgres" / "seed.sql"

# Seed for reproducibility
RANDOM_SEED = 42


class FMCGDataGenerator:
    """
    Orchestrates generation of all FMCG data using level-based dependencies.

    Generation follows 15 levels (0-14) based on FK dependencies.
    Uses modular level generators from data_generation.generators package.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        """
        Initialize generator with reproducible random state.

        Args:
            seed: Random seed for reproducibility
        """
        # Initialize random state
        self.seed = seed
        random.seed(seed)
        rng = np.random.default_rng(seed)
        fake = Faker()
        Faker.seed(seed)

        # Performance modules
        pool = StaticDataPool(seed=seed)
        pooled_faker = PooledFaker(seed=seed)
        realism_monitor = RealismMonitor(manifest_path=BENCHMARK_MANIFEST_PATH)
        risk_manager = RiskEventManager(manifest_path=BENCHMARK_MANIFEST_PATH, seed=seed)
        quirks_manager = QuirksManager(manifest_path=BENCHMARK_MANIFEST_PATH, seed=seed)

        # Trigger risk events and get stochastic mode
        triggered_events = risk_manager.trigger_all()
        stochastic_mode = risk_manager.get_stochastic_mode()
        
        # Configure monitor with active quirks
        if realism_monitor:
            realism_monitor.set_active_quirks(quirks_manager.get_enabled_quirks())
            realism_monitor.set_stochastic_mode(stochastic_mode)

        # Create shared context for all generators
        self.ctx = GeneratorContext(
            seed=seed,
            rng=rng,
            fake=fake,
            pooled_faker=pooled_faker,
            base_year=2024,
            named_entities=create_named_entities(),
            pool=pool,
            realism_monitor=realism_monitor,
            risk_manager=risk_manager,
            quirks_manager=quirks_manager,
            stochastic_mode=stochastic_mode,
            triggered_events=triggered_events,
        )

        # Initialize empty data tables
        self.ctx.init_data_tables()

        # Instantiate level generators
        self._generators = [
            Level0Generator(self.ctx),
            Level1Generator(self.ctx),
            Level2Generator(self.ctx),
            Level3Generator(self.ctx),
            Level4Generator(self.ctx),
            Level5Generator(self.ctx),
            Level6Generator(self.ctx),
            Level7Generator(self.ctx),
            Level8Generator(self.ctx),
            Level9Generator(self.ctx),
            Level10Generator(self.ctx),
            Level11Generator(self.ctx),
            Level12Generator(self.ctx),
            Level13Generator(self.ctx),
            Level14Generator(self.ctx),
        ]

    def generate_all(self) -> None:
        """
        Generate all data in dependency order (levels 0-14).

        Includes performance tracking and inline realism monitoring.
        """
        print("=" * 60)
        print("Prism Consumer Goods - FMCG Data Generation")
        print("=" * 60)
        print(f"Seed: {self.seed}")
        print(f"Target: ~{sum(TARGET_ROW_COUNTS.values()):,} rows across 67 tables")
        print()

        # Chaos injection status
        print("Chaos Injection Active:")
        print(f"  Risk events: {self.ctx.triggered_events}")
        print(f"  Quirks enabled: {self.ctx.quirks_manager.get_enabled_quirks()}")
        print(f"  Stochastic mode: {self.ctx.stochastic_mode.value}")
        print()

        print("Generating levels 0-14...")
        print()

        gen_start = time.time()
        self.generate_from_level(0)
        gen_elapsed = time.time() - gen_start

        print()
        print("=" * 60)
        print("Generation Summary")
        print("=" * 60)
        total_rows = sum(len(rows) for rows in self.ctx.data.values())
        rows_per_sec = total_rows / gen_elapsed if gen_elapsed > 0 else 0
        print(f"Total rows: {total_rows:,}")
        print(f"Total time: {gen_elapsed:.2f}s ({rows_per_sec:,.0f} rows/sec)")

        # Performance breakdown by level
        if self.ctx._level_times:
            print()
            print("Level Performance:")
            for level in sorted(self.ctx._level_times.keys()):
                t = self.ctx._level_times[level]
                r = self.ctx._level_rows.get(level, 0)
                rps = r / t if t > 0 else 0
                print(f"  Level {level:2d}: {t:6.2f}s - {r:8,} rows ({rps:,.0f}/sec)")

        # Feed all generated data to the RealismMonitor for benchmark validation
        if self.ctx.realism_monitor:
            self._feed_data_to_monitor()

        # Inline realism monitoring report
        if self.ctx.realism_monitor:
            report = self.ctx.realism_monitor.get_reality_report()
            if not report["is_realistic"]:
                print()
                print("WARNING: Realism violations detected:")
                for violation in report["violations"][:5]:
                    print(f"  - {violation}")

    def _feed_data_to_monitor(self) -> None:
        """Feed all generated data to the RealismMonitor for benchmark validation."""
        monitor = self.ctx.realism_monitor
        if not monitor:
            return

        # Tables that the monitor knows how to process
        # NOTE: Order matters! shipments must come before shipment_lines
        # so the destination lookup is populated for mass balance tracking
        monitored_tables = [
            "pos_sales", "orders", "order_lines", "work_orders",
            "batches", "batch_ingredients", "inventory",
            "shipments", "shipment_legs", "shipment_lines",  # shipments before shipment_lines!
            "demand_forecasts", "forecast_accuracy",
            "returns", "osa_metrics", "kpi_actuals",
        ]

        for table in monitored_tables:
            rows = self.ctx.data.get(table, [])
            if rows:
                monitor.observe_batch("gen", table, rows)

        # Run the benchmark checks
        monitor.check_benchmarks()

    def generate_from_level(self, start_level: int) -> None:
        """
        Generate (or regenerate) from a specific level through level 14.

        Args:
            start_level: Level to start generation from (0-14)
        """
        if start_level < 0 or start_level > 14:
            raise ValueError(f"start_level must be 0-14, got {start_level}")

        # If regenerating, clear affected levels
        if start_level > 0:
            print(f"Regenerating levels {start_level}-14 (cascade)...")
            self._clear_levels(start_level, 14)

        # Generate each level in order
        for level in range(start_level, 15):
            level_start = time.time()

            # Build lookup cache for levels that need it
            self.ctx.build_cache(level)

            # Run level generator
            self._generators[level].generate()

            # Track performance
            level_elapsed = time.time() - level_start
            self._report_level_stats(level, level_elapsed)

    def _clear_levels(self, start: int, end: int) -> None:
        """Clear data for specified levels (for regeneration)."""
        level_tables = {
            0: ["divisions", "channels", "products", "packaging_types", "ports",
                "carriers", "emission_factors", "kpi_thresholds", "business_rules",
                "ingredients"],
            1: ["suppliers", "plants", "production_lines", "carrier_contracts",
                "route_segments"],
            2: ["supplier_ingredients", "certifications", "formulas",
                "formula_ingredients", "carrier_rates", "routes",
                "route_segment_assignments"],
            3: ["retail_accounts", "retail_locations", "distribution_centers"],
            4: ["skus", "sku_costs", "sku_substitutes", "promotions",
                "promotion_skus", "promotion_accounts"],
            5: ["purchase_orders", "goods_receipts", "work_orders",
                "supplier_esg_scores", "sustainability_targets",
                "modal_shift_opportunities"],
            6: ["purchase_order_lines", "goods_receipt_lines", "work_order_materials",
                "batches", "batch_cost_ledger"],
            7: ["batch_ingredients", "inventory"],
            8: ["pos_sales", "demand_forecasts", "forecast_accuracy",
                "consensus_adjustments", "orders", "replenishment_params",
                "demand_allocation", "capacity_plans"],
            9: ["order_lines", "order_allocations", "supply_plans",
                "plan_exceptions", "pick_waves"],
            10: ["pick_wave_orders", "shipments", "shipment_legs"],
            11: ["shipment_lines"],
            12: ["rma_authorizations", "returns", "return_lines"],
            13: ["disposition_logs"],
            14: ["kpi_actuals", "osa_metrics", "risk_events", "audit_log"],
        }

        for level in range(start, end + 1):
            for table in level_tables.get(level, []):
                if table in self.ctx.data:
                    self.ctx.data[table] = []
            self.ctx.generated_levels.discard(level)

    def _report_level_stats(self, level: int, elapsed: float) -> None:
        """Report statistics for a completed level."""
        level_row_count = sum(
            len(self.ctx.data.get(t, []))
            for t in self._get_level_tables(level)
        )

        self.ctx._level_times[level] = elapsed
        self.ctx._level_rows[level] = level_row_count

        rows_per_sec = level_row_count / elapsed if elapsed > 0 else 0
        print(f"    {elapsed:.2f}s ({rows_per_sec:,.0f} rows/sec)")

    def _get_level_tables(self, level: int) -> list[str]:
        """Get the list of tables for a given level."""
        level_tables = {
            0: ["divisions", "channels", "products", "packaging_types", "ports",
                "carriers", "emission_factors", "kpi_thresholds", "business_rules",
                "ingredients"],
            1: ["suppliers", "plants", "production_lines", "carrier_contracts",
                "route_segments"],
            2: ["supplier_ingredients", "certifications", "formulas",
                "formula_ingredients", "carrier_rates", "routes",
                "route_segment_assignments"],
            3: ["retail_accounts", "retail_locations", "distribution_centers"],
            4: ["skus", "sku_costs", "sku_substitutes", "promotions",
                "promotion_skus", "promotion_accounts"],
            5: ["purchase_orders", "goods_receipts", "work_orders",
                "supplier_esg_scores", "sustainability_targets",
                "modal_shift_opportunities"],
            6: ["purchase_order_lines", "goods_receipt_lines", "work_order_materials",
                "batches", "batch_cost_ledger"],
            7: ["batch_ingredients", "inventory"],
            8: ["pos_sales", "demand_forecasts", "forecast_accuracy",
                "consensus_adjustments", "orders", "replenishment_params",
                "demand_allocation", "capacity_plans"],
            9: ["order_lines", "order_allocations", "supply_plans",
                "plan_exceptions", "pick_waves"],
            10: ["pick_wave_orders", "shipments", "shipment_legs"],
            11: ["shipment_lines"],
            12: ["rma_authorizations", "returns", "return_lines"],
            13: ["disposition_logs"],
            14: ["kpi_actuals", "osa_metrics", "risk_events", "audit_log"],
        }
        return level_tables.get(level, [])

    def validate_realism(self) -> dict[str, tuple[bool, str]]:
        """
        Validate generated data meets realism requirements.

        Returns dict of {check_name: (passed, message)}
        """
        print()
        print("=" * 60)
        print("Validation Suite")
        print("=" * 60)

        # Use modular DataValidator
        validator = DataValidator(self.ctx)

        results = {
            "row_counts": validator.validate_row_counts(),
            "pareto": validator.validate_pareto(),
            "hub_concentration": validator.validate_hub_concentration(),
            "named_entities": validator.validate_named_entities(),
            "spof": validator.validate_spof(),
            "multi_promo": validator.validate_multi_promo(),
            "referential_integrity": validator.validate_referential_integrity(),
            "chaos_injection": validator.validate_chaos_injection(),
        }

        # Print summary
        print()
        print("-" * 40)
        passed_checks = sum(1 for p, _ in results.values() if p)
        total_checks = len(results)
        print(f"Validation: {passed_checks}/{total_checks} checks passed")

        for name, (ok, msg) in results.items():
            status = "+" if ok else "x"
            print(f"  {status} {name}: {msg}")

        # Check RealismMonitor benchmarks
        benchmark_passed = True
        if self.ctx.realism_monitor:
            print()
            print("-" * 40)
            print("Benchmark Comparison:")
            # _print_benchmark_comparison returns False if any benchmark failed
            benchmark_passed = self._print_benchmark_comparison()

        # Add benchmark result to return dict
        results["benchmarks"] = (benchmark_passed, "RealismMonitor benchmarks")

        return results

    def _print_benchmark_comparison(self) -> bool:
        """
        Print detailed benchmark comparison from RealismMonitor.
        
        Returns:
            True if all benchmarks passed, False otherwise.
        """
        report = self.ctx.realism_monitor.get_reality_report()
        stats = report.get("statistics", {})
        tol = self.ctx.realism_monitor._tolerances
        all_passed = True

        def get_range(key, default):
            val = tol.get(key, default)
            return (val[0], val[1]) if isinstance(val, list) else default

        # Helper to print status
        def check_stat(label, value, target_range, is_pct=True):
            nonlocal all_passed
            if isinstance(target_range, tuple):
                passed = target_range[0] <= value <= target_range[1]
                t_str = f"[{target_range[0]:.0%}-{target_range[1]:.0%}]" if is_pct else f"[{target_range[0]}-{target_range[1]}]"
            else:
                passed = value <= target_range
                t_str = f"[<{target_range:.0%}]" if is_pct else f"[<{target_range}]"
            
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            
            v_str = f"{value:.1%}" if is_pct else f"{value:.3f}"
            print(f"  {label:<22} {v_str:<8} {t_str:<12} {status}")

        # Distribution
        if "order_sku" in stats.get("frequencies", {}):
            pareto = stats["frequencies"]["order_sku"].get("pareto_20_80", 0)
            check_stat("Pareto (top 20%)", pareto, get_range("pareto_top20_range", (0.75, 0.85)))

        if stats.get("hub_concentration", {}).get("top_accounts"):
            pct = stats["hub_concentration"]["top_accounts"][0].get("pct", 0)
            check_stat("Hub concentration", pct, get_range("hub_concentration_range", (0.20, 0.30)))

        # Bullwhip
        bw = stats.get("bullwhip", {})
        if bw:
            check_stat("POS CV", bw.get("pos_cv", 0), get_range("pos_cv_range", (0.15, 0.80)), False)
            check_stat("Order CV", bw.get("order_cv", 0), get_range("order_cv_range", (0.20, 0.80)), False)
            check_stat("Bullwhip multiplier", bw.get("multiplier", 0), get_range("bullwhip_multiplier_range", (0.3, 3.0)), False)
            check_stat("Promo Lift", bw.get("promo_lift", 0), get_range("promo_lift_range", (1.5, 3.5)), False)

        # Forecast bias
        fc = stats.get("forecast", {})
        if fc:
            check_stat("Forecast bias", fc.get("bias_pct", 0), tol.get("forecast_bias_max", 0.20))

        # Production
        prod = stats.get("production", {})
        if prod:
            check_stat("Yield mean", prod.get("yield_mean", 0), get_range("yield_mean_range", (0.96, 0.99)))
            
            # QC Rejection Rate - Check for dynamic tolerance due to data_decay
            qc_rate = prod.get("qc_rejection_rate", 0)
            qc_range = get_range("qc_rejection_rate_range", (0.01, 0.04))
            
            # If data_decay is active, elevate tolerance
            if self.ctx.quirks_manager and self.ctx.quirks_manager.is_enabled("data_decay"):
                decay_config = self.ctx.quirks_manager.get_params("data_decay")
                base = decay_config.get("base_rejection_rate", 0.02)
                elevated = decay_config.get("elevated_rejection_rate", 0.08)
                # Weighted average depends on age distribution, but max shouldn't exceed elevated
                # Allow up to 1.5x elevated rate to be safe
                qc_range = (qc_range[0], elevated * 1.5)
                
            check_stat("QC rejection rate", qc_rate, qc_range)

        # Logistics
        log = stats.get("logistics", {})
        if log:
            check_stat("OTIF rate", log.get("otif_rate", 0), get_range("otif_range", (0.85, 0.99)))
            check_stat("Delay mean", log.get("delay_mean", 0), tol.get("delay_mean_max_hours", 24), False)

        # Returns
        ret = stats.get("returns", {})
        if ret:
            check_stat("Return rate", ret.get("return_rate", 0), get_range("return_rate_range", (0.01, 0.06)))

        # OSA
        osa = stats.get("osa", {})
        if osa:
            check_stat("OSA rate", osa.get("osa_rate", 0), get_range("osa_range", (0.88, 0.96)))

        # Expert Metrics
        expert = stats.get("expert", {})
        if expert:
            print()
            print("  Expert Reality Checks:")
            check_stat("Schedule Adherence", expert.get("schedule_adherence_days", 0), tol.get("schedule_adherence_tolerance_days", 1.0), False)
            check_stat("Truck Fill Rate", expert.get("truck_fill_rate", 0), (tol.get("truck_fill_rate_target", 0.70), 1.0))
            check_stat("SLOB Inventory", expert.get("slob_pct", 0), tol.get("slob_inventory_max_pct", 0.15))
            check_stat("OEE", expert.get("oee", 0), get_range("oee_range", (0.65, 0.85)))
            check_stat("Inventory Turns", expert.get("inventory_turns", 0), get_range("inventory_turns_range", (6.0, 14.0)), False)
            check_stat("Forecast MAPE", expert.get("forecast_mape", 0), get_range("mape_range", (0.20, 0.50)))

            # Cost-to-Serve metrics (display as $ values, not percentages)
            cost_per_case = expert.get("cost_per_case", 0)
            cost_range = get_range("cost_per_case_range", (1.00, 3.00))
            cost_passed = cost_range[0] <= cost_per_case <= cost_range[1]
            if not cost_passed:
                all_passed = False
            print(f"  {'Cost-to-Serve':<22} ${cost_per_case:<7.2f} [${cost_range[0]:.2f}-${cost_range[1]:.2f}]  {'PASS' if cost_passed else 'FAIL'}")

            # Cost variance (P90/P50 ratio)
            cost_variance = expert.get("cost_p90_p50_ratio", 1.0)
            max_variance = tol.get("cost_variance_max_ratio", 4.0)
            variance_passed = cost_variance <= max_variance
            if not variance_passed:
                all_passed = False
            print(f"  {'Cost Variance (P90/P50)':<22} {cost_variance:<7.1f}x [<{max_variance:.1f}x]       {'PASS' if variance_passed else 'FAIL'}")

        # Mass Balance (Physics) checks
        mb = stats.get("mass_balance", {})
        if mb and mb.get("batch_count", 0) > 0:
            print()
            print("  Mass Balance (Physics):")
            max_drift = tol.get("mass_balance_drift_max", 0.02)

            # Ingredient → Batch (kg) - negative drift is normal (yield loss)
            ing_drift = mb.get("ingredient_to_batch_drift", 0)
            ing_passed = ing_drift <= max_drift  # Can't have more output than input
            if not ing_passed:
                all_passed = False
            print(f"  {'Ingredient→Batch (kg)':<22} {ing_drift:+7.1%}  [<+{max_drift:.0%}]       {'PASS' if ing_passed else 'FAIL'}")

            # Batch → Ship+Inv (cases) - should balance within tolerance
            batch_drift = mb.get("batch_to_shipment_drift", 0)
            batch_passed = abs(batch_drift) <= max_drift * 5  # 10% tolerance
            if not batch_passed:
                all_passed = False
            print(f"  {'Batch→Ship+Inv (cases)':<22} {batch_drift:+7.1%}  [±{max_drift*5:.0%}]       {'PASS' if batch_passed else 'FAIL'}")

            # Order → Fulfill (cases) - can't ship more than ordered
            order_drift = mb.get("order_to_fulfill_drift", 0)
            fill_rate = mb.get("fill_rate", 0)
            order_passed = order_drift <= max_drift  # Can't over-fulfill
            if not order_passed:
                all_passed = False
            print(f"  {'Order→Fulfill (cases)':<22} {order_drift:+7.1%}  [<+{max_drift:.0%}]       {'PASS' if order_passed else 'FAIL'}  (Fill: {fill_rate:.1%})")

        # Chaos effects summary
        chaos = stats.get("chaos_effects", {})
        if chaos:
            print()
            print("Chaos Effects Applied:")
            print(f"  Port strike legs:      {chaos.get('port_strike_delayed_legs', 0):,}")
            print(f"  Congestion legs:       {chaos.get('congestion_correlated_legs', 0):,}")
            print(f"  Batched promo orders:  {chaos.get('batched_promo_orders', 0):,}")
            
        return all_passed

    def write_sql(self, output_path: Path = OUTPUT_PATH) -> None:
        """
        Write generated data to SQL file using COPY format.
        """
        print()
        print(f"Writing SQL to {output_path}...")
        write_start = time.time()

        total_rows = sum(len(rows) for rows in self.ctx.data.values())
        rows_written = 0

        with open(output_path, "w") as f:
            # Header
            f.write("-- ============================================\n")
            f.write("-- Prism Consumer Goods - FMCG Seed Data\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Seed: {self.seed}\n")
            f.write(f"-- Total rows: {total_rows:,}\n")
            f.write("-- ============================================\n\n")

            # Disable triggers for bulk load
            f.write("SET session_replication_role = replica;\n\n")

            # Write each table's data with progress reporting
            table_count = len([t for t, rows in self.ctx.data.items() if rows])
            tables_written = 0

            for table_name, rows in self.ctx.data.items():
                if not rows:
                    continue
                self._write_table_copy(f, table_name, rows)
                rows_written += len(rows)
                tables_written += 1

                # Progress every 10 tables
                if tables_written % 10 == 0:
                    pct = rows_written / total_rows * 100
                    print(f"    Writing... {pct:.0f}% ({tables_written}/{table_count} tables)")

            # Re-enable triggers
            f.write("\nSET session_replication_role = DEFAULT;\n")

            # Update sequences
            f.write("\n-- Update sequences\n")
            for table_name, rows in self.ctx.data.items():
                if rows and "id" in rows[0]:
                    max_id = max(r.get("id", 0) for r in rows)
                    f.write(f"SELECT setval('{table_name}_id_seq', {max_id});\n")

        write_elapsed = time.time() - write_start
        rows_per_sec = total_rows / write_elapsed if write_elapsed > 0 else 0
        print(f"Done. {total_rows:,} rows written in {write_elapsed:.2f}s ({rows_per_sec:,.0f} rows/sec)")

    def _write_table_copy(self, f: Any, table_name: str, rows: list[dict]) -> None:
        """Write a table's data using COPY format."""
        if not rows:
            return

        # Get columns from first row
        columns = list(rows[0].keys())

        f.write(f"\n-- {table_name}: {len(rows):,} rows\n")
        f.write(f"COPY {table_name} ({', '.join(columns)}) FROM stdin;\n")

        for row in rows:
            values = []
            for col in columns:
                val = row.get(col)
                if val is None:
                    values.append("\\N")
                elif isinstance(val, bool):
                    values.append("t" if val else "f")
                elif isinstance(val, (date, datetime)):
                    values.append(val.isoformat())
                elif isinstance(val, str):
                    # Escape tabs, newlines, backslashes
                    values.append(
                        val.replace("\\", "\\\\")
                        .replace("\t", "\\t")
                        .replace("\n", "\\n")
                    )
                else:
                    values.append(str(val))
            f.write("\t".join(values) + "\n")

        f.write("\\.\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic FMCG supply chain data for Prism Consumer Goods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full generation + validation (no file write)
  python generate_data.py --validate-only

  # Full generation + write seed.sql
  python generate_data.py

  # Regenerate from specific level (cascade)
  python generate_data.py --from-level 8 --validate-only

  # Custom output path
  python generate_data.py --output /tmp/seed.sql

  # Different random seed
  python generate_data.py --seed 123
""",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Generate and validate data without writing SQL file",
    )

    parser.add_argument(
        "--from-level",
        type=int,
        choices=range(0, 15),
        metavar="N",
        help="Regenerate from level N through 14 (cascade). Requires prior generation.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path for seed.sql (default: {OUTPUT_PATH})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks after generation",
    )

    return parser.parse_args()


def main() -> int:
    """
    Generate FMCG data with CLI interface.

    Returns:
        0 on success, 1 on validation failure
    """
    args = parse_args()

    # Create generator
    generator = FMCGDataGenerator(seed=args.seed)

    # Generate data
    if args.from_level is not None:
        # Cascade regeneration from specific level
        if not generator.ctx.generated_levels:
            print("Warning: No prior generation found. Running full generation.")
            generator.generate_all()
        else:
            generator.generate_from_level(args.from_level)
    else:
        # Full generation
        generator.generate_all()

    # Run validation
    if not args.skip_validation:
        results = generator.validate_realism()
        all_passed = all(passed for passed, _ in results.values())
    else:
        all_passed = True
        print("\nValidation skipped.")

    # Write SQL if not validate-only
    if not args.validate_only:
        generator.write_sql(args.output)
        print(f"\nOutput: {args.output}")

    # Return status
    if all_passed:
        print("\nSuccess!")
        return 0
    else:
        print("\nValidation failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
