"""ERP configuration loader.

Loads cost_master.json, simulation_config.json, and world_definition.json
into a typed ErpConfig dataclass for use throughout the ETL pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EntityResolutionConfig:
    """Tier 1: Duplicate/variant entity friction."""

    duplicate_supplier_rate: float = 0.10
    sku_rename_rate: float = 0.05


@dataclass
class ThreeWayMatchConfig:
    """Tier 2: PO/GR/Invoice mismatch friction."""

    price_variance_rate: float = 0.08
    price_variance_pct_range: tuple[float, float] = (0.02, 0.15)
    qty_variance_rate: float = 0.05
    qty_variance_pct_range: tuple[float, float] = (0.01, 0.10)


@dataclass
class DataQualityConfig:
    """Tier 3: Null FKs, duplicates, status inconsistencies."""

    null_fk_rate_ap: float = 0.02
    null_fk_rate_gl: float = 0.01
    duplicate_invoice_rate: float = 0.005
    status_inconsistency_rate: float = 0.03


@dataclass
class PaymentTimingConfig:
    """Tier 4: Cash cycle — payments, receipts, discounts, bad debt."""

    ap_payment_lag_mean_days: float = 0.0
    ap_payment_lag_std_days: float = 5.0
    ar_receipt_lag_mean_days: float = 0.0
    ar_receipt_lag_std_days: float = 7.0
    early_payment_discount_rate: float = 0.10
    early_payment_discount_pct: float = 0.02
    early_payment_window_days: int = 10
    bad_debt_rate: float = 0.005


@dataclass
class FrictionConfig:
    """Top-level friction configuration with global toggle."""

    enabled: bool = False
    seed: int = 42
    entity_resolution: EntityResolutionConfig = field(
        default_factory=EntityResolutionConfig
    )
    three_way_match: ThreeWayMatchConfig = field(
        default_factory=ThreeWayMatchConfig
    )
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    payment_timing: PaymentTimingConfig = field(
        default_factory=PaymentTimingConfig
    )


@dataclass
class ErpConfig:
    """Aggregated configuration for ERP generation."""

    # Logistics cost parameters
    route_costs: dict[str, dict] = field(default_factory=dict)
    warehouse_costs: dict[str, float] = field(default_factory=dict)

    # Manufacturing cost splits
    labor_pct: dict[str, float] = field(default_factory=dict)
    overhead_pct: dict[str, float] = field(default_factory=dict)

    # Working capital
    dso_by_channel: dict[str, int] = field(default_factory=dict)
    dpo_days: float = 45.0

    # Channel definitions from world_definition
    channels: dict[str, dict] = field(default_factory=dict)

    # Plant parameters from simulation_config
    plant_parameters: dict[str, dict] = field(default_factory=dict)

    # Chart of accounts (seed)
    chart_of_accounts: list[dict[str, str]] = field(default_factory=list)

    # Friction layer (v0.78.0)
    friction: FrictionConfig = field(default_factory=FrictionConfig)


# Fixed Chart of Accounts — 14 accounts for GL journal
_A = "asset"
_L = "liability"
_R = "revenue"
_E = "expense"
CHART_OF_ACCOUNTS = [
    {"account_code": "1000", "account_name": "Cash", "account_type": _A},
    {"account_code": "1100", "account_name": "Raw Material Inventory", "account_type": _A},
    {"account_code": "1120", "account_name": "Work In Process", "account_type": _A},
    {"account_code": "1130", "account_name": "Finished Goods Inventory", "account_type": _A},
    {"account_code": "1140", "account_name": "In-Transit Inventory", "account_type": _A},
    {"account_code": "1200", "account_name": "Accounts Receivable", "account_type": _A},
    {"account_code": "2100", "account_name": "Accounts Payable", "account_type": _L},
    {"account_code": "4100", "account_name": "Revenue", "account_type": _R},
    {"account_code": "4200", "account_name": "Discount Income", "account_type": _R},
    {"account_code": "5100", "account_name": "Cost of Goods Sold", "account_type": _E},
    {"account_code": "5200", "account_name": "Returns Expense", "account_type": _E},
    {"account_code": "5300", "account_name": "Freight Expense", "account_type": _E},
    {"account_code": "5400", "account_name": "Manufacturing Overhead", "account_type": _E},
    {"account_code": "5500", "account_name": "Bad Debt Expense", "account_type": _E},
]


def _load_friction_config(raw: dict) -> FrictionConfig:
    """Parse the friction section from cost_master.json."""
    fc = FrictionConfig(
        enabled=raw.get("enabled", False),
        seed=raw.get("seed", 42),
    )

    er = raw.get("entity_resolution", {})
    fc.entity_resolution = EntityResolutionConfig(
        duplicate_supplier_rate=er.get("duplicate_supplier_rate", 0.10),
        sku_rename_rate=er.get("sku_rename_rate", 0.05),
    )

    twm = raw.get("three_way_match", {})
    pv_range = twm.get("price_variance_pct_range", [0.02, 0.15])
    qv_range = twm.get("qty_variance_pct_range", [0.01, 0.10])
    fc.three_way_match = ThreeWayMatchConfig(
        price_variance_rate=twm.get("price_variance_rate", 0.08),
        price_variance_pct_range=(pv_range[0], pv_range[1]),
        qty_variance_rate=twm.get("qty_variance_rate", 0.05),
        qty_variance_pct_range=(qv_range[0], qv_range[1]),
    )

    dq = raw.get("data_quality", {})
    fc.data_quality = DataQualityConfig(
        null_fk_rate_ap=dq.get("null_fk_rate_ap", 0.02),
        null_fk_rate_gl=dq.get("null_fk_rate_gl", 0.01),
        duplicate_invoice_rate=dq.get("duplicate_invoice_rate", 0.005),
        status_inconsistency_rate=dq.get("status_inconsistency_rate", 0.03),
    )

    pt = raw.get("payment_timing", {})
    fc.payment_timing = PaymentTimingConfig(
        ap_payment_lag_mean_days=pt.get("ap_payment_lag_mean_days", 0.0),
        ap_payment_lag_std_days=pt.get("ap_payment_lag_std_days", 5.0),
        ar_receipt_lag_mean_days=pt.get("ar_receipt_lag_mean_days", 0.0),
        ar_receipt_lag_std_days=pt.get("ar_receipt_lag_std_days", 7.0),
        early_payment_discount_rate=pt.get("early_payment_discount_rate", 0.10),
        early_payment_discount_pct=pt.get("early_payment_discount_pct", 0.02),
        early_payment_window_days=pt.get("early_payment_window_days", 10),
        bad_debt_rate=pt.get("bad_debt_rate", 0.005),
    )

    return fc


def load_erp_config(
    config_dir: Path | None = None,
) -> ErpConfig:
    """Load all config files and return ErpConfig."""
    if config_dir is None:
        config_dir = Path("src/prism_sim/config")

    cfg = ErpConfig()
    cfg.chart_of_accounts = CHART_OF_ACCOUNTS

    # --- cost_master.json ---
    cost_path = config_dir / "cost_master.json"
    if cost_path.exists():
        cm = json.loads(cost_path.read_text())
        lc = cm.get("logistics_costs", {})
        cfg.route_costs = lc.get("routes", {})
        cfg.warehouse_costs = lc.get("warehouse_cost_per_case_per_day", {})

        mfg = cm.get("manufacturing_costs", {})
        cfg.labor_pct = mfg.get("labor_pct_of_material", {"default": 0.25})
        cfg.overhead_pct = mfg.get("overhead_pct_of_material", {"default": 0.22})

        wc = cm.get("working_capital", {})
        cfg.dso_by_channel = wc.get("dso_days_by_channel", {})
        cfg.dpo_days = wc.get("dpo_days", 45.0)

        # Friction layer (v0.78.0)
        friction_raw = cm.get("friction", {})
        if friction_raw:
            cfg.friction = _load_friction_config(friction_raw)

    # --- world_definition.json ---
    wd_path = config_dir / "world_definition.json"
    if wd_path.exists():
        wd = json.loads(wd_path.read_text())
        cfg.channels = wd.get("topology", {}).get("channels", {})

    # --- simulation_config.json ---
    sc_path = config_dir / "simulation_config.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        sim_params = sc.get("simulation_parameters", {})
        cfg.plant_parameters = sim_params.get("manufacturing", {}).get(
            "plant_parameters", {}
        )

    return cfg
