"""ERP configuration loader.

Loads cost_master.json, simulation_config.json, and world_definition.json
into a typed ErpConfig dataclass for use throughout the ETL pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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


# Fixed Chart of Accounts — 12 accounts for GL journal
CHART_OF_ACCOUNTS = [
    {"account_code": "1100", "account_name": "Raw Material Inventory", "account_type": "asset"},
    {"account_code": "1120", "account_name": "Work In Process", "account_type": "asset"},
    {"account_code": "1130", "account_name": "Finished Goods Inventory", "account_type": "asset"},
    {"account_code": "1140", "account_name": "In-Transit Inventory", "account_type": "asset"},
    {"account_code": "1200", "account_name": "Accounts Receivable", "account_type": "asset"},
    {"account_code": "1000", "account_name": "Cash", "account_type": "asset"},
    {"account_code": "2100", "account_name": "Accounts Payable", "account_type": "liability"},
    {"account_code": "4100", "account_name": "Revenue", "account_type": "revenue"},
    {"account_code": "5100", "account_name": "Cost of Goods Sold", "account_type": "expense"},
    {"account_code": "5200", "account_name": "Returns Expense", "account_type": "expense"},
    {"account_code": "5300", "account_name": "Freight Expense", "account_type": "expense"},
    {"account_code": "5400", "account_name": "Manufacturing Overhead", "account_type": "expense"},
]


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
