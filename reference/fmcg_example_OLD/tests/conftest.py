"""
Pytest fixtures for FMCG example tests.

Spec: magical-launching-forest.md Phase 5

Provides:
- Database connections (PostgreSQL, Neo4j)
- Ontology accessor
- Test data fixtures (named entities)
"""

import pytest
from pathlib import Path

# Connection settings (different ports from supply_chain_example)
PG_DSN = "postgresql://virt_graph:dev_password@localhost:5433/prism_fmcg"
NEO4J_URI = "bolt://localhost:7688"
NEO4J_AUTH = ("neo4j", "dev_password")

ONTOLOGY_PATH = Path(__file__).parent.parent / "ontology" / "prism_fmcg.yaml"


@pytest.fixture(scope="session")
def pg_connection():
    """
    PostgreSQL connection for integration tests.

    TODO: Implement after schema.sql is complete
    """
    pytest.skip("PostgreSQL connection pending schema implementation")
    # import psycopg2
    # conn = psycopg2.connect(PG_DSN)
    # yield conn
    # conn.close()


@pytest.fixture(scope="session")
def neo4j_driver():
    """
    Neo4j driver for benchmark comparison tests.

    TODO: Implement after Neo4j migration is complete
    """
    pytest.skip("Neo4j connection pending migration implementation")
    # from neo4j import GraphDatabase
    # driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    # yield driver
    # driver.close()


@pytest.fixture(scope="session")
def ontology():
    """
    OntologyAccessor for schema-driven tests.

    TODO: Implement after prism_fmcg.yaml is complete
    """
    pytest.skip("Ontology accessor pending ontology implementation")
    # import sys
    # sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    # from virt_graph.ontology import OntologyAccessor
    # return OntologyAccessor(ONTOLOGY_PATH)


# =============================================================================
# Named Entity Fixtures (Deterministic Testing)
# =============================================================================

@pytest.fixture
def contaminated_batch_code() -> str:
    """Batch code for recall trace testing."""
    return "B-2024-RECALL-001"


@pytest.fixture
def megamart_account_code() -> str:
    """MegaMart account code for hub stress testing."""
    return "ACCT-MEGA-001"


@pytest.fixture
def palm_oil_supplier_code() -> str:
    """Single-source Palm Oil supplier for SPOF detection."""
    return "SUP-PALM-MY-001"


@pytest.fixture
def chicago_dc_code() -> str:
    """Bottleneck DC Chicago for centrality testing."""
    return "DC-NAM-CHI-001"


@pytest.fixture
def black_friday_promo_code() -> str:
    """Black Friday 2024 promotion for bullwhip testing."""
    return "PROMO-BF-2024"


@pytest.fixture
def seasonal_lane_code() -> str:
    """Seasonal Shanghai->LA lane for temporal routing."""
    return "LANE-SH-LA-001"


# =============================================================================
# Performance Thresholds (from spec)
# =============================================================================

@pytest.fixture
def performance_thresholds() -> dict:
    """
    Beast mode query performance targets.

    From spec magical-launching-forest.md "Success Criteria".
    """
    return {
        "recall_trace_seconds": 5.0,  # 1 batch -> 47,500 orders in <5s
        "landed_cost_seconds": 2.0,   # Full path aggregation in <2s
        "spof_detection_seconds": 1.0,  # Find all single-source ingredients in <1s
        "osa_root_cause_seconds": 3.0,  # Correlate low-OSA with DC bottlenecks in <3s
    }
