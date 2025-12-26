import pytest

from prism_sim.network.core import Link, Node, NodeType
from prism_sim.product.core import Product, ProductCategory


def test_node_creation():
    node = Node(
        id="RDC-NAM-NE",
        name="Northeast RDC",
        type=NodeType.DC,
        location="Pennsylvania",
        storage_capacity=50000
    )
    assert node.id == "RDC-NAM-NE"
    assert node.type == NodeType.DC
    assert node.storage_capacity == 50000

def test_link_creation():
    link = Link(
        id="LANE-PA-NY",
        source_id="RDC-NAM-NE",
        target_id="STORE-NY-001",
        distance_km=150,
        lead_time_days=1
    )
    assert link.source_id == "RDC-NAM-NE"
    assert link.distance_km == 150

def test_product_physics():
    # Test "The Brick" (Soap)
    soap = Product(
        id="SKU-SOAP-001",
        name="Prism Bar Soap",
        category=ProductCategory.PERSONAL_WASH,
        weight_kg=8.5,
        length_cm=30,
        width_cm=20,
        height_cm=15,
        cases_per_pallet=80
    )

    assert soap.category == ProductCategory.PERSONAL_WASH
    assert soap.weight_kg == 8.5
    assert soap.volume_m3 == pytest.approx(0.009, rel=1e-3)
