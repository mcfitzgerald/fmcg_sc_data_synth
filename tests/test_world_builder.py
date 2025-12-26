from prism_sim.config.loader import load_manifest
from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.builder import WorldBuilder


def test_world_builder_initialization():
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # Check Nodes
    assert world.get_node("RDC-NAM-CHI") is not None
    assert world.get_node("RDC-NAM-CHI").location == "Chicago, IL"
    assert world.get_node("SUP-SURF-SPEC").type == NodeType.SUPPLIER

    # Check Products
    soap = world.get_product("SKU-SOAP-001")
    assert soap is not None
    assert soap.category == ProductCategory.PERSONAL_WASH
    assert soap.weight_kg == 8.5

    paste = world.get_product("SKU-PASTE-001")
    assert paste.category == ProductCategory.ORAL_CARE

    det = world.get_product("SKU-DET-001")
    assert det.category == ProductCategory.HOME_CARE

    # Check Recipe
    recipe = world.get_recipe("SKU-DET-001")
    assert recipe is not None
    assert recipe.ingredients["ING-SURF-SPEC"] == 0.05
    assert recipe.run_rate_cases_per_hour == 1200

def test_named_entities_exist():
    # Verify the specific "Named Entities" from the Intent are present
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # West RDC check
    assert world.get_node("RDC-NAM-CAL") is not None
