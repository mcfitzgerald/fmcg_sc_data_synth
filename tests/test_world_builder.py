from prism_sim.config.loader import load_manifest
from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.builder import WorldBuilder


def test_world_builder_initialization():
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # Check Nodes (Updated for Deep NAM Static World)
    # RDC-MW is the Chicago DC
    assert world.get_node("RDC-MW") is not None
    assert world.get_node("RDC-MW").location == "Chicago, IL"
    
    # Check Suppliers (generated with random names but fixed IDs SUP-001...)
    # Actually NetworkGenerator uses SUP-{i+1:03d}
    # Let's just check type of a known supplier if possible, or search for one
    suppliers = [n for n in world.nodes.values() if n.type == NodeType.SUPPLIER]
    assert len(suppliers) > 0
    assert suppliers[0].type == NodeType.SUPPLIER

    # Check Products (Updated for ProductGenerator)
    # i=0 -> ORAL -> SKU-ORAL-001
    # i=1 -> PERSONAL -> SKU-PERSONAL-002
    # i=2 -> HOME -> SKU-HOME-003
    
    soap = world.get_product("SKU-PERSONAL-002")
    assert soap is not None
    assert soap.category == ProductCategory.PERSONAL_WASH
    # Weight is randomized around 8.5
    assert 5.0 < soap.weight_kg < 12.0

    paste = world.get_product("SKU-ORAL-001")
    assert paste is not None
    assert paste.category == ProductCategory.ORAL_CARE

    det = world.get_product("SKU-HOME-003")
    assert det is not None
    assert det.category == ProductCategory.HOME_CARE

    # Check Recipe
    # Recipes are generated for all FG.
    recipe = world.get_recipe("SKU-HOME-003")
    assert recipe is not None
    # Check for ingredients (keys exist)
    assert "ING-SURF-SPEC" in recipe.ingredients
    # Run rate is 1200 for HOME_CARE in generator
    assert recipe.run_rate_cases_per_hour == 1200


def test_named_entities_exist():
    # Verify the specific "Named Entities" from the Intent are present
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # West RDC check (Reno, NV -> RDC-WE)
    assert world.get_node("RDC-WE") is not None
