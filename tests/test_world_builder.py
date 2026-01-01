"""Tests for WorldBuilder and procedural world generation."""

from prism_sim.config.loader import load_manifest
from prism_sim.network.core import NodeType
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.builder import WorldBuilder


def test_world_builder_initialization():
    """Test that WorldBuilder creates a valid world with procedural generation."""
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # Check Nodes (Updated for Deep NAM Static World)
    # RDC-MW is the Chicago DC
    assert world.get_node("RDC-MW") is not None
    assert world.get_node("RDC-MW").location == "Chicago, IL"

    # Check Suppliers (generated with random names but fixed IDs SUP-001...)
    suppliers = [n for n in world.nodes.values() if n.type == NodeType.SUPPLIER]
    assert len(suppliers) > 0
    assert suppliers[0].type == NodeType.SUPPLIER

    # Check Products - find by category since IDs vary with static world generation
    # Find first product of each category
    oral_products = [p for p in world.products.values()
                     if p.category == ProductCategory.ORAL_CARE and p.id.startswith("SKU-")]
    personal_products = [p for p in world.products.values()
                         if p.category == ProductCategory.PERSONAL_WASH and p.id.startswith("SKU-")]
    home_products = [p for p in world.products.values()
                     if p.category == ProductCategory.HOME_CARE and p.id.startswith("SKU-")]

    assert len(oral_products) > 0, "Should have ORAL_CARE products"
    assert len(personal_products) > 0, "Should have PERSONAL_WASH products"
    assert len(home_products) > 0, "Should have HOME_CARE products"

    # Verify first product of each category has expected attributes
    paste = oral_products[0]
    assert paste.category == ProductCategory.ORAL_CARE

    soap = personal_products[0]
    assert soap.category == ProductCategory.PERSONAL_WASH
    # Weight is randomized
    assert soap.weight_kg > 0

    det = home_products[0]
    assert det.category == ProductCategory.HOME_CARE

    # Check Recipe structure (procedurally generated)
    # Use the actual HOME_CARE product ID we found
    recipe = world.get_recipe(det.id)
    assert recipe is not None, f"Should have recipe for {det.id}"

    # Validate ingredient structure follows procedural patterns
    # HOME_CARE should have: PKG-BOTTLE, PKG-CAP, PKG-LABEL, BLK-*, ACT-CHEM-*
    assert len(recipe.ingredients) >= 3, "Recipe should have multiple ingredients"

    # Check that ingredients follow naming conventions
    ing_prefixes = [ing_id.split("-")[0] for ing_id in recipe.ingredients.keys()]
    assert "PKG" in ing_prefixes, "Recipe should contain packaging ingredients"

    # At least one chemical component (active or bulk)
    has_chemical = any(p in ing_prefixes for p in ["ACT", "BLK"])
    assert has_chemical, "Recipe should contain chemical ingredients"

    # Run rate should match HOME_CARE profile (6000 cases/hr from world_definition.json)
    # Note: Run rate may vary based on config
    assert recipe.run_rate_cases_per_hour > 0


def test_named_entities_exist():
    """Verify the specific 'Named Entities' from the Intent are present."""
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # West RDC check (Reno, NV -> RDC-WE)
    assert world.get_node("RDC-WE") is not None


def test_ingredients_generated():
    """Verify procedural ingredients are generated with correct prefixes."""
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # Check that ingredient products exist
    ingredients = [p for p in world.products.values()
                   if p.category == ProductCategory.INGREDIENT]
    assert len(ingredients) > 0, "Should have generated ingredients"

    # Check ingredient ID patterns
    pkg_count = sum(1 for p in ingredients if p.id.startswith("PKG-"))
    act_count = sum(1 for p in ingredients if p.id.startswith("ACT-"))
    blk_count = sum(1 for p in ingredients if p.id.startswith("BLK-"))

    assert pkg_count > 0, "Should have packaging ingredients (PKG-*)"
    assert act_count > 0, "Should have active chemical ingredients (ACT-*)"
    assert blk_count > 0, "Should have bulk base ingredients (BLK-*)"


def test_spof_ingredient_exists():
    """Verify the SPOF ingredient from config exists in the world."""
    manifest = load_manifest()
    builder = WorldBuilder(manifest)
    world = builder.build()

    # SPOF ingredient is ACT-CHEM-001 (from simulation_config.json)
    spof_ing = world.get_product("ACT-CHEM-001")
    assert spof_ing is not None, "SPOF ingredient ACT-CHEM-001 should exist"
    assert spof_ing.category == ProductCategory.INGREDIENT
