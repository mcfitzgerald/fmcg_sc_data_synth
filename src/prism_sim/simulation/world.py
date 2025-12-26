from typing import Dict, Optional
from prism_sim.network.core import Node, Link
from prism_sim.product.core import Product, Recipe


class World:
    """
    The container for the static structure of the supply chain.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.links: Dict[str, Link] = {}
        self.products: Dict[str, Product] = {}
        self.recipes: Dict[str, Recipe] = {}

    def add_node(self, node: Node):
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node

    def add_link(self, link: Link):
        if link.id in self.links:
            raise ValueError(f"Link {link.id} already exists")
        self.links[link.id] = link

    def add_product(self, product: Product):
        if product.id in self.products:
            raise ValueError(f"Product {product.id} already exists")
        self.products[product.id] = product

    def add_recipe(self, recipe: Recipe):
        if recipe.product_id in self.recipes:
            # Assuming one recipe per product for now, or use a composite key later
            raise ValueError(f"Recipe for {recipe.product_id} already exists")
        self.recipes[recipe.product_id] = recipe

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_link(self, link_id: str) -> Optional[Link]:
        return self.links.get(link_id)

    def get_product(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)

    def get_recipe(self, product_id: str) -> Optional[Recipe]:
        return self.recipes.get(product_id)
