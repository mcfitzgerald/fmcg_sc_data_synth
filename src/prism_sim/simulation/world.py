from prism_sim.network.core import Link, Node
from prism_sim.product.core import Product, Recipe


class World:
    """
    The container for the static structure of the supply chain.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.links: dict[str, Link] = {}
        self.products: dict[str, Product] = {}
        self.recipes: dict[str, Recipe] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node

    def add_link(self, link: Link) -> None:
        if link.id in self.links:
            raise ValueError(f"Link {link.id} already exists")
        self.links[link.id] = link

    def add_product(self, product: Product) -> None:
        if product.id in self.products:
            raise ValueError(f"Product {product.id} already exists")
        self.products[product.id] = product

    def add_recipe(self, recipe: Recipe) -> None:
        if recipe.product_id in self.recipes:
            # Assuming one recipe per product for now, or use a composite key later
            raise ValueError(f"Recipe for {recipe.product_id} already exists")
        self.recipes[recipe.product_id] = recipe

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_link(self, link_id: str) -> Link | None:
        return self.links.get(link_id)

    def get_product(self, product_id: str) -> Product | None:
        return self.products.get(product_id)

    def get_recipe(self, product_id: str) -> Recipe | None:
        return self.recipes.get(product_id)
