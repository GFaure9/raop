import numpy as np

from typing import Callable, Tuple, List
# from anytree import Node, RenderTree, PostOrderIter

# ########### old method using anytree Node class (significantly longer computation time) ###########
# (this method is for European options only)
# def create_binary_tree(
#         n_layers: int,
#         u: float,
#         d: float,
#         init_underlying_price: float,
# ) -> Tuple[Node, List[Node]]:
#     node_num = 0
#
#     root = Node(node_num, S=init_underlying_price)
#     current_layer = [root]
#
#     for _ in range(n_layers - 1):
#         new_layer = []
#         for node in current_layer:
#             s_up = node.S * u
#             s_down = node.S * d
#             node.up = Node(node_num + 1, parent=node, S=s_up)
#             node.down = Node(node_num + 2, parent=node, S=s_down)
#             new_layer.extend([node.up, node.down])
#             node_num += 2
#
#         current_layer = new_layer
#
#     last_layer = current_layer
#
#     return root, last_layer
#
#
# def compute_last_option_prices(
#         last_layer_nodes: List[Node],
#         payoff_func: Callable,
#         k: float,
#         option_type: str,
# ):
#     for node in last_layer_nodes:
#         node.option_price = payoff_func(s=node.S, k=k, option_type=option_type)
#
#
# def compute_previous_option_prices(root: Node, p: float, discount_factor: float):
#     for node in PostOrderIter(root):
#         if len(node.children) > 0:
#             c_up, c_down = node.up.option_price, node.down.option_price
#             node.option_price = discount_factor * (p * c_up + (1 - p) * c_down)
#
#
# def print_tree(root):
#     for pre, _, node in RenderTree(root):
#         print(f"{pre}{node.name}[{node.S}]")
# ###############################################################################################


def create_binary_tree(
        n_layers: int,
        u: float,
        d: float,
        init_underlying_price: float,
) -> List[np.ndarray]:

    current_layer = np.array([init_underlying_price])

    tree = [current_layer]

    for k in range(n_layers):
        u_d = np.tile([u, d], (2 ** k, 1))
        new_layer = np.array([current_layer]).T * u_d
        new_layer = np.concatenate(new_layer)
        current_layer = new_layer

        tree.append(new_layer)

    return tree


def compute_last_option_prices(
        tree: np.ndarray,
        payoff_func: Callable,
        k: float,
        option_type: str,
        american: bool,
) -> List[np.ndarray]:
    if american:
        payoffs_tree = [
            payoff_func(s=s, k=k, option_type=option_type) for s in tree
        ]
        return payoffs_tree
    else:
        return [payoff_func(s=tree[-1], k=k, option_type=option_type)]


def compute_previous_option_prices(
        payoffs_tree: np.ndarray,
        p: float,
        discount_factor: float,
        american: bool,
) -> float:
    probabilities = np.array([p, 1-p])

    current_option_prices = payoffs_tree[-1]

    k = len(payoffs_tree) - 2
    while len(current_option_prices) > 1:
        option_prices_by_pairs = np.reshape(current_option_prices, (-1, 2))
        previous_option_prices = discount_factor * option_prices_by_pairs.dot(probabilities)
        if american:
            previous_option_prices = np.amax([payoffs_tree[k], previous_option_prices], axis=0)
        current_option_prices = previous_option_prices
        k -= 1

    return current_option_prices[0]
