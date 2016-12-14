import numpy as np
from itertools import product
from util.analysis import mul_prod
from util.quadrature.helpers import multi_index_bounded_sum


class QuadratureRule:

    def get_nodes_count(self):
        raise NotImplementedError

    def apply(self, function):
        raise NotImplementedError


class FullTensorQuadrature(QuadratureRule):
    def __init__(self, orders_1d, chaos_list):
        assert len(orders_1d) == len(chaos_list)
        # contains [([n11,n12,n13],[w11,w12,w13]), ([n21,n22],[w21,w22])]
        nodes_weights_pairs = [chaos.poly_basis.nodes_and_weights(order) for order, chaos in zip(orders_1d, chaos_list)]
        nodes_list = [nodes for nodes, _ in nodes_weights_pairs]
        weights_list = [weights for _, weights in nodes_weights_pairs]
        # use full tensor product of all dimensions by using 'product'
        self.nodes = np.array([grid_nodes for grid_nodes in product(*nodes_list)])
        self.weights = np.array([mul_prod(grid_weights) for grid_weights in product(*weights_list)])

    def get_nodes_count(self):
        return len(self.weights)

    def apply(self, function):
        return self.weights.dot(np.apply_along_axis(function, 1, self.nodes))


def centralize_index(index, length):
    center = (length - 1) // 2
    return center + index // 2 + 1 if index % 2 == 1 else center - index // 2


class CentralizedDiamondQuadrature(QuadratureRule):
    def __init__(self, chaos_list, sum_bound, even):
        nodes_list, weights_list = [], []
        length = sum_bound + 1  # ignore lengths and use sum_bound+1 for every dimension to ensure we can index!
        nodes_weights_pairs = [chaos.nodes_and_weights(length) for chaos in chaos_list]
        # for every multi index we add one nodes tuple to the list, so we will later have the same
        # amount of nodes/weights as we have basis polynomials.
        if even:
            indices = list(multi_index_bounded_sum(len(chaos_list),
                                                   sum_bound + 1 if sum_bound % 2 == 1 else sum_bound))
        else:
            indices = multi_index_bounded_sum(len(chaos_list), sum_bound)

        for multi_index in indices:
            current_nodes, current_weights = [], []
            for (nodes, weights), index in zip(nodes_weights_pairs, multi_index):
                # here it is important that we have enough nodes to use the multi_index's index!
                centralized = centralize_index(index, length)  # important as nodes are symmetric around the center
                current_nodes.append(nodes[centralized])
                current_weights.append(weights[centralized])
            nodes_list.append(current_nodes)
            weights_list.append(mul_prod(current_weights))
        self.nodes = np.array(nodes_list)
        self.weights = np.array(weights_list)

    def get_nodes_count(self):
        return len(self.weights)

    def apply(self, function):
        return self.weights.dot(np.apply_along_axis(function, 1, self.nodes))


# adapted from code from https://people.sc.fsu.edu/~jburkardt/m_src/sandia_sparse/sandia_sparse.html
class SparseQuadrature(QuadratureRule):

    def __init__(self, level_max: int, nesting, nodes_and_weights_funcs):
        self.dim_num = len(nodes_and_weights_funcs)
        self.level_max = level_max
        self.nesting = nesting
        self.nodes, self.weights = self.nesting.calculate_nodes_and_weights(self.dim_num, self.level_max,
                                                                            nodes_and_weights_funcs)

    def get_nodes_count(self):
        return len(self.weights)

    def apply(self, function):
        return self.weights.dot(np.apply_along_axis(function, 1, self.nodes))
