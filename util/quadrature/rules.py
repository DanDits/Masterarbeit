import numpy as np
from itertools import product
from util.analysis import mul_prod
from util.quadrature.helpers import multi_index_bounded_sum
from scipy.integrate import nquad


class QuadratureRule:
    def apply_to_all_nodes_simultaneously(self, function):
        return self.get_weights().dot(function(self.get_nodes()))

    def apply(self, function):
        return self.get_weights().dot(np.apply_along_axis(function, 1, self.get_nodes()))

    def supports_simultaneous_application(self):
        return True

    def get_nodes(self):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def get_nodes_count(self):
        raise NotImplementedError


class GeneralPurposeQuadrature(QuadratureRule):
    def __init__(self, distributions):
        self.supports = [distr.support for distr in distributions]

    def apply_to_all_nodes_simultaneously(self, function):
        raise ValueError("Cannot be used with general purpose integrator as it has no fixed nodes.")

    def supports_simultaneous_application(self):
        return False

    def apply(self, function):
        return nquad(function, self.supports)[0]

    def get_nodes(self):
        pass

    def get_weights(self):
        pass

    def get_nodes_count(self):
        pass


class FullTensorQuadrature(QuadratureRule):
    def __init__(self, orders_1d, nodes_and_weights_funcs):
        assert len(orders_1d) == len(nodes_and_weights_funcs)
        # contains [([n11,n12,n13],[w11,w12,w13]), ([n21,n22],[w21,w22])]
        nodes_weights_pairs = [nodes_and_weights(order) for order, nodes_and_weights in zip(orders_1d,
                                                                                            nodes_and_weights_funcs)]
        nodes_list = [nodes for nodes, _ in nodes_weights_pairs]
        weights_list = [weights for _, weights in nodes_weights_pairs]
        # use full tensor product of all dimensions by using 'product'
        self.nodes = np.array([grid_nodes for grid_nodes in product(*nodes_list)])
        self.weights = np.array([mul_prod(grid_weights) for grid_weights in product(*weights_list)])

    def get_nodes(self):
        return self.nodes

    def get_nodes_count(self):
        return len(self.weights)

    def get_weights(self):
        return self.weights


def centralize_index(index, length):
    center = (length - 1) // 2
    return center + index // 2 + 1 if index % 2 == 1 else center - index // 2


# this is no real quadrature formula, it just behaves as one, its main purpose is to use its nodes
# as they offer great performance for multivariate matrix inversion collocation
class CentralizedDiamondQuadrature(QuadratureRule):
    def __init__(self, nodes_and_weights_funcs, sum_bound, even):
        nodes_list, weights_list = [], []
        length = sum_bound + 1  # ignore lengths and use sum_bound+1 for every dimension to ensure we can index!
        nodes_weights_pairs = [nodes_and_weights_func(length) for nodes_and_weights_func in nodes_and_weights_funcs]
        # for every multi index we add one nodes tuple to the list, so we will later have the same
        # amount of nodes/weights as we have basis polynomials.
        if even:
            indices = list(multi_index_bounded_sum(len(nodes_and_weights_funcs),
                                                   sum_bound + 1 if sum_bound % 2 == 1 else sum_bound))
        else:
            indices = multi_index_bounded_sum(len(nodes_and_weights_funcs), sum_bound)

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

    def get_nodes(self):
        return self.nodes

    def get_nodes_count(self):
        return len(self.weights)

    def get_weights(self):
        return self.weights


# Not a quadrature rule, this is just a way to associate the subset of the full tensor product that is restricted
# by the sum_bound with a set of nodes that does a good job for the matrix inversion approach, similiar to
# CentralizedDiamond but treating Laguerre and Jacobi differently as their nodes are not symmetric around 0
# one can think of this approach using t
class PseudoSparseDiamond(QuadratureRule):
    def __init__(self, nodes_and_weights_funcs, poly_names, sum_bound):
        nodes_list, weights_list = [], []
        length = sum_bound + 1
        nodes_weights_pairs = [nodes_and_weights_func(length) for nodes_and_weights_func in nodes_and_weights_funcs]
        # for every multi index we add one nodes tuple to the list, so we will later have the same
        # amount of nodes/weights as we have basis polynomials with the same sum_bound
        indices = multi_index_bounded_sum(len(nodes_and_weights_funcs), sum_bound)

        for multi_index in indices:
            current_nodes, current_weights = [], []
            for (nodes, weights), index, poly_name in zip(nodes_weights_pairs, multi_index, poly_names):
                # here it is important that we have enough nodes to use the multi_index's index!
                if poly_name in ["Hermite", "Legendre"]:
                    index = centralize_index(index, length)  # important as nodes are symmetric around the center
                elif poly_name == "Jacobi":
                    index = length - index - 1  # TODO maybe depending on if alpha > beta ?
                elif poly_name == "Laguerre":
                    pass  # do not change index
                else:
                    raise ValueError("Not supported polynomials:", poly_name)
                current_nodes.append(nodes[index])
                current_weights.append(weights[index])
            nodes_list.append(current_nodes)
            weights_list.append(mul_prod(current_weights))
        self.nodes = np.array(nodes_list)
        self.weights = np.array(weights_list)

    def get_nodes(self):
        return self.nodes

    def get_nodes_count(self):
        return len(self.weights)

    def get_weights(self):
        return self.weights


# implementation for weakly and fully nested grids inspired and adapted
# from code from https://people.sc.fsu.edu/~jburkardt/m_src/sandia_sparse/sandia_sparse.html
class SparseQuadrature(QuadratureRule):
    def __init__(self, level_max: int, nesting, nodes_and_weights_funcs):
        self.dim_num = len(nodes_and_weights_funcs)
        self.level_max = level_max
        self.nesting = nesting
        self.nodes, self.weights = self.nesting.calculate_nodes_and_weights(self.dim_num, self.level_max,
                                                                            nodes_and_weights_funcs)

    def get_nodes(self):
        return self.nodes

    def get_nodes_count(self):
        return len(self.weights)

    def get_weights(self):
        return self.weights
