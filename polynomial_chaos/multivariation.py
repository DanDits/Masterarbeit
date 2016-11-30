# methods for multidimensional polynomials, simple tensor products of 1d polynomials or sparse grid variants

# TODO how to get stochastic polynomial basis? how to build tensor product of 1d interpolations?
# TODO use sparse grids for higher dimensional case (n>=5)
from itertools import combinations, product
from math import factorial

import math

from util.analysis import mul_prod
from functools import lru_cache
from polynomial_chaos.poly_chaos_distributions import PolyChaosDistribution
from polynomial_chaos.poly import PolyBasis

def multi_index_bounded_sum_length(dimension, sum_bound):
    """
    See multi_index_bounded_sum. This returns the length of the generator's sequence
    which is (dimension+sum_bound)!/(dimension!sum_bound!)
    :param dimension: The dimension of the multi index.
    :param sum_bound: The bound for the sum.
    :return: The sequence length
    """
    return factorial(dimension + sum_bound) // (factorial(dimension) * factorial(sum_bound))


def multi_index_bounded_sum(dimension, sum_bound):
    """
    An iterable generator that returns the multi index i=(i_1,i_2,i_3,...i_dimension)
    with the i_ being greater than or equal to zero and the sum over all i_ is smaller than or equal to
    sum_bound. Indices are sorted by sum and for equal summed indices lexicographically.
    :param dimension: Dimension = length of the index.
    :param sum_bound: The bound of the sum.
    :return: A generator for the indices.
    """
    d, P = dimension, sum_bound
    # compute every possible d-length list whose sum is smaller than or equal to P with summands >= 0
    # it is easier to understand if we imagine we want sums that are smaller than or equal to P+d with summands >=1
    # (and at the end subtract 1 from each position) that there are (P+d over P)=(P+d)!/(P!d!)
    for i in range(d, d + P + 1):
        # compute d-length lists whose sum is exactly equal to i
        current_to_order = []
        for comb in combinations(range(i - 1), d - 1):
            # imagine i being written as i=1_1_1_..._1 with i times a 1 and (i-1) blanks
            # replace d-1 blanks (_) with a comma, the rest with a plus to get all possibilities
            # therefore get all possible i length combinations
            comb += (i - 1,)  # add a virtual comma after the last 1 to simplify the next loop
            last_pos = -1
            index = []
            for comma_pos in comb:
                index.append(comma_pos - last_pos - 1)  # each reduced by 1 as we do not want [1,1,2] but [0,0,1]
                last_pos = comma_pos
            current_to_order.append(index)
        for ind in sorted(current_to_order):
            yield ind


def poly_basis_multify(basis_list, sum_bound, nodes_and_weights, multi_indices=None):
    """
    Generates a multivariate polynomial basis from the given list of univariate polynomial basis.
    The returned basis polynomials are again indexed by a simple integer i which corresponds to the multi index
    given by the i-th element of the multi_index_bounded_sum sequence.
    Does not use the full tensor product basis where each component would be bounded by the given bound as this
    would be way too huge for higher dimensions.
    :param basis_list: A list of functions that take one index parameter and return the corresponding basis
    polynomial. The length of this list determines the dimension of the multi index and the new multivariate basis.
    :param sum_bound: The multi index sum is bounded by this value.
    :param nodes_and_weights The nodes and weights
    :param multi_indices: (Optional) A list of multi indices if already computed, else this will be computed internally.
    :return: The multivariate PolyBasis.
    """
    if multi_indices is None:
        multi_indices = list(multi_index_bounded_sum(len(basis_list), sum_bound))

    @lru_cache(maxsize=None)
    def poly(simple_index):
        multi_index = multi_indices[simple_index]

        def multi_poly(ys):
            return mul_prod(basis.polys(index)(y) for y, index, basis in zip(ys, multi_index, basis_list))
        return multi_poly

    basis = PolyBasis(",".join(basis.name for basis in basis_list),
                      poly,
                      nodes_and_weights)
    return basis


def center_iterator(data):
    total = len(data)
    direction = 1
    i = (total - 1) // 2
    while 0 <= i < total:
        yield data[i]
        i += direction
        direction = int(-math.copysign(abs(direction) + 1, direction))


def centralize_index(index, length):
    center = (length - 1) // 2
    return center + index // 2 + 1 if index % 2 == 1 else center - index // 2


def chaos_multify(chaos_list, sum_bound):
    basis_list = [chaos.poly_basis for chaos in chaos_list]
    multi_indices = list(multi_index_bounded_sum(len(basis_list), sum_bound))

    def nodes_and_weights_multify(lengths, use_full_tensor_product=False):
        if use_full_tensor_product:
            # contains [([n11,n12,n13],[w11,w12,w13]), ([n21,n22],[w21,w22])]
            assert len(lengths) == len(chaos_list)
            nodes_weights_pairs = [chaos.nodes_and_weights(length) for length, chaos in zip(lengths, chaos_list)]
            nodes_list = [nodes for nodes, _ in nodes_weights_pairs]
            weights_list = [weights for _, weights in nodes_weights_pairs]
            # use full tensor product of all dimensions by using 'product'
            nodes_list = [grid_nodes for grid_nodes in product(*nodes_list)]
            weights_list = [grid_weights for grid_weights in product(*weights_list)]
        else:
            nodes_list, weights_list = [], []
            length = sum_bound + 1  # ignore lengths and use sum_bound+1 for every dimension to ensure we can index!
            odes_weights_pairs = [chaos.nodes_and_weights(length) for chaos in chaos_list]
            # for every multi index we add one nodes tuple to the list, so we will later have the same
            # amount of nodes/weights as we have basis polynomials.
            for multi_index in multi_indices:
                current_nodes, current_weights = [], []
                for (nodes, weights), index in zip(odes_weights_pairs, multi_index):
                    # here it is important that we have enough nodes to use the multi_index's index!
                    centralized = centralize_index(index, length)  # important as nodes are symmetric around the center
                    current_nodes.append(nodes[centralized])
                    current_weights.append(weights[centralized])
                nodes_list.append(current_nodes)
                weights_list.append(current_weights)
        return nodes_list, weights_list

    def gamma_multify(n):
        prod = 1.
        multi_index = multi_indices[n]
        for dim_index, chaos in zip(multi_index, chaos_list):
            prod *= chaos.normalization_gamma(dim_index)
        return prod

    multi_chaos = PolyChaosDistribution(poly_basis_multify(basis_list, sum_bound, nodes_and_weights_multify,
                                                           multi_indices),
                                        [chaos.distribution for chaos in chaos_list],
                                        gamma_multify)
    return multi_chaos

if __name__ == "__main__":
    count = 0
    dim, bound = 3, 2
    for current in multi_index_bounded_sum(dim, bound):
        print(current)
        count += 1
    assert count == multi_index_bounded_sum_length(dim, bound)
