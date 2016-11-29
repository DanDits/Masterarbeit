# methods for multidimensional polynomials, simple tensor products of 1d polynomials or sparse grid variants

# TODO how to get stochastic polynomial basis? how to build tensor product of 1d interpolations?
# TODO use sparse grids for higher dimensional case (n>=5)
from itertools import combinations, product
from math import factorial
from util.analysis import mul_prod
from functools import lru_cache
from polynomial_chaos.poly_chaos_distributions import PolyChaosDistribution


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


def poly_basis_multify(basis_list, sum_bound, multi_indices=None):
    """
    Generates a multivariate polynomial basis from the given list of univariate polynomial basis.
    The returned basis is again indexed by a simple integer i which corresponds to the multi index
    given by the i-th element of the multi_index_bounded_sum sequence.
    Does not use the full tensor product basis where each component would be bounded by the given bound as this
    would be way too huge for higher dimensions.
    :param basis_list: A list of functions that take one index parameter and return the corresponding basis
    polynomial. The length of this list determines the dimension of the multi index and the new multivariate basis.
    :param sum_bound: The multi index sum is bounded by this value.
    :param multi_indices: (Optional) A list of multi indices if already computed, else this will be computed internally.
    :return: The basis which is again indexed by a single integer.
    """
    if multi_indices is None:
        multi_indices = list(multi_index_bounded_sum(len(basis_list), sum_bound))

    @lru_cache(maxsize=None)
    def poly(simple_index):
        multi_index = multi_indices[simple_index]

        def multi_poly(ys):
            return mul_prod(basis(index)(y) for y, index, basis in zip(ys, multi_index, basis_list))
        return multi_poly
    return poly


def chaos_multify(chaos_list, sum_bound):
    def nodes_and_weights_multify(lengths):
        # contains [([n11,n12,n13],[w11,w12,w13]), ([n21,n22],[w21,w22])]
        assert len(lengths) == len(chaos_list)
        nodes_weights_pairs = [chaos.nodes_and_weights(length) for length, chaos in zip(lengths, chaos_list)]
        nodes_list = [nodes for nodes, _ in nodes_weights_pairs]
        weights_list = [weights for _, weights in nodes_weights_pairs]
        # use full tensor product of all dimensions by using 'product'
        nodes_list = [grid_nodes for grid_nodes in product(*nodes_list)]
        weights_list = [grid_weights for grid_weights in product(*weights_list)]
        return nodes_list, weights_list

    basis_list = [chaos.poly_basis for chaos in chaos_list]
    multi_indices = list(multi_index_bounded_sum(len(basis_list), sum_bound))

    def gamma_multify(n):
        prod = 1.
        multi_index = multi_indices[n]
        for dim_index, chaos in zip(multi_index, chaos_list):
            prod *= chaos.normalization_gamma(dim_index)
        return prod

    multi_chaos = PolyChaosDistribution(",".join(chaos.poly_name for chaos in chaos_list),
                                        poly_basis_multify(basis_list, sum_bound, multi_indices),
                                        [chaos.distribution for chaos in chaos_list],
                                        gamma_multify,
                                        nodes_and_weights_multify)
    return multi_chaos

if __name__ == "__main__":
    count = 0
    dim, bound = 3, 2
    for current in multi_index_bounded_sum(dim, bound):
        print(current)
        count += 1
    assert count == multi_index_bounded_sum_length(dim, bound)
