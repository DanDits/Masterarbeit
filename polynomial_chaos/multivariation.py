# methods for multidimensional polynomials, simple tensor products of 1d polynomials or sparse grid variants
from util.analysis import mul_prod
from functools import lru_cache
from polynomial_chaos.poly_chaos_distributions import PolyChaosDistribution
from polynomial_chaos.poly import PolyBasis
from polynomial_chaos.distributions import Distribution
from util.quadrature.nesting import get_nesting_for_multiple_names
from util.quadrature.helpers import multi_index_bounded_sum


class MultivariatePolyChaosDistribution(PolyChaosDistribution):
    def __init__(self, poly_basis, distribution, normalization_gamma, chaos_list):
        super().__init__(poly_basis, distribution, normalization_gamma)
        self.chaos_list = chaos_list

    def get_distributions(self):
        return [chaos.distribution for chaos in self.chaos_list]

    def get_nodes_and_weights(self):
        return [chaos.poly_basis.nodes_and_weights for chaos in self.chaos_list]

    def get_nesting(self):
        return get_nesting_for_multiple_names([chaos.poly_basis.name for chaos in self.chaos_list])


def poly_basis_multify(basis_list, multi_indices):
    """
    Generates a multivariate polynomial basis from the given list of univariate polynomial basis.
    The returned basis polynomials are again indexed by a simple integer i which corresponds to the multi index
    given by the i-th element of the multi_index_bounded_sum sequence.
    Does not use the full tensor product basis where each component would be bounded by the given bound as this
    would be way too huge for higher dimensions.
    :param basis_list: A list of functions that take one index parameter and return the corresponding basis
    polynomial. The length of this list determines the dimension of the multi index and the new multivariate basis.
    :param multi_indices: A list of multi indices.
    :return: The multivariate PolyBasis.
    """

    @lru_cache(maxsize=None)
    def poly(simple_index):
        multi_index = multi_indices[simple_index]

        def multi_poly(ys):
            return mul_prod(basis.polys(index)(y) for y, index, basis in zip(ys, multi_index, basis_list))
        return multi_poly

    return PolyBasis(",".join(basis.name for basis in basis_list), poly, None)


def chaos_multify(chaos_list, sum_bound):
    basis_list = [chaos.poly_basis for chaos in chaos_list]
    multi_indices = list(multi_index_bounded_sum(len(basis_list), sum_bound))

    def gamma_multify(n):
        prod = 1.
        multi_index = multi_indices[n]
        for dim_index, chaos in zip(multi_index, chaos_list):
            prod *= chaos.normalization_gamma(dim_index)
        return prod

    multi_poly_basis = poly_basis_multify(basis_list, multi_indices)
    multi_distribution = Distribution.multi_from_univariates([chaos.distribution for chaos in chaos_list])
    multi_chaos = MultivariatePolyChaosDistribution(multi_poly_basis, multi_distribution, gamma_multify, chaos_list)
    return multi_chaos

