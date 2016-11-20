import math
import polynomial_chaos.poly as poly
import polynomial_chaos.distributions as distr
from functools import lru_cache

chaos = []


class PolyChaosDistribution:
    def __init__(self, poly_name, poly_basis, distribution, normalization_gamma):
        self.poly_name = poly_name
        self.poly_basis = poly_basis
        self.distribution = distribution
        self.normalization_gamma = normalization_gamma
        chaos.append(self)

    @lru_cache(maxsize=None)
    def normalized_basis(self, degree):
        return lambda x: (self.poly_basis(degree)(x) / math.sqrt(self.normalization_gamma(degree)))


hermiteChaos = PolyChaosDistribution("Hermite", poly.hermite_basis(),
                                     distr.gaussian, lambda n: math.factorial(n))
legendreChaos = PolyChaosDistribution("Legendre", poly.legendre_basis(),
                                      distr.make_uniform(-1, 1),
                                      # this is reduced by factor 1/2 as 1/2 is the density function of the distribution
                                      lambda n: 1 / (2 * n + 1))


# Other chaos pairs, not implemented:
# ("Gamma", "Laguerre", [0, math.inf]),
# ("Beta","Jacobi", [a, b]),
# ("Poisson", "Charlier", support_Naturals),
# ("Binomial", "Krawtchouk", support_NaturalsFinite),
# ("Negative Binomial", "Meixner", support_Naturals),
# ("Hypergeometric", "Hahn", support_NaturalsFinite)


def get_chaos_by_poly(poly_name):
    for curr in chaos:
        if curr.poly_name == poly_name:
            return curr
    raise ValueError("No polynomial chaos distribution for polynomial name", poly_name)


def get_chaos_by_distribution(distribution_name):
    for curr in chaos:
        if curr.distribution.name == distribution_name:
            return curr
    raise ValueError("No polynomial chaos distribution for distribution name", distribution_name)
