import math
import polynomial_chaos.poly as poly
import polynomial_chaos.distributions as distr
from functools import lru_cache, partial

chaos = []


class PolyChaosDistribution:
    def __init__(self, poly_name, poly_basis, distribution, normalization_gamma, nodes):
        self.poly_name = poly_name
        self.poly_basis = poly_basis
        self.distribution = distribution
        self.normalization_gamma = normalization_gamma
        self.interpolation_nodes = nodes
        chaos.append(self)

    @lru_cache(maxsize=None)
    def normalized_basis(self, degree):
        return lambda x: (self.poly_basis(degree)(x) / math.sqrt(self.normalization_gamma(degree)))


hermiteChaos = PolyChaosDistribution("Hermite", poly.hermite_basis(),
                                     distr.gaussian, lambda n: math.factorial(n),
                                     poly.hermite_nodes)
legendreChaos = PolyChaosDistribution("Legendre", poly.legendre_basis(),
                                      distr.make_uniform(-1, 1),
                                      # this is reduced by factor 1/2 as 1/2 is the density function of the distribution
                                      lambda n: 1 / (2 * n + 1),
                                      poly.legendre_nodes)


# Notation hint for literature: Pochhammer symbol for falling factorial.. was hard to find!
# Xiu seems to use falling instead of rising?
# Falling: alpha_n = alpha*(alpha-1)*...*(alpha-n+1)
def rising_factorial(alpha, n):
    prod = 1
    for i in range(n):
        prod *= alpha + i
    return prod


def make_laguerreChaos(alpha):
    return PolyChaosDistribution("Laguerre", poly.laguerre_basis(alpha),
                                 distr.make_gamma(alpha, 1),
                                 lambda n: rising_factorial(alpha, n) / math.factorial(n),
                                 partial(poly.laguerre_nodes, alpha=alpha))


# Other chaos pairs, not implemented:
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
