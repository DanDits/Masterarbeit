import math
import polynomial_chaos.poly as poly
import polynomial_chaos.distributions as distr
from functools import lru_cache
from util.analysis import rising_factorial


class PolyChaosDistribution:
    def __init__(self, poly_basis, distribution, normalization_gamma):
        self.poly_basis = poly_basis
        self.distribution = distribution
        self.normalization_gamma = normalization_gamma
        self.nodes_and_weights = poly_basis.nodes_and_weights

    @lru_cache(maxsize=None)
    def normalized_basis(self, degree):
        return lambda x: (self.poly_basis.polys(degree)(x) / math.sqrt(self.normalization_gamma(degree)))


hermiteChaos = PolyChaosDistribution(poly.make_hermite(),
                                     distr.gaussian, lambda n: math.factorial(n))
legendreChaos = PolyChaosDistribution(poly.make_legendre(),
                                      distr.make_uniform(-1, 1),
                                      # this is reduced by factor 1/2 as 1/2 is the density function of the distribution
                                      lambda n: 1 / (2 * n + 1))


# TODO convergence very zigzac like when using interpolation by matrix inversion as this uses a method that
# TODO does require some symmetry of the nodes around the middle node which is not fulfilled for laguerre nodes
# nodes are not symmetric around 0., unstable fast,...
def make_laguerreChaos(alpha):  # alpha > 0
    assert alpha > 0
    return PolyChaosDistribution(poly.make_laguerre(alpha),
                                 distr.make_gamma(alpha, 1),
                                 lambda n: 1)


def make_jacobiChaos(alpha, beta):  # alpha, beta > -1
    return PolyChaosDistribution(poly.make_jacobi(alpha, beta),
                                 distr.make_beta(alpha, beta),
                                 lambda n: ((rising_factorial(alpha + 1, n) * rising_factorial(beta + 1, n)
                                            / (math.factorial(n) * (2 * n + alpha + beta + 1)
                                               * rising_factorial(alpha + beta + 2, n - 1))) if n > 0 else 1))
# Other chaos pairs (with discrete distributions); not implemented:
# ("Poisson", "Charlier")
# ("Binomial", "Krawtchouk")
# ("Negative Binomial", "Meixner")
# ("Hypergeometric", "Hahn")


def get_chaos_by_distribution(find_distr):
    if find_distr.name == "Gaussian":
        chaos = hermiteChaos
    elif find_distr.name == "Uniform":
        chaos = legendreChaos
    elif find_distr.name == "Gamma":
        chaos = make_laguerreChaos(find_distr.parameters[0])
    elif find_distr.name == "Beta":
        chaos = make_jacobiChaos(find_distr.parameters[0], find_distr.parameters[1])
    else:
        raise ValueError("Not supported distribution:", find_distr.name)
    return chaos
