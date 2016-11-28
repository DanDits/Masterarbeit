import math
import polynomial_chaos.poly as poly
import polynomial_chaos.distributions as distr
from functools import lru_cache, partial


class PolyChaosDistribution:
    def __init__(self, poly_name, poly_basis, distribution, normalization_gamma, nodes_and_weights):
        self.poly_name = poly_name
        self.poly_basis = poly_basis
        self.distribution = distribution
        self.normalization_gamma = normalization_gamma
        self.nodes_and_weights = nodes_and_weights

    @lru_cache(maxsize=None)
    def normalized_basis(self, degree):
        return lambda x: (self.poly_basis(degree)(x) / math.sqrt(self.normalization_gamma(degree)))


hermiteChaos = PolyChaosDistribution("Hermite", poly.hermite_basis(),
                                     distr.gaussian, lambda n: math.factorial(n),
                                     poly.hermite_nodes_and_weights)
legendreChaos = PolyChaosDistribution("Legendre", poly.legendre_basis(),
                                      distr.make_uniform(-1, 1),
                                      # this is reduced by factor 1/2 as 1/2 is the density function of the distribution
                                      lambda n: 1 / (2 * n + 1),
                                      poly.legendre_nodes_and_weights)


# Notation hint for literature: Pochhammer symbol for falling factorial.. was hard to find!
# Xiu and other authors define this to be rising factorial in contrast to wikipedia
# Falling: alpha_n = alpha*(alpha-1)*...*(alpha-n+1)
def rising_factorial(alpha, n):
    prod = 1
    for i in range(n):
        prod *= alpha + i
    return prod


def make_laguerreChaos(alpha):  # alpha > 0
    return PolyChaosDistribution("Laguerre", poly.laguerre_basis(alpha),
                                 distr.make_gamma(alpha, 1),
                                 lambda n: rising_factorial(alpha, n) / math.factorial(n),
                                 partial(poly.laguerre_nodes_and_weights, alpha=alpha))


def make_jacobiChaos(alpha, beta): # alpha, beta > -1
    return PolyChaosDistribution("Jacobi", poly.jacobi_basis(alpha, beta),
                                 distr.make_beta(alpha, beta),
                                 lambda n: ((rising_factorial(alpha + 1, n) * rising_factorial(beta + 1, n)
                                            / (math.factorial(n) * (2 * n + alpha + beta + 1)
                                               * rising_factorial(alpha + beta + 2, n - 1))) if n > 0 else 1),
                                 partial(poly.jacobi_nodes_and_weights, alpha=alpha, beta=beta))
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


if __name__ == "__main__":
    from scipy.integrate import quad

    # for testing the orthogonality and normalization of the polynomial basis
    alpha, beta = 2.5, -0.5
    chaos = hermiteChaos  # make_jacobiChaos(alpha, beta)
    basis = [chaos.normalized_basis(i) for i in range(20)]

    for b1 in basis:
        row = []
        for b2 in basis:
            result = quad(lambda x: b1(x) * b2(x) * chaos.distribution.weight(x),
                          chaos.distribution.support[0], chaos.distribution.support[1])[0]
            row.append(result)
        print(row)
