import math
import polynomial_chaos.poly as poly
import polynomial_chaos.distributions as distr
from functools import lru_cache
from util.analysis import rising_factorial
from util.quadrature.rules import CentralizedDiamondQuadrature, FullTensorQuadrature, SparseQuadrature, PseudoSparseDiamond, \
    GeneralPurposeQuadrature
from util.quadrature.nesting import get_nesting_for_name
from util.quadrature.closed_fully_nested import ClosedFullNesting
from util.quadrature.glenshaw_curtis import calculate_transformed_nodes_and_weights


class PolyChaosDistribution:
    def __init__(self, poly_basis, distribution, normalization_gamma):
        self.poly_basis = poly_basis
        self.distribution = distribution
        self.normalization_gamma = normalization_gamma
        self.quadrature_rule = None

    @lru_cache(maxsize=None)
    def normalized_basis(self, degree):
        norm_factor = 1. / math.sqrt(self.normalization_gamma(degree))
        return lambda x: (self.poly_basis.polys(degree)(x) * norm_factor)

    def get_nodes_and_weights(self):
        return [self.poly_basis.nodes_and_weights]

    def get_distributions(self):
        return [self.distribution]

    def get_nesting(self):
        return get_nesting_for_name(self.poly_basis.name)

    def init_quadrature_rule(self, method, param):
        if method == "sparse":
            nesting = self.get_nesting()
            nodes_and_weights_funcs = self.get_nodes_and_weights()
            level = param
            self.quadrature_rule = SparseQuadrature(level, nesting, nodes_and_weights_funcs)
        elif method == "full_tensor":
            orders_1d = param
            nodes_and_weights_funcs = self.get_nodes_and_weights()
            self.quadrature_rule = FullTensorQuadrature(orders_1d, nodes_and_weights_funcs)
        elif method == "centralized":
            sum_bound, even = param
            self.quadrature_rule = CentralizedDiamondQuadrature(self.get_nodes_and_weights(), sum_bound, even)
        elif method == "pseudo_sparse":
            sum_bound, poly_names = param
            self.quadrature_rule = PseudoSparseDiamond(self.get_nodes_and_weights(), poly_names, sum_bound)
        elif method == "sparse_gc":
            distrs = self.get_distributions()
            nesting = ClosedFullNesting()
            level = param
            nodes_and_weights_funcs = [calculate_transformed_nodes_and_weights(d) for d in distrs]
            self.quadrature_rule = SparseQuadrature(level, nesting, nodes_and_weights_funcs)
        elif method == "general_purpose":
            distrs = self.get_distributions()
            self.quadrature_rule = GeneralPurposeQuadrature(distrs)
        else:
            raise ValueError("Unknown quadrature method:" + method)

    def integrate(self, function, function_parameter_is_nodes_matrix=False):
        if self.quadrature_rule is None:
            raise ValueError("Quadrature rule not yet initialized.")
        if function_parameter_is_nodes_matrix:
            return self.quadrature_rule.apply_to_all_nodes_simultaneously(function)
        return self.quadrature_rule.apply(function)


hermiteChaos = PolyChaosDistribution(poly.make_hermite(),
                                     distr.gaussian, lambda n: math.factorial(n))
legendreChaos = PolyChaosDistribution(poly.make_legendre(),
                                      distr.make_uniform(-1, 1),
                                      # this is reduced by factor 1/2 as 1/2 is the density function of the distribution
                                      lambda n: 1 / (2 * n + 1))


# nodes are not symmetric around 0., unstable fast,...
def make_laguerreChaos(alpha):  # alpha > 0
    assert alpha > 0
    return PolyChaosDistribution(poly.make_laguerre(alpha),
                                 distr.make_gamma(alpha, 1),
                                 lambda n: 1)  # we use normalized polynomials here


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


def get_chaos_name_by_distribution(find_distr):
    if find_distr.name == "Gaussian":
        return "Hermite"
    elif find_distr.name == "Uniform":
        return "Legendre"
    elif find_distr.name == "Gamma":
        return "Laguerre"
    elif find_distr.name == "Beta":
        return "Jacobi"
    else:
        raise ValueError("Unknown distribution:", find_distr.name)


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
