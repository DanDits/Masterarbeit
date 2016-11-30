import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly_chaos_distributions as chaos
from polynomial_chaos.distributions import make_exponential, inverse_gaussian


# Setup configurable parameters
approx_expo = False  # if True approximates exponential distribution, else uniform distribution (in [-1,1])
approx_orders_P = [1, 3, 5, 7, 9]  # orders to approximate and plot


# Load required distributions and hermite polynomial basis
uniform_legendre_chaos = chaos.legendreChaos
uniform = uniform_legendre_chaos.distribution

expo = make_exponential(1.)

gaussian_hermite_chaos = chaos.hermiteChaos
gaussian = gaussian_hermite_chaos.distribution
hermite_basis = gaussian_hermite_chaos.poly_basis.polys
hermite_normalization = gaussian_hermite_chaos.normalization_gamma


def factors_by_order(order):
    return order + 1  # In one dimensional setting (N=1) it holds M=P+1

dimension_M = factors_by_order(max(approx_orders_P))
to_approximate = expo if approx_expo else uniform

# Calculate gpc factors
gpc_factors = [scipy.integrate.quad(lambda y: (to_approximate.inverse_distribution(y)
                                               * hermite_basis(m)(gaussian.inverse_distribution(y))
                                               / hermite_normalization(m)),
                                    0, 1) for m in range(dimension_M)]

gpc_factors = list(factor[0] for factor in gpc_factors)
print(gpc_factors)


# gpc approximation of uniform distribution uses the hermite basis and the factors obtained by projecting the uniform
# distribution orthogonally (using gaussian weighted integral as scalar product)
# on the subspace spanned by the first M hermite basis polynomials.
def gpc_approx(y, factors):
    return sum(factor * hermite_basis(m)(y) for factor, m in zip(factors, range(len(factors))))

bin_centers = None
for approx_order in approx_orders_P:
    # Calculate discrete distribution (=histogram) of the gpc approximation by sampling
    samples_count = 2000  # how many samples of the normal distribution to draw
    # samples = rnd.normal(0., 1., samples_count)  # directly generate normally distributed samples,
    # would require about 10 times more samples

    uniform_samples = np.arange(1. / samples_count, 1. - 1. / samples_count, 1. / samples_count)  # 0 and 1 exclusive!
    samples = np.vectorize(inverse_gaussian)(uniform_samples)
    output = np.vectorize(lambda y: gpc_approx(y, gpc_factors[:factors_by_order(approx_order)]))(samples)

    # to see the stochastic gibbs effect when approximating the uniform distribution more clear, set bins=100
    hist, bin_edges = np.histogram(output, bins='auto', density=True)
    bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    # Plotting of the demo approximation
    plt.plot(bin_centers, hist, label="Hermite approximation of order " + str(approx_order))


plt.title("gPC approximation of a distribution by hermite polynomials")
plt.plot(bin_centers, np.vectorize(to_approximate.weight)(bin_centers),
         label="Exact " + str(to_approximate) + " distribution")
plt.legend()
# plt.axis([-6, 14, 0, 1])
plt.show()
