import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly_chaos_distributions as chaos
from polynomial_chaos.distributions import make_exponential, inverse_gaussian


# Setup configurable parameters
approx_expo = False  # if True approximates exponential distribution, else uniform distribution (in [-1,1])
approx_orders_P = [1, 3, 5, 7, 9]  # orders to approximate and plot


# Load required distributions and hermite polynomial basis
uniform_legendre_chaos = chaos.get_chaos_by_distribution("Uniform")
uniform = uniform_legendre_chaos.distribution

expo = make_exponential(1.)

gaussian_hermite_chaos = chaos.get_chaos_by_distribution("Gaussian")
gaussian = gaussian_hermite_chaos.distribution
hermite_basis = gaussian_hermite_chaos.poly_basis
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

# Relevant for visualizing results and data
left_bound = -6
right_bound = 6
output_resolution = 0.1  # width of the discrete distribution

t = np.arange(left_bound, right_bound, output_resolution)
t += 0.5 * output_resolution  # to adjust that samples are from all the interval [L, L+resolution] set t in the middle
for approx_order in approx_orders_P:
    # Calculate discrete distribution (=histogram) of the gpc approximation by sampling
    samples_count = 2000  # how many samples of the normal distribution to draw
    # samples = rnd.normal(0., 1., samples_count)  # directly generate normally distributed samples,
    # would require about 10 times more samples

    uniform_samples = np.arange(1. / samples_count, 1. - 1. / samples_count, 1. / samples_count)  # 0 and 1 exclusive!
    samples = np.vectorize(inverse_gaussian)(uniform_samples)
    output = sorted(np.vectorize(lambda y: gpc_approx(y, gpc_factors[:factors_by_order(approx_order)]))(samples))

    output_distribution = np.zeros(shape=t.shape)
    for value in output:
        ind = np.clip([int((value - left_bound) / output_resolution)], 0, len(output_distribution) - 1)
        output_distribution[ind] += 1
    output_distribution /= (samples_count * output_resolution)

    # Plotting of the demo approximation
    plt.plot(t, output_distribution, label="Hermite approximation of order " + str(approx_order))


plt.title("gPC approximation of a distribution by hermite polynomials")
plt.plot(t, np.vectorize(to_approximate.weight)(t), label="Exact " + str(to_approximate) + " distribution")
plt.legend()
# plt.axis([-6, 14, 0, 1])
plt.show()
