import numpy.random as rnd
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly_chaos_distributions as chaos
from polynomial_chaos.distributions import make_exponential, inverse_gaussian

uniform_legendre_chaos = chaos.get_chaos_by_distribution("Uniform")
uniform = uniform_legendre_chaos["distribution"]

expo = make_exponential(1.)

gaussian_hermite_chaos = chaos.get_chaos_by_distribution("Gaussian")
gaussian = gaussian_hermite_chaos["distribution"]
hermite_basis = gaussian_hermite_chaos["poly_basis"]
hermite_normalization = gaussian_hermite_chaos[chaos.attribute_normalization_gamma]

# Setup parameters
approx_expo = True
approx_order_P = 7

dimension_M = approx_order_P + 1  # In one dimensional setting (N=1) it holds M=P+1
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
def gpc_approx(y):
    return sum(factor * hermite_basis(m)(y) for factor, m in zip(gpc_factors, range(dimension_M)))


# Calculate discrete distribution of the gpc approximation by sampling

left_bound = -6
right_bound = 6

samples_count = 2000  # how many samples of the normal distribution to draw
# samples = rnd.normal(0., 1., samples_count)  # directly generate normally distributed samples, requires factor 10 more
uniform_samples = np.arange(1. / samples_count, 1. - 1. / samples_count, 1. / samples_count)  # make 0 and 1 exclusive
samples = np.vectorize(inverse_gaussian)(uniform_samples)
output = sorted(np.vectorize(gpc_approx)(samples))
output_resolution = 0.1  # width of the discrete distribution

t = np.arange(left_bound, right_bound, output_resolution)
output_distribution = np.zeros(shape=t.shape)
for value in output:
    ind = np.clip(int((value - left_bound) / output_resolution), 0, len(output_distribution) - 1)
    output_distribution[ind] += 1
output_distribution /= (samples_count * output_resolution)
t += 0.5 * output_resolution  # to adjust that samples are from all the interval [L, L+resolution] set t in the middle

# Plotting of the demo approximation

plt.plot(t, np.vectorize(to_approximate.weight)(t), label="Exact " + to_approximate.name)
plt.plot(t, output_distribution, label="Approximation of order " + str(approx_order_P))
plt.legend()
# plt.axis([-6, 14, 0, 1])
plt.show()
