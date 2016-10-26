import numpy.random as rnd
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import polynomials.distributions as distr
from polynomials.poly import distributions, \
    attribute_polybasis, attribute_weight, attribute_polybasis_normalization_gamma


uniform_weight = distributions["Uniform"][attribute_weight]  # in [-1,1]
expo_weight = distributions["Gamma"][attribute_weight]  # expo is special case of gamma
distr_gaussian = distributions["Gaussian"]
gaussian_weight = distr_gaussian[attribute_weight]
hermite_basis = distr_gaussian[attribute_polybasis]  # Hermite basis
hermite_normalization = distr_gaussian[attribute_polybasis_normalization_gamma]

approx_order_P = 7
dimension_M = approx_order_P + 1  # In one dimensional setting (N=1) it holds M=P+1

gpc_factors = [scipy.integrate.quad(lambda y: distr.inverse_uniform(y) * hermite_basis(m)(distr.inverse_gaussian(y))
                                    / hermite_normalization(m),
                                    0, 1) for m in range(dimension_M)]

gpc_factors = list(factor[0] for factor in gpc_factors)
print(gpc_factors)


# gpc approximation of uniform distribution uses the hermite basis and the factors obtained by projecting the uniform
# distribution orthogonally (using gaussian weighted integral as scalar product)
# on the subspace spanned by the first M hermite basis polynomials.
def gpc_approx(y):
    return sum(factor * hermite_basis(m)(y) for factor, m in zip(gpc_factors, range(dimension_M)))


left_bound = -6
right_bound = 6

samples_count = 20000  # how many samples of the normal distribution to draw
samples = rnd.normal(0., 1., samples_count)
output = sorted(np.vectorize(gpc_approx)(samples))
output_resolution = 0.1  # width of the discrete distribution

t = np.arange(left_bound, right_bound, output_resolution)
output_distribution = np.zeros(shape=t.shape)
for value in output:
    ind = int((value - left_bound) / output_resolution)
    if 0 <= ind < len(output_distribution):
        output_distribution[ind] += 1
    elif ind <= 0:
        output_distribution[0] += 1
    else:
        output_distribution[-1] += 1
output_distribution /= (samples_count * output_resolution)
t += 0.5 * output_resolution  # to adjust that samples are from all the interval [L, L+resolution] set t in the middle

# Plotting of the demo approximation

expo_t = np.arange(output_resolution, right_bound, output_resolution)
#plt.plot(expo_t, np.vectorize(expo_weight)(expo_t))
plt.plot(t, np.vectorize(uniform_weight)(t))
plt.plot(t, output_distribution)
# plt.axis([-6, 14, 0, 1])
plt.show()
