# Here we want to show that we do not need to use the full tensor product space of polynomials to approximate
# a function well enough

from polynomial_chaos.poly_chaos_distributions import hermiteChaos, legendreChaos
import numpy as np
from scipy.integrate import nquad
from util.quadrature.helpers import multi_index_bounded_sum
from itertools import product
import matplotlib.pyplot as plt


def to_approx(*ys):
    return np.sin(ys[0] * ys[1]) ** 4  # one zero row and column after a non zero one, good example
    # return np.sin(ys[0] + ys[1]) ** 2  # chess pattern, declining to top right corner, takes longer for x axis

chaoses = [legendreChaos, hermiteChaos]
distrs = [chaos.distribution for chaos in chaoses]
basises = [chaos.normalized_basis for chaos in chaoses]
supports = [distr.support for distr in distrs]

N = len(chaoses)
M = 15


def calculate_factor(polys):
    def integrate_func(*ys):
        weight = 1
        for y, distr, poly in zip(ys, distrs, polys):
            weight *= distr.weight(y) * poly(y)
        return to_approx(*ys) * weight

    factor = nquad(integrate_func, supports)[0]
    return factor

Z = np.zeros((M+1, M+1))
for multi_index in product(range(M + 1), repeat=N):
    print(multi_index)
    curr_polys = [basis(ind) for ind, basis in zip(multi_index, basises)]
    Z[multi_index] = abs(calculate_factor(curr_polys))

print(Z)
plt.figure()
plt.title("Koeffizienten $|\hat{f}_m|$")
plt.xlabel("$m_1$")
plt.ylabel("$m_2$")
plt.xlim((0, M))
plt.ylim((0, M))
plt.pcolor(Z, vmin=Z.min(), vmax=Z.max())
plt.colorbar()
plt.show()
