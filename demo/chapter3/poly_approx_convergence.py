import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly_chaos_distributions as pcd
from scipy.integrate import quad
from util.analysis import error_l2_relative as error
from itertools import cycle

chaos = pcd.legendreChaos  # pcd.make_jacobiChaos(-0.5, -0.6)  # hermite chaos doesn't work
weight = np.vectorize(chaos.distribution.weight)
functions = [("$\sin(\pi x)$", list(range(1, 19, 2)), lambda x: np.sin(np.pi * x)),
             ("$\sin(\pi x)^2$", list(range(1, 19, 2)), lambda x: np.sin(np.pi * x) ** 2),
             ("$|x-1/4|$", list(range(1, 40, 2)), lambda x: np.abs(x - 1/4)),
             ("$|\sin(\pi x)|^3$", list(range(1, 40, 2)), lambda x: np.abs(np.sin(np.pi * x))**3)]
N = 10
lbound, rbound = chaos.distribution.support

if np.isinf(lbound):
    lbound = -20
if np.isinf(rbound):
    rbound = 20
plt.figure()
x_data = np.arange(lbound, rbound, 0.001)

errors_over_N = []
for name, Ns, func in functions:
    curr_errors = []
    # calculate fourier coefficients
    total_basis = [chaos.normalized_basis(n) for n in range(max(Ns) + 1)]
    total_coeffs = [quad(lambda x: func(x) * poly(x) * weight(x),
                    *chaos.distribution.support)[0] for poly in total_basis]
    for N in Ns:
        basis = total_basis[:N+1]
        coeffs = total_coeffs[:N+1]
        reconstructed = lambda x: sum(coeff * poly(x) for coeff, poly in zip(coeffs, basis))
        curr_errors.append(error(reconstructed(x_data), func(x_data)))
    errors_over_N.append((name, Ns, curr_errors))
marker = cycle(('D', '*', 'o', '>', '<'))
for name, Ns, errors in errors_over_N:
    plt.plot(Ns, errors, "-" + next(marker), label=name)
rel_size = 'x-large'
plt.title("Spektrale Konvergenz der Bestapproximation verschiedener Funktionen mit {} Polynomen".format(chaos.poly_basis.name),
          fontsize=rel_size)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1e-13, plt.ylim()[1]))
plt.xlabel('Maximaler Polynom Grad', fontsize=rel_size)
plt.ylabel('Relativer Fehler in diskreter L2-Norm', fontsize=rel_size)
plt.legend(fontsize=rel_size)
plt.show()

