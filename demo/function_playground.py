import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn
import polynomial_chaos.poly as poly
import polynomial_chaos.poly_chaos_distributions as ch

chaos = ch.hermiteChaos
basis = chaos.normalized_basis

def ref_poly(degree, values):
    if degree == 0:
        return np.ones(shape=values.shape)
    elif degree == 1:
        return values
    elif degree == 2:
        return (values ** 2 - 1) / np.sqrt(2)
    elif degree == 3:
        return values ** 3 - 3 * values

def ref_poly2(degree, values):
    if degree == 0:
        return np.ones(shape=values.shape)
    elif degree == 1:
        return values
    elif degree == 2:
        return 1.5 * values ** 2 - 0.5

x = np.arange(-2, 2, 0.01)
for i in range(0, 3):
    plt.plot(x, np.vectorize(basis(i))(x), label="i={}".format(i))
    plt.plot(x, ref_poly(i, x), ".", label="ref i={}".format(i))
plt.legend()
plt.ylim((-2, 2))
plt.show()
