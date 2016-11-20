import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly as poly
import polynomial_chaos.poly_chaos_distributions as ch

basis = poly.legendre_basis()

def legendre_ref(degree, values):
    if degree == 0:
        return np.ones(shape=values.shape)
    elif degree == 1:
        return values
    elif degree == 2:
        return 1.5 * values ** 2 - 0.5
    elif degree == 3:
        return 0.5 * (5 * values ** 3 - 3 * values)


x = np.arange(-2, 2, 0.01)
for i in range(0, 4):
    plt.plot(x, np.vectorize(basis(i))(x), label="i={}".format(i))
    plt.plot(x, legendre_ref(i, x), ".", label="ref i={}".format(i))
print(np.vectorize(basis(3))(x) / legendre_ref(3, x))
plt.legend()
plt.ylim((-2, 2))
plt.show()
