from math import pi
import numpy as np
from pseudospectral.solver import spectral_derivative
import matplotlib.pyplot as plt

bound = [-1, 1]


def testfun_solution(eks):  # second derivative of testfun
    var = pi * (eks + 1)
    return pi ** 2 * (np.cos(var) ** 2 * (np.sin(var) + 3) - np.sin(var) - 1) * np.exp(np.sin(var))


def testfun(x):
    return np.sin(pi * (x + 1)) * np.exp(np.sin(pi * (x + 1)))

result_x, result_y = spectral_derivative(bound, 512, testfun, 2)
x_highres = np.linspace(bound[0], bound[1], endpoint=False, num=512)  # higher resolution
plt.plot(result_x, result_y, x_highres, testfun_solution(x_highres))
plt.legend(["Pseudospectral solution", "Original solution"])
plt.show()
