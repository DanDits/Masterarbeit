from math import pi
import numpy as np
from diff_equation.pseudospectral_solver import spectral_derivative
import matplotlib.pyplot as plt

bound = [-1, 1]


def testfun_solution(eks):  # second derivative of testfun
    var = pi * (eks + 1)
    return pi ** 2 * (np.cos(var) ** 2 * (np.sin(var) + 3) - np.sin(var) - 1) * np.exp(np.sin(var))


def testfun(x):
    return np.sin(pi * (x + 1)) * np.exp(np.sin(pi * (x + 1)))


@np.vectorize
def testfun2(eks):
    if eks < 0:
        return -eks
    else:
        return eks


@np.vectorize
def testfun2_solution(eks):
    if eks < 0:
        return -1
    else:
        return 1

fun = testfun
funsol = testfun_solution
deriv = 2 if fun == testfun else 1
fun_descr = "$f(x)=-\sin(x)e^{-\sin(x)}$" if fun == testfun else "$f(x)=|x|$"

result_x, result_y = spectral_derivative(bound, 2**8, fun, deriv)
x_highres = np.linspace(bound[0], bound[1], endpoint=False, num=512)  # higher resolution
plt.plot(result_x, result_y, x_highres, funsol(x_highres))
plt.legend(["Pseudospectral solution", "Original solution"])

from util.analysis import error_l2_relative as error
errors = []
Ns = [2 ** r for r in range(1, 8)]
for N in Ns:
    result_x, result_y = spectral_derivative(bound, N, fun, deriv)
    errors.append(error(result_y, funsol(result_x)))
plt.figure()
font = 28
plt.title("Fehler der spektralen Ableitung der Ordnung {} von {}".format(deriv, fun_descr),
          fontsize=font)
plt.plot(Ns, errors, "-o")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Anzahl Fourier Punkte', fontsize=font)
plt.ylabel('Relativer Fehler in diskreter L2-Norm', fontsize=font)
plt.show()
