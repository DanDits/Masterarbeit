import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn

n = 100
x = np.linspace(-np.pi, np.pi, endpoint=False, num=n)


def function(t):
    return np.sqrt(1 / t)


def function2(var1):
    return np.cos(var1)

y = function(np.linspace(1, n, num=n))
plt.plot(range(n), y)
plt.yscale('log')
plt.show()
