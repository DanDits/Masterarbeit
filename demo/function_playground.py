import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn

n = 512
x = np.linspace(-np.pi, np.pi, endpoint=False, num=n)

# we get: 1296.45145882


def function(var1):
    return 2*np.exp(-np.cos(var1))


def function2(var1):
    return np.cos(var1)

y = function(x)
fourier = fftn(y)
print("x(", x.shape, ")")
print("First Fourier is=", fourier[0], "scaled=", fourier[0] / n)
fourier[0] = 0
plt.plot(x, ifftn(fourier), x, function(x))
#print("Difference:", ifftn(fourier) - function(x))
plt.show()
