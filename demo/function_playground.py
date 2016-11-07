import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn

x = np.linspace(-np.pi, np.pi, endpoint=False, num=32)


def function(var1):
    return 2*np.exp(-np.cos(x))

def function2(var1):
    return np.cos(var1)

y = function(x)
fourier = fftn(y)
print("x(", x.shape, "):", x)
print("First Fourier is=", fourier[0])
fourier[0] = 0
plt.plot(x, ifftn(fourier), x, function(x))
print("Difference:", ifftn(fourier) - function(x))
plt.show()
