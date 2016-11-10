import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn

n = 512
x = np.linspace(-np.pi, np.pi, endpoint=False, num=n)


alpha_small = 1 / 4
def function(xs):
    return 1 + alpha_small * (-1 + 4 * np.sin(sum(xs)) ** 2 - 2 * np.cos(sum(xs)) ** 2
                                                          + 4 * np.sin(sum(xs)) ** 2 * np.cos(sum(xs)) ** 2
                                                          + 2 * np.sin(sum(xs)) ** 2)


def function2(var1):
    return np.cos(var1)

y = function([x])
print(min(y), " and ", max(y))
plt.show()
