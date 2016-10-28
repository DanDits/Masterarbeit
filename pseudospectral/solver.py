import numpy as np
import math
from numpy.fft import ifft, fft
import matplotlib.pyplot as plt

bound_left = -math.pi  # left border of interval
bound_right = math.pi  # right border of interval
N = 8  # number of grid points

h = (bound_right - bound_left) / (N - 1)  # spatial step size
x = np.linspace(bound_left, bound_right, endpoint=False, num=N)
scale = 2 * math.pi / (bound_right - bound_left)
kxx = -(scale * np.append(np.arange(0, N / 2 + 1), np.arange(- N / 2 + 1, 0))) ** 2

print(h, len(kxx), "kxx=", kxx)

# TODO read pseudospectral_2.pdf
y = np.sin(x)
yxx = ifft(kxx * fft(y))
print(yxx)

plt.plot(x, yxx)
plt.show()