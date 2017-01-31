import numpy as np
from scipy.fftpack import fft


def test(z):
    result = np.zeros(shape=z.shape, dtype=np.complex64)
    N = len(z)
    for i in range(N):
        curr = 0.
        for j in range(N):
            print(np.exp(-1j*i*np.pi))
            curr += np.exp(-2*np.pi*1j * j * i / N - 1j*i*np.pi) * z[j]
        result[i] = curr
    return result

start = np.array([3, 4, 7, 9])
result = test(start)
wanted = fft(start)

print(result)
print(wanted)
