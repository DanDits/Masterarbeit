import numpy as np
from scipy.fftpack import ifft


def get_nodes(order):
    if order == 1:
        return np.zeros(1)
    return np.cos((order - 1 - np.arange(order)) * np.pi / (order - 1))


def get_weights(order):
    if order == 1:
        return np.ones(1) * 2.0
    theta = np.arange(order) * np.pi / (order - 1)

    weights = np.ones(order)
    for i in range(order):
        for j in range(1, (order - 1) // 2 + 1):
            b = 2.0
            if 2 * j == order - 1:
                b = 1.0
            weights[i] -= b * np.cos(2.0 * j * theta[i]) / (4 * j * j - 1)

    weights /= (order - 1)
    weights[1:order - 1] *= 2.0
    return weights


# implementation from http://www.scientificpython.net/pyblog/clenshaw-curtis-quadrature
# uses fft to compute weights in O(nlogn) instead of O(n^2)
def nodes_and_weights(n1):
    """ Computes the Clenshaw Curtis nodes and weights """
    if n1 == 1:
        x = 0
        w = 2
    elif n1 == 2:
        x = np.array([-1., 1.])
        w = np.array([1., 1.])
    else:
        n = n1 - 1
        C = np.zeros((n1, 2))
        k = 2 * (1 + np.arange(np.floor(n/2)))
        C[::2, 0] = 2 / np.hstack((1, 1-k*k))
        C[1, 1] = -n
        V = np.vstack((C, np.flipud(C[1:n, :])))
        F = np.real(ifft(V, n=None, axis=0))
        x = F[0:n1, 1]
        w = np.hstack((F[0, 0], 2*F[1:n, 0], F[n, 0]))
    return x, w

