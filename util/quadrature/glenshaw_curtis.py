import numpy as np
from scipy.fftpack import ifft
from polynomial_chaos.distributions import Distribution
from functools import lru_cache


def get_nodes(order):
    if order == 1:
        return np.zeros(1)
    return np.cos((order - 1 - np.arange(order)) * np.pi / (order - 1))


# direct implementation to compute weights which is O(n^2)
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
        x = np.array([0.])
        w = np.array([2.])
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


def calculate_transformed_nodes_and_weights(distribution: Distribution):
    @lru_cache(maxsize=None)
    def transformed_nodes_and_weights(order):
        weight = distribution.weight
        support = distribution.support
        # possible extensions to be more general: bounded interval, onesided unbounded left, onesisded unbounded right
        if support == (-1, 1):
            nodes, weights = nodes_and_weights(order)
            # we do not need to transform the nodes as they already are correct for glenshaw curtis
            # so we pretend we integrate a function which is multiplied by the distribution's weight function
            weights *= np.vectorize(weight)(nodes)
        elif support == (-np.Inf, np.Inf):
            # now we got a problem as the support's are not equal and the support isn't even compact
            # so there is no bijection between [-1,1] and (-Inf, Inf)
            # therefore we drop the first and last node and use the bijection (-Inf,Inf)->(-1,1) z->2/pi*atan(z)
            # Then: Integral_{-Inf}^{Inf}(f(z)weight(z))dz
            #       =Integral_{-1}^{1}(f(tan(pi/2x))weight(tan(pi/2x))pi/2(tan^2(pi/2x)+1)dx
            # The latter we cant integrate with glenshaw curtis quadrature, except tan(+-pi/2) is +-Inf...
            # this substitution introduces a factor pi/2*(tan^2(pi/2x)+1)
            # and also a scaling of the nodes by tan(pi/2x)
            nodes, weights = nodes_and_weights(order)
            # we cannot really delete the first and last node, but we set it to something harmless
            # and zero out their weight
            nodes[0] = 0.
            nodes[-1] = 0.
            weights[0] = 0.
            weights[-1] = 0.
            pi_half = np.pi / 2.
            nodes = np.tan(pi_half * nodes)
            weights *= np.vectorize(weight)(nodes)
            weights *= pi_half * (nodes ** 2 + 1)
        elif support == (0, np.Inf):
            nodes, weights = nodes_and_weights(order)
            # same as (-Inf,Inf) support, except now we use the substitution x=1-2e^(-z)
            # which transforms z=-log((1-z)/2) and introduces a factor (after shortening) of 1/(1-x)
            # now we do not need to worry about the leftmost point, only the rightmost=1.
            # as long as the weight function is defined on all [0,Inf)
            # if the weight function is not defined on 0 it should implicitly zero out the weight for this node
            nodes[-1] = 0.
            weights[-1] = 0.
            old_nodes = nodes
            nodes = abs(np.log((1 - nodes) / 2.))  # same as negating, but prevents a node=-0.
            weights *= np.vectorize(weight)(nodes)
            weights /= (1 - old_nodes)
        else:
            raise ValueError("Unsupported support (he!):", support)
        return nodes, weights
    return transformed_nodes_and_weights
