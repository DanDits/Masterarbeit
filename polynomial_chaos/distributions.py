import math
from functools import partial
import random
from numpy import inf
from scipy.special import gamma
from scipy.special import beta as beta_func


# See "The Wiener--Askey Polynomial Chaos for Stochastic Differential Equations"
# Chapter 6.3.1 and references for this definition
def inverse_gaussian(u):
    # u is expected to be uniformly distributed in [0,1]
    h = math.sqrt(-math.log((min(u, 1 - u)) ** 2))
    return (math.copysign(1, u - 0.5)
            * (h - (2.515517 + 0.802853 * h + 0.010328 * h * h)
               / (1 + 1.432788 * h + 0.189269 * h * h + 0.001308 * h * h * h)))


def make_inverse_exponential(lamb):
    return lambda u: -math.log(1. - u) / lamb


def make_inverse_uniform(left_bound, right_bound):
    return lambda u: (right_bound - left_bound) * u + left_bound


class Distribution:
    def __init__(self, name, weight, support, sample_generator,
                 inverse_distribution=None, show_name=None, parameters=None):
        self.name = name
        self.parameters = parameters
        self.sample_generator = sample_generator
        self.show_name = name if show_name is None else show_name
        self.weight = weight
        self.support = support
        self.inverse_distribution = inverse_distribution

    def generate(self):
        return self.sample_generator()

    def __repr__(self):
        return self.show_name


gaussian = Distribution("Gaussian",
                        lambda x: math.exp(-x * x / 2.) / math.sqrt(2. * math.pi),
                        (-inf, inf),
                        partial(random.gauss, 0, 1),
                        inverse_gaussian,
                        parameters=(0, 1))


def make_uniform(left_bound, right_bound):
    if right_bound <= left_bound:
        raise ValueError("Left bound must be smaller than right bound:", left_bound, right_bound)
    return Distribution("Uniform",
                        lambda x: 1. / (right_bound - left_bound) if left_bound <= x <= right_bound else 0.,
                        (left_bound, right_bound),
                        partial(random.uniform, left_bound, right_bound),
                        make_inverse_uniform(left_bound, right_bound),
                        parameters=(left_bound, right_bound))


def make_gamma(shape, rate):
    if shape <= 0 or rate <= 0:
        raise ValueError("Requires positive parameters for gamma distribution.", shape, rate)
    gamma_shape = gamma(shape)
    return Distribution("Gamma",
                        lambda x: ((rate ** shape) * (x ** (shape - 1)) * math.exp(-rate * x) / gamma_shape
                                   if x >= 0 else 0.),
                        (0, inf),
                        partial(random.gammavariate, shape, 1. / rate),
                        show_name="Gamma({}, {})".format(shape, rate),
                        parameters=(shape, rate))


def make_exponential(lamb=1.):
    if lamb <= 0:
        raise ValueError("Requires positive lambda for exponential distribution.", lamb)
    # is special case of Gamma distribution (so parameters=(1,lambda))
    distr = make_gamma(1, lamb)
    distr.show_name = "Exponential({})".format(lamb)
    distr.inverse_distribution = make_inverse_exponential(lamb)
    return distr


def make_beta(alpha, beta):
    if alpha <= -1 or beta <= -1:
        raise ValueError("Parameters must be greater than -1", alpha, beta)
    beta_01 = partial(random.betavariate, beta + 1, alpha + 1)  # random uses switched notation...

    def generator():
        return beta_01() * 2 - 1  # scaling from [0,1] to [-1,1]
    return Distribution("Beta",
                        lambda x: (((1 - x) ** alpha) * ((1 + x) ** beta) / (2 ** (alpha + beta + 1))
                                   / beta_func(alpha + 1, beta + 1) if -1 <= x <= 1 else 0),
                        (-1, 1),
                        generator,
                        show_name="Beta({}, {})".format(alpha, beta),
                        parameters=(alpha, beta))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    test_distribution = make_beta(1., 2.)

    x = [test_distribution.sample_generator() for _ in range(100000)]
    hist, bin_edges = np.histogram(x, bins='auto', density=True)
    bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    plt.figure()
    plt.plot(bin_centers, hist, label="Calculated distribution")
    plt.plot(bin_centers, np.vectorize(test_distribution.weight)(bin_centers), label="PDF")
    plt.ylim((0, plt.ylim()[1]))
    plt.legend()
    plt.show()
