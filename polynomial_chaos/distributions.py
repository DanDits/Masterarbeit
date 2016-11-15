import math
from functools import partial
import random
from numpy import inf


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
    def __init__(self, name, weight, support, sample_generator, inverse_distribution=None, show_name=None):
        self.name = name
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
                        [-inf, inf],
                        partial(random.gauss, 0, 1),
                        inverse_gaussian)


def make_uniform(left_bound, right_bound):
    if right_bound <= left_bound:
        raise ValueError("Left bound must be smaller than right bound:", left_bound, right_bound)
    return Distribution("Uniform",
                        lambda x: 1. / (right_bound - left_bound) if left_bound <= x <= right_bound else 0.,
                        [left_bound, right_bound],
                        partial(random.uniform, left_bound, right_bound),
                        make_inverse_uniform(left_bound, right_bound))


def make_exponential(lamb=1.):
    if lamb <= 0:
        raise ValueError("Requires positive lambda for exponential distribution.", lamb)
    # is special case of Gamma distribution (so parameters=(1,lambda=1))
    return Distribution("Gamma",
                        lambda x: lamb * math.exp(-lamb * x) if x >= 0 else 0.,
                        [0, inf],
                        partial(random.expovariate, lamb),
                        make_inverse_exponential(lamb),
                        "Exponential({})".format(lamb))
