import math


# See "The Wiener--Askey Polynomial Chaos for Stochastic Differential Equations"
# Chapter 6.3.1 and references for this definition
def inverse_gaussian(u):
    # u is expected to be uniformly distributed in [0,1]
    h = math.sqrt(-math.log((min(u, 1 - u)) ** 2))
    return (math.copysign(1, u - 0.5)
            * (h - (2.515517 + 0.802853 * h + 0.010328 * h * h)
               / (1 + 1.432788 * h + 0.189269 * h * h + 0.001308 * h * h * h)))


def inverse_exponential(u):
    # for lambda = 1, which is a special case for a gamma distribution
    # u is expected to be uniformly distributed in [0,1]
    return -math.log(1 - u)  # inverse of exponential distribution 1-exp(-x)


def inverse_uniform(u):
    # u is expected to be uniformly distributed in [0,1]
    return 2 * u - 1  # inverse of uniform distribution in [-1,1]
