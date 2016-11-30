import numpy as np
import operator
from functools import reduce


def mul_prod(factors):
    """
    As python does not include a "prod" equivalent to "sum", here it is.
    :param factors: Factors to multiply by using the operator.mul
    :return: The product of all factors, 1 if empty.
    """
    return reduce(operator.mul, factors, 1)


# Notation hint for literature: Pochhammer symbol for falling factorial.. was hard to find!
# Xiu and other authors define this to be rising factorial in contrast to wikipedia
# Falling: alpha_n = alpha*(alpha-1)*...*(alpha-n+1)
def rising_factorial(alpha, n):
    prod = 1
    for i in range(n):
        prod *= alpha + i
    return prod


# noinspection PyTypeChecker
def error_l2(approx_y, solution_y):
    """
    Calculates the discrete L2 distance between the given nd-arrays of the same shape.
    Does not take into account to length of the interval this approximation lives on. To get
    the exact integral multiply by sqrt of the area of the domain.
    :param approx_y: First nd-array
    :param solution_y: Second nd-array
    :return: The real distance, which is the (up to a constant factor)
    the discrete L2 integral between the functions.
    """
    assert approx_y.shape == solution_y.shape
    return np.sqrt(np.sum(np.abs(approx_y - solution_y) ** 2) / approx_y.size)
