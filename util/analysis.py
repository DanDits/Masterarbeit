import numpy as np
import operator
from functools import reduce

from util.general import coroutine


_invalid_usage_msg = "Probably forgot to next() the coroutine after sending last value."
# based on http://www.johndcook.com/blog/standard_deviation/ from Knut
@coroutine
def running_mean_variance():
    """
    Returns a started coroutine that calculates the running mean and running variance.
    When a new value is calculated use send(value) to give it to the coroutine. The returned value
    of send is the current (including the given) mean and variance estimation.
    After sending a value, make sure to invoke next(coroutine).
    :return: A coroutine.
    """
    mean = (yield)
    k = 1
    s = 0 * mean  # to ensure the type and shape is correct
    ret = yield (mean, s)  # special case of first value, cannot estimate much
    if ret is not None:
        raise ValueError(_invalid_usage_msg)
    while True:
        k += 1
        value = (yield)
        delta = value - mean
        mean += delta / k
        delta2 = value - mean
        s += delta2 * delta
        ret = yield (mean, s / (k - 1))
        if ret is not None:
            raise ValueError(_invalid_usage_msg)


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
def rising_factorial(alpha: float, n: int):
    """
    Calculates the rising factorial (sometimes called Pochhammer symbol) of alpha and n which
    is alpha*(alpha+1)*...*(alpha+n-1). For n=0 returns 1.
    :param alpha:
    :param n:
    :return:
    """
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


def error_l2_relative(approx_y, solution_y):
    """
    Like error_l2 this calculates the distance between of the two given nd-arrays.
    Furthermore this scaled the returned error by dividing through the discrete l2 error of solution_y.
    :param approx_y: The approximation
    :param solution_y: The exact solution
    :return: The distance (relativized) between the given nd-arrays.
    """
    assert approx_y.shape == solution_y.shape
    solution_error = np.sqrt(np.sum(np.abs(solution_y) ** 2.) / solution_y.size)
    return error_l2(approx_y, solution_y) / solution_error


def error_maximum(approx_y, solution_y):
    return np.max(np.abs(approx_y - solution_y))
