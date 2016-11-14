import numpy as np


# noinspection PyTypeChecker
def error_l2(approx_y, solution_y):
    """
    Calculates the discrete L2 distance between the given nd-arrays of the same shape.
    :param approx_y: First nd-array
    :param solution_y: Second nd-array
    :return: The real distance, which is the discrete L2 integral between the functions.
    """
    assert approx_y.shape == solution_y.shape
    return np.sqrt(np.sum(np.abs(approx_y - solution_y) ** 2) / approx_y.size)
