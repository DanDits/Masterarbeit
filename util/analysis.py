import numpy as np


# noinspection PyTypeChecker
def error_l2(approx_y, solution_y):
    assert approx_y.shape == solution_y.shape
    return np.sqrt(np.sum(np.abs(approx_y - solution_y) ** 2) / approx_y.size)
