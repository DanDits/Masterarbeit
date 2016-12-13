import numpy as np


def integer_log(value, base):
    value = abs(int(value))
    if value == 0:
        return float('NaN')
    log_check = 0
    while base <= value:
        value /= base
        log_check += 1
    return log_check


def product_weights(dim_num: int, order_1d: np.array, order_nd: int, nodes_and_weights_1d_func):
    weight_nd = np.ones(order_nd)
    contig = 1  # the number of consecutive values to set
    skip = 1  # distance from current value of START to the next location of a block values to set
    rep = order_nd  # number of blocks of values to set

    for dim in range(dim_num):
        factor_order = order_1d[dim]
        weights_1d = nodes_and_weights_1d_func(factor_order)[1]
        rep //= factor_order
        skip *= factor_order
        for j in range(factor_order):
            start = j * contig
            for k in range(rep):
                weight_nd[start:start+contig] *= weights_1d[j]
                start += skip
        contig *= factor_order
    return weight_nd


def colexical_vectors(dim_num: int, bases: list):
    assert dim_num == len(bases)
    current = np.zeros(dim_num, dtype=int)
    while True:
        yield current
        # try to increment the leftmost index, if not possible reset it to zero and increment right neighbor
        for i in range(0, dim_num):
            if current[i] < bases[i] - 1:
                current[i] += 1  # successfully incremented
                break
            elif current[i] == bases[i] - 1:
                current[i] = 0
                # increment right neighbor implicitly by not breaking loop
                if i == dim_num - 1:  # if we would reset the last index stop creation
                    raise StopIteration


def compositions(summed_value: int, parts_count: int):
    t = summed_value
    h = 0
    current = np.zeros(parts_count, dtype=int)
    current[0] = summed_value
    yield current
    while current[-1] != summed_value:
        if t > 1:
            h = 0
        t = current[h]
        current[h] = 0
        current[0] = t - 1
        current[h + 1] += 1
        h += 1
        yield current


def level_to_order_open(level_1d: np.array):
    return 2 ** (level_1d + 1) - 1
