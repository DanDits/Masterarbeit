import numpy as np


def integer_log(value, base):
    """
    Returns the integer logarithm of the given value to the given base. If there is no exact integer logarithm,
    returns the next smaller value. Values < 1 will return NaN.
    :param value: The value to get the logarithm for.
    :param base: The base to use.
    :return: Result. It holds base**Result <= value for the greatest possible integer Result.
    """
    value = int(value)
    if value <= 0:
        return float('NaN')
    log_check = 0
    while base <= value:
        value /= base
        log_check += 1
    return log_check


def product_weights(dim_num: int, order_1d: np.array, order_nd: int, nodes_and_weights_1d_funcs):
    """
    Calculates the product of the weights for a multivariate quadrature rule for dimension dim_num.
    The ordering is made such that for dim_num=3, order_1d=[4,3,2], order_nd=4*3*2=24 and three given 1d rules
     Rule 1: 
       Order = 4
       W(1:4) = ( 2, 3, 5, 7 )
 
     Rule 2:
       Order = 3
       W(1:3) = ( 11, 13, 17 )
 
     Rule 3:
       Order = 2
       W(1:2) = ( 19, 23 )

     Product Rule:
       Order = 24
       W(1:24) =
         ( 2 * 11 * 19 )
         ( 3 * 11 * 19 )
         ( 5 * 11 * 19 )
         ( 7 * 11 * 19 )
         ( 2 * 13 * 19 )
         ( 3 * 13 * 19 )
         ( 5 * 13 * 19 )
         ( 7 * 13 * 19 )
         ( 2 * 17 * 19 )
         ( 3 * 17 * 19 )
         ( 5 * 17 * 19 )
         ( 7 * 17 * 19 )
         ( 2 * 11 * 23 )
         ( 3 * 11 * 23 )
         ( 5 * 11 * 23 )
         ( 7 * 11 * 23 )
         ( 2 * 13 * 23 )
         ( 3 * 13 * 23 )
         ( 5 * 13 * 23 )
         ( 7 * 13 * 23 )
         ( 2 * 17 * 23 )
         ( 3 * 17 * 23 )
         ( 5 * 17 * 23 )
         ( 7 * 17 * 23 )
    :param dim_num: The dimension of the space.
    :param order_1d: The integer vector of orders which is the amount of nodes per dimension.
    :param order_nd: The product of the orders which is the amount of total nodes for this quadrature.
    :param nodes_and_weights_1d_funcs: A list of functions, each taking an integer and returning a tuple of nodes and
    weights. The weights have to be a vector of length equal to the given integer and are
    the 1d quadrature rule weights.
    :return: A vector of length order_nd containing the weights belonging to this multivariate quadrature rule.
    """
    weight_nd = np.ones(order_nd)
    contig = 1  # the number of consecutive values to set
    skip = 1  # distance from current value of START to the next location of a block values to set
    rep = order_nd  # number of blocks of values to set

    assert len(nodes_and_weights_1d_funcs) == dim_num
    for dim, nodes_and_weights_1d_func in enumerate(nodes_and_weights_1d_funcs):
        factor_order = order_1d[dim]
        _, weights_1d = nodes_and_weights_1d_func(factor_order)
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
    """
    A generator for generating the colexical ordered vectors if length dim_num. where
    each entry is an non negative integer and smaller than the corresponding value in bases.
    If bases values are equal to 2 this can be thought of as reversed binary up counting from
    0 to (2^dim_num)-1.
    E.g. for dim_num = 3, bases=[3,2,3]:
    [0,0,0]->[1,0,0]->[2,0,0]->[0,1,0]->[1,1,0]->[2,1,0]->[0,0,1]->
    [1,0,1]->[2,0,1]->[0,1,1]->[1,1,1]->[2,1,1]->[0,0,2]->
    [1,0,2]->[2,0,2]->[0,1,2]->[1,1,2]->[2,1,2]
    :param dim_num: The length of the vector
    :param bases: A list of positive numbers of length equal to dim_num.
    :return: A generator returning integer vectors.
    """
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
    """
    A generator that generates all possible compositions of the given number into the
    given amount of parts. So for example if the summed value is 3, these compositions for 2 parts would be
    [3,0]->[2,1]->[1,2]->[0,3]
    :param summed_value: The summed value of all entries.
    :param parts_count: The amount of parts which is the length of the returned vector.
    :return: A generator that returns integer vectors of length parts_count which sum up to summed_value.
    """
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
    """
    Returns the order vector, which is the amount of points, for the given vector of levels.
    This is a component wise function which returns by definition 2^(level+1)-1
    :param level_1d: The integer vector of non negative levels.
    :return: A integer level containing the orders for each level.
    """
    return 2 ** (level_1d + 1) - 1
