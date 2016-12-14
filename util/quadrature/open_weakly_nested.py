from util.quadrature.helpers import compositions, level_to_order_open, colexical_vectors, product_weights
from util.quadrature.nesting import Nesting

import numpy as np
from util.analysis import mul_prod
from scipy.misc import comb
from itertools import chain


def calculate_point_num(dim_num: int, level_min: int, level_max: int, init_point_num: int, repetition_handler=None):
    if level_max == 0:
        return 1
    point_num = init_point_num
    for level in range(level_min, level_max + 1):
        for level_1d in compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            if repetition_handler is not None:
                order_1d = repetition_handler(order_1d)
            point_num += mul_prod(order_1d)  # the number of points is the full product of all 1d points
    return point_num


def index_level(level: int, level_max: int, dim_num: int, point_num: int, grid_index: np.array,
                grid_base: np.array):
    # open weakly nested points can only share the zero indexed point, so check each point in grid for each dimension
    level_min = 0
    if dim_num == 1:
        level_min = level_max
    # "if a point has a dim-th component whose index is 0, then the value of LEVEL at which this point would first
    # be generated is less than LEVEL, unless the dim-th component of GRID_BASE is 0"
    grid_level = np.zeros(point_num, dtype=int)
    for point in range(point_num):
        grid_level[point] = max(level, level_min)
        for dim in range(dim_num):
            if grid_index[dim, point] == 0:
                grid_level[point] = max(grid_level[point] - grid_base[dim], level_min)
    return grid_level


def multigrid_index(dim_num: int, order_1d: np.array, order_nd: int, index_offset):
    grid_index = np.zeros((dim_num, order_nd), dtype=int)
    for p, colex_vector in enumerate(colexical_vectors(dim_num, order_1d)):
        grid_index[:, p] = colex_vector + index_offset
    return grid_index


def multigrid_point(dim_num, grid_index, order_1d, offset, nodes_and_weights_funcs):
    grid_point = np.zeros(dim_num)
    assert len(nodes_and_weights_funcs) == dim_num
    for dim, nodes_and_weights_func in enumerate(nodes_and_weights_funcs):
        nodes, _ = nodes_and_weights_func(order_1d[dim])
        grid_point[dim] = nodes[grid_index[dim] - offset[dim]]
    return grid_point


def sparse_grid(dim_num: int, level_min2: int, level_max: int, point_num: int, nodes_and_weights_funcs,
                calculate_grid_base2, calculate_offset, calculate_index_level):
    grid_weight = np.zeros(point_num)
    grid_point = np.zeros((dim_num, point_num))

    point_num2 = 0
    level_min = max(0, level_max + 1 - dim_num)
    last_point3 = -1
    for level in range(level_min2, level_max + 1):
        for level_1d in compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            grid_base2 = calculate_grid_base2(order_1d)
            order_nd = mul_prod(order_1d)
            grid_weights2 = product_weights(dim_num, order_1d, order_nd, nodes_and_weights_funcs)
            coeff = ((-1) ** ((level_max - level) % 2)) * comb(dim_num - 1, level_max - level)
            offset = calculate_offset(order_1d)
            grid_index2 = multigrid_index(dim_num, order_1d, order_nd, offset)

            # level is used to find out if the point is new or already existed as it sets the lowest level
            # of appearance of each point. If grid_level is None then every point is new
            grid_level = calculate_index_level(level, level_max, dim_num, order_nd, grid_index2, grid_base2)
            for point in range(order_nd):
                if grid_level is None or grid_level[point] == level:
                    # new point!
                    point_num2 += 1
                    assert point_num2 <= point_num
                    grid_point[:, point_num2 - 1] = multigrid_point(dim_num, grid_index2[:, point],
                                                                    order_1d, offset, nodes_and_weights_funcs)
                    if level_min <= level:
                        grid_weight[point_num2 - 1] = coeff * grid_weights2[point]
                elif level_min <= level:
                    # already existing point!
                    grid_point_temp = multigrid_point(dim_num, grid_index2[:, point],
                                                      order_1d, offset, nodes_and_weights_funcs)
                    # find the index
                    point3 = -1
                    # it is very common that the point is the last_point3+1, so instead of using
                    # bruteforce search range(point_num2) we make the order smarter
                    # this SIGNIFICANTLY improves speed, as np.allclose is in the innermost loop and quite costly
                    for point2 in chain([last_point3 + 1], range(last_point3 + 1), range(last_point3 + 2, point_num2)):
                        if np.allclose(grid_point[:, point2], grid_point_temp):
                            point3 = point2
                            last_point3 = point3
                            break
                    assert point3 != -1  # we know it has to be somewhere as grid_level[point] != level
                    grid_weight[point3] += coeff * grid_weights2[point]
    assert point_num2 == point_num
    return np.rollaxis(grid_point, 1), grid_weight

# For open weakly nestings:
# A table of amount of nodes used for a combination of dimension and level:
# level\dimension
#   1   2       3       4       5
# 0	1	1	    1	    1	    1
# 1	3	5	    7	    9	    11
# 2	7	21	    37	    57	    81
# 3	15	73	    159	    289	    471
# 4	31	225	    597	    1,265	2,341
# 5	63	637	    2,031	4,969	10,363
# 6	127	1,693	6,405	17,945	41,913
# 7	255	4,289	19,023	60,577	157,583
# 8	511	10,473	53,829	193,457	557,693

# Testing values: Integrating with hermite nodes and weights (so the integral is implicitly over (-Inf,Inf)^dimension):
# dimension=1: f(x)=1 integrated is exactly 1.
# dimension=1: f(x)=sin(x)**2 integrated is (1/2 - 1/(2*e**2))=0.43233235838169365
# dimension=1: f(x)=x^n integrated is 0 for n odd, and (n-1)!! for n even
# dimension=2: f(x)=sin(x[0])**2*sin(x[1])**2 integrated is (1/2 - 1/(2*e**2)) ** 2 = 0.1869112681038772


# open weakly nested include Gauss Hermite and Gauss Legendre. These can also use the OpenNonNesting but this
# only results in more required nodes
class OpenWeaklyNesting(Nesting):
    def __init__(self):
        super().__init__("OpenWeakly")

    def calculate_point_num(self, dim_num: int, level_max: int):
        # "Oddly enough, in order to count the number of points, we will
        # behave as though LEVEL_MIN was zero. This is because our computation concentrates on throwing away all
        # points generated at lower levels, but, in fact, if we start a a nonzero level, we need to include on that
        # level all the points that would have been generated on lower levels."
        if dim_num == 1:
            level_min = level_max
            point_num = 1
        else:
            level_min = 0
            point_num = 0

        ones = np.ones(dim_num, dtype=int)

        def repetition_handler(order_1d):
            # to account for the center point which would else be repeated, subtract 1 for higher levels
            return np.where(order_1d > 1, order_1d - ones, order_1d)

        return calculate_point_num(dim_num, level_min, level_max, point_num, repetition_handler)

    def calculate_sparse_grid(self, dim_num: int, level_max: int, point_num: int, nodes_and_weights_funcs):
        level_min = max(0, level_max + 1 - dim_num)
        if dim_num != 1:
            level_min = 0

        def calculate_grid_base2(order_1d):
            return (order_1d - 1) // 2  # implicitly this is the offset

        def calculate_offset(order_1d: np.array):
            return -calculate_grid_base2(order_1d)

        return sparse_grid(dim_num, level_min, level_max, point_num, nodes_and_weights_funcs,
                           calculate_grid_base2, calculate_offset, index_level)


# For open non nestings:
# A table of amount of nodes used for a combination of dimension and level:
# level\dimension
#   1   2       3       4       5
# 0	1	1	    1 	    1	    1
# 1	3	7	    10	    13	    16
# 2	7	29	    58	    95	    141
# 3	15	95	    255	    515	    906
# 4	31	273	    945	    2,309	4,746
# 5	63	723	    3,120	9,065	21,503
# 6	127	1,813	9,484	32,259	87,358
# 7	255	4,375	27,109	106,455	325,943
# 8	511	10,265	73,915	330,985	1,135,893
#
# Testing vlaues: Integrating with laguerre nodes and weights (so the integral is implicitly over (0,Inf)^dimension):
# assuming alpha=0.3 for laguerre with alpha>0
# dimension=1: f(x)=x^2 integrated is 0.39
# dimension=1: f(x)=x^3 integrated is 0.897
# dimension=2: f(x)=x[0]^2*x[1] integrated is 0.3*0.39=0.117

# open non nested rules include Gauss Laguerre
class OpenNonNesting(Nesting):
    def __init__(self):
        super().__init__("OpenNon")

    def calculate_point_num(self, dim_num: int, level_max: int):
        level_min = max(0, level_max + 1 - dim_num)
        point_num = 0
        return calculate_point_num(dim_num, level_min, level_max, point_num)

    def calculate_sparse_grid(self, dim_num: int, level_max: int, point_num: int, nodes_and_weights_funcs):
        level_min = max(0, level_max + 1 - dim_num)

        def calculate_grid_base2(order_1d):
            return order_1d

        def none_func(*args, **kwargs):  # we are absolutely not nested, so we do not need to index the levels of points
            return None
        zeros = np.zeros(dim_num, dtype=int)
        return sparse_grid(dim_num, level_min, level_max, point_num, nodes_and_weights_funcs,
                           calculate_grid_base2, lambda order_1d: zeros, none_func)
