import util.sparse_quadrature.rules as rules
from util.sparse_quadrature.helpers import compositions, level_to_order_open, colexical_vectors, product_weights
import numpy as np
from util.analysis import mul_prod
from scipy.misc import comb


# open weakly nested include Gauss Hermite and Gauss Legendre.
def calculate_point_num(dim_num: int, level_max: int):
    if level_max == 0:
        return 1
    # normally level_min = max ( 0, level_max + 1 - dim_num )
    if dim_num == 1:
        level_min = level_max
        point_num = 1
    else:
        level_min = 0
        point_num = 0
    ones = np.ones(dim_num, dtype=int)
    for level in range(level_min, level_max + 1):
        for level_1d in compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            # to account for the center point which would else be repeated, subtract 1 for higher levels
            order_1d = np.where(order_1d > 1, order_1d - ones, order_1d)
            point_num += mul_prod(order_1d)
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


def multigrid_index(dim_num: int, order_1d: np.array, order_nd: int):
    grid_index = np.zeros((dim_num, order_nd), dtype=int)
    for p, colex_vector in enumerate(colexical_vectors(dim_num, order_1d)):
        grid_index[:, p] = colex_vector - (order_1d - 1) // 2
    return grid_index


def levels_index(dim_num: int, level_max: int, point_num: int):
    grid_index = np.zeros((dim_num, point_num), dtype=int)
    grid_base = np.zeros((dim_num, point_num), dtype=int)

    level_min = 0
    if dim_num == 1:
        level_min = level_max
    point_num2 = 0
    for level in range(level_min, level_max + 1):
        for level_1d in compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            np.testing.assert_allclose(order_1d % 2, np.ones(order_1d.shape))
            grid_base2 = (order_1d - 1) // 2
            order_nd = mul_prod(order_1d)
            # grid indices will be between -M to +M where 2*M+1=order_1d(dim) for each dim
            grid_index2 = multigrid_index(dim_num, order_1d, order_nd)
            # determine the first level of appearance of each of the points to flag points being repeats of others
            grid_level = index_level(level, level_max, dim_num, order_nd, grid_index2, grid_base2)
            # finally only keep these points where they first appeared
            for point in range(order_nd):
                if grid_level[point] == level:
                    point_num2 += 1
                    assert point_num2 <= point_num
                    grid_index[:, point_num2 - 1] = grid_index2
                    grid_base[:, point_num2 - 1] = grid_base2
    assert point_num2 == point_num
    return grid_index, grid_base


def multigrid_point(dim_num, grid_base, grid_index, order_1d, nodes_and_weights_func):
    grid_point = np.zeros(dim_num)
    for dim in range(dim_num):
        nodes, _ = nodes_and_weights_func(order_1d[dim])
        grid_point[dim] = nodes[grid_index[dim] + grid_base[dim]]
    return grid_point


def sparse_grid(dim_num: int, level_max: int, point_num: int, nodes_and_weights_func):
    grid_weight = np.zeros(point_num)
    grid_point = np.zeros((dim_num, point_num))

    point_num2 = 0
    level_min = max(0, level_max + 1 - dim_num)
    if dim_num == 1:
        level_min2 = level_min
    else:
        level_min2 = 0
    for level in range(level_min2, level_max + 1):
        print("LEVEL=", level)
        for level_1d in compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            print("LEVEL1D=", level_1d, "ORDER1D=", order_1d)
            grid_base2 = (order_1d - 1) // 2
            order_nd = mul_prod(order_1d)
            grid_weights2 = product_weights(dim_num, order_1d, order_nd, nodes_and_weights_func)
            coeff = ((-1) ** ((level_max - level) % 2)) * comb(dim_num - 1, level_max - level)
            grid_index2 = multigrid_index(dim_num, order_1d, order_nd)
            grid_level = index_level(level, level_max, dim_num, order_nd, grid_index2, grid_base2)
            for point in range(order_nd):
                if grid_level[point] == level:
                    # new point!
                    point_num2 += 1
                    assert point_num2 <= point_num
                    grid_point[:, point_num2 - 1] = multigrid_point(dim_num, grid_base2, grid_index2[:, point],
                                                                    order_1d, nodes_and_weights_func)
                    if level_min <= level:
                        grid_weight[point_num2 - 1] = coeff * grid_weights2[point]
                else:
                    # already existing point!
                    if level_min <= level:
                        grid_point_temp = multigrid_point(dim_num, grid_base2, grid_index2[:, point],
                                                                    order_1d, nodes_and_weights_func)
                        point3 = -1
                        for point2 in range(point_num2):
                            if np.allclose(grid_point[:, point2], grid_point_temp):
                                point3 = point2
                                break
                        if point3 == -1:
                            print("OHOH:", grid_point)
                            print("LOOKING FOR:", grid_point_temp)
                        assert point3 != -1
                        grid_weight[point3] += coeff * grid_weights2[point]
    assert point_num2 == point_num
    print("GridPoints:", grid_point)
    print("Grid_weight:", grid_weight)
    return np.rollaxis(grid_point, 1), grid_weight

nesting_open_weakly = rules.Nesting(calculate_point_num, levels_index, sparse_grid)
