from util.quadrature.helpers import compositions, level_to_order_closed, colexical_vectors, product_weights
from util.quadrature.nesting import Nesting

import numpy as np
from util.analysis import mul_prod
from scipy.misc import comb


def calculate_point_num(dim_num: int, level_max: int):
    if level_max == 0:
        return 1
    # the amount of new nodes appearing per level, so 1 new for level=0, 2 new for level=1, afterwards 2^(level-1) new
    new_1d = np.array([1, 2] + [2 ** i for i in range(1, level_max)])
    point_num = 0
    for level in range(level_max + 1):
        for level_1d in compositions(level, dim_num):
            point_num += mul_prod(new_1d[level_1d])
    return point_num


def multigrid_index(dim_num: int, order_1d: np.array, order_nd: int):
    grid_index = np.zeros((dim_num, order_nd), dtype=int)
    for p, colex_vector in enumerate(colexical_vectors(dim_num, order_1d)):
        grid_index[:, p] = colex_vector
    return grid_index


def multigrid_scale_closed(dim_num: int, level_max: int, level_1d: np.array, grid_index: np.array):
    for dim in range(dim_num):
        if level_1d[dim] == 0:
            if level_max == 0:
                order_max = 1
            else:
                order_max = (2 ** level_max) + 1
            grid_index[dim, :] = (order_max - 1) // 2
        else:
            factor = 2 ** (level_max - level_1d[dim])
            grid_index[dim, :] *= factor
    return grid_index


def index_to_level_closed(dim_num: int, t, order, level_max: int):
    value = 0
    for dim in range(dim_num):
        s = t[dim] % order
        assert s >= 0
        if s == 0:
            level = 0
        else:
            level = level_max
            while s % 2 == 0:
                s //= 2
                level -= 1

        if level == 0:
            value += 1
        elif level != 1:
            value += level
    return value


def abscissa_level_closed_nd(level_max: int, dim_num: int, order_nd: int, grid_index: np.array):
    if level_max == 0:
        return np.zeros(order_nd)

    order = 2 ** level_max + 1
    level = np.empty(order_nd)
    for j in range(order_nd):
        level[j] = index_to_level_closed(dim_num, grid_index[:, j], order, level_max)
    return level


def levels_index(dim_num: int, level_max: int, point_num: int):
    grid_index = np.zeros((dim_num, point_num), dtype=int)
    grid_base = np.zeros((dim_num, point_num), dtype=int)

    print("PointNum=", point_num, "Dim_num=", dim_num, "LevelMax=", level_max)
    point_num2 = 0
    for level in range(level_max + 1):
        for level_1d in compositions(level, dim_num):
            print("Current level=", level_1d)
            order_1d = level_to_order_closed(level_1d)
            order_nd = mul_prod(order_1d)
            print("Current order=", order_1d, "Current order_nd=", order_nd)
            grid_index2 = multigrid_index(dim_num, order_1d, order_nd)
            prev_shape = grid_index2.shape
            grid_index2 = multigrid_scale_closed(dim_num, level_max, level_1d, grid_index2)
            grid_level = abscissa_level_closed_nd(level_max, dim_num, order_nd, grid_index2)
            print("Grid shape=", grid_index.shape, "Grid_index2 shape=", grid_index2.shape, "Prevshape=", prev_shape)
            print("Grid levels=", grid_level, "Current level=", level)
            for point in range(order_nd):
                if grid_level[point] == level:
                    point_num2 += 1
                    assert point_num2 <= point_num
                    grid_base[:, point_num2 - 1] = order_1d
                    grid_index[:, point_num2 - 1] = grid_index2[:, point]
    print("Point_num2=", point_num2)
    assert point_num2 == point_num
    return grid_index, grid_base


def sparse_grid_weights(dim_num: int, level_max: int, point_num: int, grid_index, nodes_and_weights_funcs):

    grid_weight = np.zeros(point_num)
    level_min = max(0, level_max + 1 - dim_num)
    for level in range(level_min, level_max + 1):
        for level_1d in compositions(level, dim_num):
            order_1d = level_to_order_closed(level_1d)
            order_nd = mul_prod(order_1d)
            grid_index2 = multigrid_index(dim_num, order_1d, order_nd)
            grid_weight2 = product_weights(dim_num, order_1d, order_nd, nodes_and_weights_funcs)
            grid_index2 = multigrid_scale_closed(dim_num, level_max, level_1d, grid_index2)

            coeff = ((-1) ** ((level_max - level) % 2)) * comb(dim_num - 1, level_max - level, exact=True)
            for point2 in range(order_nd):
                for point in range(point_num):

                    if np.all(grid_index2[:, point2] == grid_index[:, point]):
                        grid_weight[point] += coeff * grid_weight2[point2]
                        break
    return grid_weight


def sparse_grid(dim_num: int, level_max: int, point_num: int, nodes_and_weights_funcs):
    grid_index, _ = levels_index(dim_num, level_max, point_num)

    if level_max == 0:
        order_max = 1
    else:
        order_max = 2 ** level_max + 1

    grid_point = np.zeros((dim_num, point_num))

    assert len(nodes_and_weights_funcs) == dim_num
    all_nodes = [nodes_and_weights_func(order_max)[0] for nodes_and_weights_func in nodes_and_weights_funcs]
    for point in range(point_num):
        for dim, nodes in enumerate(all_nodes):
            current_point_index = grid_index[dim, point]
            grid_point[dim, point] = all_nodes[dim][current_point_index]
    grid_weight = sparse_grid_weights(dim_num, level_max, point_num, grid_index, nodes_and_weights_funcs)
    return np.rollaxis(grid_point, 1), grid_weight


# closed fully nested rules include Glenshaw Curtis
class ClosedFullNesting(Nesting):
    def __init__(self):
        super().__init__("ClosedFull")

    def calculate_point_num(self, dim_num: int, level_max: int):
        return calculate_point_num(dim_num, level_max)

    def calculate_sparse_grid(self, dim_num: int, level_max: int, point_num: int, nodes_and_weights_funcs):
        return sparse_grid(dim_num, level_max, point_num, nodes_and_weights_funcs)
