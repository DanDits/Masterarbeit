import numpy as np


# adapted from code from https://people.sc.fsu.edu/~jburkardt/m_src/sandia_sparse/sandia_sparse.html
class QuadratureRule:

    def __init__(self, level_max: int, nesting, nodes_and_weights_funcs):
        self.dim_num = len(nodes_and_weights_funcs)
        self.level_max = level_max
        self.nesting = nesting
        self.nodes, self.weights = self.nesting.calculate_nodes_and_weights(self.dim_num, self.level_max,
                                                                            nodes_and_weights_funcs)

    def get_nodes_count(self):
        return len(self.weights)

    def apply(self, function):
        return np.inner(self.weights, np.apply_along_axis(function, 1, self.nodes))


class Nesting:
    def __init__(self, name):
        self.name = name

    def calculate_point_num(self, dim_num: int, level_max: int):
        raise NotImplementedError("Cannot calculate point num of abstract base class.")

    def calculate_sparse_grid(self, dim_num: int, level_max: int, point_num: int, nodes_and_weights_funcs):
        raise NotImplementedError("Cannot calculate sparse grid of abstract base class.")

    def calculate_nodes_and_weights(self, dim_num: int, level_max: int, nodes_and_weights_funcs):
        # another big speedup could be achieved by precalculating and caching the results of nodes_and_weights
        point_num = self.calculate_point_num(dim_num, level_max)
        return self.calculate_sparse_grid(dim_num, level_max, point_num, nodes_and_weights_funcs)
