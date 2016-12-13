# adapted from code from https://people.sc.fsu.edu/~jburkardt/m_src/sandia_sparse/sandia_sparse.html
class QuadratureRule:

    def __init__(self, dim_num: int, level_max: int, nesting, nodes_and_weights_func):
        self.dim_num = dim_num
        self.level_max = level_max
        self.nesting = nesting
        self.nodes_and_weights_func = nodes_and_weights_func

    def apply(self, function):
        nodes, weights = self.nesting.calculate_nodes_and_weights(self.dim_num, self.level_max,
                                                                  [self.nodes_and_weights_func] * self.dim_num)
        print("Used nodes:", len(nodes))
        print(nodes)
        summed = 0
        for node, weight in zip(nodes, weights):
            summed += weight * function(node)
        return summed

def test():
    from polynomial_chaos.poly import make_hermite
    hermite = make_hermite()
    import util.sparse_quadrature.open_weakly_nested as own
    quad = QuadratureRule(1, 6, own.nesting_open_weakly, hermite.nodes_and_weights)
    import numpy as np
    print("RESULT=", quad.apply(lambda x: x[0] ** 6))


if __name__ == "__main__":
    test()

class Nesting:
    def __init__(self, calculate_point_num_func, levels_index_func, sparse_grid_func):
        self.point_num_func = calculate_point_num_func
        self.levels_index_func = levels_index_func
        self.sparse_grid_func = sparse_grid_func

    def calculate_point_num(self, dim_num: int, level_max: int):
        return self.point_num_func(dim_num, level_max)

    def calculate_levels_index(self, dim_num: int, level_max: int):
        point_num = self.calculate_point_num(dim_num, level_max)
        return self.levels_index_func(dim_num, level_max, point_num)

    def calculate_nodes_and_weights(self, dim_num: int, level_max: int, nodes_and_weights_funcs):
        point_num = self.calculate_point_num(dim_num, level_max)
        return self.sparse_grid_func(dim_num, level_max, point_num, nodes_and_weights_funcs)
