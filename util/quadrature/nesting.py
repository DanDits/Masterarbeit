from itertools import count


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

    def get_minimum_level_with_point_num(self, dim_num: int, point_num: int):
        return next(l for l in count() if self.calculate_point_num(dim_num, l) >= point_num)


from util.quadrature.open_weakly_nested import OpenWeaklyNesting, OpenNonNesting


def get_nesting_for_name(name):
    if name in ["Hermite", "Legendre"]:
        nesting = OpenWeaklyNesting()
    elif name in ["Laguerre", "Jacobi"]:
        nesting = OpenNonNesting()
    else:
        raise ValueError("Not sure what nesting to use for poly basis", name)
    return nesting


def get_nesting_for_multiple_names(names):
    if "Laguerre" in names or "Jacobi" in names:
        return OpenNonNesting()
    elif set(names) <= {"Hermite", "Legendre"}:
        return OpenWeaklyNesting()
    raise ValueError("Not sure what nesting to use for poly basis", names)
