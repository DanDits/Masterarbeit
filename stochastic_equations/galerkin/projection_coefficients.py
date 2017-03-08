from functools import partial

import numpy as np

import polynomial_chaos.multivariation as mv
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from util.quadrature.helpers import multi_index_bounded_sum_length


def get_projection_coefficients(function, project_trial, basis, chaos):
    def vectorized_func(ys):
        vec_result = function(project_trial.transform_values(ys))
        return vec_result

    def integrate_func(nodes_matrix, i):
        res1 = np.apply_along_axis(vectorized_func, 1, nodes_matrix)
        res2 = np.apply_along_axis(basis[i], 1, nodes_matrix)
        return res1 * res2

    return [chaos.integrate(partial(integrate_func, i=i), function_parameter_is_nodes_matrix=True)
            for i in range(len(basis))]


def _get_solution_coefficients(time, xs_mesh, project_trial, basis, chaos):
    coefficients = []
    for i, x in enumerate(xs_mesh[0]):
        def current_func(ys):
            return project_trial.reference([x], time, ys)

        # if starting_value_func does not depend on ys, then only the first coeff will be nonzero
        coeff = get_projection_coefficients(current_func, project_trial, basis, chaos)
        coefficients.append(coeff)
    return np.array(coefficients)


def get_solution_coefficients(time, xs_mesh, trial, max_poly_degree, quadrature_method, quadrature_param):
    old_flag = trial.flag_raw_attributes
    trial.flag_raw_attributes = True
    distrs = trial.variable_distributions
    random_dim = len(distrs)
    sum_bound = max_poly_degree

    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)
    chaos.init_quadrature_rule(quadrature_method, quadrature_param)

    poly_count = multi_index_bounded_sum_length(random_dim, sum_bound)
    basis = [chaos.normalized_basis(i) for i in range(poly_count)]
    coeffs = _get_solution_coefficients(time, xs_mesh, trial, basis, chaos)
    trial.flag_raw_attributes = old_flag
    return coeffs
