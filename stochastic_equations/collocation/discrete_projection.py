import numpy as np

from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from stochastic_equations.collocation.coll_util import check_distribution_assertions
import polynomial_chaos.multivariation as mv
from util.quadrature.helpers import multi_index_bounded_sum_length
from stochastic_equations.collocation.coll_util import cached_collocation_point


def discrete_projection(trial, max_poly_degree, method, method_param, spatial_domain,
                        grid_size, start_time, stop_time, delta_time, wave_weight=0.5):
    sum_bound = max_poly_degree
    distrs = trial.variable_distributions
    for distr in distrs:
        check_distribution_assertions(distr)
    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)

    # the centralized minimal node approach does not work for quadrature, only for interpolation
    chaos.init_quadrature_rule(method, method_param)
    quad_points = chaos.quadrature_rule.get_nodes_count()

    poly_count = multi_index_bounded_sum_length(len(distrs), sum_bound)
    print("Poly count=", poly_count)
    basis = [chaos.poly_basis.polys(degree) for degree in range(poly_count)]

    poly_weights = []
    splitting_xs = None
    splitting_xs_mesh = None
    solution_shape = None
    debug_counter = 0

    def solution_at_node(nodes):
        nonlocal splitting_xs, splitting_xs_mesh, solution_shape, debug_counter
        debug_counter += 1
        last_solution, splitting, actual_shape = cached_collocation_point(spatial_domain, grid_size, trial, wave_weight,
                                                                          start_time, stop_time, delta_time, nodes,
                                                                          real_only=True, flatten=True,
                                                                          ensure_is_finite=True)
        if splitting_xs is None:
            splitting_xs = splitting.get_xs()
            splitting_xs_mesh = splitting.get_xs_mesh()
            solution_shape = actual_shape
        return last_solution

    for i, poly in enumerate(basis):

        def to_integrate(nodes):
            return solution_at_node(tuple(nodes)) * poly(nodes)

        curr_poly_weight = chaos.integrate(to_integrate)
        poly_weights.append(curr_poly_weight / chaos.normalization_gamma(i))
        if i % 100 == 0:
            print("Projection progress:", i + 1, "/", len(basis))
    expectancy = np.reshape(np.sqrt(chaos.normalization_gamma(0)) * poly_weights[0], solution_shape)
    variance = np.reshape(sum((weight ** 2) * chaos.normalization_gamma(i) for i, weight in enumerate(poly_weights))
                          - chaos.normalization_gamma(0) * (poly_weights[0]) ** 2,
                          solution_shape)

    def poly_approximation(y):
        return np.reshape(sum(w * p(y) for w, p in zip(poly_weights, basis)), solution_shape)

    return splitting_xs, splitting_xs_mesh, expectancy, variance, quad_points, len(basis)
