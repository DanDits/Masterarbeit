import numpy as np

from diff_equation.splitting import Splitting
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from stochastic_equations.collocation.coll_util import check_distribution_assertions
import polynomial_chaos.multivariation as mv
import diff_equation.klein_gordon as kg
from util.quadrature.helpers import multi_index_bounded_sum_length
from functools import lru_cache


def discrete_projection_expectancy(trial, max_poly_degree, method, method_param, spatial_domain,
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

    @lru_cache(maxsize=None)
    def solution_at_node(nodes):
        nonlocal splitting_xs, splitting_xs_mesh, solution_shape, debug_counter
        debug_counter += 1
        print("...Node counter", debug_counter)
        trial.set_random_values(nodes)
        configs = kg.make_klein_gordon_wave_linhyp_configs(spatial_domain, [grid_size], trial.alpha,
                                                           trial.beta, wave_weight)
        splitting = Splitting.make_fast_strang(*configs, "FastStrang",
                                               start_time, trial.start_position, trial.start_velocity, delta_time)
        splitting.progress(stop_time, delta_time, 0)
        last_solution = splitting.solutions()[-1]
        if splitting_xs is None:
            splitting_xs = splitting.get_xs()
            splitting_xs_mesh = splitting.get_xs_mesh()
            solution_shape = last_solution.shape
        sol = last_solution.real.flatten()
        if method == "sparse" and not np.all(np.isfinite(sol)):
            print("Solution at node", nodes, "is not finite:", sol)
        return sol

    for i, poly in enumerate(basis):
        print("Integrationg poly with number", i)
        def to_integrate(nodes):
            return solution_at_node(tuple(nodes)) * poly(nodes)
        curr_poly_weight = chaos.integrate(to_integrate)
        poly_weights.append(curr_poly_weight / chaos.normalization_gamma(i))
        print("Projection progress:", i + 1, "/", len(basis))
    expectancy = np.reshape(np.sqrt(chaos.normalization_gamma(0)) * poly_weights[0], solution_shape)
    variance = np.reshape(sum((weight ** 2) * chaos.normalization_gamma(i) for i, weight in enumerate(poly_weights))
                          - chaos.normalization_gamma(0) * (poly_weights[0]) ** 2,
                          solution_shape)

    def poly_approximation(y):
        return np.reshape(sum(w * p(y) for w, p in zip(poly_weights, basis)), solution_shape)

    return splitting_xs, splitting_xs_mesh, expectancy, variance, quad_points, len(basis)
