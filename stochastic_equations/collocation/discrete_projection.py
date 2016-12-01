import numpy as np

from diff_equation.splitting import make_klein_gordon_leapfrog_fast_splitting
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from stochastic_equations.collocation.util import check_distribution_assertions
import polynomial_chaos.multivariation as mv
from util.analysis import mul_prod


def discrete_projection_expectancy(trial, max_poly_degree, random_space_quadrature_nodes_counts, spatial_domain,
                                   grid_size, start_time, stop_time, delta_time,
                                   expectancy_only=False):
    sum_bound = max_poly_degree
    distrs = trial.variable_distributions
    for distr in distrs:
        check_distribution_assertions(distr)
    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)

    # TODO instead of full tensor product try to use a sparse grid quadrature (see xiu book)
    # the symmetrized minimal node approach does not work for quadrature, only for interpolation
    quad_nodes_list, quad_weights_list = chaos.nodes_and_weights(random_space_quadrature_nodes_counts,
                                                                 use_full_tensor_product=True)
    if expectancy_only:
        poly_count = 1
    else:
        poly_count = mv.multi_index_bounded_sum_length(len(distrs), sum_bound)
    basis = [chaos.poly_basis.polys(degree) for degree in range(poly_count)]

    poly_weights = []
    splitting_xs = None
    splitting_xs_mesh = None
    solution_shape = None
    for i, poly in enumerate(basis):
        print("Projection progress:", i, "/", len(basis))
        curr_poly_weight = 0
        for nodes, weights in zip(quad_nodes_list, quad_weights_list):
            trial.set_random_values(nodes)
            splitting = make_klein_gordon_leapfrog_fast_splitting(spatial_domain, [grid_size], start_time,
                                                                  trial.start_position,
                                                                  trial.start_velocity, trial.alpha, trial.beta,
                                                                  delta_time)
            splitting.progress(stop_time, delta_time, 0)
            last_solution = splitting.solutions()[-1]
            if splitting_xs is None:
                splitting_xs = splitting.get_xs()
                splitting_xs_mesh = splitting.get_xs_mesh()
                solution_shape = last_solution.shape
            curr_poly_weight += poly(nodes) * mul_prod(weights) * last_solution.real.flatten()
        poly_weights.append(curr_poly_weight / chaos.normalization_gamma(i))
    expectancy = np.reshape(np.sqrt(chaos.normalization_gamma(0)) * poly_weights[0], solution_shape)
    variance = np.reshape(sum((weight ** 2) * chaos.normalization_gamma(i) for i, weight in enumerate(poly_weights))
                          - chaos.normalization_gamma(0) * (poly_weights[0]) ** 2,
                          solution_shape)

    def poly_approximation(y):
        return np.reshape(sum(w * p(y) for w, p in zip(poly_weights, basis)), solution_shape)

    return splitting_xs, splitting_xs_mesh, expectancy, variance
