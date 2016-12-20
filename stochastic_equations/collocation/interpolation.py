import numpy as np
from diff_equation.splitting import Splitting
import diff_equation.klein_gordon as kg
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from stochastic_equations.collocation.coll_util import check_distribution_assertions
from numpy.linalg import lstsq
import polynomial_chaos.multivariation as mv
from util.quadrature.helpers import multi_index_bounded_sum_length


def matrix_inversion_expectancy(trial, max_poly_degree, quadrature_method, quadrature_param, spatial_domain, grid_size,
                                start_time, stop_time, delta_time, wave_weight=1.):
    sum_bound = max_poly_degree
    distrs = trial.variable_distributions
    for distr in distrs:
        check_distribution_assertions(distr)
    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)

    chaos.init_quadrature_rule(quadrature_method, quadrature_param)
    nodes_list = chaos.quadrature_rule.get_nodes()

    # for uniform or beta distribution you could also use chebyshev (or slightly worse glenshaw) nodes
    # not always optimal performance, but pretty good
    # nodes_list = [[node] for node in chebyshev_nodes(random_space_nodes_counts[0])]  # if 1d

    poly_count = multi_index_bounded_sum_length(len(distrs), sum_bound)
    basis = [chaos.normalized_basis(degree) for degree in range(poly_count)]

    solution_at_nodes = []  # used to build right hand side (simultaneously at every grid point)
    splitting_xs = None
    splitting_xs_mesh = None
    solution_shape = None
    for nodes in nodes_list:
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
        solution_at_nodes.append(last_solution.real.flatten())

    rhs_u = np.array(solution_at_nodes)

    vandermonde_A = []
    for nodes in nodes_list:
        row = []
        for basis_poly in basis:
            row.append(basis_poly(nodes))
        vandermonde_A.append(row)
    vandermonde_A = np.array(vandermonde_A)

    # computes the weights which are the factors for representing the random solution by the given basis polynomials
    # each column corresponds to the the factors of a spatial grid point
    result = lstsq(vandermonde_A, rhs_u)
    rank = result[2] / min(vandermonde_A.shape)
    print("Vandermonde_A:", vandermonde_A.shape, "u:", rhs_u.shape, "Weights:", result[0].shape)
    print("Condition:", abs(result[3][0] / result[3][-1]))
    weights = result[0]
    # normalize weights if the basis wasn't normalized. use math.sqrt as this "works" aka doesn't crash for big integers
    # weights = np.diag([math.sqrt(chaos.normalization_gamma(i)) for i in range(weights.shape[0])]).dot(weights)

    def poly_approximation(y):
        vectorized_basis = np.array([_basis_poly(y) for _basis_poly in basis])
        return np.transpose(weights).dot(vectorized_basis).reshape(solution_shape)

    # E[w_N]=sum_0^N(weight_k*E[phi_k])=weight_0*E[1*phi_0]=weight_0*E[phi_0*phi_0]*sqrt(gamma_0)=weight_0*sqrt(gamma_0)
    expectancy = np.reshape(weights[0, :], solution_shape) * np.sqrt(chaos.normalization_gamma(0))

    # Var[w_N]=E[(w_N)^2]-E[w_N]^2
    variance = np.reshape(np.sum(weights ** 2, axis=0) - (chaos.normalization_gamma(0) * (weights[0, :] ** 2)),
                          solution_shape)
    return (splitting_xs, splitting_xs_mesh, expectancy, variance, rank,
            poly_count, chaos.quadrature_rule.get_nodes_count())
