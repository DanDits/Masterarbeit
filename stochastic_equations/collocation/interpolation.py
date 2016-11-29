import numpy as np
from diff_equation.splitting import make_klein_gordon_leapfrog_fast_splitting
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from stochastic_equations.collocation.util import check_distribution_assertions
from numpy.linalg import lstsq
import math
import polynomial_chaos.multivariation as mv

# nodes in interval (-1,1), increasingly dense at boundary
def glenshaw_curtis_nodes(size):
    size += 2  # as we do not want first and last point which would be -1 and 1
    return -np.cos(np.pi * ((np.arange(2, size) - 1) / (size - 1)))


# nodes in interval (-1,1), increasingly dense at boundary. Minimize polynomial prod(x-node_i) in [-1,1]
def chebyshev_nodes(size):
    return np.cos(np.pi * (np.arange(1, size + 1) * 2 - 1) / (2 * size))


def matrix_inversion_expectancy(trial, max_poly_degree, random_space_nodes_counts, spatial_domain, grid_size,
                                start_time, stop_time, delta_time):
    sum_bound = max_poly_degree
    distrs = trial.variable_distributions
    for distr in distrs:
        check_distribution_assertions(distr)
    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)

    nodes_list = chaos.nodes_and_weights(random_space_nodes_counts)[0]
    poly_count = mv.multi_index_bounded_sum_length(len(distrs), sum_bound)
    basis = [chaos.poly_basis(degree) for degree in range(poly_count)]

    solution_at_nodes = []  # used to build right hand side (simultaneously at every grid point)
    splitting_xs = None
    splitting_xs_mesh = None
    solution_shape = None
    for nodes in nodes_list:
        trial.set_random_values(nodes)
        splitting = make_klein_gordon_leapfrog_fast_splitting(spatial_domain, [grid_size], start_time,
                                                              trial.start_position, trial.start_velocity,
                                                              trial.alpha, trial.beta, delta_time)
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
    print("VandA:", vandermonde_A.shape, "rhsu:", rhs_u.shape)
    result = lstsq(vandermonde_A, rhs_u)
    rank = result[2] / min(vandermonde_A.shape)
    print("Vandermonde_A:", vandermonde_A.shape, "u:", rhs_u.shape, "Weights:", result[0].shape)
    weights = result[0]
    # normalize weights as the basis wasn't normalized. use math.sqrt as this "works" aka doesn't crash for big integers
    weights = np.diag([math.sqrt(chaos.normalization_gamma(i)) for i in range(weights.shape[0])]).dot(weights)

    def poly_approximation(y):
        vectorized_basis = np.array([_basis_poly(y) for _basis_poly in basis])
        return np.transpose(weights).dot(vectorized_basis).reshape(solution_shape)

    # E[w_N]=sum_0^N(weight_k*E[phi_k])=weight_0*E[1*phi_0]=weight_0*E[phi_0*phi_0]*sqrt(gamma_0)=weight_0*sqrt(gamma_0)
    expectancy = np.reshape(weights[0, :], solution_shape) * np.sqrt(chaos.normalization_gamma(0))

    # Var[w_N]=E[(w_N)^2]-E[w_N]^2
    variance = np.reshape(np.sum(weights ** 2, axis=0) - (chaos.normalization_gamma(0) * (weights[0, :] ** 2)),
                          solution_shape)
    return splitting_xs, splitting_xs_mesh, expectancy, variance, rank


def matrix_inversion_expectancy_1d(trial, max_poly_degree, random_space_nodes_count, spatial_domain, grid_size,
                                start_time, stop_time, delta_time):
    # if poly degree gets too big, the vandermonde matrix will become singular (with increasingly decreasing rank)
    # and this will make the calculated expectancy become unusable for high degrees
    # For uniform distribution this effect occurs around degree 63, for gaussian around 36, for gamma(2.5) around 22
    distr = trial.variable_distributions[0]
    chaos = get_chaos_by_distribution(distr)
    check_distribution_assertions(distr)

    # clenshaw curtis nodes are pretty good for uniform distribution, but not as good as legendre nodes
    nodes = chaos.nodes_and_weights(random_space_nodes_count)[0]
    # nodes = chebyshev_nodes(random_space_nodes_count)
    # nodes /= 1 - nodes ** 2
    # nodes = glenshaw_curtis_nodes(random_space_nodes_count)

    # could normalize basis first, then the weights are fine after solving linear system
    # basis = [chaos.normalized_basis(degree) for degree in range(max_poly_degree + 1)]
    basis = [chaos.poly_basis(degree) for degree in range(max_poly_degree + 1)]

    # for each node in random space calculate solution u(t,x) in discrete grid at some time T

    # use matrix inversion method to calculate polynomial approximation to random solution u
    solution_at_nodes = []  # used to build right hand side (simultaneously at every grid point)
    splitting_xs = None
    splitting_xs_mesh = None
    for node in nodes:
        trial.set_random_values([node])
        splitting = make_klein_gordon_leapfrog_fast_splitting(spatial_domain, [grid_size], start_time,
                                                              trial.start_position, trial.start_velocity,
                                                              trial.alpha, trial.beta, delta_time)
        splitting.progress(stop_time, delta_time, 0)
        if splitting_xs is None:
            splitting_xs = splitting.get_xs()
            splitting_xs_mesh = splitting.get_xs_mesh()
        solution_at_nodes.append(splitting.solutions()[-1].real)

    rhs_u = np.array(solution_at_nodes)

    vandermonde_A = []
    for node in nodes:
        row = []
        for basis_poly in basis:
            row.append(basis_poly(node))
        vandermonde_A.append(row)
    vandermonde_A = np.array(vandermonde_A)

    # computes the weights which are the factors for representing the random solution by the given basis polynomials
    # each column corresponds to the the factors of a spatial grid point

    result = lstsq(vandermonde_A, rhs_u)
    rank = result[2] / min(vandermonde_A.shape)
    weights = result[0]
    # normalize weights as the basis wasn't normalized. use math.sqrt as this "works" aka doesn't crash for big integers
    weights = np.diag([math.sqrt(chaos.normalization_gamma(i)) for i in range(weights.shape[0])]).dot(weights)
    # weights = np.linalg.solve(vandermonde_A, rhs_u)  # only works for square vandermonde matrix

    def poly_approximation(y):
        vectorized_basis = np.array([_basis_poly(y) for _basis_poly in basis])
        return np.transpose(weights).dot(vectorized_basis)

    # E[w_N]=sum_0^N(weight_k*E[phi_k])=weight_0*E[1*phi_0]=weight_0*E[phi_0*phi_0]*sqrt(gamma_0)=weight_0*sqrt(gamma_0)
    expectancy = np.reshape(weights[0, :], (grid_size,)) * np.sqrt(chaos.normalization_gamma(0))

    # Var[w_N]=E[(w_N)^2]-E[w_N]^2
    variance = np.reshape(np.sum(weights ** 2, axis=0) - (chaos.normalization_gamma(0) * (weights[0, :] ** 2)),
                          (grid_size,))
    return splitting_xs, splitting_xs_mesh, expectancy, variance, rank
