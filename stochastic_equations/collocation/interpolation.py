# TODO extend to multidim case, how to get stochastic polynomial basis? how to build tensor product of 1d interpolations?
# TODO use sparse grids for higher dimensional case

import numpy as np
from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from polynomial_chaos.poly_chaos_distributions import legendreChaos, hermiteChaos
from numpy.linalg import lstsq
from polynomial_chaos.distributions import inverse_gaussian

# TODO try interpolation approach by quadrature formula, so find good weights and nodes for gauss quadrature in 1d

def glenshaw_curtis_nodes(size):
    size += 2  # as we do not want first and last point which would be -1 and 1
    return -np.cos(np.pi * ((np.array(range(2, size)) - 1) / (size - 1)))


def matrix_inversion_expectancy(trial, max_poly_degree, random_space_nodes_count, spatial_domain, grid_size,
                         start_time, stop_time, delta_time):
    if trial.variable_distributions[0].name == "Gaussian":
        #uniform_simple = np.linspace(1. / (random_space_nodes_count + 1), 1,
        #                             endpoint=False, num=random_space_nodes_count)
        #nodes = np.vectorize(inverse_gaussian)(uniform_simple)
        chaos = hermiteChaos
    else:
        chaos = legendreChaos
    # TODO hermite chaos seems to bring worse results even for gaussian distributions?
    chaos = legendreChaos  # belongs to uniform distribution in [-1,1] (-> for easy evaluation of expectancy,...)
    nodes = glenshaw_curtis_nodes(random_space_nodes_count)  # in [-1,1]
    print("Distr:", trial.variable_distributions[0].name,
          "Polynomials:", chaos.poly_name, ", Nodes:", nodes)
    basis = [chaos.normalized_basis(degree) for degree in range(max_poly_degree + 1)]
    # for each node in random space calculate solution u(t,x) in discrete grid at some time T

    # use matrix inversion method to calculate polynomial approximation to random solution u
    solution_at_nodes = []  # used to build right hand side (simultaneously at every grid point)
    splitting_xs = None
    splitting_xs_mesh = None
    for node in nodes:
        trial.set_random_values([node])
        splitting = make_klein_gordon_leapfrog_splitting(spatial_domain, [grid_size], start_time, trial.start_position,
                                                         trial.start_velocity, trial.alpha, trial.beta)
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

    weights = lstsq(vandermonde_A, rhs_u)[0]
    # weights = np.linalg.solve(vandermonde_A, rhs_u)  # only works for square vandermonde matrix

    def poly_approximation(y):
        vectorized_basis = np.array([_basis_poly(y) for _basis_poly in basis])
        return np.transpose(weights).dot(vectorized_basis)

    expectancy = np.reshape(weights[0, :], (grid_size,)) / np.sqrt(chaos.normalization_gamma(0))
    #expectancy = np.zeros((grid_size,))
    #for i in range(max_poly_degree):
    #    expectancy += np.reshape(weights[i, :], (grid_size,)) / np.sqrt(chaos.normalization_gamma(i))

    return splitting_xs, splitting_xs_mesh, expectancy


