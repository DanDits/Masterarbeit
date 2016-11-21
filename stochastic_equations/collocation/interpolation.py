# TODO extend to multidim case, how to get stochastic polynomial basis? how to build tensor product of 1d interpolations?
# TODO use sparse grids for higher dimensional case

import numpy as np
from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from polynomial_chaos.poly_chaos_distributions import legendreChaos, hermiteChaos
import polynomial_chaos.poly as p
from numpy.linalg import lstsq


# TODO try interpolation approach by quadrature formula, so find good weights and nodes for gauss quadrature in 1d

def glenshaw_curtis_nodes(size):
    size += 2  # as we do not want first and last point which would be -1 and 1
    return -np.cos(np.pi * ((np.array(range(2, size)) - 1) / (size - 1)))


def matrix_inversion_expectancy(trial, max_poly_degree, random_space_nodes_count, spatial_domain, grid_size,
                                start_time, stop_time, delta_time):
    # TODO check out jblevins.org on "Numerical Quadrature Rules for Commmon Distributions"
    # TODO also support exponential distribution
        # TODO so far the convergence for higher poly degree reverses itself for every nodes we tried so far
    #TODO this effect occurs faster for gaussian (trial2_1 about degree 20-30), later for uniform (trial3 degree 65)
    if trial.variable_distributions[0].name == "Gaussian":
        # nodes = glenshaw_curtis_nodes(random_space_nodes_count)
        # nodes /= 1 - nodes ** 2  # variable transformation of the nodes in (-1, 1) to interval (-Inf, Inf)

        # Taken from http://keisan.casio.com/exec/system/1281195844 the zeros of the hermite polynomial H_30
        nodes = [-6.863345293529891581061,
                 -6.138279220123934620395,
                 -5.533147151567495725118,
                 -4.988918968589943944486,
                 -4.483055357092518341887,
                 -4.003908603861228815228,
                 -3.544443873155349886925,
                 -3.099970529586441748689,
                 -2.667132124535617200571,
                 -2.243391467761504072473,
                 -1.826741143603688038836,
                 -1.415527800198188511941,
                 -1.008338271046723461805,
                 -0.6039210586255523077782,
                 -0.2011285765488714855458,
                 0.2011285765488714855458,
                 0.6039210586255523077782,
                 1.008338271046723461805,
                 1.415527800198188511941,
                 1.826741143603688038836,
                 2.243391467761504072473,
                 2.667132124535617200571,
                 3.099970529586441748689,
                 3.544443873155349886925,
                 4.003908603861228815228,
                 4.483055357092518341887,
                 4.988918968589943944486,
                 5.533147151567495725118,
                 6.138279220123934620395,
                 6.863345293529891581061]
        nodes = p.hermite_nodes(random_space_nodes_count)
        chaos = hermiteChaos
    elif trial.variable_distributions[0].name == "Uniform":
        # belongs to uniform distribution in [-1,1] (-> for easy evaluation of expectancy,...)
        nodes = glenshaw_curtis_nodes(random_space_nodes_count)  # in (-1,1)
        """nodes = [-1/3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)),
                 - 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                 0,
                 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                 1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))]"""
        chaos = legendreChaos
    else:
        raise ValueError("Not supported distribution:", trial.variable_distributions[0].name)
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

    # E[w_N]=sum_0^N(weight_k*E[phi_k])=weight_0*E[1*phi_0]=weight_0*E[phi_0*phi_0]*sqrt(gamma_0)=weight_0*sqrt(gamma_0)
    expectancy = np.reshape(weights[0, :], (grid_size,)) * np.sqrt(chaos.normalization_gamma(0))

    return splitting_xs, splitting_xs_mesh, expectancy
