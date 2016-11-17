# TODO first 1D random space, use matrix inversion approach, gridpoints clenshaw curtis nodes
# TODO extend to multidim case, how to get stochastic polynomial basis? how to build tensor product of 1d interpolations?
# TODO how to obtain expectancy/ variance from this interpolation?
# TODO use sparse grids for higher dimensional case

import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
import polynomial_chaos.distributions as distributions
from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from polynomial_chaos.poly_chaos_distributions import legendreChaos

from stochastic_equations.stochastic_trial import StochasticTrial
from numpy.linalg import lstsq


def glenshaw_curtis_nodes(size):
    return -np.cos(np.pi * ((np.array(range(1, size - 1)) - 1) / (size - 1)))


left_3, right_3 = 2.5, 3  # y[0] bigger than 2
trial = StochasticTrial([distributions.make_uniform(-1, 1)],  # y[0] bigger than 2 enforced by random variable
                        lambda xs, ys: 1 / (np.sin(sum(xs)) + ys[0]),
                        lambda xs, ys: np.zeros(shape=sum(xs).shape),
                        lambda xs, t, ys: np.cos(t) / (np.sin(sum(xs)) + ys[0]),
                        # from U(-1,1) to U(left_3, right_3)
                        random_variables=[lambda y: (right_3 - left_3) / 2 * (y + 1) + left_3]) \
    .add_parameters("beta", lambda xs, ys: 1 + (ys[0] - 2) * (np.sin(sum(xs)) / (np.sin(sum(xs)) + ys[0])
                                                              + 2 * np.cos(sum(xs)) ** 2
                                                              / (np.sin(sum(xs)) + ys[0]) ** 2),
                    "alpha", lambda ys: ys[0] - 2,
                    "expectancy", lambda xs, t: np.cos(t) / (right_3 - left_3)
                                                * (np.log(np.sin(sum(xs)) + right_3)
                                                   - np.log(np.sin(sum(xs)) + left_3)))

N = 5  # maximum degree of the polynomial, so N+1 polynomials
M = 6  # number of nodes in random space, >= N+1
spatial_dimension = 1
grid_size = 128
spatial_domain = list(repeat([-np.pi, np.pi], spatial_dimension))
start_time = 0
stop_time = 0.5
delta_time = 0.001

nodes = glenshaw_curtis_nodes(M)  # in [-1,1]
chaos = legendreChaos  # belongs to uniform distribution in [-1,1] (-> for easy evaluation of expectancy,...)
basis = [chaos.poly_basis(degree) for degree in range(N + 1)]
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
    solution_at_nodes.append(splitting.solutions()[-1])

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
weights = lstsq(vandermonde_A, rhs_u)


def poly_approximation(y):
    vectorized_basis = np.array([_basis_poly(y) for _basis_poly in basis])
    return np.transpose(weights).dot(vectorized_basis)


def expectancy():
    return weights[0, :] * chaos.normalization_gamma(0)


plt.figure()
plt.plot(splitting_xs[0], expectancy(), "o", label="Expectancy by interpolation")
plt.plot(splitting_xs[0], trial.expectancy(splitting_xs_mesh, stop_time), "x", label="Expectancy of trial")
plt.legend()
plt.show()
