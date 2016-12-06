import numpy as np
from stochastic_equations.stochastic_trial import StochasticTrial
from polynomial_chaos import distributions
import polynomial_chaos.poly_chaos_distributions as pcd
from numpy.linalg import eigh
from scipy.linalg import expm
from diff_equation.solver_config import SolverConfig
from diff_equation.pseudospectral_solver import WaveSolverConfig
from functools import partial

# TODO also possible to not use nquad but a quadrature formula for the specific distribution

# y[0] > 1
left_1, right_1 = 2, 5
trial_1 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: 2 * np.sin(sum(xs)),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: 2 * np.cos(t * ys[0]) * np.sin(sum(xs)),
                          # from U(-1,1) to U(left_1, right_1)
                          random_variables=[lambda y: (right_1 - left_1) / 2 * (y + 1) + left_1],
                          name="Trial1") \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    "expectancy", lambda xs, t: (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                 * (np.sin(right_1 * t) - np.sin(left_1 * t))),
                    "variance", lambda xs, t: (1 / (t * (right_1 - left_1)) * np.sin(sum(xs)) ** 2
                                               * (2 * t * right_1 + np.sin(2 * t * right_1)
                                                  - 2 * t * left_1 - np.sin(2 * t * left_1))
                                               - (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                  * (np.sin(right_1 * t) - np.sin(left_1 * t))) ** 2))

# y[0] in (0,1), is enforced by random variable which can take any real value!
trial_2_1 = StochasticTrial([distributions.gaussian],
                            lambda xs, ys: np.zeros(shape=sum(xs).shape),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_1")
trial_2_1.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])

trial = trial_2_1
trial.flag_raw_attributes = True
N = 10

# calculate symmetric matrix A where a_ik=E[alpha(y)phi_i(y)phi_k(y)]
chaos = pcd.get_chaos_by_distribution(trial.variable_distributions[0])
basis = [chaos.normalized_basis(i) for i in range(N + 1)]


def calculate_expectancy_matrix_sym(to_expect):  # to_expect symmetric in i and k!
    matrix = []
    for i in range(N + 1):
        row = [0] * i  # fill start of row with zeros as we do not want to calculate symmetric entries twice
        # first only calculate upper triangle
        for k in range(i, N + 1):
            row.append(trial.calculate_expectancy_simple(partial(to_expect, i=i, k=k)))
        matrix.append(row)
    matrix = np.array(matrix)
    matrix += np.triu(matrix, 1).transpose()  # make upper triangle matrix symmetric
    return matrix

# one could use the projected alpha and beta(x) coefficients but
# if they are no polynomials and the degree is low this leads to negative eigenvalues
# projected_alpha_coefficients = [trial.calculate_expectancy_simple(lambda ys: (trial.alpha(trial.transform_values(ys))
#                                                                               * basis[i](ys)))
#                                for i in range(N + 1)]
matrix_a = calculate_expectancy_matrix_sym(lambda ys, i, k: (trial.alpha(trial.transform_values(ys))
                                                             * basis[i](ys) * basis[k](ys)))

# diagonalize A=SDS'
diag, transform_s = eigh(matrix_a)  # now holds A=S*D*S', S is orthonormal, D a diagonal matrix
# eigenvalues in D are positive and bounded by the extrema of alpha(y)
print("Eigenvalues:", diag)
print("Wavespeeds:", np.sqrt(diag))


# calculate symmetric matrix B(x) where b_ik(x)=E[beta(x,y)phi_i(y)phi_k(y)]
def matrix_b(xs):
    return calculate_expectancy_matrix_sym(lambda ys, i, k: (trial.beta(xs, trial.transform_values(ys))
                                                             * basis[i](ys) * basis[k](ys)))

# TODO test MultiWave solver and make sure the input/output formats fit together
# TODO then implement Lie splitting (therefore transform initial values by leftmultiplying S'
# TODO calculate initial values as projection coefficients of initial values
class MultiWaveSolver(SolverConfig):
    def __init__(self, intervals, grid_points_list, wave_speeds):
        super().__init__(intervals, grid_points_list)
        self.wave_speeds = wave_speeds
        self.configs = [WaveSolverConfig(intervals, grid_points_list, wave_speed) for wave_speed in wave_speeds]

    # u0s are expected to be a list of (N+1) u0 vectors of length 'grid_size', each one corresponding to in index k
    # same for u0ts; t0 is a float for the start time
    def init_solver(self, t0, u0s, u0ts):
        assert len(u0s) == len(self.configs)
        assert len(u0ts) == len(self.configs)
        self.init_initial_values(t0, u0s, u0ts)
        for config, u0, u0t in zip(self.configs, u0s, u0ts):
            config.init_solver(t0, u0, u0t)
        # each individual config solver returns a list of vectors of length 'grid_size'. We pack the lists all in a list
        # then we transpose this construct using list(zip(*...))
        # then we get a list whose first entry contains a list of positions, the second a list of velocities,...
        self.solver = lambda time: list(zip(*[curr_config.solver(time) for curr_config in self.configs]))

def test_multilinear():
    B = lambda xs: np.array([[3, 0], [0, 2]])
    S = np.eye(2)
    t0 = 0
    u0s = [np.array([-1, 1, 2]), np.array([3, 3, 4])]
    u0ts = [np.array([0, 2, 1]), np.array([2, 0, 1])]
    config = MultiLinearOdeSolver([(-np.pi, np.pi)], [3], B, S)
    config.init_solver(t0, u0s, u0ts)
    config.solve([1])
    # Should be [[array([ 0.16055654,  0.97916366,  0.24874702]), array([ 1.86474308,  0.46783108,  1.32223078])]]
    print("Result:", config.solutions())


# solves w_tt=-B(x)Sw for each x in grid using matrix exponential
class MultiLinearOdeSolver(SolverConfig):
    def __init__(self, intervals, grid_points_list, function_b, matrix_s):
        super().__init__(intervals, grid_points_list)
        self.function_b = function_b
        self.matrix_s = matrix_s

    # input format see MultiWaveSolver: u0s a list of u0 vectors
    def init_solver(self, t0, u0s, u0ts):
        self.init_initial_values(t0, u0s, u0ts)

        # Construct a matrix [u0_0 u0_1   ..u0_grid_size ] (one row per element in u0s)
        #                    [ ...                       ]
        #                    [u0t_0 u0t_1 ..u0t_grid_size] (one row per element in u0ts)
        #                    [ ...                       ]
        starting_value_matrix = np.array(u0s + u0ts)
        print("SVM:", starting_value_matrix)
        w_length = len(u0s)

        # TODO currently only for one spatial dimension
        def solution_at(time):
            positions, velocities = [], []
            # as the matrix b is dependent on x, we have to solve it for every x individually
            for i, x in enumerate(self.xs[0]):
                z = np.zeros((2 * w_length,) * 2)
                print("Shapes:", z.shape, w_length, self.function_b([x]).shape)
                z[:w_length, w_length:] = np.eye(w_length)
                z[w_length:, :w_length] = -self.function_b([x]) * self.matrix_s
                solution = expm(z * (time - self.start_time)).dot(starting_value_matrix[:, i])
                print("Solution", i, "=", solution, "SVM at i=", starting_value_matrix[:, i])
                positions.append(solution[:w_length])
                velocities.append(solution[w_length:])
            # positions currently a list of length 'grid_size' with each entry a vector of length 'N+1'
            # we want the transposed of this and also to map the tuple to a vector
            positions = list(map(np.array, zip(*positions)))
            velocities = list(map(np.array, zip(*velocities)))
            return positions, velocities
        self.solver = solution_at

test_multilinear()