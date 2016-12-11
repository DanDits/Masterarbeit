import numpy as np
from stochastic_equations.stochastic_trial import StochasticTrial
from polynomial_chaos import distributions
import polynomial_chaos.poly_chaos_distributions as pcd
from numpy.linalg import eigh
from scipy.linalg import expm
from diff_equation.solver_config import SolverConfig
from diff_equation.pseudospectral_solver import WaveSolverConfig
from functools import partial
from diff_equation.splitting import Splitting

param_g1 = 2
alpha_1 = 1
assert param_g1 ** 2 - alpha_1 > 1E-7
trial_0 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, ys: param_g1 * np.cos(sum(xs)),
                          lambda xs, t, ys: np.sin(sum(xs) + param_g1 * t),
                          name="Trial0") \
    .add_parameters("beta", lambda xs, ys: param_g1 ** 2 - alpha_1,
                    "alpha", lambda ys: alpha_1)
# y[0] > 1
left_1, right_1 = 2., 5.
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

trial_5 = StochasticTrial([distributions.gaussian],
                          lambda xs, ys: np.cos(sum(xs)),
                          lambda xs, ys: np.sin(sum([x ** 2 for x in xs])),
                          name="Trial5") \
    .add_parameters("beta", lambda xs, ys: 3 + np.sin(xs[0] * ys[0]) + np.sin(xs[0] + ys[0]),
                    "alpha", lambda ys: 1 + np.exp(ys[0]),
                    "expectancy_data", "../../data/qmc_100000, Trial5, 0.5, 128.npy")


# TODO try other harder trials (like mc5), also try to calculate variance


# calculates 1d expectancy using quadrature rule defined by chaos' poly basis
def calculate_expectancy_fast(function, quadrature_nodes, quadrature_weights):
    return np.inner(quadrature_weights[:, 0], function(quadrature_nodes)[:, 0])


def calculate_function_expectancy(function, expectancy_params):
    return calculate_expectancy_fast(function, expectancy_params[0], expectancy_params[1])


def get_evaluated_poly(cache, basis, i, expectancy_params):
    quadrature_nodes = expectancy_params[0]
    if len(quadrature_nodes) != cache.get("quadrature_nodes_count", 0):
        cache.clear()
        cache["quadrature_nodes_count"] = len(quadrature_nodes)
    cached_evaluated_polys = cache.get("evaluated_polys", [])
    if len(cached_evaluated_polys) <= i:
        for j in range(len(cached_evaluated_polys), i + 1):
            cached_evaluated_polys.append(np.apply_along_axis(basis[i], 1, quadrature_nodes))
    return cached_evaluated_polys[i]


# max_poly_degree is also called N in comments in order to conform with literature and because it is shorter
def galerkin_expectancy(trial, max_poly_degree, domain, grid_size, start_time, stop_time, delta_time,
                        quadrature_nodes_count, wave_weight):
    trial.flag_raw_attributes = True

    # calculate symmetric matrix A where a_ik=E[alpha(y)phi_i(y)phi_k(y)]
    chaos = pcd.get_chaos_by_distribution(trial.variable_distributions[0])
    basis = [chaos.normalized_basis(i) for i in range(max_poly_degree + 1)]

    nodes, weights = chaos.nodes_and_weights(quadrature_nodes_count)
    nodes = nodes.reshape((nodes.shape[0], 1))  # to allow for future multivariate nodes and weights
    weights = weights.reshape((weights.shape[0], 1))
    expectancy_params = (nodes, weights)
    cache = {}
    wave_speeds, matrix_s = calculate_wave_speed_transform(trial, basis, max_poly_degree, expectancy_params, cache)
    function_b = lambda xs: matrix_b(trial, xs, basis, max_poly_degree, expectancy_params, cache)

    splitting = make_splitting(domain, grid_size, wave_speeds,
                               basis, function_b, matrix_s, start_time, trial, expectancy_params, cache, delta_time,
                               wave_weight)
    xs = splitting.solver_configs[0].xs

    exp = None
    if trial.has_parameter("expectancy"):
        exp = trial.expectancy(xs, stop_time)
    elif trial.raw_reference is not None:
        exp = trial.calculate_expectancy(xs, stop_time, trial.raw_reference)
    elif trial.has_parameter("expectancy_data"):
        try:
            exp = np.load(trial.expectancy_data)
        except FileNotFoundError:
            print("No expectancy data found, should be here!?")
    return (xs,
            calculate_expectancy(splitting, chaos.normalization_gamma, start_time, stop_time, delta_time),
            exp)


def calculate_expectancy_matrix_sym(expectancy_params, to_expect, max_poly_degree):  # to_expect symmetric in i and k!
    matrix = []
    for i in range(max_poly_degree + 1):
        row = [0] * i  # fill start of row with zeros as we do not want to calculate symmetric entries twice
        # first only calculate upper triangle
        for k in range(i, max_poly_degree + 1):
            exp_value = calculate_function_expectancy(partial(to_expect, i=i, k=k), expectancy_params)
            row.append(exp_value)
        matrix.append(row)
    matrix = np.array(matrix)
    matrix += np.triu(matrix, 1).transpose()  # make upper triangle matrix symmetric
    return matrix


def calculate_wave_speed_transform(trial, basis, max_poly_degree, expectancy_params, cache):
    # one could use the projected alpha and beta(x) coefficients but
    # if they are no polynomials and the degree is low this leads to negative eigenvalues
    def transformed_alpha(ys):
        return [trial.alpha(trial.transform_values(ys))]

    def alpha_expectancy_func(all_ys, i, k):
        res1 = np.apply_along_axis(transformed_alpha, 1, all_ys)
        res2 = get_evaluated_poly(cache, basis, i, expectancy_params)
        res3 = get_evaluated_poly(cache, basis, k, expectancy_params)
        return res1 * res2 * res3

    matrix_a = calculate_expectancy_matrix_sym(expectancy_params, alpha_expectancy_func, max_poly_degree)

    # diagonalize A=SDS'
    diag, transform_s = eigh(matrix_a)  # now holds A=S*D*S', S is orthonormal, D a diagonal matrix
    # eigenvalues in D are positive and bounded by the extrema of alpha(y)
    wave_speeds = np.sqrt(diag)
    # print("DIag:", diag, "Transform:", transform_s)
    # print("Should be zero matrix:", transform_s.dot(np.diag(diag)).dot(transform_s.transpose()) - matrix_a)
    return wave_speeds, transform_s


# calculate symmetric matrix B(x) where b_ik(x)=E[beta(x,y)phi_i(y)phi_k(y)]
def matrix_b(trial, xs, basis, max_poly_degree, expectancy_params, cache):
    def transformed_beta(ys):
        return [trial.beta(xs, trial.transform_values(ys))]

    return calculate_expectancy_matrix_sym(expectancy_params,
                                           lambda all_ys, i, k: (np.apply_along_axis(transformed_beta, 1, all_ys)
                                                                 * get_evaluated_poly(cache, basis, i,
                                                                                      expectancy_params)
                                                                 * get_evaluated_poly(cache, basis, k,
                                                                                      expectancy_params)),
                                           max_poly_degree)


class MultiWaveSolver(SolverConfig):
    def __init__(self, intervals, grid_points_list, wave_speeds, splitting_factor):
        super().__init__(intervals, grid_points_list)
        print("Got wave_speeds:", wave_speeds)
        self.wave_speeds = wave_speeds
        self.splitting_factor = splitting_factor
        self.configs = [WaveSolverConfig(intervals, grid_points_list, wave_speed, splitting_factor)
                        for wave_speed in self.wave_speeds]

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


# solves w_tt=-B(x)Sw for each x in grid using matrix exponential
class MultiLinearOdeSolver(SolverConfig):
    def __init__(self, intervals, grid_points_list, function_b, matrix_s, splitting_factor):
        super().__init__(intervals, grid_points_list)
        self.function_b = function_b
        self.splitting_factor = splitting_factor
        self.matrix_s = matrix_s
        self.w_length = matrix_s.shape[1]
        self.base_matrix_z = np.zeros((2 * self.w_length,) * 2)
        self.base_matrix_z[:self.w_length, self.w_length:] = np.eye(self.w_length) * self.splitting_factor
        self.matrix_z_for_xs = []
        negative_transposed_s = -self.matrix_s.transpose()
        for x in self.xs[0]:
            current_z = np.copy(self.base_matrix_z)
            current_z[self.w_length:, :self.w_length] = negative_transposed_s.dot(self.function_b([x])
                                                                                  .dot(self.matrix_s))
            self.matrix_z_for_xs.append(current_z)
        self.last_delta_time = 0
        self.matrix_expm_z_dt_for_xs = None

    def calculate_matrix_exponentials(self, time_step_size):
        self.last_delta_time = time_step_size
        self.matrix_expm_z_dt_for_xs = [expm(current_z * time_step_size) for current_z in self.matrix_z_for_xs]

    # input format see MultiWaveSolver: u0s a list of u0 vectors
    def init_solver(self, t0, u0s, u0ts):
        self.init_initial_values(t0, u0s, u0ts)

        # Construct a matrix [u0_0 u0_1   ..u0_grid_size ] (one row per element in u0s)
        #                    [ ...                       ]
        #                    [u0t_0 u0t_1 ..u0t_grid_size] (one row per element in u0ts)
        #                    [ ...                       ]
        starting_value_matrix = np.array(u0s + u0ts)

        # TODO currently only for one spatial dimension (like lots of other code for galerkin...)
        def solution_at(time):
            positions, velocities = [], []
            # as the matrix b is dependent on x, we have to solve it for every x individually
            if self.is_new_delta_time(time - self.start_time):
                self.calculate_matrix_exponentials(time - self.start_time)
            for i, matrix_exp in enumerate(self.matrix_expm_z_dt_for_xs):
                solution = matrix_exp.dot(starting_value_matrix[:, i])
                positions.append(solution[:self.w_length])
                velocities.append(solution[self.w_length:])
            # positions currently a list of length 'grid_size' with each entry a vector of length 'N+1'
            # we want the transposed of this and also to map the tuple to a vector
            positions = list(map(np.array, zip(*positions)))
            velocities = list(map(np.array, zip(*velocities)))
            return positions, velocities

        self.solver = solution_at


def get_projection_coefficients(function, project_trial, basis, expectancy_params, cache):
    def transformed_function(ys):
        return [function(project_trial.transform_values(ys))]

    return [calculate_function_expectancy(lambda all_ys: (np.apply_along_axis(transformed_function, 1, all_ys)
                                                          * get_evaluated_poly(cache, basis, i, expectancy_params)),
                                          expectancy_params)
            for i in range(len(basis))]


def get_starting_value_coefficients(xs_mesh, starting_value_func, project_trial, matrix_s_transposed,
                                    basis, expectancy_params, cache):
    # TODO only for one spatial dimension currently
    coefficients = []
    for x in xs_mesh[0]:
        current_func = lambda ys: starting_value_func([x], ys)
        # if starting_value_func does not depend on ys, then only the first coeff will be nonzero
        coeff = get_projection_coefficients(current_func, project_trial, basis, expectancy_params, cache)

        coeffs = matrix_s_transposed.dot(coeff)
        coefficients.append(coeffs)
    return list(map(np.array, zip(*coefficients)))  # transpose and convert to vectors, we want a list of length N+1


def make_splitting(domain, grid_size, wave_speeds, basis, function_b, matrix_s, start_time, trial,
                   expectancy_params, cache, delta_time, wave_weight=0.5):
    multi_wave_config = MultiWaveSolver(domain, [grid_size], wave_speeds, wave_weight)
    multi_ode_config = MultiLinearOdeSolver(domain, [grid_size], function_b, matrix_s, 1. - wave_weight)
    transposed_s = matrix_s.transpose()
    start_positions = get_starting_value_coefficients(multi_wave_config.xs_mesh, trial.start_position,
                                                      trial, transposed_s, basis, expectancy_params, cache)
    start_velocities = get_starting_value_coefficients(multi_wave_config.xs_mesh, trial.start_velocity,
                                                       trial, transposed_s, basis, expectancy_params, cache)
    splitting = Splitting.make_fast_strang(multi_wave_config, multi_ode_config, "FastStrangGalerkin",
                                           start_time, start_positions, start_velocities, delta_time)
    splitting.matrix_s = matrix_s
    return splitting


def calculate_expectancy(splitting, normalization_gamma, start_time, stop_time, delta_time):
    splitting.progress(stop_time, delta_time, 0)
    last_coefficients = splitting.solutions()[-1]
    # this is now a list of length N+1, one vector for each index k, each of length 'grid_size'
    # first we need to transpose this construct in order to be able to retransform the coefficients with the matrix S
    grid_coeffs = list(map(np.array, zip(*last_coefficients)))  # transpose
    grid_coeffs = [splitting.matrix_s.dot(curr) for curr in grid_coeffs]
    # for expectancy we only need to use the zeroth coefficient
    # in literature this sqrt(gamma(0)) is often forgotten, but this is mainly because it is 1 for most(all?) chaos'
    expectancy = np.sqrt(normalization_gamma(0)) * np.array([coeffs[0] for coeffs in grid_coeffs])
    return expectancy


def test(plot=True):
    from util.analysis import error_l2
    domain = [(-np.pi, np.pi)]
    grid_size = 128
    start_time, stop_time, delta_time = 0., 0.5, 0.001
    max_poly_degree = 15
    # trial 1, Q=50, D=15,dt=0.0001->error=7.5E-10  # perfect order 10 convergence
    # trial 1, Q=50, D=5,dt=0.0001->error=7.5E-10  # perfect order 10 convergence
    # trial 1, Q=50, D=3,dt=0.0001->error=5.8E-9, for dt<=0.001 perfect order 10 convergence
    # trial 1, Q=50, D=1, independent of dt, wont get better than 2.4E-4
    wave_weight = 0.5  # does not seem to have much influence (at least on trial5); but can have on stability as this problem is kinda irregular!
    quadrature_nodes_counts = [50]
    errors = []
    trial = trial_1  # trial_5 saved expectancy is not bette than 1.4E-4
    for quadrature_nodes_count in quadrature_nodes_counts:
        xs, exp, trial_exp = galerkin_expectancy(trial, max_poly_degree, domain, grid_size,
                                                 start_time, stop_time, delta_time, quadrature_nodes_count,
                                                 wave_weight)

        error = error_l2(exp, trial_exp)
        errors.append(error)
        print("Degree={}, Q nodes={}, rel. error={}".format(max_poly_degree, quadrature_nodes_count, error))
    print("Relative errors for {}".format(trial.name), errors)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Galerkin Error to expectancy for {}, PolyDegree={}, T={}, dt={}, grid={}, wave_weight={}"
                  .format(trial.name, max_poly_degree, stop_time, delta_time, grid_size, wave_weight))
        plt.xlabel("Quadrature nodes count")
        plt.ylabel("discrete l2 error")
        plt.yscale('log')
        plt.plot(quadrature_nodes_counts, errors, "o")
        plt.show()

        plt.figure()
        plt.plot(xs[0], exp, label="Calculated exp")
        plt.plot(xs[0], trial_exp, label="Exact exp")
        plt.legend()
        plt.show()


# TODO maybe instead of using beta(x,y) split equation further into spectral coefficients sum beta'_i(x)phi_i(y)
# TODO and use another splitting method for this (chained sum of strang splittings, always group 2)


import tests.profile_execute as profile
profile.profile_function(partial(test, False), 'galerkin.dump')
#test()
