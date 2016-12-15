import numpy as np
from numpy.linalg import eigh
from scipy.linalg import expm
from diff_equation.solver_config import SolverConfig
from diff_equation.pseudospectral_solver import WaveSolverConfig
from functools import partial
from diff_equation.splitting import Splitting
import polynomial_chaos.multivariation as mv
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from util.quadrature.helpers import multi_index_bounded_sum_length
from util.quadrature.rules import QuadratureRule


# TODO currently only for one spatial dimension

def get_evaluated_poly(cache, basis, i, chaos):
    rule = chaos.quadrature_rule  # type: QuadratureRule
    if (rule.get_nodes_count() != cache.get("quadrature_nodes_count", 0)
        or chaos.poly_basis.name != cache.get("poly_basis_name", "")):
        cache.clear()
        cache["quadrature_nodes_count"] = rule.get_nodes_count()
        cache["poly_basis_name"] = chaos.poly_basis.name
        print("CLEARING CACHE!")
    cached_evaluated_polys = cache.get("evaluated_polys")
    if cached_evaluated_polys is None:
        cached_evaluated_polys = []
        cache["evaluated_polys"] = cached_evaluated_polys
    if len(cached_evaluated_polys) <= i:
        quadrature_nodes = rule.get_nodes()
        for j in range(len(cached_evaluated_polys), i + 1):
            print("NEW IN CACHE:", j)
            cached_evaluated_polys.append(np.apply_along_axis(basis[i], 1, quadrature_nodes))
    return cached_evaluated_polys[i]


# max_poly_degree is also called N in comments in order to conform with literature and because it is shorter
def galerkin_expectancy(trial, max_poly_degree, domain, grid_size, start_time, stop_time, delta_time, wave_weight,
                        quadrature_method, quadrature_param, cache=None):
    trial.flag_raw_attributes = True
    sum_bound = max_poly_degree
    distrs = trial.variable_distributions
    # calculate symmetric matrix A where a_ik=E[alpha(y)phi_i(y)phi_k(y)]
    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)
    chaos.init_quadrature_rule(quadrature_method, quadrature_param)

    poly_count = multi_index_bounded_sum_length(len(distrs), sum_bound)
    basis = [chaos.normalized_basis(i) for i in range(poly_count)]
    if cache is None:
        cache = {}
    wave_speeds, matrix_s = calculate_wave_speed_transform(trial, basis, poly_count, chaos, cache)
    function_b = lambda xs: matrix_b(trial, xs, basis, poly_count, chaos, cache)

    splitting = make_splitting(domain, grid_size, wave_speeds,
                               basis, function_b, matrix_s, start_time, trial, chaos, cache, delta_time,
                               wave_weight)
    xs = splitting.solver_configs[0].xs
    xs_mesh = splitting.solver_configs[0].xs_mesh
    return (xs, xs_mesh,
            calculate_expectancy(splitting, chaos.normalization_gamma, stop_time, delta_time))


def calculate_expectancy_matrix_sym(chaos, to_expect, poly_count):  # to_expect symmetric in i and k!
    matrix = []

    for i in range(poly_count):
        row = [0] * i  # fill start of row with zeros as we do not want to calculate symmetric entries twice
        # first only calculate upper triangle
        for k in range(i, poly_count):
            exp_value = chaos.integrate(partial(to_expect, i=i, k=k), function_parameter_is_nodes_matrix=True)
            row.append(exp_value)
        matrix.append(row)
    matrix = np.array(matrix)
    matrix += np.triu(matrix, 1).transpose()  # make upper triangle matrix symmetric
    return matrix


def calculate_wave_speed_transform(trial, basis, poly_count, chaos, cache):
    # one could use the projected alpha and beta(x) coefficients but
    # if they are no polynomials and the degree is low this leads to negative eigenvalues

    def alpha_expectancy_func(nodes_matrix, i, k):
        res1 = np.apply_along_axis(lambda ys: trial.alpha(trial.transform_values(ys)), 1, nodes_matrix)
        res2 = get_evaluated_poly(cache, basis, i, chaos)
        res3 = get_evaluated_poly(cache, basis, k, chaos)
        return res1 * res2 * res3

    matrix_a = calculate_expectancy_matrix_sym(chaos, alpha_expectancy_func, poly_count)

    # diagonalize A=SDS'
    diag, transform_s = eigh(matrix_a)  # now holds A=S*D*S', S is orthonormal, D a diagonal matrix
    # eigenvalues in D are positive and bounded by the extrema of alpha(y)
    wave_speeds = np.sqrt(diag)
    # print("DIag:", diag, "Transform:", transform_s)
    # print("Should be zero matrix:", transform_s.dot(np.diag(diag)).dot(transform_s.transpose()) - matrix_a)
    return wave_speeds, transform_s


# calculate symmetric matrix B(x) where b_ik(x)=E[beta(x,y)phi_i(y)phi_k(y)]
def matrix_b(trial, xs, basis, poly_count, chaos, cache):

    def beta_expectancy_func(nodes_matrix, i, k):
        res1 = np.apply_along_axis(lambda ys: trial.beta(xs, trial.transform_values(ys)), 1, nodes_matrix)
        res2 = get_evaluated_poly(cache, basis, i, chaos)
        res3 = get_evaluated_poly(cache, basis, k, chaos)
        return res1 * res2 * res3
    return calculate_expectancy_matrix_sym(chaos, beta_expectancy_func, poly_count)


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


def get_projection_coefficients(function, project_trial, basis, chaos, cache):
    def vectorized_func(ys):
        vec_result = function(project_trial.transform_values(ys))
        return vec_result

    def integrate_func(nodes_matrix, i):
        res1 = np.apply_along_axis(vectorized_func, 1, nodes_matrix)
        res2 = get_evaluated_poly(cache, basis, i, chaos)
        return res1 * res2
    return [chaos.integrate(partial(integrate_func, i=i), function_parameter_is_nodes_matrix=True)
            for i in range(len(basis))]


def get_starting_value_coefficients(xs_mesh, starting_value_func, project_trial, matrix_s_transposed,
                                    basis, chaos, cache):
    coefficients = []
    for x in xs_mesh[0]:
        def current_func(ys):
            return starting_value_func([x], ys)
        # if starting_value_func does not depend on ys, then only the first coeff will be nonzero
        coeff = get_projection_coefficients(current_func, project_trial, basis, chaos, cache)
        coeffs = matrix_s_transposed.dot(coeff)
        coefficients.append(coeffs)
    return list(map(np.array, zip(*coefficients)))  # transpose and convert to vectors, we want a list of length N+1


def make_splitting(domain, grid_size, wave_speeds, basis, function_b, matrix_s, start_time, trial,
                   chaos, cache, delta_time, wave_weight=0.5):
    multi_wave_config = MultiWaveSolver(domain, [grid_size], wave_speeds, wave_weight)
    multi_ode_config = MultiLinearOdeSolver(domain, [grid_size], function_b, matrix_s, 1. - wave_weight)
    transposed_s = matrix_s.transpose()
    start_positions = get_starting_value_coefficients(multi_wave_config.xs_mesh, trial.start_position,
                                                      trial, transposed_s, basis, chaos, cache)

    start_velocities = get_starting_value_coefficients(multi_wave_config.xs_mesh, trial.start_velocity,
                                                       trial, transposed_s, basis, chaos, cache)
    splitting = Splitting.make_fast_strang(multi_wave_config, multi_ode_config, "FastStrangGalerkin",
                                           start_time, start_positions, start_velocities, delta_time)
    splitting.matrix_s = matrix_s
    return splitting


def calculate_expectancy(splitting, normalization_gamma, stop_time, delta_time):
    print("Starting progressing splitting.")
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
