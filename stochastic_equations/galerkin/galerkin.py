from util.quadrature.helpers import multi_index_bounded_sum_length
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
import polynomial_chaos.multivariation as mv
from numpy.linalg import eigh
import numpy as np
from functools import partial
from util.caching import cache_by_first_parameter, clear_caches_by_name
from diff_equation.solver_config import SolverConfig
from diff_equation.pseudospectral_solver import WaveSolverConfig
from diff_equation.ode_solver import LinhypSolverConfig
from diff_equation.splitting import Splitting
import collections
from util.analysis import mul_prod

parameter_validation_for_cache = None
cache_name_poly_count = "Poly"
cache_name_quadrature = "Quad"


def galerkin_approximation(trial, max_poly_degree, domain, grid_size, start_time, steps_list, delta_time, wave_weight,
                           quadrature_method, quadrature_param, retrieve_coeffs=False):
    global parameter_validation_for_cache
    if parameter_validation_for_cache is None:
        parameter_validation_for_cache = {"grid_size": grid_size,
                                          "quadrature_method": quadrature_method,
                                          "quadrature_param": quadrature_param,
                                          "trial": trial,
                                          "poly_degree": max_poly_degree}
    assert parameter_validation_for_cache["grid_size"] == grid_size
    assert parameter_validation_for_cache["trial"] == trial
    if max_poly_degree != parameter_validation_for_cache["poly_degree"]:
        clear_caches_by_name(cache_name_poly_count)
    if (parameter_validation_for_cache["quadrature_method"] != quadrature_method
            or parameter_validation_for_cache["quadrature_param"] != quadrature_param):
        clear_caches_by_name(cache_name_quadrature)
    assert len(domain) == 1  # can only handle 1d domains so far, else would need to flatten arrays and other reshaping

    trial.flag_raw_attributes = True
    distrs = trial.variable_distributions
    random_dim = len(distrs)
    sum_bound = max_poly_degree

    chaos = mv.chaos_multify([get_chaos_by_distribution(distr) for distr in distrs], sum_bound)
    chaos.init_quadrature_rule(quadrature_method, quadrature_param)
    print("Quadrature nodes count:", chaos.quadrature_rule.get_nodes_count())

    poly_count = multi_index_bounded_sum_length(random_dim, sum_bound)
    basis = [chaos.normalized_basis(i) for i in range(poly_count)]
    xs, xs_mesh = SolverConfig.make_spatial_discretization(domain, [grid_size])

    # diagonalize symmetric positive definite matrix A where a_ik=E[alpha(y)phi_i(y)phi_k(y)], A=SDS^T
    wave_speeds, matrix_s = calculate_wave_speed_transform(poly_count, trial, basis, chaos)
    print("Wave speeds calculation finished:", wave_speeds)
    # diagonalize transformed symmetric positive definite matrix B(x) for each x, so S^TB(x)S=R(x)D(x)R(x)^T
    # the dependence on x will be performed for every x in xs and the results stacked in a layer of matrices
    # assumes that xs do not change between runs
    betas, matrix_r = calculate_transformed_betas(poly_count, trial, basis, chaos, matrix_s, xs)
    print("Betas calculation finished.")
    splitting = make_splitting(domain, grid_size, wave_speeds,
                               basis, betas, matrix_r, matrix_s, start_time, trial, chaos, delta_time, wave_weight)
    if not isinstance(steps_list, collections.Iterable):
        steps_list = [steps_list]
    for steps in steps_list:
        expectancy, variance, coeffs = calculate_expectancy_variance(splitting, steps, delta_time)
        results = xs, xs_mesh, expectancy, variance, chaos.quadrature_rule.get_nodes_count()
        if retrieve_coeffs:
            yield (*results, coeffs)
        else:
            yield (*results,)


@cache_by_first_parameter([cache_name_quadrature])
def get_evaluated_poly(i, basis, chaos):
    return np.apply_along_axis(basis[i], 1, chaos.quadrature_rule.get_nodes())


@cache_by_first_parameter([cache_name_quadrature])
def get_evaluated_polys(ik, basis, chaos):
    return get_evaluated_poly(ik[0], basis, chaos) * get_evaluated_poly(ik[1], basis, chaos)


# noinspection PyUnusedLocal
@cache_by_first_parameter([cache_name_quadrature])
def get_alpha_at_nodes_matrix(poly_count, trial, nodes_matrix):
    return np.apply_along_axis(lambda ys: trial.alpha(trial.transform_values(ys)), 1, nodes_matrix)


@cache_by_first_parameter([cache_name_quadrature, cache_name_poly_count])
def calculate_wave_speed_transform(poly_count, trial, basis, chaos):
    def alpha_expectancy_func(nodes_matrix, i, k):
        res1 = get_alpha_at_nodes_matrix(poly_count, trial, nodes_matrix)
        res2 = get_evaluated_polys((i, k), basis, chaos)
        return res1 * res2

    def alpha_expectancy_func_simple(*ys, i, k):
        res1 = trial.alpha(trial.transform_values(ys))
        res2 = basis[i](ys) * basis[k](ys)
        res3 = mul_prod(distr.weight(y) for y, distr in zip(ys, chaos.get_distributions()))
        return res1 * res2 * res3

    matrix_a = calculate_expectancy_matrix_sym(chaos, (alpha_expectancy_func, alpha_expectancy_func_simple), poly_count)
    diag, transform_s = eigh(matrix_a)  # now holds A=S*D*S^T, S is orthonormal, D a diagonal matrix
    wave_speeds = np.sqrt(diag)
    return wave_speeds, transform_s


@cache_by_first_parameter([cache_name_poly_count])
def get_beta_at_nodes_matrix(poly_count_and_x, trial, nodes_matrix):
    return np.apply_along_axis(lambda ys: trial.beta([poly_count_and_x[1]], trial.transform_values(ys)),
                               1, nodes_matrix)


@cache_by_first_parameter([cache_name_poly_count])
def calculate_transformed_betas(poly_count, trial, basis, chaos, transform_s, xs):
    def beta_expectancy_func(nodes_matrix, i, k, x):
        res1 = get_beta_at_nodes_matrix((poly_count, x), trial, nodes_matrix)
        res2 = get_evaluated_polys((i, k), basis, chaos)
        return res1 * res2

    def beta_expectancy_func_simple(*ys, i, k, x):
        res3 = mul_prod(distr.weight(y) for y, distr in zip(ys, chaos.get_distributions()))
        return trial.beta([x], trial.transform_values(ys)) * basis[i](ys) * basis[k](ys) * res3

    x_nodes = xs[0]  # one dimensional only
    # tensor.shape = (nodes_count,poly_count,poly_count), so one matrix B(x) per layer
    tensor = calculate_expectancy_tensor(chaos, (beta_expectancy_func, beta_expectancy_func_simple), poly_count, x_nodes)
    # apply the transformation matrix s on the tensor
    tensor = np.matmul(tensor, transform_s)  # application of transform_s on the right for each layer
    tensor = np.matmul(transform_s.T, tensor)  # application of transposed transform_s on the left for each layer

    # diag.shape = (nodes_count,poly_count), transform_r.shape = (nodes_count, poly_count, poly_count)
    diag, transform_r = eigh(tensor)
    betas = diag.T
    # last two dimensions need to be square, calculates eigenvalues and vectors for symmetric matrix
    return betas, transform_r


@cache_by_first_parameter([cache_name_quadrature])
def calculate_beta_integral(ikj, x, chaos, to_expect):  # j needs to be cached as well as this corresponds to x
    if chaos.quadrature_rule.supports_simultaneous_application():
        return chaos.integrate(partial(to_expect[0], i=ikj[0], k=ikj[1], x=x), function_parameter_is_nodes_matrix=True)
    else:
        return chaos.integrate(partial(to_expect[1], i=ikj[0], k=ikj[1], x=x))


def calculate_expectancy_tensor(chaos, to_expect, poly_count, x_nodes):
    nodes_count = len(x_nodes)
    tensor = np.empty((nodes_count, poly_count, poly_count))
    for i in range(poly_count):
        for k in range(i, poly_count):
            for j, x in enumerate(x_nodes):
                exp_value = calculate_beta_integral((i, k, j), x, chaos, to_expect)
                # use symmetry to fill matrix
                tensor[j, i, k] = exp_value
                tensor[j, k, i] = exp_value
        print("Tensor calculation:", i+1, "/", poly_count)
    return tensor


def calculate_expectancy_matrix_sym(chaos, to_expect, poly_count):
    print("Starting calculating symmetric expectancy matrix")
    matrix = np.empty((poly_count, poly_count))
    for i in range(poly_count):
        # we know it is symmetric and therefore only calculate upper triangle
        for k in range(i, poly_count):
            if chaos.quadrature_rule.supports_simultaneous_application():
                exp_value = chaos.integrate(partial(to_expect[0], i=i, k=k), function_parameter_is_nodes_matrix=True)
            else:
                exp_value = chaos.integrate(partial(to_expect[1], i=i, k=k))
            # use symmetry to fill matrix
            matrix[i, k] = exp_value
            matrix[k, i] = exp_value
        print("Matrix A calculation", i+1, "/", poly_count)
    return matrix


def get_projection_coefficients(name_and_poly_count, function, project_trial, basis, chaos):
    def vectorized_func(ys):
        vec_result = function(project_trial.transform_values(ys))
        return vec_result

    # noinspection PyUnusedLocal
    @cache_by_first_parameter([cache_name_quadrature])
    def get_vectorized_with_nodes_matrix(namepoly, nodes_matrix):
        return np.apply_along_axis(vectorized_func, 1, nodes_matrix)

    def integrate_func(nodes_matrix, i):
        res1 = get_vectorized_with_nodes_matrix(name_and_poly_count, nodes_matrix)
        res2 = get_evaluated_poly(i, basis, chaos)
        return res1 * res2

    def integrate_func_simple(*ys, i):
        res3 = mul_prod(distr.weight(y) for y, distr in zip(ys, chaos.get_distributions()))
        return function(project_trial.transform_values(ys)) * basis[i](ys) * res3

    supports_multiple = chaos.quadrature_rule.supports_simultaneous_application()
    to_integrate = integrate_func if supports_multiple else integrate_func_simple
    return [chaos.integrate(partial(to_integrate, i=i),
                            function_parameter_is_nodes_matrix=supports_multiple)
            for i in range(len(basis))]


@cache_by_first_parameter([cache_name_poly_count, cache_name_quadrature])
def get_starting_value_coefficients(name_and_poly_count, xs_mesh, starting_value_func,
                                    project_trial, matrix_s_transposed,
                                    basis, chaos):
    coefficients = []
    for i, x in enumerate(xs_mesh[0]):
        def current_func(ys):
            return starting_value_func([x], ys)

        # if starting_value_func does not depend on ys, then only the first coeff will be nonzero
        coeff = get_projection_coefficients(name_and_poly_count, current_func, project_trial, basis, chaos)
        coeffs = matrix_s_transposed.dot(coeff)
        coefficients.append(coeffs)
    return list(map(np.array, zip(*coefficients)))  # transpose and convert to vectors, we want a list of length N+1


# ---------------------------SPLITTING---------------------------


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


class MultiLinhypSolver(SolverConfig):
    def __init__(self, intervals, grid_points_list, betas, tensor_r, splitting_factor):
        super().__init__(intervals, grid_points_list)
        self.betas = betas
        self.tensor_r = tensor_r
        self.tensor_r_transposed = np.transpose(self.tensor_r, axes=[0, 2, 1])  # transpose the matrices of each layer
        self.splitting_factor = splitting_factor
        self.configs = [LinhypSolverConfig(intervals, grid_points_list, beta, splitting_factor)
                        for beta in self.betas]

    def init_solver(self, t0, u0s, u0ts):
        assert len(u0s) == len(self.configs)
        assert len(u0ts) == len(self.configs)
        self.init_initial_values(t0, u0s, u0ts)

        # multiply by R^T on the left, corresponding entries that belong to the same x value are multiplied
        transformed_positions = np.einsum('kij,jk->ik', self.tensor_r_transposed, np.array(u0s))
        transformed_velocities = np.einsum('kij,jk->ik', self.tensor_r_transposed, np.array(u0ts))

        for config, u0, u0t in zip(self.configs, transformed_positions, transformed_velocities):
            config.init_solver(t0, u0, u0t)

        # each individual config solver returns a list of vectors of length 'grid_size'. We pack the lists all in a list
        # then we transpose this construct using list(zip(*...))
        # then we get a list whose first entry contains a list of positions, the second a list of velocities,...

        def solution_at(time):
            results = (curr_config.solver(time) for curr_config in self.configs)
            positions, velocities = zip(*results)  # transpose
            transformed_result_positions = np.einsum('kij,jk->ik', self.tensor_r, np.array(positions))
            transformed_result_velocities = np.einsum('kij,jk->ik', self.tensor_r, np.array(velocities))
            return transformed_result_positions, transformed_result_velocities

        self.solver = solution_at


def make_splitting(domain, grid_size, wave_speeds,
                   basis, betas, matrix_r, matrix_s, start_time, trial, chaos, delta_time, wave_weight):
    multi_wave_config = MultiWaveSolver(domain, [grid_size], wave_speeds, wave_weight)
    multi_linhyp_config = MultiLinhypSolver(domain, [grid_size], betas, matrix_r, 1 - wave_weight)

    print("Calculating starting position coefficients...")
    start_positions = get_starting_value_coefficients(("position", len(basis)), multi_wave_config.xs_mesh,
                                                      trial.start_position,
                                                      trial, matrix_s.T, basis, chaos)
    print("Calculating starting velocity coefficients...")
    start_velocities = get_starting_value_coefficients(("velocity", len(basis)), multi_wave_config.xs_mesh,
                                                       trial.start_velocity,
                                                       trial, matrix_s.T, basis, chaos)
    splitting = Splitting.make_fast_strang(multi_wave_config, multi_linhyp_config, "FastStrangGalerkin",
                                           start_time, start_positions, start_velocities, delta_time)
    splitting.matrix_s = matrix_s
    return splitting


def calculate_expectancy_variance(splitting, steps, delta_time):
    print("Starting progressing splitting: Current time:", repr(splitting.get_current_time()),
          "Making", steps, "steps with dt=", delta_time)
    splitting.progress(steps, delta_time, 0)
    print("Finished progressing splitting: Current time:", repr(splitting.get_current_time()))
    last_coefficients = splitting.solutions()[-1]
    # this is now a list of length N+1, one vector for each index k, each of length 'grid_size'
    # first we need to transpose this construct in order to be able to retransform the coefficients with the matrix S
    grid_coeffs = list(map(np.array, zip(*last_coefficients)))  # transpose
    grid_coeffs = np.array([splitting.matrix_s.dot(curr) for curr in grid_coeffs])
    # for expectancy we only need to use the zeroth coefficient, polynomials are normalized so no need for gammas

    expectancy = grid_coeffs[:, 0]

    variance = np.zeros(shape=expectancy.shape)
    if grid_coeffs.shape[1] > 1:
        # for variance we need all coefficients except for the zeroth one,
        # polynomials are normalized so no need for gammas
        variance = np.sum(np.square(grid_coeffs[:, 1:]), axis=1, out=variance)

    return expectancy, variance, grid_coeffs
