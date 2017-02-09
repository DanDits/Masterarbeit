# directly comparing matrix inversion and discrete projection over the amount of used collocation points
from itertools import repeat
import numpy as np
import demo.stochastic_trials as st
from polynomial_chaos.poly_chaos_distributions import get_chaos_name_by_distribution
from util.quadrature.helpers import multi_index_bounded_sum_length
from stochastic_equations.collocation.interpolation import matrix_inversion_expectancy
import util.quadrature.nesting as nst
from util.quadrature.closed_fully_nested import ClosedFullNesting
import matplotlib.pyplot as plt
from util.analysis import error_l2_relative
from stochastic_equations.collocation.discrete_projection import discrete_projection


trial = st.trial_1  # the trial to use

# for 4 dimensions: max(N)=10 (x system) fits together with max_level=4 (625 nodes for full_tensor)
# for 1 dimension: chooses discrete projection parameter automatically the same to those of matrix inversion
N = list(range(35))  # for mi maximum degree of the univariate polynomial

# for full_tensor(s)
M = [(int(np.ceil(multi_index_bounded_sum_length(len(trial.variable_distributions), n)
                  ** (1 / len(trial.variable_distributions)))),) * len(trial.variable_distributions)
     for n in N]

P = [0, 2, 7]   # for dp
max_level = 4  # for dp

chaos_names = [get_chaos_name_by_distribution(distr) for distr in trial.variable_distributions]
random_dim = len(trial.variable_distributions)


def minimum_full_tensor_param_for_count(count):
    return (int(np.ceil(count ** (1 / random_dim))),) * random_dim

# general config that probably does not need to be changed
spatial_dimension = 1
grid_size = trial.get_parameter("grid_size", 128)
spatial_domain = tuple(repeat(tuple([-np.pi, np.pi]), spatial_dimension))
start_time = 0
stop_time = trial.get_parameter("stop_time", 0.5)
delta_time = trial.get_parameter("delta_time", 0.001)


# methods to use for matrix inversion
methods_mi = ["full_tensor"]#, "pseudo_sparse"]
method_params_mi = [M]#, [(n, chaos_names) for n in N]]
method_markers_mi = ["o-", "D-"]

# calculate approximation by using matrix inversion
exp_var_results_mi, quad_points_mi = dict(), dict()
for method, params in zip(methods_mi, method_params_mi):
    exp_var_results_mi[method] = []
    quad_points_mi[method] = []
    print("Starting matrix inversion method", method, "...")
    for n, quadrature_param in zip(N, params):
        print("n,param=", n, quadrature_param)
        result_xs, result_xs_mesh, mi_expectancy, mi_variance, mi_rank_frac, curr_poly_count, curr_quad_points = \
            matrix_inversion_expectancy(trial, n, method, quadrature_param, spatial_domain, grid_size,
                                        start_time, stop_time, delta_time)
        exp_var_results_mi[method].append((n, quadrature_param, mi_expectancy, mi_variance))
        quad_points_mi[method].append(curr_quad_points)

# calculate matrix inversion errors
trial_expectancy = trial.obtain_evaluated_expectancy(result_xs, result_xs_mesh, stop_time)
trial_variance = trial.obtain_evaluated_variance(result_xs, result_xs_mesh, stop_time)
errors_exp_mi, errors_variance_mi = dict(), dict()
error_func = error_l2_relative
for method in methods_mi:
    errors_exp_mi[method] = []
    errors_variance_mi[method] = []
    for n, param, expectancy, variance in exp_var_results_mi[method]:
        error = -1
        if trial_expectancy is not None:
            error = error_func(expectancy, trial_expectancy)
            errors_exp_mi[method].append(error)
            print("Error expectancy", n, param, "=", error)
        if trial_variance is not None:
            error_var = error_func(variance, trial_variance)
            errors_variance_mi[method].append(error_var)
            print("Error variance", n, param, "=", error_var)

plt.figure()
# plot matrix inversion errors
print("Errors matrix inversion expectancy:", errors_exp_mi)
print("Quad points matrix inversion:", quad_points_mi)  # can have multiple times same amount as full_tensor is chosen to be just bigger than multi_indexed_bounded_sum
for method, marker in zip(methods_mi, method_markers_mi):
    if len(errors_exp_mi[method]) > 0 or len(errors_variance_mi[method]) > 0:
        x_data = quad_points_mi[method]
        if len(errors_exp_mi[method]) == len(x_data):
            plt.plot(x_data, errors_exp_mi[method], marker, color="b", label="Erwartungswert MI ({})".format(method))
        if len(errors_variance_mi[method]) == len(x_data):
            plt.plot(x_data, errors_variance_mi[method], marker, color="r", label="Varianz MI ({})".format(method))


nesting = nst.get_nesting_for_multiple_names(chaos_names)
wanted_levels = list(range(max_level + 1))
wanted_quad_points = [nesting.calculate_point_num(random_dim, level) for level in wanted_levels]
nesting = ClosedFullNesting()
L_gc = [nesting.get_minimum_level_with_point_num(random_dim, count) for count in wanted_quad_points]
if random_dim > 1:
    quadrature_methods_dp = {"full_tensor": list(map(minimum_full_tensor_param_for_count, wanted_quad_points[:-1]))
                             , "sparse": wanted_levels}
                             #, "sparse_gc": L_gc}
else:
    quadrature_methods_dp = {"full_tensor": M}

expectancy_already_plotted_by_method = set()  # plot expectancy only once per method as it does not depend on p
# colors red to green, then shades of grey
variance_colors = ["#FF0000", "#FF6200", "#F5C711", "#54C41B", "#00FF00", "#000000", "#AAAAAA"]
if len(P) > len(variance_colors):
    raise ValueError("Too many values for P... reduce or add more colors.")

for (method, method_params), marker in zip(quadrature_methods_dp.items(), ["->", "-x", "-<", "-."]):
    print("Starting dp method", method, "with params", method_params)
    exp_var_results_dp = dict()
    quad_points_dp = dict()

    # for each sum bound for the polynomial basis use all possible quadratures
    for p in P:
        print("P=", p, "of", P)
        exp_var_results_dp[p] = []
        quad_points_dp[p] = []
        # for each parameter of the method do a discrete projection and save it
        for param in method_params:
            print("Current param:", param)
            result_xs, result_xs_mesh, dp_expectancy, dp_variance, points, poly_count = \
                discrete_projection(trial, p, method, param, spatial_domain, grid_size, start_time, stop_time,
                                    delta_time)
            exp_var_results_dp[p].append((param, dp_expectancy, dp_variance))
            quad_points_dp[p].append(points)

    # calculate and plot errors depending on the amount of quadrature points
    for p, variance_color in zip(P, variance_colors):
        errors_exp = []
        errors_var = []
        for param, expectancy, variance in exp_var_results_dp[p]:
            if method not in expectancy_already_plotted_by_method and trial_expectancy is not None:
                error = error_func(expectancy, trial_expectancy)
                errors_exp.append(error)
                print("Error expectancy dp", method, param, "=", error)
            if trial_variance is not None:
                error = error_func(variance, trial_variance)
                errors_var.append(error)
                print("Error variance dp", method, param, "=", error)

        x_values = quad_points_dp[p]
        if len(errors_exp) > 0 and len(errors_exp) == len(x_values):
            expectancy_already_plotted_by_method.add(method)
            plt.plot(x_values, errors_exp, marker, color="b", label="Erwartungswert DP P={} ({})".format(p, method))
        if len(errors_var) > 0 and len(errors_var) == len(x_values):
            plt.plot(x_values, errors_var, marker, color=variance_color, label="Varianz DP P={} ({})".format(p, method))

plt.title("Vergleich verschiedener Collocationsansätze")
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('Anzahl an Collocationspunkten')
plt.ylabel("Relativer Fehler in diskreter L2-Norm")
plt.show()
