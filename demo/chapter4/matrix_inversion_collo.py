import numpy as np
from itertools import repeat
from util.quadrature.helpers import multi_index_bounded_sum_length
from stochastic_equations.collocation.interpolation import matrix_inversion_expectancy
import matplotlib.pyplot as plt
from util.analysis import error_l2_relative, error_maximum
import demo.stochastic_trials as st
import util.quadrature.nesting as nst
from util.quadrature.closed_fully_nested import ClosedFullNesting
from polynomial_chaos.poly_chaos_distributions import get_chaos_name_by_distribution

trial = st.trial_8

# "High order is not the same as high accuracy. High order translates to high accuracy only when the integrand
# is very smooth" (http://apps.nrbook.com/empanel/index.html?pg=179#)
N = list(range(15))  # maximum degree of the univariate polynomial

# from n+1 to n+10 notably difference for most examples
# number of nodes in random space, >= N+1, higher CAN give more accuracy (for higher polys)
# M = [(n + 10,) * len(trial.variable_distributions) for n in N]

# minimum possible value that the full tensor product of nodes is still bigger than the number of basis polynomials
# if minimal: rank of vandermonde decreases by 10-30% and solution does not improve; very fast!
# if minimal+1:
M = [(int(np.ceil(multi_index_bounded_sum_length(len(trial.variable_distributions), n)
                  ** (1 / len(trial.variable_distributions)))),) * len(trial.variable_distributions)
     for n in N]

chaos_names = [get_chaos_name_by_distribution(distr) for distr in trial.variable_distributions]
dim = len(chaos_names)
nesting = nst.get_nesting_for_multiple_names(chaos_names)
poly_count_pre = [multi_index_bounded_sum_length(dim, n) for n in N]
L = [nesting.get_minimum_level_with_point_num(dim, count) for count in poly_count_pre]
nesting = ClosedFullNesting()
L_gc = [nesting.get_minimum_level_with_point_num(dim, count) for count in poly_count_pre]

spatial_dimension = 1
use_max_error = False
show_point_count_plot = False
plot_variance = True
grid_size = trial.get_parameter("grid_size", 128)
spatial_domain = tuple(repeat(tuple([-np.pi, np.pi]), spatial_dimension))
start_time = 0
stop_time = trial.get_parameter("stop_time", 0.5)
# if grid_size is bigger this needs to be smaller, especially for higher poly degrees
delta_time = trial.get_parameter("delta_time", 0.001)
methods = ["full_tensor", "pseudo_sparse"]  # , "chebyshev"]  # ["full_tensor", "sparse", "sparse_gc", "centralized"]
method_params = [M, [(n, chaos_names) for n in N]]  # , M]  # [M, L, L_gc, [(n, False) for n in N]]
method_markers = ["-o", "-D", "-x", "-."]
rank_frac = None
exp_var_results = dict()
rank_fractions, quad_points, poly_count = dict(), dict(), dict()
for method, params in zip(methods, method_params):
    exp_var_results[method] = []
    rank_fractions[method] = []
    quad_points[method] = []
    poly_count[method] = []
    print("Starting method", method, "...")
    for n, quadrature_param in zip(N, params):
        print("n,m=", n, quadrature_param)
        result_xs, result_xs_mesh, mi_expectancy, mi_variance, mi_rank_frac, curr_poly_count, curr_quad_points = \
            matrix_inversion_expectancy(trial, n, method, quadrature_param, spatial_domain, grid_size,
                                        start_time, stop_time, delta_time)
        exp_var_results[method].append((n, quadrature_param, mi_expectancy, mi_variance))
        rank_fractions[method].append(mi_rank_frac)
        quad_points[method].append(curr_quad_points)
        poly_count[method].append(curr_poly_count)
    rank_fractions[method] = list(map(lambda frac: 10 ** ((frac - 1) * 10),
                              rank_fractions[method]))  # rescale to make visible in logarithmic scale

print("Plotting:")
trial_expectancy = trial.obtain_evaluated_expectancy(result_xs, result_xs_mesh, stop_time)
trial_variance = trial.obtain_evaluated_variance(result_xs, result_xs_mesh, stop_time)

if use_max_error:
    error_func = error_maximum
    error_descr = "Fehler in Maximumsnorm"
else:
    error_func = error_l2_relative
    error_descr = "Relativer Fehler in diskreter L2-Norm"

plt.figure()
print(trial.name)
plt.title("Collocation durch Interpolation, $\\tau={}$, $T={}$".format(delta_time, stop_time))
errors_exp, errors_variance = dict(), dict()
for method, marker in zip(methods, method_markers):
    errors_exp[method] = []
    errors_variance[method] = []
    for n, param, expectancy, variance in exp_var_results[method]:
        error = -1
        if trial_expectancy is not None:
            error = error_func(expectancy, trial_expectancy)
            errors_exp[method].append(error)
            print("Error expectancy", n, param, "=", error)
        if trial_variance is not None:
            error_var = error_func(variance, trial_variance)
            errors_variance[method].append(error_var)
            print("Error variance", n, param, "=", error_var)
        plt.plot(result_xs[0], expectancy, marker, label="deg={}, Fehler={:.5E}"
                 .format(n, error))
ref = trial_expectancy
if ref is not None:
    plt.plot(result_xs[0], ref, label="Exakte Referenz")
# plt.ylim((0, 1))
plt.legend()
# save_fig(plt.axes(), "../data/interpol_invmat_trial5_512_0.00005.pickle")
print("Errors exp=", errors_exp)
print("Errors var=", errors_variance)

plt.figure()
if show_point_count_plot:
    plt.subplot(121)  # 1 row, 2 columns, subplot 1
else:
    plt.subplot(111)
for method, marker in zip(methods, method_markers):
    if len(errors_exp[method]) > 0 or len(errors_variance[method]) > 0:
        x_data = poly_count[method]
        if len(errors_exp[method]) == len(x_data):
            plt.plot(x_data, errors_exp[method], marker, color="b", label="Erwartungswert ({})".format(method))
        if plot_variance and len(errors_variance[method]) == len(x_data):
            plt.plot(x_data, errors_variance[method], marker, color="r", label="Varianz ({})".format(method))
        if len(rank_fractions) == len(x_data):
            plt.plot(x_data, rank_fractions, marker, color="g", label="Vandermonde Rangdefizit {}".format(method))
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('Anzahl an Basispolynomen $M$')
plt.ylabel(error_descr)
plt.title("Collocation durch Interpolation, $\\tau={}$, $T={}$".format(delta_time, stop_time))
if show_point_count_plot:
    plt.subplot(122)
    for method, marker in zip(methods, method_markers):
        x_data = poly_count[method]
        plt.plot(x_data, quad_points[method], marker, color="b", label="{}".format(method))
    plt.xlabel('Anzahl an Basispolynomen $M$')
    plt.ylabel('Anzahl an Interpolationspunkten')
    plt.legend(loc='best')

plt.show()
print("Poly count pre=", poly_count_pre)
print("Poly counts:", poly_count)

