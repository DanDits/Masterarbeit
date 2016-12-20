import numpy as np
from itertools import repeat
from util.quadrature.helpers import multi_index_bounded_sum_length
from stochastic_equations.collocation.interpolation import matrix_inversion_expectancy
import matplotlib.pyplot as plt
from util.analysis import error_l2
import demo.stochastic_trials as st

trial = st.trial_discont

# "High order is not the same as high accuracy. High order translates to high accuracy only when the integrand
# is very smooth" (http://apps.nrbook.com/empanel/index.html?pg=179#)
N = list(range(40))  # maximum degree of the univariate polynomial

# from n+1 to n+10 notably difference for most examples
# number of nodes in random space, >= N+1, higher CAN give more accuracy (for higher polys)
# M = [(n + 10,) * len(trial.variable_distributions) for n in N]

# minimum possible value that the full tensor product of nodes is still bigger than the number of basis polynomials
# if minimal: rank of vandermonde decreases by 10-30% and solution does not improve; very fast!
# if minimal+1:
M = [(int(np.ceil(multi_index_bounded_sum_length(len(trial.variable_distributions), n)
                  ** (1 / len(trial.variable_distributions)))),) * len(trial.variable_distributions)
     for n in N]

spatial_dimension = 1
grid_size = trial.get_parameter("grid_size", 128)
spatial_domain = list(repeat([-np.pi, np.pi], spatial_dimension))
start_time = 0
stop_time = trial.get_parameter("stop_time", 0.5)
# if grid_size is bigger this needs to be smaller, especially for higher poly degrees
delta_time = trial.get_parameter("delta_time", 0.001)

rank_frac = None
exp_var_results_mi, rank_fracs = [], []
for n, m in zip(N, M):
    print("n,m=", n, m)
    quadrature_method = "full_tensor"
    quadrature_param = m
    result_xs, result_xs_mesh, mi_expectancy, mi_variance, mi_rank_frac = \
        matrix_inversion_expectancy(trial, n, quadrature_method, quadrature_param, spatial_domain, grid_size,
                                    start_time, stop_time, delta_time)
    exp_var_results_mi.append((n, m, mi_expectancy, mi_variance))
    rank_fracs.append(mi_rank_frac)
rank_fractions = list(map(lambda frac: 10 ** ((frac - 1) * 10),
                          rank_fracs))  # rescale to make visible in logarithmic scale

print("Plotting:")
trial_expectancy = trial.obtain_evaluated_expectancy(result_xs, result_xs_mesh, stop_time)
trial_variance = trial.obtain_evaluated_variance(result_xs, result_xs_mesh, stop_time)

plt.figure()
plt.title("Expectancies matrix inversion, spatial grid size={}, {}, T={}".format(grid_size, trial.name, stop_time))
errors_mi, errors_variance_mi = [], []
for n, m, expectancy, variance in exp_var_results_mi:
    error = -1
    if trial_expectancy is not None:
        error = error_l2(trial_expectancy, expectancy)
        errors_mi.append(error)
        print("Error expectancy mi", n, m, "=", error)
    if trial_variance is not None:
        error_var = error_l2(trial_variance, variance)
        errors_variance_mi.append(error_var)
        print("Error variance mi", n, m, "=", error_var)
    plt.plot(result_xs[0], expectancy, "D" if n < 7 else ".", label="deg={}, error={:.5E}"
             .format(n, error))
ref = trial_expectancy
if ref is not None:
    plt.plot(result_xs[0], ref, label="Exact reference")
# plt.ylim((0, 1))
plt.legend()
# save_fig(plt.axes(), "../data/interpol_invmat_trial5_512_0.00005.pickle")
print("Errors mi=", errors_mi)
if len(errors_mi) > 0 or len(errors_variance_mi) > 0:
    plt.figure()
    plt.title("Collocation interpolation for {}, gridsize={}, dt={}, T={}".format(trial.name, grid_size,
                                                                                  delta_time, stop_time))
    if len(errors_mi) == len(N):
        plt.plot(N, errors_mi, label="Errors to expectancy")
    if len(errors_variance_mi) == len(N):
        plt.plot(N, errors_variance_mi, label="Errors to variance")
    if len(rank_fractions) == len(N):
        plt.plot(N, rank_fractions, label="Vandermonde rank deficiency")
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Maximum polynom degree')
plt.show()
