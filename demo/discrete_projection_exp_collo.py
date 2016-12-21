import numpy as np
from itertools import repeat
from stochastic_equations.collocation.discrete_projection import discrete_projection_expectancy
import matplotlib.pyplot as plt
from util.analysis import error_l2
import demo.stochastic_trials as st
import util.quadrature.nesting as nst
from polynomial_chaos.poly_chaos_distributions import get_chaos_name_by_distribution
from util.quadrature.closed_fully_nested import ClosedFullNesting
from itertools import chain

trial = st.trial_5

# to calculate expectancy we only need polynomial of degree 0 as the equations are not coupled like for matrix inversion
n = 0

# the accuracy only depends on how good we can integrate, compare different methods and
# show error depending on the amount of nodes
levels_count = 5
chaos_names = [get_chaos_name_by_distribution(distr) for distr in trial.variable_distributions]
nesting = nst.get_nesting_for_multiple_names(chaos_names)

# for method sparse: level used, will result in about 2^(level+1)+1 quadrature nodes
L = [n for n in range(levels_count)]

# for method full_tensor: number of nodes and weights used for discrete projection's quadrature formula per dimension
dim = len(trial.variable_distributions)
# set full tensors param so that the full tensor node count is the smallest bigger value than sparse's node count
Q = [int(np.ceil(nesting.calculate_point_num(dim, l) ** (1./dim))) for l in L]
Q = list(x for x in chain.from_iterable((q - 1, q) for q in Q) if x > 0)  # and also include one smaller value >0 for each
Q = [[q] * dim for q in Q]  # excepts one for each dimension

# for method sparse_gc: level used, will result in 2^level-1 quadrature nodes
nesting_gc = ClosedFullNesting()
L_gc = [nesting_gc.get_minimum_level_with_point_num(dim, nesting.calculate_point_num(dim, l)) for l in L]

# TODO for visualizing something equivalent for variance, we need to also increase n, do in another demo
# TODO create a demo for comparing exp for MI and DP, especially to show (when) that they are equal

spatial_dimension = 1
grid_size = trial.get_parameter("grid_size", 128)
spatial_domain = list(repeat([-np.pi, np.pi], spatial_dimension))
start_time = 0
stop_time = trial.get_parameter("stop_time", 0.5)
# if grid_size is bigger this needs to be smaller, especially for higher poly degrees
delta_time = trial.get_parameter("delta_time", 0.001)

exp_var_results, quad_points = {}, {}

methods = ["full_tensor", "sparse"]
all_params = [Q, L, L_gc]
method_markers = ["-o", "-D", "-x"]
trial_expectancy, trial_variance = None, None

for method, method_params in zip(methods, all_params):
    exp_var_results[method] = []
    quad_points[method] = []
    for param in method_params:
        result_xs, result_xs_mesh, dp_expectancy, dp_variance, points, poly_count = \
            discrete_projection_expectancy(trial, n, method, param, spatial_domain, grid_size, start_time, stop_time,
                                           delta_time)
        exp_var_results[method].append((n, param, dp_expectancy, dp_variance))
        quad_points[method].append(points)

if trial_expectancy is None:
    trial_expectancy = trial.obtain_evaluated_expectancy(result_xs, result_xs_mesh, stop_time)
if trial_variance is None:
    trial_variance = trial.obtain_evaluated_variance(result_xs, result_xs_mesh, stop_time)


plt.figure()
plt.title("Discrete projection colloc. for {}, gridsize={}, dt={}, T={}".format(trial.name, grid_size,
                                                                                delta_time, stop_time))
for method, marker in zip(methods, method_markers):
    errors_exp = []
    for n, param, expectancy, variance in exp_var_results[method]:
        if np.any(np.isnan(expectancy)):
            print("Expectancy contains NaN:", expectancy)
        error = -1
        if trial_expectancy is not None:
            error = error_l2(trial_expectancy, expectancy)
            errors_exp.append(error)
            print("Error expectancy dp", method, n, param, "=", error)

    if len(errors_exp) > 0:
        x_values = quad_points[method]
        if len(errors_exp) == len(x_values):
            plt.plot(x_values, errors_exp, marker, color="b", label="Errors of {} to expectancy".format(method))

print("Quadrature points:", quad_points)
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('Amount of quadrature points')
plt.show()
