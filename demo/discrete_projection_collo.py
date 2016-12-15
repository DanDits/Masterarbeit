import numpy as np
from itertools import repeat
from stochastic_equations.collocation.discrete_projection import discrete_projection_expectancy
import matplotlib.pyplot as plt
from util.analysis import error_l2
import demo.stochastic_trials as st


trial = st.trial_6

N = list(range(6))  # maximum degree of the univariate polynomial

# for method full_tensor: number of nodes and weights used for discrete projection's quadrature formula per dimension
Q = [(n + 1,) * len(trial.variable_distributions) for n in N]
# for method sparse: level used, will result in about 2^(level+1) quadrature nodes
L = [n for n in N]
# for method centralized: bound of sum and if even option is to be used
S = [(n + 1, False) for n in N]

spatial_dimension = 1
grid_size = trial.get_parameter("grid_size", 128)
spatial_domain = list(repeat([-np.pi, np.pi], spatial_dimension))
start_time = 0
stop_time = trial.get_parameter("stop_time", 0.5)
# if grid_size is bigger this needs to be smaller, especially for higher poly degrees
delta_time = trial.get_parameter("delta_time", 0.001)

exp_var_results, quad_points = {}, {}

methods = ["sparse", "full_tensor", "centralized"]
trial_expectancy, trial_variance = None, None

for method in methods:
    if method == "sparse":
        method_params = L
    elif method == "full_tensor":
        method_params = Q
    else:
        method_params = S
    exp_var_results[method] = []
    quad_points[method] = []
    for n, param in zip(N, method_params):
        result_xs, result_xs_mesh, dp_expectancy, dp_variance, points = \
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
for method, marker in zip(methods, ["-D", "-o", "-x"]):
    errors_exp, errors_var = [], []
    for n, param, expectancy, variance in exp_var_results[method]:
        error = -1
        if trial_expectancy is not None:
            error = error_l2(trial_expectancy, expectancy)
            errors_exp.append(error)
            print("Error expectancy dp", n, param, "=", error)
        if trial_variance is not None:
            error_var = error_l2(trial_variance, variance)
            errors_var.append(error_var)
            print("Error variance dp", n, param, "=", error_var)

    if len(errors_exp) > 0 or len(errors_var) > 0:
        if len(errors_exp) == len(N):
            plt.plot(N, errors_exp, marker, color="b", label="Errors of {} to expectancy".format(method))
        if len(errors_var) == len(N):
            plt.plot(N, errors_var, marker, color="r", label="Errors of {} to variance".format(method))

        """for error, n, q in zip(errors_exp, N, quad_points[method]):
            plt.text(n, error, "Q={}".format(q))

        for error, n, q in zip(errors_var, N, quad_points[method]):
            plt.text(n, error, "Q={}".format(q))"""
print(quad_points)
plt.yscale('log')
plt.legend()
plt.xlabel('Maximum univariate polynom degree')
plt.show()
