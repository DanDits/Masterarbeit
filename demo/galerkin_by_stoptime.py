# Plots the error of the galerkin approximation over different stop times using the same delta time and approximation
# degree (which should be good enough so that the splitting error does not fall into account)
from util.analysis import error_l2_relative
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.galerkin import galerkin_approximation


domain = [(-np.pi, np.pi)]
trial = st.trial_2_1  # requires us to get expectancy and variances at all stop_times!!
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
stop_times = [0.1, 0.2, 0.3, 0.4, 0.5]
delta_time = 0.0001
max_poly_degree = 3
wave_weight = 1.

if len(trial.variable_distributions) == 1:
    quadrature_method = "full_tensor"
    quadrature_param = [max_poly_degree + 5] * len(trial.variable_distributions)
else:
    quadrature_method = "sparse"
    quadrature_param = max_poly_degree

errors_exp, errors_var = [], []

for stop_time, (xs, xs_mesh, exp, var, quadrature_nodes_count) in zip(stop_times,
                                                                      galerkin_approximation(trial, max_poly_degree,
                                                                                             domain,
                                                                           grid_size, start_time, stop_times,
                                                                           delta_time, wave_weight,
                                                                           quadrature_method, quadrature_param)):
    error_var = None

    trial_exp = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
    trial_var = trial.obtain_evaluated_variance(xs, xs_mesh, stop_time)
    error_exp = error_l2_relative(exp, trial_exp)
    errors_exp.append(error_exp)
    if var is not None and trial_var is not None:
        error_var = error_l2_relative(var, trial_var)
        errors_var.append(error_var)
    print("Error for ", trial.name, "dt=", delta_time, "stop_time=", stop_time, "degree", max_poly_degree,
          "quad=", quadrature_param, "EXP:", error_exp, "VAR:", error_var)


def target(p):
    return lambda time: np.exp((p / 2 + 1) * time)

plt.figure()
plt.title("Galerkin-Approximation, $P={}$, $\\tau={}$".format(max_poly_degree, delta_time))
plt.plot(stop_times, errors_exp, label="Erwartungswert")
if len(stop_times) == len(errors_var):
    plt.plot(stop_times, errors_var, label="Varianz")
target_func = target(max_poly_degree)
plt.plot(stop_times, errors_exp[1] / target_func(stop_times[1]) * target_func(np.array(stop_times)),
         label="Erwartungswertziel")
plt.plot(stop_times, errors_var[1] / target_func(stop_times[1]) * target_func(np.array(stop_times)),
         label="Varianzziel")
plt.yscale('log')
plt.xlabel('Stopzeit $T$')
plt.legend(loc='best')
plt.show()
