# plots the galerkin error over the degree using a fixed delta time and fixed stop time
from util.analysis import error_l2_relative
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.galerkin import galerkin_approximation


domain = [(-np.pi, np.pi)]
trial = st.trial_1
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
stop_time = trial.get_parameter("stop_time", 5)
delta_time = 0.0001
steps = stop_time / delta_time
max_poly_degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
wave_weight = 1.

if len(trial.variable_distributions) == 1:
    # needs to be by one bigger than max(max_poly_degrees) in 1D
    quadrature_method = "full_tensor"
    quadrature_params = [[n + 3] * len(trial.variable_distributions) for n in max_poly_degrees]

    # might be slightly better, but curve is more irregular for trial1
    #quadrature_params = [[max(max_poly_degrees) + 2] * len(trial.variable_distributions) for _ in max_poly_degrees]
else:
    quadrature_method = "sparse"
    quadrature_params = max_poly_degrees

trial_exp, trial_var = None, None
errors_exp, errors_var = [], []
for max_poly_degree, quadrature_param in zip(max_poly_degrees, quadrature_params):
    print("Curr degree", max_poly_degree, "of all", max_poly_degrees)

    error_var = None

    xs, xs_mesh, exp, var, quadrature_nodes_count = next(galerkin_approximation(trial, max_poly_degree, domain,
                                                                           grid_size, start_time, steps,
                                                                           delta_time, wave_weight,
                                                                           quadrature_method, quadrature_param))
    if trial_exp is None:
        trial_exp = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
    if trial_var is None:
        trial_var = trial.obtain_evaluated_variance(xs, xs_mesh, stop_time)
    error_exp = error_l2_relative(exp, trial_exp)
    errors_exp.append(error_exp)
    if var is not None and trial_var is not None:
        error_var = error_l2_relative(var, trial_var)
        errors_var.append(error_var)
    print("Error for ", trial.name, "dt=", delta_time, "stop_time=", stop_time, "degree", max_poly_degree,
          "quad=", quadrature_param, "EXP:", error_exp, "VAR:", error_var)

plt.figure()
plt.title("Galerkin-Approximation, $T={}$, $\\tau={}$".format(stop_time, delta_time))
plt.plot(max_poly_degrees, errors_exp, label="Erwartungswert")
if len(max_poly_degrees) == len(errors_var):
    plt.plot(max_poly_degrees, errors_var, label="Varianz")
plt.yscale('log')
plt.xlabel('Maximaler Polynomgrad $P$')
plt.ylabel('Relativer Fehler in diskreter L2-Norm')
plt.ylim(ymax=1)
plt.legend(loc='best')
plt.show()
