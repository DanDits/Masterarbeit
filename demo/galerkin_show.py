from util.analysis import error_l2_relative
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.galerkin import galerkin_approximation


# OBSERVATIONS:
# trial1: with delta time = 0.000001 it loses accuracy,
# this loss seems independent of quadrature count and is the same for poly degree 2,3,4; does not depend on wave_weight
domain = [(-np.pi, np.pi)]
trial = st.trial_7_t2
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
stop_time = trial.get_parameter("stop_time", 1.)
delta_times = [0.1, 0.01, 0.001, 0.0001]
max_poly_degrees = [1, 5, 10, 20, 40]
wave_weight = 1.

# trial7: >=10 if degree <=6, >=15 if degree <= 10,  if degree <= 15
if len(trial.variable_distributions) == 1:
    # needs to be by one bigger than max(max_poly_degrees) in 1D
    quadrature_method = "full_tensor"
    quadrature_params = [[n + 1] * len(trial.variable_distributions) for n in max_poly_degrees]
else:
    quadrature_method = "sparse"
    quadrature_params = max_poly_degrees

# if quadrature method or parameter is changed, the caches need to be cleared as they use the quadrature nodes
print("Max polys:", max_poly_degrees)
print("Quad=", quadrature_method, "params=", quadrature_params)
print("Waveweight=", wave_weight)
print("DT=", delta_times)
plt.figure()
print(trial.name)
plt.xlabel("$\\frac{1}{\\tau}$")
plt.ylabel("Relativer Fehler in diskreter L2-Norm")
plt.xscale('log')
plt.yscale('log')
cache = {}
trial_exp, trial_var = None, None

for max_poly_degree, quadrature_param in zip(max_poly_degrees, quadrature_params):
    print("Curr degree", max_poly_degree, "of all", max_poly_degrees)
    errors_exp, errors_var = [], []
    error_var = None
    for delta_time in delta_times:
        print("Curr dt", delta_time, "of all", delta_times)
        steps = int(stop_time / delta_time)
        xs, xs_mesh, exp, var, quadrature_nodes_count = galerkin_approximation(trial, max_poly_degree, domain,
                                                                               grid_size, start_time, steps,
                                                                               delta_time, wave_weight,
                                                                               quadrature_method, quadrature_param)
        if trial_exp is None:
            trial_exp = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
        if trial_var is None:
            trial_var = trial.obtain_evaluated_variance(xs, xs_mesh, stop_time)
        error_exp = error_l2_relative(exp, trial_exp)
        errors_exp.append(error_exp)
        if var is not None:
            error_var = error_l2_relative(var, trial_var)
            errors_var.append(error_var)
        print("Degree={}, dt={}, error_exp={}, error_var={}".format(max_poly_degree, delta_time, error_exp, error_var))
    print("exp errors for {} {}".format(quadrature_method, trial.name), errors_exp)
    print("var errors for {} {}".format(quadrature_method, trial.name), errors_var)

    plt.plot(1. / np.array(delta_times), errors_exp, "-", label="Erwartungswert, $P={}$".format(max_poly_degree))
    if len(delta_times) == len(errors_var):
        plt.plot(1. / np.array(delta_times), errors_var, "-o", label="Varianz, $P={}$".format(max_poly_degree))
    print("Plotting done.")


plt.title("Galerkin-Approximation, T={}, $\\kappa={}$, {} mit {} Punkten"
          .format(stop_time, wave_weight, quadrature_method, quadrature_nodes_count))
plt.ylim((1E-13, 1.))
plt.legend()
plt.show()
