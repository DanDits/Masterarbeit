from util.analysis import error_l2_relative
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.newgalerkin import galerkin_approximation


# OBSERVATIONS:
# trial1: with delta time = 0.000001 it loses accuracy, this loss seems independent of quadrature count and is the same for poly degree 2,3,4
#         does not depend on wave_weight
domain = [(-np.pi, np.pi)]
trial = st.trial_8
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
stop_time = trial.get_parameter("stop_time", 0.5)
delta_times = [0.1, 0.01, 0.001]
max_poly_degrees = [0, 1, 2]
wave_weight = 1.

quadrature_method = "full_tensor"
# trial7: >=10 if degree <=6, >=15 if degree <= 10,  if degree <= 15
quadrature_param = [5] * len(trial.variable_distributions)
quadrature_method = "sparse"  # TODO compare sparse and full_tensor, how to handle different caching?
quadrature_param = 2
# if quadrature method or parameter is changed, the caches need to be cleared as they use the quadrature nodes
print("Max polys:", max_poly_degrees)
print("Quad=", quadrature_method, "params=", quadrature_param)
print("Waveweight=", wave_weight)
print("DT=", delta_times)
plt.figure()
plt.title("Galerkin Error to expectancy for {}, T={}, grid={}, wave_weight={}, quad={}, {}"
          .format(trial.name, stop_time, grid_size, wave_weight, quadrature_method, quadrature_param))
plt.xlabel("1/Delta_time")
plt.ylabel("discrete l2 error")
plt.xscale('log')
plt.yscale('log')
cache = {}
trial_exp, trial_var = None, None

for max_poly_degree in max_poly_degrees:
    print("Curr degree", max_poly_degree, "of all", max_poly_degrees)
    errors_exp, errors_var = [], []
    error_var = None
    for delta_time in delta_times:
        print("Curr dt", delta_time, "of all", delta_times)
        xs, xs_mesh, exp, var = galerkin_approximation(trial, max_poly_degree, domain, grid_size,
                                               start_time, stop_time, delta_time, wave_weight,
                                               quadrature_method, quadrature_param)
        if trial_exp is None:
            trial_exp = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
        if trial_var is None:
            trial_var = trial.obtain_evaluated_variance(xs, xs_mesh, stop_time)
        error_exp = error_l2_relative(exp, trial_exp)
        errors_exp.append(error_exp)
        if var is not None:
            print(var.shape, trial_var.shape)
            error_var = error_l2_relative(var, trial_var)
            errors_var.append(error_var)
        print("Degree={}, dt={}, error_exp={}, error_var={}".format(max_poly_degree, delta_time, error_exp, error_var))
    print("exp errors for {} {}".format(quadrature_method, trial.name), errors_exp)
    print("var errors for {} {}".format(quadrature_method, trial.name), errors_var)

    plt.plot(1. / np.array(delta_times), errors_exp, "-", label="max poly degree={}, {}".format(max_poly_degree,
                                                                                            quadrature_method))
    if len(delta_times) == len(errors_var):
        plt.plot(1. / np.array(delta_times), errors_var, "-o", label="VAR max poly degree={}, {}".format(max_poly_degree,
                                                                                                quadrature_method))

plt.ylim((1E-13, 1.))
plt.legend()
plt.show()
