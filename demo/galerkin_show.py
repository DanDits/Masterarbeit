from util.analysis import error_l2
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.galerkin import galerkin_expectancy
from stochastic_equations.galerkin.newgalerkin import galerkin_approximation


# OBSERVATIONS:
# trial1: with delta time = 0.000001 it loses accuracy, this loss seems independent of quadrature count and is the same for poly degree 2,3,4
#         does not depend on wave_weight
domain = [(-np.pi, np.pi)]
trial = st.trial_1
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
stop_time = trial.get_parameter("stop_time", 0.5)
delta_times = [0.1, 0.05, 0.01, 0.001]
max_poly_degrees = [0, 1, 2, 3, 4]
wave_weight = 1.  # does not seem to have much influence (at least on trial5); but can have on stability as this problem is kinda irregular!

quadrature_method = "full_tensor"
# trial7: >=10 if degree <=6, >=15 if degree <= 10,  if degree <= 15
quadrature_param = [20] * len(trial.variable_distributions)
#quadrature_method = "sparse"
#quadrature_param = 4
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
trial_exp = None



for max_poly_degree in max_poly_degrees:
    errors = []
    for delta_time in delta_times:
        xs, xs_mesh, exp = galerkin_approximation(trial, max_poly_degree, domain, grid_size,
                                               start_time, stop_time, delta_time, wave_weight,
                                               quadrature_method, quadrature_param)
        if trial_exp is None:
            trial_exp = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
        error = error_l2(exp, trial_exp)
        errors.append(error)
        print("Degree={}, dt={}, error={}".format(max_poly_degree, delta_time, error))
    print("errors for {} {}".format(quadrature_method, trial.name), errors)
    plt.plot(1. / np.array(delta_times), errors, "-", label="max poly degree={}, {}".format(max_poly_degree,
                                                                                            quadrature_method))

plt.ylim((1E-13, 1.))
plt.legend()
plt.show()
