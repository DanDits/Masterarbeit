
from util.analysis import error_l2_relative
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.galerkin import galerkin_approximation
import stochastic_equations.galerkin.projection_coefficients as pc
import diff_equation.solver_config as config
from numpy.linalg import norm

domain = [(-np.pi, np.pi)]
trial = st.trial_1  # requires us to get expectancy and variances at all stop_times!!
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
delta_time = 0.0001
stop_time = 3
max_poly_degree = 15
wave_weight = 1.
xs, xs_mesh = config.SolverConfig.make_spatial_discretization(domain, [grid_size])

if len(trial.variable_distributions) == 1:
    quadrature_method = "full_tensor"
    quadrature_param = [max_poly_degree + 5] * len(trial.variable_distributions)
else:
    quadrature_method = "sparse"
    quadrature_param = max_poly_degree


coeffs_calc = pc.solution_coefficients_calculator(trial, max_poly_degree, quadrature_method, quadrature_param)
coeffs = coeffs_calc(stop_time, xs_mesh)
coeffs_norm = norm(coeffs, 2, axis=0)

plt.figure()
plt.plot(range(len(coeffs_norm)), coeffs_norm)
plt.yscale('log')
plt.show()
print(coeffs_norm)
