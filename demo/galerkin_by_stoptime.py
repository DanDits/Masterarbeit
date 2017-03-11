# Plots the error of the galerkin approximation over different stop times using the same delta time and approximation
# degree (which should be good enough so that the splitting error does not fall into account)
from itertools import accumulate
from stochastic_equations.galerkin.projection_coefficients import solution_coefficients_calculator
from util.analysis import error_l2
import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
from stochastic_equations.galerkin.galerkin import galerkin_approximation


domain = [(-np.pi, np.pi)]
trial = st.trial_3  # requires us to get expectancy and variances at all stop_times!!
grid_size = trial.get_parameter("grid_size", 128)
start_time = 0.
delta_time = 0.0001
steps_delta = 300  # if 1, this will plot for every single delta_time step making fast strang splitting to a normal strang, if higher the plot is refined worse but calculation is faster
stop_time = 10
steps_list = [steps_delta] * int(stop_time / delta_time / steps_delta)
max_poly_degree = 25
wave_weight = 1.
plot_coeff_error_only = False

if len(trial.variable_distributions) == 1:
    quadrature_method = "full_tensor"
    quadrature_param = [max_poly_degree + 3] * len(trial.variable_distributions)

    quadrature_method = "general_purpose"
    quadrature_param = None
else:
    quadrature_method = "sparse"
    quadrature_param = max_poly_degree

errors_exp, errors_var, errors_coeffs = [], [], []
stop_times = []
coeff_calc = None
for total_steps, (xs, xs_mesh, exp, var, quadrature_nodes_count, coeffs) \
        in zip(accumulate(steps_list), galerkin_approximation(trial, max_poly_degree, domain,
                                                              grid_size, start_time, steps_list,
                                                              delta_time, wave_weight,
                                                              quadrature_method, quadrature_param,
                                                              retrieve_coeffs=True)):
    if not coeff_calc:
        coeff_calc = solution_coefficients_calculator(trial, max_poly_degree, quadrature_method, quadrature_param)
    stop_time = total_steps * delta_time

    reference_coeffs = coeff_calc(stop_time, xs_mesh)

    coeffs_norm = error_l2(coeffs, reference_coeffs, axis=0) ** 2  # errors for each degree
    assert len(coeffs_norm) == max_poly_degree + 1  # in 1d
    coeffs_norm = np.sum(coeffs_norm)  # sum over all degrees
    errors_coeffs.append(coeffs_norm)

    stop_times.append(stop_time)
    error_var = None

    trial_exp = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
    trial_var = trial.obtain_evaluated_variance(xs, xs_mesh, stop_time)
    error_exp = error_l2(exp, trial_exp)
    errors_exp.append(error_exp)
    if var is not None and trial_var is not None:
        error_var = error_l2(var, trial_var)
        errors_var.append(error_var)
    print("Error for ", trial.name, "dt=", delta_time, "stop_time=", stop_time, "degree", max_poly_degree,
          "quad=", quadrature_param, "EXP:", error_exp, "VAR:", error_var, "COEFFS:", coeffs_norm)


#def target(p):
#    return lambda time: np.exp((p / 2 + 1) * time)


def fit_exponential(x_data, y_data):
    # fits y=Ae^(Bx) <=> log(y)=log(A)+Bx
    coeffs = np.polyfit(x_data, np.log(y_data), deg=1)  # highest power first
    A = np.exp(coeffs[1])
    B = coeffs[0]
    print("Fitting result: {}*e^({}x)".format(A, B))
    return lambda x: A * np.exp(B * x)

plt.figure()
plt.title("Galerkin-Approximation, $P={}$, $\\tau={}$".format(max_poly_degree, delta_time))
if not plot_coeff_error_only:
    plt.plot(stop_times, errors_exp, label="Erwartungswert $\\left\\Vert\\hat{u}_0-\\hat{v}_0\\right\\Vert$")
    if len(stop_times) == len(errors_var):
        plt.plot(stop_times, errors_var, label="Varianz $\\left\\Vert\\sum_{m=1}^P\\hat{u}_m^2-\\hat{v}_m^2\\right\\Vert$")
    #fitted_func = fit_exponential(stop_times, errors_exp)
    #fitted_x = np.linspace(min(stop_times), max(stop_times), num=50, endpoint=True)
    #plt.plot(fitted_x, fitted_func(fitted_x), label="Fitted")

print("StopTimes:", stop_times)
print("Errors coeffs:", errors_coeffs)
plt.plot(stop_times, errors_coeffs, linestyle="dotted",
         label="Fourier $\\sum_{m=0}^P\\left\\Vert\\hat{u}_m-\\hat{v}_m\\right\\Vert^2$")

plt.yscale('log')
plt.xlabel('Stoppzeit $T$')
plt.ylabel('Absoluter Fehler in diskreter L2-Norm')
plt.ylim(ymax=1)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
