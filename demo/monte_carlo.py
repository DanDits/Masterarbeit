import numpy as np
import demo.stochastic_trials as st
from stochastic_equations.sampling.monte_carlo import simulate
from itertools import repeat
import matplotlib.pyplot as plt

from util.analysis import error_l2
from util.animate import animate_2d_surface

dimension = 1
grid_size_N = 128 if dimension >= 2 else 128
do_calculate_expectancy = True  # dimension == 1  # for 128*128 in dim=2 already takes about 30s
domain = list(repeat([-np.pi, np.pi], dimension))
delta_time = 0.001
start_time = 0.
stop_time = 0.5
save_last_solution = True
simulations_count = 200000
do_quasi_monte_carlo = True

# the factor of the step number between two consecutive solutions used to estimate order of convergence
order_factor = 10  # >=2, integer

trial = st.trial_7

splitting_xs, splitting_xs_mesh, expectancy, errors, solutions, solutions_for_order_estimate = \
    simulate(trial, simulations_count, [simulations_count // 3, simulations_count // 2, simulations_count],
             domain, grid_size_N, start_time,
             stop_time, delta_time,
             do_calculate_expectancy=do_calculate_expectancy, order_factor=order_factor,
             quasi_monte_carlo=do_quasi_monte_carlo)
print("Quasi Monte Carlo:", do_quasi_monte_carlo)
if dimension == 1:
    plt.figure()
    plt.title("Solutions at time={}, dt={}, qMC={}".format(stop_time, delta_time, do_quasi_monte_carlo))
    if trial.has_parameter("orientation_func"):
        plt.plot(*splitting_xs, trial.orientation_func(splitting_xs_mesh, stop_time), 'o', label="...for orientation")
    if expectancy is not None:
        plt.plot(*splitting_xs, expectancy, label="Expectancy")
    for step, solution in solutions:
        plt.plot(*splitting_xs, solution, label="Mean with N={}".format(step))
    if save_last_solution:
        np.save('../data/{}_{}, {}, {}, {}'.format("qmc" if do_quasi_monte_carlo else "mc",
                                                   solutions[-1][0], trial.name, stop_time, grid_size_N),
                solutions[-1][1])
    plt.legend()
elif dimension == 2:
    animate_2d_surface(splitting_xs[0], splitting_xs[1], [sol for _, sol in solutions],
                       [step for step, sol in solutions], 100)

if len(errors) > 0:
    plt.figure()
    plt.title("Errors to expectancy, dt={}, qMC={}".format(delta_time, do_quasi_monte_carlo))
    plt.plot(range(len(errors)), errors)
    plt.xlabel("Simulation")
    plt.ylabel("Error")
    plt.xscale('log')
    plt.yscale('log')

if len(solutions_for_order_estimate) == 3:
    if expectancy is None:
        order = np.log((solutions_for_order_estimate[0] - solutions_for_order_estimate[1])
                       / (solutions_for_order_estimate[1] - solutions_for_order_estimate[2])) / np.log(order_factor)
        total_order = np.log(np.sum(1. / order_factor ** (order * 2)) / order.size) / (np.log(1 / order_factor) * 2)
        print("Total convergence order without expectancy:", total_order)
    else:
        err_order = [error_l2(sol_order, expectancy) for sol_order in solutions_for_order_estimate]
        print("Errors for order:", err_order)
        total_order = np.log((err_order[0] - err_order[1]) / (err_order[1] - err_order[2])) / np.log(order_factor)
        print("Total convergence order of error (estimate using expectancy):", total_order)

plt.show()
