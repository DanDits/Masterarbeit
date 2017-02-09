import numpy as np
import demo.stochastic_trials as st
from stochastic_equations.sampling.monte_carlo import simulate
from itertools import repeat
import matplotlib.pyplot as plt

from util.analysis import error_l2
from util.animate import animate_2d_surface
from itertools import cycle


dimension = 1
grid_size_N = 128 if dimension >= 2 else 128
do_calculate_expectancy = True  # dimension == 1  # for 128*128 in dim=2 already takes about 30s
domain = list(repeat([-np.pi, np.pi], dimension))
delta_time = 0.001
start_time = 0.
stop_time = 0.5
save_last_solution = True
simulations_count = 100000
do_quasi_monte_carlo = True

# the factor of the step number between two consecutive solutions used to estimate order of convergence
order_factor = 10  # >=2, integer

trial = st.trial_discont_simple_gauss

splitting_xs, splitting_xs_mesh, expectancy, variance, expectancy_errors, variance_errors, \
    solutions, solutions_for_order_estimate = \
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
    if variance is not None:
        plt.plot(*splitting_xs, variance, "D", label="Variance")
    for (step, current_expectancy, current_variance), color in zip(solutions,
                                                                   cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
        plt.plot(*splitting_xs, current_expectancy, color=color, label="Exp. with N={}".format(step))
        plt.plot(*splitting_xs, current_variance, "D", color=color, label="Var. with N={}".format(step))
    if save_last_solution:
        np.save('../data/{}_exp, {}, {}, {}, {}'.format("qmc" if do_quasi_monte_carlo else "mc",
                                                   solutions[-1][0], trial.name, stop_time, grid_size_N),
                solutions[-1][1])
        np.save('../data/{}_var, {}, {}, {}, {}'.format("qmc" if do_quasi_monte_carlo else "mc",
                                                   solutions[-1][0], trial.name, stop_time, grid_size_N),
                solutions[-1][2])
    plt.legend()
elif dimension == 2:
    animate_2d_surface(splitting_xs[0], splitting_xs[1], [sol for _, sol, __ in solutions],
                       [step for step, sol, _ in solutions], 100)

if len(expectancy_errors) > 0 or len(variance_errors) > 0:
    plt.figure()
    print(trial.name)
    plt.title("{}Monte-Carlo-Methode, $T={}$".format("Quasi-" if do_quasi_monte_carlo else "",
                                                     stop_time),
              fontsize="x-large")
    if len(expectancy_errors) > 0:
        Ns = np.array(range(len(expectancy_errors)))
        plt.plot(Ns, expectancy_errors, label="Erwartungswert")
        plt.plot(Ns, 0.08 * Ns ** (-0.5), label="Referenzlinie $0.08R^{-0.5}$")
    if len(variance_errors) > 0:
        plt.plot(range(len(variance_errors)), variance_errors, label="Varianz")

    plt.xlabel("Anzahl an Durchläufen", fontsize="x-large")
    plt.ylabel("Fehler in diskreter L2-Norm", fontsize="x-large")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize="x-large")

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
