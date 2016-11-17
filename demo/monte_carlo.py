import numpy as np
from stochastic_equations.stochastic_trial import StochasticTrial
from polynomial_chaos import distributions
from stochastic_equations.sampling.monte_carlo import simulate
from itertools import repeat
import matplotlib.pyplot as plt
from scipy.integrate import quad
from util.animate import animate_2d_surface

dimension = 1
grid_size_N = 128 if dimension >= 2 else 512
do_calculate_expectancy = True  # dimension == 1  # for 128*128 in dim=2 already takes about 30s
domain = list(repeat([-np.pi, np.pi], dimension))
delta_time = 0.001
start_time = 0.
stop_time = 0.5
simulations_count = 300

# the factor of the step number between two consecutive solutions used to estimate order of convergence
order_factor = 2  # >=2, integer

# y[0] > 1
trial_1 = StochasticTrial([distributions.make_uniform(2, 3)],
                          lambda xs, ys: 2 * np.sin(sum(xs)),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: np.sin(sum(xs) + t * ys[0]) + np.sin(sum(xs) - t * ys[0])) \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    # at t=0.5: 0.624096 sin(x)
                    "expectancy", lambda xs, t: 1 / t * (np.cos(sum(xs) - 3 * t) - np.cos(sum(xs) + 3 * t)
                                                         + np.cos(sum(xs) + 2 * t) - np.cos(sum(xs) - 2 * t)),
                    # at t=0.5: 0.630645 sin(x), the solution evaluated at the expectancy of y[0]
                    "orientation_func", lambda xs, t: np.sin(sum(xs) + t * 2.5) + np.sin(sum(xs) - t * 2.5))
# y[0] in (0,1), if smaller, the time step size needs to be smaller as well
left_2, right_2 = 0.25, 0.75
trial_2 = StochasticTrial([distributions.make_uniform(left_2, right_2)],
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0]) \
    .add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                    "alpha", lambda ys: 1 / ys[0],
                    "expectancy", lambda xs, t: (np.sin(sum(xs)) * (1 / (right_2 - left_2))
                                                 * quad(lambda y: np.sin(t / y) * y, left_2, right_2)[0]))
# y[0] in (0,1), is enforced by random variable which can take any real value!
trial_2_1 = StochasticTrial([distributions.gaussian],
                            lambda xs, ys: np.zeros(shape=sum(xs).shape),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.2 + 0.7 * np.sin(y) ** 2])
trial_2_1.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])
left_3, right_3 = 2.5, 3  # y[0] bigger than 2
trial_3 = StochasticTrial([distributions.make_uniform(-1, 1)],  # y[0] bigger than 2 enforced by random variable
                        lambda xs, ys: 1 / (np.sin(sum(xs)) + ys[0]),
                        lambda xs, ys: np.zeros(shape=sum(xs).shape),
                        lambda xs, t, ys: np.cos(t) / (np.sin(sum(xs)) + ys[0]),
                        # from U(-1,1) to U(left_3, right_3)
                        random_variables=[lambda y: (right_3 - left_3) / 2 * (y + 1) + left_3]) \
    .add_parameters("beta", lambda xs, ys: 1 + (ys[0] - 2) * (np.sin(sum(xs)) / (np.sin(sum(xs)) + ys[0])
                                                              + 2 * np.cos(sum(xs)) ** 2
                                                              / (np.sin(sum(xs)) + ys[0]) ** 2),
                    "alpha", lambda ys: ys[0] - 2,
                    "expectancy", lambda xs, t: np.cos(t) / (right_3 - left_3)
                                                * (np.log(np.sin(sum(xs)) + right_3)
                                                   - np.log(np.sin(sum(xs)) + left_3)))
trial_4 = StochasticTrial([distributions.gaussian, distributions.make_uniform(0, 1),
                           distributions.make_exponential(1)],
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, ys: np.sin(sum(xs)) ** 2,
                          random_variables=[lambda y: np.exp(y)]) \
    .add_parameters("beta", lambda xs, ys: 2 + np.sin(xs[0] + ys[2]),
                    "alpha", lambda ys: 1 + 0.5 * ys[0] + 3 * ys[1])

trial = trial_3

splitting_xs, splitting_xs_mesh, expectancy, errors, solutions, solutions_for_order_estimate = \
    simulate(trial, simulations_count, [simulations_count // 3, simulations_count // 2, simulations_count],
             domain, grid_size_N, start_time,
             stop_time, delta_time,
             do_calculate_expectancy=do_calculate_expectancy, order_factor=order_factor)

if dimension == 1:
    plt.figure()
    plt.title("Solutions at time={}, dt={}".format(stop_time, delta_time))
    if trial.has_parameter("orientation_func"):
        plt.plot(*splitting_xs, trial.orientation_func(splitting_xs_mesh, stop_time), 'o', label="...for orientation")
    if expectancy is not None:
        plt.plot(*splitting_xs, expectancy, label="Expectancy")
    for step, solution in solutions:
        plt.plot(*splitting_xs, solution, label="Mean with N={}".format(step))
    plt.legend()
elif dimension == 2:
    animate_2d_surface(splitting_xs[0], splitting_xs[1], [sol for _, sol in solutions],
                       [step for step, sol in solutions], 100)

if len(errors) > 0:
    plt.figure()
    plt.title("Errors to expectancy, dt={}".format(delta_time))
    plt.plot(range(len(errors)), errors)
    plt.xlabel("Simulation")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.show()

if len(solutions_for_order_estimate) == 3:
    order = np.log((solutions_for_order_estimate[0] - solutions_for_order_estimate[1])
                   / (solutions_for_order_estimate[1] - solutions_for_order_estimate[2])) / np.log(order_factor)

    total_order = np.log(np.sum(1. / order_factor ** (order * 2)) / order.size) / (np.log(1 / order_factor) * 2)

    if dimension == 1:
        plt.figure()
        plt.title("Monte Carlo convergence rate to expectancy={:.2E}, dt={}, N={}"
                  .format(total_order.real, delta_time, grid_size_N))
        plt.plot(*splitting_xs, order, label="Point wise estimated convergence rate")
        plt.legend()

        plt.show()
