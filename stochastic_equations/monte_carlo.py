import random
import numpy as np
from stochastic_equations.stochastic_trial import StochasticTrial
from polynomial_chaos import distributions
from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from itertools import repeat
import matplotlib.pyplot as plt
from scipy.integrate import quad
from util.analysis import error_l2

dimension = 1
grid_size_N = 128 if dimension >= 2 else 512
domain = list(repeat([-np.pi, np.pi], dimension))
delta_time = 0.001
start_time = 0.
stop_time = 0.5
simulations_count = 1000

order_factor = 2  # the factor of the step number between two consecutive solutions used to estimate order of converg.
steps_for_order_estimate = [int(simulations_count / (order_factor ** i)) for i in range(3)]
solutions_for_order_estimate = []

# y[0] > 1
trial_1 = StochasticTrial([distributions.make_uniform(2, 3)],
                          lambda xs, ys: 2 * np.sin(sum(xs)),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: np.sin(sum(xs) + t * ys[0]) + np.sin(sum(xs) - t * ys[0])) \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    "expectancy", lambda xs, t: 1 / t * (np.cos(sum(xs) - 3 * t) - np.cos(sum(xs) + 3 * t)
                                                         + np.cos(sum(xs) + 2 * t) - np.cos(sum(xs) - 2 * t)))
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
trial_3 = StochasticTrial([distributions.make_uniform(left_3, right_3)],  # y[0] bigger than 2
                          lambda xs, ys: 1 / (np.sin(sum(xs)) + ys[0]),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: np.cos(t) / (np.sin(sum(xs)) + ys[0])) \
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

# TODO 2d trial and visualization
trial = trial_2_1

last_solutions_sum = None
splitting_xs = None
splitting_xs_mesh = None
expectancy = None
errors = []
solution_at_step = []
random.seed(1)
for i in range(1, simulations_count + 1):
    if i % 100 == 0:
        print("Simulation", i)  # about 7s for 100 simulations with leapfrog (1min with lie trotter), 512, 0.001, [0,.5]
    trial.randomize()

    lie_splitting = make_klein_gordon_leapfrog_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                                         trial.start_velocity, trial.alpha,
                                                         trial.beta)
    lie_splitting.progress(stop_time, delta_time)

    last_solution = lie_splitting.solutions()[-1]
    if last_solutions_sum is not None:
        last_solutions_sum += last_solution
    else:
        splitting_xs = lie_splitting.get_xs()
        splitting_xs_mesh = lie_splitting.get_xs_mesh()
        if trial.has_parameter("expectancy"):
            expectancy = trial.expectancy(splitting_xs_mesh, lie_splitting.times()[-1])
        elif trial.reference is not None:
            expectancy = trial.calculate_expectancy(splitting_xs, lie_splitting.times()[-1], trial.reference)
        last_solutions_sum = last_solution
    if i % 1000 == 0:
        solution_at_step.append((i, last_solutions_sum / (i + 1)))
    if i in steps_for_order_estimate:
        solutions_for_order_estimate.append(last_solutions_sum / (i + 1))
    if expectancy is not None:
        errors.append(error_l2(expectancy, last_solutions_sum / (i + 1)))
last_solutions_sum *= 1 / simulations_count
solution_at_step.append((simulations_count, last_solutions_sum))

plt.figure()
plt.title("Solutions at time={}, dt={}".format(stop_time, delta_time))
if expectancy is not None:
    plt.plot(*splitting_xs, expectancy, label="Expectancy")
for step, solution in solution_at_step:
    plt.plot(*splitting_xs, solution, label="Mean with N={}".format(step))
plt.legend()
#plt.savefig("/home/daniel/PycharmProjects/Masterarbeit/images/mc_plots_trial4.png")

if len(errors) > 0:
    plt.figure()
    plt.title("Errors to expectancy, dt={}".format(delta_time))
    plt.plot(range(len(errors)), errors)
    plt.xlabel("Simulation")
    plt.ylabel("Error")
    plt.yscale('log')

if len(solutions_for_order_estimate) == 3:
    order = np.log((solutions_for_order_estimate[0] - solutions_for_order_estimate[1])
                   / (solutions_for_order_estimate[1] - solutions_for_order_estimate[2])) / np.log(order_factor)

    total_order = np.log(sum(1. / order_factor ** (order * 2)) / order.size) / (np.log(1 / order_factor) * 2)
    print("Estimated convergence rate:", total_order.real)
    plt.figure()
    plt.title("Monte Carlo convergence rate to expectancy={:.2E}, dt={}, N={}"
              .format(total_order.real, delta_time, grid_size_N))
    plt.plot(*splitting_xs, order, label="Point wise estimated convergence rate")
    #plt.savefig("/home/daniel/PycharmProjects/Masterarbeit/images/mc_convergence_trial4.png")
    plt.legend()

plt.show()

# TODO how to enforce constraints for alpha, beta>0 ? implicitly? discard if invalid?
