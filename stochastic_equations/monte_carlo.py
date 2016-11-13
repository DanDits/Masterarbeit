import numpy as np
from stochastic_equations.stochastic_trial import StochasticTrial
from polynomial_chaos import distributions
from diff_equation.splitting import make_klein_gordon_lie_trotter_splitting
from itertools import repeat
import matplotlib.pyplot as plt

from util.analysis import error_l2

distribs = [distributions.make_uniform(2, 3)]
dimension = 1
grid_size_N = 128 if dimension >= 2 else 512
domain = list(repeat([-np.pi, np.pi], dimension))
delta_time = 0.001
start_time = 0.
stop_time = 1.
simulations_count = 10000

trial = StochasticTrial(distribs,
                        lambda xs, ys: 2 * np.sin(sum(xs)),
                        lambda xs, ys: np.zeros(shape=sum(xs).shape),
                        lambda xs, t, ys: np.sin(sum(xs) + t * ys[0]) + np.sin(sum(xs) - t * ys[0])) \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    "expectancy", lambda xs, t: 1 / t * (np.cos(sum(xs) - 3 * t) - np.cos(sum(xs) + 3 * t)
                                                         + np.cos(sum(xs) + 2 * t) - np.cos(sum(xs) - 2 * t)))
last_solutions_sum = None
splitting_xs = None
splitting_xs_mesh = None
expectancy = None
errors = []
for i in range(simulations_count):
    if i % 100 == 0:
        print("Simulation", i)  # about one minute for 100 simulations with lie trotter, 512, 0.001, [0,1]
    trial.randomize()

    lie_splitting = make_klein_gordon_lie_trotter_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                                            trial.start_velocity, trial.alpha,
                                                            trial.beta)
    lie_splitting.progress(stop_time, delta_time)

    last_solution = lie_splitting.solutions()[-1]
    if last_solutions_sum is not None:
        last_solutions_sum += last_solution
    else:
        splitting_xs = lie_splitting.get_xs()
        splitting_xs_mesh = lie_splitting.get_xs_mesh()
        expectancy = trial.expectancy(splitting_xs_mesh, stop_time)
        last_solutions_sum = last_solution
    errors.append(error_l2(expectancy, last_solutions_sum / (i + 1)))
last_solutions_sum *= 1 / simulations_count

plt.figure()
plt.title("Solutions at time={}, dt={}".format(stop_time, delta_time))
plt.plot(*splitting_xs, expectancy, label="Expectancy")
plt.plot(*splitting_xs, last_solutions_sum, label="Mean with N={}".format(simulations_count))
plt.legend()

plt.figure()
plt.title("Errors to expectancy, dt={}".format(delta_time))
plt.plot(range(len(errors)), errors)
plt.xlabel("Simulation")
plt.ylabel("Error")
plt.yscale('log')
plt.show()

# TODO how to enforce constraints for alpha, beta>0 ? implicitly? discard if invalid?
