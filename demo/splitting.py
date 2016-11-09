import numpy as np
from itertools import repeat, cycle
from math import pi
import matplotlib.pyplot as plt

from diff_equation.splitting import make_klein_gordon_lie_trotter_splitting, make_klein_gordon_strang_splitting
from util.trial import Trial

dimension = 1
grid_size_N = 512
domain = list(repeat([-pi, pi], dimension))
delta_time = 0.001
save_every_x_solution = 1
plot_solutions_count = 5
start_time = 0.
stop_time = 4
show_errors = True
show_reference = True


# Splitting resources for the book: https://www.math.ntnu.no/~holden/operatorsplitting/
# TODO find a trial where beta depends on x, find a trial for two dimensional case
param_g1 = 3  # some parameter greater than one
trial_1 = Trial(lambda xs: np.sin(sum(xs)),
                lambda xs: param_g1 * np.cos(sum(xs)),
                lambda xs, t: np.sin(sum(xs) + param_g1 * t)) \
    .add_parameters("beta", lambda xs: param_g1 ** 2 - 1,
                    "alpha", 1)

# still invalid example since the time derivatives 0th fourier is now only zero at the start (t=0), but only there
offset = 15.9098530420256905490264393 / (2 * pi)  # is (normalized) integral from -pi to pi over exp(-cos(x))
trial_2 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: 2 * np.exp(-np.cos(sum(xs))) - offset,
                lambda xs, t: np.sin(2 * t) * np.exp(-np.cos(sum(xs))) - offset * t) \
    .add_parameters("beta", lambda xs: 4 + np.cos(sum(xs)) + np.sin(sum(xs)) ** 2,
                    "alpha", 1)

param_1, param_2, param_n1, param_3, alpha_g0 = 0.3, 0.5, 2, 1.2, 0.3
assert param_n1 * alpha_g0 < param_3  # to ensure beta > 0
trial_3 = Trial(lambda xs: param_1 * np.cos(param_n1 * sum(xs)),
                lambda xs: param_2 * param_3 * np.cos(param_n1 * sum(xs)),
                lambda xs, t: np.cos(param_n1 * sum(xs)) * (param_1 * np.cos(param_3 * t)
                                                            + param_2 * np.sin(param_3 * t))) \
    .add_parameters("beta", lambda xs: -alpha_g0 * (param_n1 ** 2) + param_3 ** 2,
                    "alpha", alpha_g0)

param_g2 = 2  # some parameter greater than one
trial_4 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: param_g2 * np.sin(sum(xs)),
                lambda xs, t: np.sin(sum(xs)) * np.sin(param_g2 * t)) \
    .add_parameters("beta", lambda xs: param_g2 ** 2 - 1,
                    "alpha", 1)

trial = trial_4

plt.figure()

lie_splitting = make_klein_gordon_lie_trotter_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                                        trial.start_velocity, trial.param["alpha"], trial.param["beta"])
lie_splitting.progress(stop_time, delta_time, save_every_x_solution)
strang_splitting = make_klein_gordon_strang_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                                      trial.start_velocity, trial.param["alpha"], trial.param["beta"])
strang_splitting.progress(stop_time, delta_time, save_every_x_solution)
xs = lie_splitting.get_xs()
xs_mesh = lie_splitting.get_xs_mesh()

plot_counter = 0
plot_every_x_solution = ((stop_time - start_time) / delta_time) / plot_solutions_count

if dimension == 1:
    plt.plot(*xs, trial.start_position(xs_mesh), label="Start position")
    for (t, solution_lie), (_, solution_strang), color in zip(lie_splitting.timed_solutions,
                                                              strang_splitting.timed_solutions,
                                                              cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
        plot_counter += 1
        if plot_counter == plot_every_x_solution:
            plot_counter = 0
            plt.plot(*xs, solution_lie, "o", color=color, label="Lie solution at {}".format(t))
            plt.plot(*xs, solution_strang, "+", color=color, label="Strang solution at {}".format(t))
            if show_reference:
                plt.plot(*xs, trial.reference(xs_mesh, t), color=color, label="Reference at {}".format(t))
    plt.legend()
    plt.title("Splitting methods for Klein Gordon equation, dt={}".format(delta_time))

if show_errors:
    errors_lie = [trial.error(xs_mesh, t, y) for t, y in lie_splitting.timed_solutions]
    errors_strang = [trial.error(xs_mesh, t, y) for t, y in strang_splitting.timed_solutions]
    plt.figure()
    plt.plot(lie_splitting.times(), errors_lie, label="Errors of lie in discrete L2 norm")
    plt.plot(strang_splitting.times(), errors_strang, label="Errors of strang in discrete L2 norm")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.legend()

plt.show()
