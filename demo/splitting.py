import numpy as np
from itertools import repeat, cycle
from math import pi
import matplotlib.pyplot as plt
import time
from util.animate import animate_1d, animate_2d_surface

from diff_equation.splitting import make_klein_gordon_lie_trotter_splitting, make_klein_gordon_strang_splitting, \
    make_klein_gordon_fast_strang_splitting, \
    make_klein_gordon_lie_trotter_reversed_splitting, \
    make_klein_gordon_strang_reversed_splitting
from util.trial import Trial

dimension = 1
grid_size_N = 128 if dimension >= 2 else 512
domain = list(repeat([-pi, pi], dimension))
delta_time = 0.001
save_every_x_solution = 1
plot_solutions_count = 5
start_time = 0.
stop_time = 4
show_errors = True
show_reference = True
do_animate = False

param_g1 = 3  # some parameter greater than one
alpha_1 = 0.2  # smaller than param_g1 ** 2 / dimension to ensure beta>0
trial_1 = Trial(lambda xs: np.sin(sum(xs)),
                lambda xs: param_g1 * np.cos(sum(xs)),
                lambda xs, t: np.sin(sum(xs) + param_g1 * t)) \
    .add_parameters("beta", lambda xs: param_g1 ** 2 - len(xs) * alpha_1,
                    "alpha", lambda: alpha_1)

alpha_small = 1. / 4.  # smaller than 1/3 to ensure beta>0
trial_2 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: np.exp(np.cos(sum(xs)) ** 2) * np.cos(sum(xs)),
                lambda xs, t: np.sin(t) * np.exp(np.cos(sum(xs)) ** 2) * np.cos(sum(xs))) \
    .add_parameters("beta", lambda xs: 1 + len(xs) * alpha_small *
                                           (-1 + 4 * np.sin(sum(xs)) ** 2 - 2 * np.cos(sum(xs)) ** 2
                                            + 4 * np.sin(sum(xs)) ** 2 * np.cos(sum(xs)) ** 2
                                            + 2 * np.sin(sum(xs)) ** 2),
                    "alpha", lambda: alpha_small)

# probably needs to be adapted for higher dimensional case
param_1, param_2, param_n1, param_3, alpha_g0 = 0.3, 0.5, 2, 1.2, 0.3
assert param_n1 * alpha_g0 < param_3  # to ensure beta > 0
trial_3 = Trial(lambda xs: param_1 * np.cos(param_n1 * sum(xs)),
                lambda xs: param_2 * param_3 * np.cos(param_n1 * sum(xs)),
                lambda xs, t: np.cos(param_n1 * sum(xs)) * (param_1 * np.cos(param_3 * t)
                                                            + param_2 * np.sin(param_3 * t))) \
    .add_parameters("beta", lambda xs: -alpha_g0 * (param_n1 ** 2) + param_3 ** 2,
                    "alpha", lambda: alpha_g0)

param_g2 = 2  # some parameter greater than one
alpha_4 = 0.5  # smaller than param_g2 ** 2 / dimension to ensure beta>0
trial_4 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: param_g2 * np.sin(sum(xs)),
                lambda xs, t: np.sin(sum(xs)) * np.sin(param_g2 * t)) \
    .add_parameters("beta", lambda xs: param_g2 ** 2 - len(xs) * alpha_4,
                    "alpha", lambda: alpha_4)

y = 2  # bigger than 1; gets unstable very quickly (after about 1.0)
trial_5 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: np.sin(sum(xs)),
                lambda xs, t: np.sin(sum(xs)) * np.sin(t / y) * y) \
    .add_parameters("beta", lambda xs: (1 - 1 / y) / y,
                    "alpha", lambda: 1 / y ** 2)
y = 3  # bigger than one
trial_6 = Trial(lambda xs: 2 * np.sin(sum(xs)),
                lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs, t: np.sin(sum(xs) + t * y) + np.sin(sum(xs) - t * y)) \
    .add_parameters("beta", lambda xs: y ** 2 - y,  # y^2 - alpha(y)
                    "alpha", lambda: y)

trial = trial_4

measure_start = time.time()
lie_splitting = make_klein_gordon_lie_trotter_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                                        trial.start_velocity, trial.alpha, trial.beta)
lie_splitting.progress(stop_time, delta_time, save_every_x_solution)
print("Lie took", (time.time() - measure_start))
measure_start = time.time()
strang_splitting = make_klein_gordon_strang_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                                      trial.start_velocity, trial.alpha, trial.beta)
strang_splitting.progress(stop_time, delta_time, save_every_x_solution)
print("Strang took", (time.time() - measure_start))
measure_start = time.time()
fast_strang_splitting = make_klein_gordon_fast_strang_splitting(domain, [grid_size_N], start_time,
                                                                trial.start_position, trial.start_velocity,
                                                                trial.alpha, trial.beta,
                                                                delta_time)
fast_strang_splitting.progress(stop_time, delta_time)  # intermediate solutions discarded at the end anyways
print("Fast strang took", (time.time() - measure_start))

lie_reversed_splitting = make_klein_gordon_lie_trotter_reversed_splitting(domain, [grid_size_N], start_time,
                                                                          trial.start_position,
                                                                          trial.start_velocity,
                                                                          trial.alpha, trial.beta)
lie_reversed_splitting.progress(stop_time, delta_time, save_every_x_solution)

strang_reversed_splitting = make_klein_gordon_strang_reversed_splitting(domain, [grid_size_N], start_time,
                                                                        trial.start_position,
                                                                        trial.start_velocity,
                                                                        trial.alpha, trial.beta)
strang_reversed_splitting.progress(stop_time, delta_time, save_every_x_solution)

xs = lie_splitting.get_xs()
xs_mesh = lie_splitting.get_xs_mesh()

plot_counter = 0
plot_every_x_solution = ((stop_time - start_time) / delta_time) / plot_solutions_count

if dimension == 1:
    if do_animate:
        animate_1d(xs[0], strang_splitting.solutions(), strang_splitting.times(), 1)
    else:
        plt.figure()
        plt.plot(*xs, trial.start_position(xs_mesh), label="Start position")
        for (t, solution_lie), (_, solution_strang), color in zip(lie_splitting.timed_solutions,
                                                                  strang_splitting.timed_solutions,
                                                                  cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
            plot_counter += 1
            if plot_counter == plot_every_x_solution:
                plot_counter = 0
                # not filled color circles so we can see when strang solution is very close!
                plt.plot(*xs, solution_lie, "o", markeredgecolor=color, markerfacecolor="None",
                         label="Lie solution at {}".format(t))
                plt.plot(*xs, solution_strang, "+", color=color, label="Strang solution at {}".format(t))
                if show_reference:
                    plt.plot(*xs, trial.reference(xs_mesh, t), color=color, label="Reference at {}".format(t))
        plt.legend()
        plt.title("Splitting methods for Klein Gordon equation, dt={}".format(delta_time))
elif dimension == 2:
    animate_2d_surface(xs[0], xs[1], strang_splitting.solutions(), strang_splitting.times(), 100)

if show_errors:
    errors_lie = [trial.error(xs_mesh, t, y) for t, y in lie_splitting.timed_solutions]
    errors_strang = [trial.error(xs_mesh, t, y) for t, y in strang_splitting.timed_solutions]
    errors_lie_reversed = [trial.error(xs_mesh, t, y) for t, y in lie_reversed_splitting.timed_solutions]
    errors_strang_reversed = [trial.error(xs_mesh, t, y) for t, y in strang_reversed_splitting.timed_solutions]
    plt.figure()
    plt.plot(lie_splitting.times(), errors_lie, label="Errors of lie in discrete L2 norm")
    plt.plot(strang_splitting.times(), errors_strang, label="Errors of strang in discrete L2 norm")
    plt.plot(lie_reversed_splitting.times(), errors_lie_reversed, label="Errors of reversed lie in discrete L2 norm")
    plt.plot(strang_reversed_splitting.times(), errors_strang_reversed,
             label="Errors of reversed strang in discrete L2 norm")
    plt.title("Splitting method errors for Klein Gordon, dt={}".format(delta_time))
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.legend()

    error_improved_strang = [trial.error(xs_mesh, t, y) for t, y in fast_strang_splitting.timed_solutions]
    print("Error of lie splitting at end:{:.2E}".format(errors_lie[-1]))
    print("Error of strang splitting at end:{:.2E}".format(errors_strang[-1]))
    print("Error of fast strang splitting at end:{:.2E}".format(error_improved_strang[-1]))
    print("Error of reversed lie splitting at end:{:.2E}".format(errors_lie_reversed[-1]))
    print("Error of reversed strang splitting at end:{:.2E}".format(errors_strang_reversed[-1]))

plt.show()
