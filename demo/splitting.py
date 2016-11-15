import numpy as np
from itertools import repeat, cycle
from functools import partial
from math import pi
import matplotlib.pyplot as plt
import time
from util.animate import animate_1d, animate_2d_surface

from diff_equation.splitting import make_klein_gordon_lie_trotter_splitting, make_klein_gordon_strang_splitting, \
    make_klein_gordon_fast_strang_splitting, \
    make_klein_gordon_lie_trotter_reversed_splitting, \
    make_klein_gordon_strang_reversed_splitting, make_klein_gordon_leapfrog_splitting, \
    make_klein_gordon_leapfrog_reversed_splitting
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
do_animate = True

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

y_5 = 0.95  # in (0,1), if smaller, the time step size needs to be smaller as well
trial_5 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: np.sin(sum(xs)),
                lambda xs, t: np.sin(sum(xs)) * np.sin(t / y_5) * y_5) \
    .add_parameters("beta", lambda xs: 1 / y_5 ** 2 - 1 / y_5,  # 1/y^2 - alpha(y)
                    "alpha", lambda: 1 / y_5)
y_6 = 3  # bigger than one, solution equivalent to # =2 * sin(sum(xs)) * cos(t*y_6)
trial_6 = Trial(lambda xs: 2 * np.sin(sum(xs)),
                lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs, t: np.sin(sum(xs) + t * y_6) + np.sin(sum(xs) - t * y_6)) \
    .add_parameters("beta", lambda xs: y_6 ** 2 - y_6,  # y^2 - alpha(y)
                    "alpha", lambda: y_6)
trial_frog = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: 2 * np.exp(-np.cos(sum(xs))),
                lambda xs, t: np.sin(2 * t) * np.exp(-np.cos(sum(xs)))) \
    .add_parameters("beta", lambda xs: 4 + np.cos(sum(xs)) + np.sin(sum(xs)) ** 2,
                    "alpha", lambda: 1,
                    "frog_only", True)
y_frog_2 = 3  # > 2
trial_frog2 = Trial(lambda xs: 1 / (np.sin(sum(xs)) + y_frog_2),
                    lambda xs: np.zeros(shape=sum(xs).shape),
                    lambda xs, t: np.cos(t) / (np.sin(sum(xs)) + y_frog_2)) \
    .add_parameters("beta", lambda xs: 1 + (y_frog_2 - 2) * (np.sin(sum(xs)) / (np.sin(sum(xs)) + y_frog_2)
                                                             + 2 * np.cos(sum(xs)) ** 2
                                                             / (np.sin(sum(xs)) + y_frog_2) ** 2),
                    "alpha", lambda: y_frog_2 - 2,
                    "frog_only", True)
trial_frog3 = Trial(lambda xs: np.sin(sum(xs)),
                    lambda xs: np.sin(sum(xs)) ** 2) \
    .add_parameters("beta", lambda xs: 2 + np.sin(sum(xs) + 1),
                    "alpha", lambda: 1 + 0.5 + 3 * 0.5,
                    "frog_only", True)
trial = trial_frog3

splitting_factories = [make_klein_gordon_lie_trotter_splitting, make_klein_gordon_lie_trotter_reversed_splitting,
                       make_klein_gordon_strang_splitting, make_klein_gordon_strang_reversed_splitting,
                       partial(make_klein_gordon_fast_strang_splitting, time_step_size=delta_time),
                       make_klein_gordon_leapfrog_splitting, make_klein_gordon_leapfrog_reversed_splitting]
if trial.has_parameter("frog_only"):
    splitting_factories = [make_klein_gordon_leapfrog_splitting, make_klein_gordon_leapfrog_reversed_splitting]


splittings = [factory(domain, [grid_size_N], start_time, trial.start_position,
                      trial.start_velocity, trial.alpha, trial.beta)
              for factory in splitting_factories]
for splitting in splittings:
    measure_start = time.time()
    splitting.progress(stop_time, delta_time, save_every_x_solution)
    print(splitting.name, "took", (time.time() - measure_start))


ref_splitting = splittings[0]
ref_splitting_2 = splittings[2] if len(splittings) > 2 else splittings[-1]
xs = ref_splitting.get_xs()
xs_mesh = ref_splitting.get_xs_mesh()

plot_counter = 0
plot_every_x_solution = ((stop_time - start_time) / delta_time) / plot_solutions_count

if dimension == 1:
    if do_animate:
        animate_1d(xs[0], ref_splitting.solutions(), ref_splitting.times(), 1)
    else:
        plt.figure()
        plt.plot(*xs, trial.start_position(xs_mesh), label="Start position")

        for (t, solution_0), (_, solution_1), color in zip(ref_splitting.timed_solutions,
                                                                  ref_splitting_2.timed_solutions,
                                                                  cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
            plot_counter += 1
            if plot_counter == plot_every_x_solution:
                plot_counter = 0
                # not filled color circles so we can see when strang solution is very close!
                plt.plot(*xs, solution_0, "o", markeredgecolor=color, markerfacecolor="None",
                         label="{} solution at {}".format(ref_splitting.name, t))
                plt.plot(*xs, solution_1, "+", color=color,
                         label="{} solution at {:.2E}".format(ref_splitting_2.name, t))
                if show_reference:
                    plt.plot(*xs, trial.reference(xs_mesh, t), color=color, label="Reference at {}".format(t))
        plt.legend()
        plt.title("Splitting methods for Klein Gordon equation, dt={}".format(delta_time))
elif dimension == 2:
    animate_2d_surface(xs[0], xs[1], ref_splitting.solutions(), ref_splitting.times(), 100)

if show_errors:
    plt.figure()
    for splitting in splittings:
        errors = [trial.error(xs_mesh, t, y) for t, y in splitting.timed_solutions]

        print("Error of {} splitting at end:{:.2E}".format(splitting.name, errors[-1]))
        plt.plot(splitting.times(), errors, label="Errors of {} in discrete L2 norm".format(splitting.name))

    plt.title("Splitting method errors for Klein Gordon, dt={}, N={}".format(delta_time, grid_size_N))
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.legend()

plt.show()
