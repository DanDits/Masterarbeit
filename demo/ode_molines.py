from math import pi
from itertools import repeat, cycle
import numpy as np
import matplotlib.pyplot as plt
from diff_equation.ode_solver import LinhypSolverConfig
from util.animate import animate_1d, animate_2d_surface
from util.trial import Trial

# ----- USER CONFIGS -------------------

# basic config
show_errors = True
plot_references = True  # when not animating, plot in same figure
do_animate = True
grid_n = 128  # amount of grid points per dimension
dimension = 2  # plotting only supported for one or two dimensional
domain = list(repeat([-pi, pi], dimension))  # intervals with periodic boundary conditions, so a ring in 1d, torus in 2d
anim_pause = 100  # in ms
show_times = np.arange(0, 30, anim_pause / 1000)  # times to evaluate solution for and plot it
# show_times = [0, 1, 2, 3]


# Interesting: Show trial_sq in 1d for a long time (t=70), N=128, too high frequency of sin(x(t+1)) => nice patterns!

trial_sq = Trial(lambda xs: np.sin(sum(xs)),  # start position
                 lambda xs: sum(xs) * np.cos(sum(xs)),  # start velocity
                 lambda xs, t: np.sin(sum(xs) * (t + 1)))\
                .add_parameters("beta", lambda xs: sum(xs) ** 2)

trial_one = Trial(lambda xs: np.cos(sum(xs)),
                  lambda xs: -np.sin(sum(xs)),
                  lambda xs, t: np.cos(sum(xs) + t))\
            .add_parameters("beta", lambda xs: np.ones(shape=sum(xs).shape))

trial_onezero = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                      lambda xs: np.cos(sum(xs)),
                      lambda xs, t: np.cos(sum(xs)) * np.sin(t))\
                .add_parameters("beta", lambda xs: np.ones(shape=sum(xs).shape))
param_g1 = 2  # some parameter greater than one
trial_4 = Trial(lambda xs: np.sin(sum(xs)),
                lambda xs: param_g1 * np.cos(sum(xs)),
                lambda xs, t: np.sin(sum(xs) + param_g1 * t)) \
    .add_parameters("beta", lambda xs: param_g1 ** 2)

trial = trial_sq

linhyp_config = LinhypSolverConfig(domain, [grid_n], trial.param["beta"])
linhyp_config.init_solver(0, trial.start_position, trial.start_velocity)
linhyp_config.solve(show_times)

if show_errors:
    errors = [trial.error(linhyp_config.xs_mesh, t, y) for t, y in linhyp_config.timed_solutions]
    plt.figure()
    plt.plot(linhyp_config.times(), errors, label="Errors in discrete L2 norm")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.show()

if dimension == 1:
    if do_animate:
        animate_1d(linhyp_config.xs[0], linhyp_config.solutions(), show_times, 100)  # pause between frames in ms
    else:
        # all times in one figure
        plt.figure()
        for (time, sol), color in zip(linhyp_config.timed_solutions, cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
            plt.plot(*linhyp_config.xs, sol.real, '.', color=color, label="Solution at time=" + str(time))
            if plot_references:
                plt.plot(*linhyp_config.xs, trial.reference(linhyp_config.xs_mesh, time), color=color,
                         label="Reference solution at time=" + str(time))
        plt.legend()
        plt.show()
if dimension == 2:
    animate_2d_surface(*linhyp_config.xs, linhyp_config.solutions(), show_times, anim_pause)
