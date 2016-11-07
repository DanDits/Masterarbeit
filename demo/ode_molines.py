from math import pi
from itertools import repeat, cycle
import numpy as np
import matplotlib.pyplot as plt
from diff_equation.ode_solver import linhyp_solution
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
                .set_config("beta", lambda xs: sum(xs) ** 2)

trial_one = Trial(lambda xs: np.cos(sum(xs)),
                  lambda xs: -np.sin(sum(xs)),
                  lambda xs, t: np.cos(sum(xs) + t))\
            .set_config("beta", lambda xs: np.ones(shape=sum(xs).shape))

trial_onezero = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                      lambda xs: np.cos(sum(xs)),
                      lambda xs, t: np.cos(sum(xs)) * np.sin(t))\
                .set_config("beta", lambda xs: np.ones(shape=sum(xs).shape))

trial = trial_onezero

x_result, t_result, y_result = linhyp_solution(domain, [grid_n],
                                               0, trial.start_position, trial.start_velocity, trial.config["beta"],
                                               show_times)
x_mesh = np.meshgrid(*x_result, sparse=True)

if show_errors:
    errors = [trial.error(x_mesh, t, y) for t, y in zip(t_result, y_result)]
    plt.figure()
    plt.plot(t_result, errors, label="Errors in discrete L2 norm")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.show()

if dimension == 1:
    if do_animate:
        animate_1d(x_result[0], y_result, show_times, 100)  # pause between frames in ms
    else:
        # all times in one figure
        plt.figure()
        for time, sol, color in zip(t_result, y_result, cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
            plt.plot(*x_result, sol.real, '.', color=color, label="Solution at time=" + str(time))
            if plot_references:
                plt.plot(*x_result, trial.reference(x_mesh, time), color=color,
                         label="Reference solution at time=" + str(time))
        plt.legend()
        plt.show()
if dimension == 2:
    animate_2d_surface(*x_result, y_result, show_times, anim_pause)
