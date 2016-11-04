from math import pi
from itertools import repeat, cycle
import numpy as np
from util.analysis import error_l2
import matplotlib.pyplot as plt
from diff_equation.ode_solver import linhyp_solution
from util.animate import animate_1d, animate_2d_surface

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


# Interesting: Show sp1, sv1, b1 for a long time (t=70), N=128, too high frequency of sin(x(t+1)) => nice patterns!

def start_position(xs):
    return np.sin(sum(xs))  # sp1
    # return np.cos(sum(xs))  # sp2


def start_velocity(xs):
    return sum(xs) * np.cos(sum(xs))  # sv1
    # return -np.sin(sum(xs))  # sv2


def beta(xs):
    return sum(xs) ** 2  # b1
    # return np.ones(shape=sum(xs).shape)  # b2


def reference(normal_xs, t):
    xs = np.meshgrid(*normal_xs, sparse=True)
    return np.sin(sum(xs) * (t + 1))  # for sp1, sv1, b1 in any dimension
    # return np.cos(sum(xs) + t)  # for sp2, sv2, b2 in 1d


x_result, t_result, y_result = linhyp_solution(domain, [grid_n],
                                               0, start_position, start_velocity, beta, show_times)


if show_errors:
    errors = [error_l2(y, reference(x_result, t)) for t, y in zip(t_result, y_result)]
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
                plt.plot(*x_result, reference(x_result, time), color=color,
                         label="Reference solution at time=" + str(time))
        plt.legend()
        plt.show()
if dimension == 2:
    animate_2d_surface(*x_result, y_result, show_times, anim_pause)
