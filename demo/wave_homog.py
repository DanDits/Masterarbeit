from math import pi
from itertools import repeat, cycle
import numpy as np
import matplotlib.pyplot as plt
from diff_equation.pseudospectral_solver import wave_solution, error_l2
from util.animate import animate_1d, animate_2d_surface

# ----- USER CONFIGS -------------------

# basic config
show_errors = False
plot_references = True  # when not animating, plot in same figure
do_animate = True
grid_n = 256  # power of 2 for best performance of fft, 2^15 for 1d already takes some time
param_1 = 2  # parameter that can be used for start position and reference solutions
wave_speed = 1 / param_1  # > 0
dimension = 2  # plotting only supported for one or two dimensional; higher dimension will require lower grid_n
domain = list(repeat([-pi, pi], dimension))  # intervals with periodic boundary conditions, so a ring in 1d, torus in 2d
show_times = np.arange(0, 30, 0.1)  # times to evaluate solution for and plot it


def start_position(xs, delta=0):
    return np.sin(np.sqrt(sum((x + delta) ** 2 for x in xs)))  # not periodic!
    # return np.sin(param_1 * (sum(x for x in xs) + delta))
    # return 1 / np.cosh((sum(x + delta for x in xs)) * 10) ** 2
    # return np.sin(param_1 * (sum(x for x in xs) + delta)) * np.cos(param_1 * (sum(x for x in xs) + delta))


def start_velocity(xs):  # zeroth fourier coefficient must be zero! (so for example periodic function)
    sum_x = sum(x for x in xs)
    # return np.cos(sum_x) + np.sin(sum_x)
    # return np.cos(sum_x)
    # return np.zeros(shape=sum_x.shape)
    return np.cos(param_1 * sum_x) ** 2 - np.sin(param_1 * sum_x) ** 2


def start_velocity_integral(inner_xs, delta):
    sum_x = sum(x for x in inner_xs)
    return np.zeros(shape=sum_x.shape)
    # return np.sin(sum_x + delta)  # for start_velocity=cos(sum_x)
    # return (sum_x + delta + np.sin(sum_x + delta) * np.cos(sum_x + delta)) / 2  # for start_velocity = cos^2


def reference_1d_dalembert(xs, t):
    # this is the d'Alembert reference solution if the indefinite integral of the start_velocity is known
    return (start_position(xs, wave_speed * t) / 2
            + start_position(xs, -wave_speed * t) / 2
            + (start_velocity_integral(xs, wave_speed * t)
               - start_velocity_integral(xs, -wave_speed * t)) / (2 * wave_speed))


def reference_1d(xs, t):
    # either use dalembert for 1d and if integral of start velocity is known
    # return reference_1d_dalembert(xs, t)

    # or give reference solution directly
    return np.sin(param_1 * sum(x for x in xs) + t) * np.cos(param_1 * sum(x for x in xs) + t)

""" # Interesting config: difference to d'alemberts solution after some time when wave hits boundary
show_errors = True
plot_references = True
do_animate = False
grid_n = 256
wave_speed = 1
dimension = 1
domain = list(repeat([-pi, pi], dimension))
show_times = [0, 0.5, 1, 2, 3, 5, 10]
start_position = lambda xs, delta=0: 1 / np.cosh((sum(x + delta for x in xs)) * 10) ** 2
start_velocity = lambda xs: np.zeros(shape=sum(x for x in xs).shape)
start_velocity_integral = lambda xs, delta: np.zeros(shape=sum(x for x in xs).shape)
reference_1d = lambda xs, t: reference_1d_dalembert(xs, t)"""

# --- CALCULATION AND VISUALIZATION -------------

x_result, t_result, y_result = wave_solution(domain, [grid_n],
                                             0, start_position, start_velocity, wave_speed, show_times)

if dimension == 1 and show_errors:
    errors = [error_l2(y, reference_1d(x_result, t)) for t, y in zip(t_result, y_result)]
    plt.figure()
    plt.plot(t_result, errors, label="Errors in discrete L2 norm")
    plt.xlabel("Time")
    plt.ylabel("Error")

if dimension == 1:
    if do_animate:
        animate_1d(x_result[0], y_result, show_times, 100)  # pause between frames in ms
    else:
        # all times in one figure
        plt.figure()
        for time, sol, color in zip(t_result, y_result, cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
            plt.plot(*x_result, sol.real, '.', color=color, label="Solution at time=" + str(time))
            if plot_references:
                plt.plot(*x_result, reference_1d(x_result, time), color=color,
                         label="Reference solution at time=" + str(time))
        plt.legend()
        plt.title("Wave equation solution by pseudospectral spatial method and exact time solution\nwith N="
                  + str(grid_n) + " grid points")
        plt.show()
elif dimension == 2:
    animate_2d_surface(*x_result, y_result, show_times, 100)

