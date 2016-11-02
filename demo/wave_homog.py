from math import pi
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
from pseudospectral.solver import wave_solution
from util.animate import animate_1d, animate_2d_surface

do_animate = True
grid_n = 128  # power of 2 for best performance of fft
wave_speed = 0.7  # > 0
dimension = 2  # plotting only supported for one or two dimensional; higher dimension will require lower grid_n
domain = list(repeat([-pi, pi], dimension))  # intervals with periodic boundary conditions, so a torus in 2d
show_times = np.arange(0, 30, 0.1)  # times to evaluate solution for and plot it
# show_times = [0, 1, 5, 10]


def start_position(xs, delta=0):
    return np.sin(np.sqrt(sum(x ** 2 for x in xs) + delta))


def start_velocity(xs):  # zeroth fourier coefficient must be zero! (so for example periodic function)
    # return np.cos(sum(x for x in xs)) + np.sin(sum(x for x in xs))
    return np.sin(sum(x for x in xs)) + np.cos(sum(x for x in xs))


def reference_1d(xs, t):
    # this is the d'Alembert reference solution if the indefinite integral of the start_velocity is known

    def start_velocity_integral(inner_xs, delta):
        sum_x = sum(x for x in inner_xs)
        # return np.sin(sum_x + delta)  # for start_velocity=cos(sum_x)
        return (sum_x + delta + np.sin(sum_x + delta) * np.cos(sum_x + delta)) / 2  # for start_velocity = cos^2
    return (start_position(xs, wave_speed * t) / 2
            + start_position(xs, -wave_speed * t) / 2
            + (start_velocity_integral(xs, wave_speed * t)
               - start_velocity_integral(xs, -wave_speed * t)) / (2 * wave_speed))


x_result, t_result, y_result = wave_solution(domain, [grid_n],
                                             0, start_position, start_velocity, wave_speed, show_times)
if len(x_result) == 1:  # 1D plots
    if do_animate:
        animate_1d(x_result[0], y_result, show_times, 100)  # pause between frames in ms
    else:
        # all times in one figure

        plt.plot(*x_result, reference_1d(x_result, t_result[1]) + 0.05,
                 label="Reference solution at time=" + str(t_result[1]))
        plt.plot(*x_result, reference_1d(x_result, t_result[2]) + 0.05,
                 label="Reference solution at time=" + str(t_result[2]))
        for time, sol in zip(t_result, y_result):
            plt.plot(*x_result, sol.real, label="Solution at time=" + str(time))
        plt.legend()
        plt.title("Wave equation solution by pseudospectral spatial method and exact time solution\nwith N="
                  + str(grid_n) + " grid points")
        plt.show()
elif len(x_result) == 2:
    animate_2d_surface(*x_result, y_result, show_times, 100)
