from math import pi
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
from diff_equation.pseudospectral_solver import heat_solution
from util.animate import animate_1d, animate_2d


do_animate = True
grid_n = 128  # power of 2 for best performance of fft
thermal_diffusivity = 0.1  # > 0
dimension = 2  # plotting only supported for one or two dimensional, higher dimension will require lower grid_n
domain = list(repeat([-pi, pi], dimension))  # intervals with periodic boundary conditions, so a torus
show_times = np.arange(0, 20, 0.1)  # times to evaluate solution for and plot it


# starting condition for homogeneous heat equation with periodic boundary equation in given domain
def start_condition(xs):
    #return np.sin(np.sqrt(sum(x ** 2 for x in xs))) ** 2
    # return 1 / np.cosh(10 * sum(xs) / math.pi) ** 2
    return np.where(1 > np.abs(sum(xs)), np.ones(shape=sum(xs).shape), 0)  # discontinuous block in 1D


x_result, t_result, y_result = heat_solution(domain, [grid_n],
                                             0, start_condition, thermal_diffusivity, show_times)

if len(x_result) == 1:  # 1D plots
    if do_animate:
        animate_1d(x_result[0], [y_result], show_times, 100)  # pause between frames in ms
    else:
        # all times in one figure
        for time, sol in zip(t_result, y_result):
            plt.plot(*x_result, sol, label="Solution at time=" + str(time))
        plt.legend()
        plt.title("Heat equation solution by pseudospectral spatial method and exact time solution\nwith N="
                  + str(grid_n) + " grid points")
        plt.show()
elif len(x_result) == 2:  # 2D plots
    if do_animate:
        animate_2d(*x_result, y_result, show_times, 100)
    else:
        # one figure per time
        for time, sol in zip(t_result, y_result):
            plt.figure()
            plt.pcolormesh(*x_result, sol, vmin=-1, vmax=1)  # fix color bar range over different figures
            plt.colorbar()
            plt.show()
