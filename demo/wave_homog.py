from math import pi
from itertools import repeat, cycle
import numpy as np
import matplotlib.pyplot as plt
from diff_equation.pseudospectral_solver import WaveSolverConfig
from util.analysis import error_l2
from util.animate import animate_1d, animate_2d_surface
from util.trial import Trial

# ----- USER CONFIGS -------------------

# basic config
show_errors = True
plot_references = True  # when not animating, plot in same figure
do_animate = True
grid_n = 512  # power of 2 for best performance of fft, 2^15 for 1d already takes some time
dimension = 1  # plotting only supported for one or two dimensional; higher dimension will require lower grid_n
domain = list(repeat([-pi, pi], dimension))  # intervals with periodic boundary conditions, so a ring in 1d, torus in 2d
anim_pause = 100  # in ms
show_times = np.arange(0, 30, anim_pause / 1000)  # times to evaluate solution for and plot it

trial_1 = Trial(lambda xs: np.sin(sum(xs)),
                lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs, t: 0.5 * (np.sin(sum(xs) + t) + np.sin(sum(xs) - t))) \
    .add_parameters("wave_speed", 1 / np.sqrt(dimension))
wave_period_per_dim_2 = [2, 3, 4]
wave_speed_2 = 0.5
trial_2 = Trial(lambda xs: sum(np.sin(wave_period * x) for wave_period, x in zip(wave_period_per_dim_2, xs)),
                lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs, t: 0.5 * (sum(np.sin(wave_period * (x + wave_speed_2 * t))
                                         + np.sin(wave_period * (x - wave_speed_2 * t))
                                         for wave_period, x in zip(wave_period_per_dim_2, xs)))) \
    .add_parameters("wave_speed", wave_speed_2)
# remark: trial_3 1d reference solution is only valid until the peaks hits the "borders", as d'alembert solution lives
# on total real line and does not respect periodic boundary conditions
trial_3 = Trial(lambda xs: 1 / np.cosh(sum(xs) * 10 / pi) ** 2,
                lambda xs: np.zeros(shape=sum(xs).shape)) \
    .add_parameters("wave_speed", 0.1,
                    "vel_integral", lambda xs: np.zeros(shape=sum(xs).shape))
# just for the looks, no reference solution. In 2d not sure if this is even valid
trial_3_1 = Trial(lambda xs: 1 / np.cosh(sum(xs) * 10 / pi) ** 2,
                  lambda xs: np.cos(sum(xs))) \
    .add_parameters("wave_speed", 1)
wave_period_4 = 2
trial_4 = Trial(lambda xs: np.sin(wave_period_4 * sum(xs)) * np.cos(wave_period_4 * sum(xs)),
                lambda xs: np.cos(xs[0])) \
    .add_parameters("wave_speed", 0.5,
                    "vel_integral", lambda xs: np.sin(xs[0]))
# Interesting config: difference to d'alemberts solution after some time when wave hits boundary
trial_fancy = Trial(lambda xs: 1 / np.cosh(sum(xs) * 10) ** 2,
                    lambda xs: np.cos(sum(xs))) \
    .add_parameters("wave_speed", 1,
                    "vel_integral", lambda xs: np.sin(sum(xs)),
                    "show_times", [0, 0.5, 1, 2, 3, 5, 10],
                    "do_animate", False)
trial_5 = Trial(lambda xs: np.sin(sum(xs)),
                lambda xs: np.exp(np.cos(sum(xs)) ** 2)) \
    .add_parameters("wave_speed", 1.)

def reference_1d_dalembert(start_position, start_velocity_integral, wave_speed, xs, t):
    # this is the d'Alembert reference solution if the indefinite integral of the start_velocity is known
    return (start_position([x + wave_speed * t for x in xs]) / 2
            + start_position([x - wave_speed * t for x in xs]) / 2
            + (start_velocity_integral([x + wave_speed * t for x in xs])
               - start_velocity_integral([x - wave_speed * t for x in xs])) / (2 * wave_speed))


def reference(ref_trial, xs, t):
    if ref_trial.reference:
        return ref_trial.reference(xs, t)
    elif ref_trial.has_parameter("vel_integral") and len(xs) == 1:
        return reference_1d_dalembert(ref_trial.start_position, ref_trial.vel_integral,
                                      ref_trial.wave_speed, xs, t)


def has_reference(ref_trial, dim):
    return ref_trial.reference or (ref_trial.has_parameter("vel_integral") and dim == 1)


# --- CALCULATION AND VISUALIZATION -------------

trial = trial_5

wave_config = WaveSolverConfig(domain, [grid_n], trial.wave_speed)
wave_config.init_solver(0, trial.start_position, trial.start_velocity)

if trial.has_parameter("show_times"):
    show_times = trial.show_times
if trial.has_parameter("do_animate"):
    do_animate = trial.do_animate
wave_config.solve(show_times)

if show_errors and has_reference(trial, dimension):
    errors = [error_l2(y, reference(trial, wave_config.xs_mesh, t)) for t, y in wave_config.timed_solutions()]
    plt.figure()
    plt.plot(wave_config.times(), errors, label="Errors in discrete L2 norm")
    plt.xlabel("Time")
    plt.ylabel("Error")

if dimension == 1:
    if do_animate:
        animate_1d(wave_config.xs[0], [wave_config.solutions()], show_times, 100)  # pause between frames in ms
    else:
        # all times in one figure
        plt.figure()
        for (time, sol), color in zip(wave_config.timed_solutions(), cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
            plt.plot(*wave_config.xs, sol.real, '.', color=color, label="Solution at time=" + str(time))
            if plot_references:
                plt.plot(*wave_config.xs, reference(trial, wave_config.xs_mesh, time), color=color,
                         label="Reference solution at time=" + str(time))
        plt.legend()
        plt.title("Wave equation solution by pseudospectral spatial method and exact time solution\nwith N="
                  + str(grid_n) + " grid points")
        plt.show()
elif dimension == 2:
    animate_2d_surface(*wave_config.xs, wave_config.solutions(), show_times, anim_pause)

