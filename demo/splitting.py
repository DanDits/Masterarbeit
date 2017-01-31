import numpy as np
from itertools import repeat, cycle
from diff_equation.splitting import Splitting
from math import pi
import matplotlib.pyplot as plt
import time
from scipy.special import jn as bessel_first
from scipy.special import yn as bessel_second
from diff_equation.pseudospectral_solver import OffsetWaveSolverConfig
from util.animate import animate_1d, animate_2d_surface
from util.analysis import error_l2_relative
import diff_equation.klein_gordon as kg
from util.trial import Trial

dimension = 2
grid_size_N = 64 if dimension >= 2 else 128
domain = list(repeat([-pi, pi], dimension))
delta_time = 0.001
save_every_x_solution = 1
plot_solutions_count = 5
start_time = 0.
stop_time = 1
wave_weight = 0.5
show_errors = True
show_reference = True
do_animate = True

param_g1 = 7  # some parameter greater than one
alpha_1 = 2  # smaller than param_g1 ** 2 / dimension to ensure beta>0
trial_1 = Trial(lambda xs: np.sin(sum(xs)),
                lambda xs: param_g1 * np.cos(sum(xs)),
                lambda xs, t: np.sin(sum(xs) + param_g1 * t),
                "Trial1") \
    .add_parameters("beta", lambda xs: param_g1 ** 2 - len(xs) * alpha_1,
                    "alpha", lambda: alpha_1,
                    "offset", (param_g1 ** 2 - dimension * alpha_1) / 2)

alpha_small = 1. / 4.  # smaller than 1/3 to ensure beta>0
trial_2 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: np.exp(np.cos(sum(xs)) ** 2) * np.cos(sum(xs)),
                lambda xs, t: np.sin(t) * np.exp(np.cos(sum(xs)) ** 2) * np.cos(sum(xs)),
                name="Trial2") \
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
                                                            + param_2 * np.sin(param_3 * t)),
                name="Trial3") \
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
                   lambda xs, t: np.sin(2 * t) * np.exp(-np.cos(sum(xs))),
                   "TrialFrog") \
    .add_parameters("beta", lambda xs: 4 + len(xs) * (np.cos(sum(xs)) + np.sin(sum(xs)) ** 2),
                    "alpha", lambda: 1,
                    "offset", 1)
y_frog_2 = 3  # > 2
trial_frog2 = Trial(lambda xs: 1 / (np.sin(sum(xs)) + y_frog_2),
                    lambda xs: np.zeros(shape=sum(xs).shape),
                    lambda xs, t: np.cos(t) / (np.sin(sum(xs)) + y_frog_2)) \
    .add_parameters("beta", lambda xs: 1 + (y_frog_2 - 2) * (np.sin(sum(xs)) / (np.sin(sum(xs)) + y_frog_2)
                                                             + 2 * np.cos(sum(xs)) ** 2
                                                             / (np.sin(sum(xs)) + y_frog_2) ** 2),
                    "alpha", lambda: y_frog_2 - 2,
                    "offset", 0.2)
trial_frog3 = Trial(lambda xs: np.sin(sum(xs)),
                    lambda xs: np.sin(sum(xs)) ** 2) \
    .add_parameters("beta", lambda xs: 2 + np.sin(np.sqrt(sum(x ** 2 for x in xs)) + 1),
                    "alpha", lambda: 1 + 0.5 + 3 * 0.5,
                    "offset", 1)
# requires small step size and 1d!
trial_frog4 = Trial(lambda xs: np.sin(sum(xs)),
                    lambda xs: np.zeros(shape=sum(xs).shape)) \
    .add_parameters("beta", lambda xs: np.where(xs[0] > 0, xs[0], 10),
                    "alpha", lambda: 0.7)

bessel_A, bessel_B, bessel_alpha, bessel_beta, bessel_C1, bessel_C2 = 1, -1, 1, 1, 10, 0
def bessel_xi(xs, t):
    temp = (bessel_alpha ** 2) * ((t + bessel_C1) ** 2) - ((sum(xs) + bessel_C2) ** 2)
    return np.sqrt(bessel_beta) * np.sqrt(temp) / bessel_alpha
# not a valid trial as the spatial derivative is not periodic, but works for some small time spans in the center!
trial_bessel = Trial(lambda xs: (bessel_A * bessel_first(0, bessel_xi(xs, 0))
                                 + bessel_B * bessel_second(0, bessel_xi(xs, 0))),
                     lambda xs: (bessel_A * bessel_first(1, bessel_xi(xs, 0)) * (-bessel_beta * bessel_C1)
                                 / bessel_xi(xs, 0)
                                 + bessel_B * bessel_second(1, bessel_xi(xs, 0)) * (-bessel_beta * bessel_C1)
                                 / bessel_xi(xs, 0)),
                     lambda xs, t: (bessel_A * bessel_first(0, bessel_xi(xs, t))
                                    + bessel_B * bessel_second(0, bessel_xi(xs, t)))) \
    .add_parameters("beta", lambda xs: bessel_beta,
                    "alpha", lambda: bessel_alpha ** 2)

if __name__ == "__main__":
    trial = trial_frog

    trial.error_function = error_l2_relative
    offset_wave_solver = None
    if trial == trial_1:
        offset_wave_solver = OffsetWaveSolverConfig(domain, [grid_size_N], np.sqrt(alpha_1), param_g1 ** 2 - alpha_1)
        offset_wave_solver.init_solver(start_time, trial.start_position, trial.start_velocity)
    elif trial == trial_3:
        offset_wave_solver = OffsetWaveSolverConfig(domain, [grid_size_N], np.sqrt(alpha_g0),
                                                    -alpha_g0 * (param_n1 ** 2) + param_3 ** 2)
        offset_wave_solver.init_solver(start_time, trial.start_position, trial.start_velocity)


    def make_wave_linhyp_configs():
        return kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], trial.alpha, trial.beta, wave_weight)


    def make_leapfrog_configs():
        return kg.make_klein_gordon_leapfrog_configs(domain, [grid_size_N], trial.alpha, trial.beta)


    splittings = [Splitting.make_lie(*make_leapfrog_configs(), "LeapfrogLie",
                                     start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_strang(*make_leapfrog_configs(), "LeapfrogStrang",
                                        start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_fast_strang(*make_leapfrog_configs(), "LeapfrogFastStrang",
                                             start_time, trial.start_position, trial.start_velocity, delta_time),
                  Splitting.make_lie(*make_wave_linhyp_configs(), "Lie",
                                     start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_strang(*make_wave_linhyp_configs(), "Strang",
                                        start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_strang(*(reversed(make_wave_linhyp_configs())), "ReversedStrang",
                                        start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_fast_strang(*make_wave_linhyp_configs(), "FastStrang",
                                             start_time, trial.start_position, trial.start_velocity, delta_time)]

    for splitting in splittings:
        measure_start = time.time()
        splitting.progress(stop_time, delta_time, save_every_x_solution)
        print(splitting.name, "took", (time.time() - measure_start))

    ref_splitting = splittings[0]
    ref_splitting_2 = splittings[5] if len(splittings) > 5 else splittings[-1]
    result_xs = ref_splitting.get_xs()
    xs_mesh = ref_splitting.get_xs_mesh()

    ref = [trial.reference(xs_mesh, t) for t in ref_splitting.times()]
    plot_counter = 0
    plot_every_x_solution = ((stop_time - start_time) / delta_time) / plot_solutions_count

    if dimension == 1:
        if do_animate:
            animate_1d(result_xs[0], [ref_splitting.solutions(), ref_splitting_2.solutions(), ref],
                       ref_splitting.times(), 1,
                       labels=[ref_splitting.name, ref_splitting_2.name, "Exact"])
        else:
            plt.figure()
            plt.plot(*result_xs, trial.start_position(xs_mesh), label="Start position")

            for (curr_t, solution_0), (_, solution_1), color in zip(ref_splitting.timed_solutions(),
                                                                    ref_splitting_2.timed_solutions(),
                                                                    cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
                plot_counter += 1
                if plot_counter == plot_every_x_solution:
                    plot_counter = 0
                    # not filled color circles so we can see when strang solution is very close!
                    plt.plot(*result_xs, solution_0, "o", markeredgecolor=color, markerfacecolor="None",
                             label="{} solution at {}".format(ref_splitting.name, curr_t))
                    plt.plot(*result_xs, solution_1, "+", color=color,
                             label="{} solution at {:.2E}".format(ref_splitting_2.name, curr_t))
                    if offset_wave_solver is not None:
                        offset_wave_solver.solve([curr_t])
                        plt.plot(*result_xs, offset_wave_solver.solutions()[-1], ".", color=color,
                                 label="OW solution at {:.2E}".format(offset_wave_solver.times()[-1]))

                    if show_reference and trial.reference is not None:
                        plt.plot(*result_xs, trial.reference(xs_mesh, curr_t),
                                 color=color, label="Reference at {}".format(curr_t))
            plt.legend()
            plt.title("Splitting methods for Klein Gordon equation, dt={}".format(delta_time))
    elif dimension == 2:
        animate_2d_surface(result_xs[0], result_xs[1], ref_splitting.solutions(), ref_splitting.times(), 100)

    if show_errors and trial.reference is not None:
        plt.figure()
        for splitting, marker in zip(splittings, "xxoooDDDD..-----"):
            errors = [trial.error(xs_mesh, t, y) for t, y in splitting.timed_solutions()]
            print("Error of {} splitting at end:{:.2E}".format(splitting.name, errors[-1]))
            plt.plot(splitting.times(), errors, marker, label="Errors of {}".format(splitting.name))

        if offset_wave_solver is not None:
            last_time = splittings[0].times()[-1]
            offset_wave_solver.solve([last_time])
            print("Error of OffsetWaveSolver at end:{:.2E}".format(trial.error(xs_mesh, last_time,
                                                                               offset_wave_solver.solutions()[-1])))
        plt.title("Splitting method errors for Klein Gordon, dt={}, N={}".format(delta_time, grid_size_N))
        plt.xlabel("Time")
        plt.ylabel("Error in discrete L2-norm")
        plt.yscale('log')
        plt.legend()

    plt.show()
