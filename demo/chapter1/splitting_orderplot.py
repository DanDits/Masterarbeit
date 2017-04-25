from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np

import demo.chapter1.splitting as ds
import diff_equation.klein_gordon as kg
from diff_equation.splitting import Splitting
from util.analysis import error_l2_relative

dimension = 1
grid_size_N = 64 if dimension >= 2 else 1024
domain = list(repeat([-np.pi, np.pi], dimension))
trial = ds.trial_4
wave_weight = 0.9

start_time = 0.
delta_times = np.arange(0.25, 0.0001, -0.0005)
stop_time = 2.

errors_per_delta_time = []
splittings = []
xs_mesh = None


def make_wave_linhyp_configs():
    return kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], trial.alpha, trial.beta, wave_weight)


def make_leapfrog_configs():
    return kg.make_klein_gordon_leapfrog_configs(domain, [grid_size_N], trial.alpha, trial.beta)


for delta_time in delta_times:
    print(delta_time)
    # each splitting needs its own config, as the splitting re initializes the config's solvers!
    splittings = [
                  Splitting.make_lie(*make_wave_linhyp_configs(), "Lie",
                                     start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_strang(*make_wave_linhyp_configs(), "Strang",
                                        start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_strang(*(reversed(make_wave_linhyp_configs())), "ReversedStrang",
                                        start_time, trial.start_position, trial.start_velocity),
                  Splitting.make_fast_strang(*make_wave_linhyp_configs(), "FastStrang",
                                             start_time, trial.start_position, trial.start_velocity, delta_time)]

    for i, splitting in enumerate(splittings):
        splitting.progress(splitting.approx_steps_to_end_time(stop_time, delta_time), delta_time, 0)
        if xs_mesh is None:
            xs_mesh = splitting.get_xs_mesh()
        if len(errors_per_delta_time) <= i:
            errors_per_delta_time.append([])
        trial.error_function = error_l2_relative
        errors_per_delta_time[i].append(trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1]))
        if i==1:
            print(delta_time, errors_per_delta_time[i][-1])

size = "xx-large"
plt.figure()
print(trial.name)
plt.title("Konvergenz verschiedener Splitting Varianten, $N={}$, $w={}$, $T={}$"
          .format(grid_size_N, wave_weight, stop_time),
          fontsize=size)
plt.xlabel("$\\frac{1}{\\tau}$", fontsize=size)
plt.ylabel("Relativer Fehler in diskreter L2-Norm", fontsize=size)
plt.xscale('log')
plt.yscale('log')
dts = [1/delta_time for delta_time in delta_times]
for splitting, errors in zip(splittings, errors_per_delta_time):
    plt.plot(dts, errors, label=splitting.name)
plt.plot(dts, np.array(dts) ** -2, label="Referenz $\\tau^{2}$", linestyle="dashed", color="black")
plt.xlim((dts[0], dts[-1]))
plt.ylim(ymax=1.)
plt.legend(fontsize=size)
plt.show()
