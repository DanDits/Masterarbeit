import demo.splitting as ds
import diff_equation.klein_gordon as kg
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
from diff_equation.splitting import Splitting

dimension = 1
grid_size_N = 64 if dimension >= 2 else 128
domain = list(repeat([-np.pi, np.pi], dimension))
trial = ds.trial_1
wave_weight = 0.5

start_time = 0.
delta_times = np.arange(0.25, 0.0001, -0.0005)
stop_time = 1.

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

    for i, splitting in enumerate(splittings):
        splitting.progress(stop_time, delta_time, 0)
        if xs_mesh is None:
            xs_mesh = splitting.get_xs_mesh()
        if len(errors_per_delta_time) <= i:
            errors_per_delta_time.append([])
        errors_per_delta_time[i].append(trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1]))

plt.figure()
plt.title("Order plot for error of splittings for {}, grid size {}, wave_weight={}"
          .format(trial.name, grid_size_N, wave_weight))
plt.xlabel("1/delta_time")
plt.ylabel("Error at T={}".format(stop_time))
plt.xscale('log')
plt.yscale('log')
for splitting, errors in zip(splittings, errors_per_delta_time):
    plt.plot([1/delta_time for delta_time in delta_times], errors, label=splitting.name)
plt.legend()
plt.show()
