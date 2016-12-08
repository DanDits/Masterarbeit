import demo.splitting as ds
import diff_equation.splitting as sp
import numpy as np
from itertools import repeat
from functools import partial
import matplotlib.pyplot as plt

dimension = 1
grid_size_N = 64 if dimension >= 2 else 128
domain = list(repeat([-np.pi, np.pi], dimension))
trial = ds.trial_1

start_time = 0.
delta_times = np.arange(0.25, 0.0001, -0.0001)
stop_time = 1

errors_per_delta_time = []
splittings = []
xs_mesh = None
for delta_time in delta_times:
    print(delta_time)
    factories = [sp.make_klein_gordon_lie_trotter_splitting,
                 sp.make_klein_gordon_strang_splitting,
                 partial(sp.make_klein_gordon_fast_strang_splitting, time_step_size=delta_time),
                 sp.make_klein_gordon_leapfrog_splitting,
                 sp.make_klein_gordon_leapfrog_bad_splitting,
                 partial(sp.make_klein_gordon_leapfrog_fast_splitting, time_step_size=delta_time)]

    splittings = [factory(domain, [grid_size_N], start_time, trial.start_position,
                          trial.start_velocity, trial.alpha, trial.beta)
                  for factory in factories]
    for i, splitting in enumerate(splittings):
        splitting.progress(stop_time, delta_time, 0)
        if xs_mesh is None:
            xs_mesh = splitting.get_xs_mesh()
        if len(errors_per_delta_time) <= i:
            errors_per_delta_time.append([])
        errors_per_delta_time[i].append(trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1]))

plt.figure()
plt.title("Order plot for error of splittings for {}, grid size {}".format(trial.name, grid_size_N))
plt.xlabel("1/delta_time")
plt.ylabel("Error at T={}".format(stop_time))
plt.xscale('log')
plt.yscale('log')
for splitting, errors in zip(splittings, errors_per_delta_time):
    plt.plot([1/delta_time for delta_time in delta_times], errors, label=splitting.name)
plt.legend()
plt.show()
