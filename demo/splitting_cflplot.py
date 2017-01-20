import demo.splitting as ds
import diff_equation.klein_gordon as kg
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
from diff_equation.splitting import Splitting
from util.analysis import error_l2_relative


dimension = 1
grid_size_N = 64 if dimension >= 2 else 512
domain = list(repeat([-np.pi, np.pi], dimension))
trial = ds.trial_frog2


start_time = 0.
dt = 0.0005
delta_times = np.arange(0.001, 0.25, dt)
stop_time = 1.

dw = 0.02
weights = np.arange(0., 1. + dw, dw)
stability_threshold = 1.  # if bigger consider unstable
xs_mesh = None
cfl_constants = []

for weight in weights:
    def make_wave_linhyp_configs():
        return kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], trial.alpha, trial.beta, weight)


    splitting = Splitting.make_strang(*make_wave_linhyp_configs(), "Strang",
                                      start_time, trial.start_position, trial.start_velocity)
    pre_prev_error, prev_error = -1, -1
    for delta_time in delta_times:
        splitting.progress(stop_time, delta_time, 0)
        if xs_mesh is None:
            xs_mesh = splitting.get_xs_mesh()

        trial.error_function = error_l2_relative
        error = trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1])

        if error > prev_error > 0:
            if error > 1 or error / prev_error > 10:
                cfl_constants.append((2*np.pi / grid_size_N) / delta_time)
                break

        pre_prev_error = prev_error
        prev_error = error

    else:
        cfl_constants.append(0)
print(cfl_constants)
plt.figure()
plt.plot(weights, cfl_constants)
plt.show()
