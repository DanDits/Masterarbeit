import demo.splitting as ds
from diff_equation.splitting import Splitting
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
import diff_equation.klein_gordon as kg

dimension = 1
grid_size_N = 2 ** 16
domain = list(repeat([-np.pi, np.pi], dimension))


wave_weights = np.arange(0., 1.001, 0.02)
trials = [ds.trial_1, ds.trial_2, ds.trial_3, ds.trial_frog]
start_time = 0.
delta_time = 0.01
stop_time = 1

xs_mesh = None

plt.figure()
plt.title("Error per wave weight, dt={}, T={}, grid size={}".format(delta_time, stop_time, grid_size_N))
plt.xlabel("Wave weight")
plt.ylabel("Error")
plt.yscale('log')
for trial in trials:
    errors = []
    for weight in wave_weights:
        print(weight)
        configs = kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], trial.alpha, trial.beta, weight)
        splitting = Splitting.make_fast_strang(*configs, "FastStrang", start_time, trial.start_position,
                                               trial.start_velocity, delta_time)
        splitting.progress(stop_time, delta_time, 0)
        if xs_mesh is None:
            xs_mesh = splitting.get_xs_mesh()
        errors.append(trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1]))

    plt.plot(wave_weights, errors, label="{}".format(trial.name))
    print(errors)
plt.legend()
plt.show()
