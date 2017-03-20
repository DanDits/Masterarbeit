import demo.splitting as ds
from diff_equation.splitting import Splitting
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
import diff_equation.klein_gordon as kg
from util.analysis import error_l2_relative
dimension = 1
grid_size_N = 2 ** 7
domain = list(repeat([-np.pi, np.pi], dimension))


wave_weights = np.arange(0., 1.001, 0.025)
trials = [ds.trial_3]
start_time = 0.
delta_time = 0.005
stop_time = 1

xs_mesh = None

plt.figure()
size = "xx-large"
plt.title("Verschiedene Lösungen der KGG, $\\tau={}$, $N={}$, $T={}$".format(delta_time, grid_size_N, stop_time),
          fontsize=size)
plt.xlabel("Gewicht $w$", fontsize=size)
plt.ylabel("Relativer Fehler in diskreter L2-Norm", fontsize=size)
plt.yscale('log')
for trial in trials:
    errors = []
    for weight in wave_weights:
        print(weight)
        configs = kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], trial.alpha, trial.beta, weight)
        splitting = Splitting.make_fast_strang(*configs, "FastStrang", start_time, trial.start_position,
                                               trial.start_velocity, delta_time)
        splitting.progress(splitting.approx_steps_to_end_time(stop_time, delta_time), delta_time, 0)
        if xs_mesh is None:
            xs_mesh = splitting.get_xs_mesh()
        trial.error_function = error_l2_relative
        errors.append(trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1]))

    plt.plot(wave_weights, errors, label="{}, $\\alpha={}, \\beta(0)={}$".format(trial.name, trial.alpha(),
                                                                              trial.beta([0])))
    print(errors)
plt.legend()
plt.show()
