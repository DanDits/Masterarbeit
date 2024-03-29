# In correspondence to the instability observed by splitting_orderplot.py for different weights we here test
# the dependence on the weight by estimating the CFL number. Should be the same for different N, but because estimation
# is different (and not always equally accurate as these are heuristics) they do not completely fit on top of each oth
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np

import demo.chapter1.splitting as ds
import diff_equation.klein_gordon as kg
from diff_equation.splitting import Splitting
from util.analysis import error_l2_relative

dimension = 1
grid_size_Ns = [256, 512, 1024, 2048, 2048*2]
plt.figure()
plt.title("Abhängigkeit der CFL Zahl $\\tilde{c}$ vom Gewicht $w$")
plt.xlabel("Gewicht $w$")
plt.ylabel("CFL Zahl $\\tilde{c}$")
for grid_size_N in grid_size_Ns:
    print(grid_size_N)
    domain = list(repeat([-np.pi, np.pi], dimension))
    trial = ds.trial_frog2


    start_time = 0.
    dt = 0.0001
    delta_times = np.arange(0.0002, 0.1, dt)
    stop_time = 1.

    dw = 0.05
    weights = np.arange(0., 1. + dw, dw)
    xs_mesh = None
    cfl_constants = []

    for weight in weights:
        def make_wave_linhyp_configs():
            return kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], trial.alpha, trial.beta, weight)


        prev_error, prev_delta_time = -1, -1
        for delta_time in delta_times:
            splitting = Splitting.make_strang(*make_wave_linhyp_configs(), "Strang",
                                              start_time, trial.start_position, trial.start_velocity)
            splitting.progress(splitting.approx_steps_to_end_time(stop_time, delta_time), delta_time, 0)
            if xs_mesh is None:
                xs_mesh = splitting.get_xs_mesh()

            trial.error_function = error_l2_relative
            error = trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1])
            print("dt=", delta_time, "error=", error)
            if error > prev_error > 0:
                steigung = np.abs(np.log(error - prev_error) / np.log(delta_time - prev_delta_time))
                print("Steigung=", steigung)
                if error > 1 or steigung > 3. or steigung < 1.:
                    print("BREAKING, next would be:", error)
                    cfl_constants.append((2*np.pi / grid_size_N) / prev_delta_time)
                    break

            prev_error = error
            prev_delta_time = delta_time

        else:
            cfl_constants.append(0)
    print(cfl_constants)
    plt.plot(weights, cfl_constants, label="$N={}$".format(grid_size_N))
plt.legend()
plt.show()
