import demo.splitting as ds
import diff_equation.klein_gordon as kg
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
from diff_equation.splitting import Splitting
from util.analysis import error_l2_relative


dimension = 1
grid_size_Ns = [256, 512, 1024, 2048]
plt.figure()
plt.title("Abhängigkeit der CFL Zahl $\\tilde{c}$ vom Gewicht $w$", fontsize="xx-large")
plt.xlabel("Gewicht $w$", fontsize="xx-large")
plt.ylabel("CFL Zahl $\\tilde{c}$", fontsize="xx-large")
for grid_size_N in grid_size_Ns:
    domain = list(repeat([-np.pi, np.pi], dimension))
    trial = ds.trial_4


    start_time = 0.
    dt = 0.00025
    delta_times = np.arange(0.001, 0.1, dt)
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
            splitting.progress(stop_time, delta_time, 0)
            if xs_mesh is None:
                xs_mesh = splitting.get_xs_mesh()

            trial.error_function = error_l2_relative
            error = trial.error(xs_mesh, splitting.times()[-1], splitting.solutions()[-1])

            if error > prev_error > 0:
                steigung = np.abs(np.log(error - prev_error) / np.log(delta_time - prev_delta_time))
                print("dt=", delta_time, "error=", error, "Steigung=", steigung)
                if error > 1 or steigung > 2.5: # or steigung < 1.5
                    print("BREAKING, next would be:", error)
                    cfl_constants.append((2*np.pi / grid_size_N) / prev_delta_time)
                    break

            prev_error = error
            prev_delta_time = delta_time

        else:
            cfl_constants.append(0)
    print(cfl_constants)
    plt.plot(weights, cfl_constants, label="$N={}$".format(grid_size_N))
plt.legend(fontsize="xx-large")
plt.show()
