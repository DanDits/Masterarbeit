from diff_equation.pseudospectral_solver import wave_solution
from diff_equation.ode_solver import linhyp_solution
import numpy as np
from itertools import repeat
from math import pi
import matplotlib.pyplot as plt
from util.trial import Trial
from itertools import cycle


# Klein Gordon equation: u_tt=alpha*u_xx -beta(x)*u, alpha>0, beta(x)>0


# Strang splitting:
# from starting values u(t0,x)=g(x), u_t(t0,x)=h(x)
# solve wave equation to v(t0+dt/2,x), calculate v_t(t0+dt/2,x)
# with these as starting values solve linear hyperbolic ode with mol to w(t0+dt,x), calculate w_t(t0+dt,x)
# using these as starting values finally solve wave equation again to u(t0+dt,x)

# TODO how to get v_t and w_t? We want second order!? central differences (so also calculate v(t0+dt,x),...)?
# TODO instead of solving wave twice, is it faster and equally accurate to solve linhyp twice?


def klein_gordon_strang_step(intervals, grid_points_list, t0, u0, u0t, alpha, beta, t1):
    assert t1 > t0
    assert alpha > 0.
    wave_speed = np.sqrt(alpha)
    dt = t1 - t0
    print("Klein gordon step1, alpha=", alpha, "dt=", dt)

    # Step 1 of strang splitting: Solve wave equation for half of time step, also get at full time step
    x_result, _, (v, v_t1) = wave_solution(intervals, grid_points_list,
                                           t0, u0, u0t, wave_speed, [t0 + dt / 2, t1])
    xxs = np.meshgrid(*x_result, sparse=True)
    u0 = (u0(xxs) if callable(u0) else u0)

    # use central difference to get a second order estimate of the time derivative of v
    v_t = (v_t1 - u0) / dt

    if beta != 0:
        print("Klein gordon step2, alpha=", alpha, "dt=", dt)
        # Step 2 of strang splitting: Solve hyperbolic linear part for full time step, also get at 2*full time step
        _, _, (w_t1, w_t2) = linhyp_solution(intervals, grid_points_list,
                                             t0, v, v_t, beta, [t1, t1 + dt])
        # use central difference to get a second order estimate of the time derivative of w
        # w_t = (w_t2 - u0) / (2 * dt)  # TODO is it even correct to use u0 as previous value for this?
        # use backward difference to get an estimate of the time derivative of w
        w_t = (w_t1 - v) / dt
    else:
        w_t1 = v
        w_t = v_t
    print("Klein gordon step3, alpha=", alpha, "dt=", dt)
    # Step 3 of strang splitting: Solve wave equation for half of time step
    _, _, (u_t1, u_t15) = wave_solution(intervals, grid_points_list,
                                        t0 + dt / 2, w_t1, w_t, wave_speed, [t1, t1 + dt / 2])
    # use central difference to get a second order estimate of the time derivative of u at t1
    ut_t1 = (u_t15 - w_t1) / dt
    return x_result, u_t1, ut_t1


def klein_gordon_simple_step(intervals, grid_points_list, t0, u0, u0t, alpha, beta, t1):
    assert t1 > t0
    assert alpha > 0.
    wave_speed = np.sqrt(alpha)
    dt = t1 - t0
    print("Klein gordon simple step1, alpha=", alpha, "dt=", dt)

    x_result, _, (v_t1, v_t2) = wave_solution(intervals, grid_points_list,
                                           t0, u0, u0t, wave_speed, [t1, t1 + dt])
    xxs = np.meshgrid(*x_result, sparse=True)
    u0 = (u0(xxs) if callable(u0) else u0)

    # use central difference to get a second order estimate of the time derivative of v
    vt = (v_t2 - u0) / (2 * dt)

    if beta != 0:
        print("Klein gordon simple step2, alpha=", alpha, "dt=", dt)
        _, _, (w_t1, w_t2) = linhyp_solution(intervals, grid_points_list,
                                             t0, v_t1, vt, beta, [t1, t1 + dt])
        # use central difference to get a second order estimate of the time derivative of w
        wt = (w_t2 - v_t1) / (2 * dt)
    else:
        w_t1, wt = v_t1, vt
    return x_result, w_t1, wt

if __name__ == "__main__":
    dimension = 1
    grid_size_N = 128
    domain = list(repeat([-pi, pi], dimension))
    time_step_size = 0.05
    steps = 7
    start_time = 0
    show_errors = True
    show_reference = True

    param_g1 = 3  # some parameter greater than one
    trial_1 = Trial(lambda xs: np.sin(sum(xs)),
                    lambda xs: param_g1 * np.cos(sum(xs)),
                    lambda xs, t: np.sin(sum(xs) + param_g1 * t)) \
        .set_config("beta", lambda xs: param_g1 ** 2 - 1) \
        .set_config("alpha", 1)

    # TODO not a valid trial as the start velocity's zeroth fourier coefficient is not zero
    trial_2 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                    lambda xs: 2 * np.exp(-np.cos(sum(xs))),
                    lambda xs, t: np.sin(2 * t) * np.exp(-np.cos(sum(xs)))) \
        .set_config("beta", lambda xs: 4 + np.cos(sum(xs)) + np.sin(sum(xs)) ** 2) \
        .set_config("alpha", 1)

    param_1, param_2, param_n1, param_3, alpha_g0 = 0.3, 0.5, 2, 1.2, 0.3
    assert param_n1 * alpha_g0 < param_3  # to ensure beta > 0
    trial_3 = Trial(lambda xs: param_1 * np.cos(param_n1 * sum(xs)),
                    lambda xs: param_2 * param_3 * np.cos(param_n1 * sum(xs)),
                    lambda xs, t: np.cos(param_n1 * sum(xs)) * (param_1 * np.cos(param_3 * t)
                                                                + param_2 * np.sin(param_3 * t))) \
        .set_config("beta", lambda xs: -(alpha_g0 ** 2) * (param_n1 ** 2) + param_3 ** 2) \
        .set_config("alpha", alpha_g0)

    trial_wave = Trial(lambda xs: np.sin(sum(xs)),
                       lambda xs: -np.sin(sum(xs)),
                       lambda xs, t: 0.5 * (np.sin(sum(xs) + t) + np.sin(sum(xs) - t)
                                            + np.cos(sum(xs) + t) - np.cos(sum(xs) - t))) \
        .set_config("beta", 0) \
        .set_config("alpha", 1)  # wave speed squared

    trial = trial_1

    plt.figure()
    times = [start_time + (n + 1) * time_step_size for n in range(steps)]
    x_mesh = None
    solutions = []
    current_time = start_time
    current_start = trial.start_position
    current_velocity = trial.start_velocity
    for i, (time, color) in enumerate(zip(times, cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y']))):
        x_result, solution, solution_velocity = klein_gordon_simple_step(domain, [grid_size_N],
                                                                         current_time, current_start, current_velocity,
                                                                         trial.config["alpha"], trial.config["beta"],
                                                                         time)
        solutions.append(solution)
        if i == 0:
            x_mesh = np.meshgrid(*x_result, sparse=True)
            plt.plot(x_result[0], trial.start_position(x_mesh), label="Start position")
        plt.plot(x_result[0], solution, ".", color=color, label="Solution at {}".format(time))
        if show_reference:
            plt.plot(x_result[0], trial.reference(x_mesh, time), color=color, label="Reference at {}".format(time))
        current_time = time
        current_start = solution
        current_velocity = solution_velocity
    plt.legend()

    if show_errors:
        errors = [trial.error(x_mesh, t, y) for t, y in zip(times, solutions)]
        plt.figure()
        plt.plot(times, errors, label="Errors in discrete L2 norm")
        plt.xlabel("Time")
        plt.ylabel("Error")

    plt.show()
