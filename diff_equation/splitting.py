from diff_equation.pseudospectral_solver import make_wave_config, wave_solution
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

def get_derivative(previous_derivative, previous_value, current_value, next_value, time_step_size):
    # Use Taylor expansion of the first derivative f'(x)=f'(x-h)+h*f''(x-h)
    # and then use forward differences of second order to estimate f''(x-h)=(f(x+h)-2f(x)+f(x-h))/(h*h)
    return previous_derivative + (previous_value + next_value - 2 * current_value) / time_step_size


def klein_gordon_strang_step(intervals, grid_points_list, t0, u0, u0t, alpha, beta, t1, initial_call=False):
    assert t1 > t0
    assert alpha > 0.
    if initial_call:
        # due to the splitting into two operators and having a second order time derivative u_tt=...
        # there is a factor 1/2 introduced. Currently we only need to apply it once initially.
        # it cancels out with further calls and result stays correct.
        u0t_half = lambda *params: 0.5 * u0t(*params) if callable(u0t) else 0.5 * u0t
    else:
        u0t_half = u0t
    #  TODO Need yet to figure out how second order splitting (factors 0.5) works out for strang splitting
    wave_speed = np.sqrt(0.5 * alpha)
    dt = t1 - t0

    # Step 1 of strang splitting: Solve wave equation for half of time step, also get at full time step
    x_result, _, (v, v_t1) = wave_solution(intervals, grid_points_list,
                                           t0, u0, u0t_half, wave_speed, [t0 + dt / 2, t1])
    xxs = np.meshgrid(*x_result, sparse=True)
    u0 = (u0(xxs) if callable(u0) else u0)

    # use central difference to get a second order estimate of the time derivative of v
    v_t = (v_t1 - u0) / dt
    #v_t = (v - u0) / (dt / 2)  # backward difference alternative
    #v_t = (v_t1 - v) / (dt / 2)  # forward difference alternative
    #v_t = get_derivative(u0t_half, )

    beta_half = lambda *params: 0.5 * beta(*params)
    # Step 2 of strang splitting: Solve hyperbolic linear part for full time step, also get at 2*full time step
    _, _, (w_t1, w_t2) = linhyp_solution(intervals, grid_points_list,
                                         t0, v, v_t, beta_half, [t1, t1 + dt])
    # use central difference to get a second order estimate of the time derivative of w
    w_t = (w_t2 - v) / (2 * dt)
    # use backward difference to get an estimate of the time derivative of w
    w_t = (w_t1 - v) / dt  # better for trial_3, but way worse for trial_1
    w_t = (w_t2 - w_t1) / dt  # forward difference alternative

    # Step 3 of strang splitting: Solve wave equation for half of time step
    _, _, (u_t1, u_t15) = wave_solution(intervals, grid_points_list,
                                        t0 + dt / 2, w_t1, w_t, wave_speed, [t1, t1 + dt / 2])
    # use central difference to get a second order estimate of the time derivative of u at t1
    ut_t1 = (u_t15 - w_t1) / dt
    #ut_t1 = (u_t1 - w_t1) / (dt / 2)  # backward difference alternative
    ut_t1 = (u_t15 - u_t1) / (dt / 2)  # forward difference alternative
    return x_result, u_t1, ut_t1


def klein_gordon_lie_trotter_step(intervals, grid_points_list, t0, u0, u0t, alpha, beta, t1, initial_call=False):
    assert t1 > t0
    assert alpha > 0
    if initial_call:
        # due to the splitting into two operators and having a second order time derivative u_tt=...
        # there is a factor 1/2 introduced. Currently we only need to apply it once initially.
        # it cancels out with further calls and result stays correct.
        u0t_half = lambda *params: 0.5 * u0t(*params) if callable(u0t) else 0.5 * u0t
    else:
        u0t_half = u0t
    wave_speed = np.sqrt(0.5 * alpha)
    dt = t1 - t0

    x_result, _, (v_t1, v_t2) = wave_solution(intervals, grid_points_list,
                                           t0, u0, u0t_half, wave_speed, [t1, t1 + dt])
    xxs = np.meshgrid(*x_result, sparse=True)
    u0 = (u0(xxs) if callable(u0) else u0)

    # use central difference to get a second order estimate of the time derivative of v
    vt = (v_t2 - u0) / (2 * dt)

    beta_half = lambda *params: 0.5 * beta(*params)
    x_result, _, (w_t1, w_t2) = linhyp_solution(intervals, grid_points_list,
                                         t0, v_t1, vt, beta_half, [t1, t1 + dt])
    xxs = np.meshgrid(*x_result, sparse=True)
    if callable(v_t1):
        v_t1 = v_t1(xxs)
    # use central difference to get a second order estimate of the time derivative of w
    wt = (w_t2 - v_t1) / (2 * dt)

    return x_result, w_t1, wt

if __name__ == "__main__":
    dimension = 1
    grid_size_N = 512
    domain = list(repeat([-pi, pi], dimension))
    time_step_size = 0.01
    steps = 100
    plot_gap = steps / 5  # gaps between solutions to plot, use 0 to plot every solution
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

    param_g1 = 2  # some parameter greater than one
    trial_4 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                    lambda xs: param_g1 * np.sin(sum(xs)),
                    lambda xs, t: np.sin(sum(xs)) * np.sin(param_g1 * t)) \
        .set_config("beta", lambda xs: param_g1 ** 2 - 1) \
        .set_config("alpha", 1)

    # TODO try other calculation of time derivative in the splitting using the old velocity
    # trial_3 strang ziemlich gut nach 100 Schritten mit dt=0.01 mit: b,b,c slightly worse for c,b,b, not so great for b,b,b
    # trial 1 strang bad for anything but c,c,c
    # trial 4 (gut, gleich mit lie) für c,c,c, für alles andere schlecht
    trial = trial_3

    plt.figure()
    times = [start_time + (n + 1) * time_step_size for n in range(steps)]
    x_mesh = None
    solutions_lie, solutions_strang = [], []
    current_time = start_time
    current_start_lie, current_start_strang = trial.start_position, trial.start_position
    current_velocity_lie, current_velocity_strang = trial.start_velocity, trial.start_velocity
    color_it = cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])
    wave_config = make_wave_config(domain, [grid_size_N], np.sqrt(trial.config["alpha"]))
    for i, time in enumerate(times):
        print("Time:", time)
        x_result, solution, solution_velocity = klein_gordon_lie_trotter_step(domain, [grid_size_N],
                                                                         current_time, current_start_lie,
                                                                              current_velocity_lie,
                                                                         trial.config["alpha"], trial.config["beta"],
                                                                         time, initial_call=(i == 0))
        solutions_lie.append(solution)

        if i == 0:
            x_mesh = np.meshgrid(*x_result, sparse=True)
            plt.plot(x_result[0], trial.start_position(x_mesh), label="Start position")

        current_start_lie = solution
        current_velocity_lie = solution_velocity

        x_result, solution, solution_velocity = klein_gordon_strang_step(domain, [grid_size_N],
                                                                         current_time, current_start_strang,
                                                                              current_velocity_strang,
                                                                         trial.config["alpha"], trial.config["beta"],
                                                                         time, initial_call=(i == 0))
        solutions_strang.append(solution)

        current_start_strang = solution
        current_velocity_strang = solution_velocity
        current_time = time

        if i % (plot_gap + 1) == 0:
            color = next(color_it)
            plt.plot(x_result[0], solutions_strang[-1], "+", color=color, label="Strang solution at {}".format(time))
            plt.plot(x_result[0], solutions_lie[-1], "o", color=color, label="Lie solution at {}".format(time))
            if show_reference:
                plt.plot(x_result[0], trial.reference(x_mesh, time), color=color, label="Reference at {}".format(time))
    plt.legend()

    if show_errors:
        errors_lie = [trial.error(x_mesh, t, y) for t, y in zip(times, solutions_lie)]
        errors_strang = [trial.error(x_mesh, t, y) for t, y in zip(times, solutions_strang)]
        plt.figure()
        plt.plot(times, errors_lie, label="Errors of lie in discrete L2 norm")
        plt.plot(times, errors_strang, label="Errors of strang in discrete L2 norm")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.yscale('log')
        plt.legend()

    plt.show()
