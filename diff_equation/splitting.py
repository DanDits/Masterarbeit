from diff_equation.pseudospectral_solver import wave_solution
from diff_equation.ode_solver import linhyp_solution
import numpy as np
from itertools import repeat
from math import pi
import matplotlib.pyplot as plt

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

    # Step 1 of strang splitting: Solve wave equation for half of time step, also get at full time step
    x_result, _, (v, v_t1) = wave_solution(intervals, grid_points_list,
                                           t0, u0, u0t, wave_speed, [t0 + dt / 2, t1])
    xxs = np.meshgrid(*x_result, sparse=True)
    # use central difference to get a second order estimate of the time derivative of v
    v_t = (v_t1 - u0(xxs)) / dt

    # Step 2 of strang splitting: Solve hyperbolic linear part for full time step, also get at 2*full time step
    _, _, (w_t1, w_t2) = linhyp_solution(intervals, grid_points_list,
                                         t0, v, v_t, beta, [t1, t1 + dt])
    # use central difference to get a second order estimate of the time derivative of w
    w_t = (w_t2 - u0(xxs)) / (2 * dt)  # TODO is it even correct to use u0 as previous value for this?

    # Step 3 of strang splitting: Solve wave equation for half of time step
    _, _, (u_t1,) = wave_solution(intervals, grid_points_list,
                                  t0 + dt / 2, w_t1, w_t, wave_speed, [t1])
    return x_result, [t1], [u_t1]

if __name__ == "__main__":
    dimension = 1
    grid_size_N = 128
    domain = list(repeat([-pi, pi], dimension))
    test_alpha = 1
    test_beta = lambda xs: 1.5 + np.sin(sum(xs))
    time_step_size = 0.5
    start_time = 0
    start_position = lambda xs: np.sin(sum(xs))
    start_velocity = lambda xs: np.cos(sum(xs))

    plt.figure()
    for n in range(1):
        xs, _, (solution,) = klein_gordon_strang_step(domain, [grid_size_N],
                                                      start_time, start_position, start_velocity,
                                                      test_alpha, test_beta, start_time + (n + 1) * time_step_size)
        if n == 0:
            plt.plot(xs[0], start_position(xs), label="Start position")
        plt.plot(xs[0], solution, label="Solution at {}".format((n + 1) * time_step_size))
    plt.legend()
    plt.show()
