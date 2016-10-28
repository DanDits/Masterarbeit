import numpy as np
import math
from numpy.fft import ifft, fft
import matplotlib.pyplot as plt
from scipy.integrate import ode


def pseudospectral_factor_second_order(interval, grid_points):
    bound_left = interval[0]  # left border of interval
    bound_right = interval[1]  # right border of interval
    if bound_right <= bound_left or math.isinf(bound_left) or math.isinf(bound_right):
        raise ValueError("Left bound {} needs to be smaller than right bound {} and both be finite."
                         .format(bound_left, bound_right))
    N = grid_points  # should be a power of 2 for optimal performance

    scale = 2 * math.pi / (bound_right - bound_left)
    # the order is defined by the order of python's fft's result (see its documentation)
    kxx = -(scale * np.append(np.arange(0, N / 2 + 1), np.arange(- N / 2 + 1, 0))) ** 2
    return kxx


# returns (x,y) with x being the 1D grid points and y the pseudospectral solution at
# these grid points of the poisson PDE y_xx = g(x) in the given interval, with the rhs g being the second
# spatial derivative of y in the variable x. Implicitly implies solution to be periodic in the interval!
def poisson_solution(interval, grid_points, g):
    kxx = pseudospectral_factor_second_order(interval, grid_points)

    # h = (bound_right - bound_left) / (N - 1)  # spatial step size
    x = np.linspace(interval[0], interval[1], endpoint=False, num=grid_points)
    y = g(x)
    return x, ifft(kxx * fft(y))


# u_t(t,x) = u_xx(t,x) * g(t,x), u(t0,x)=y0(x)
def heat_solution(interval, grid_points, g, dt, t0, y0, t_end):
    # use explicit euler for the time integration
    # use the pseudospectral solution as approximation to spatial derivative

    kxx = pseudospectral_factor_second_order(interval, grid_points)
    x = np.linspace(interval[0], interval[1], endpoint=False, num=grid_points)
    curr_y = y0(x)
    curr_t = t0
    t, y = [curr_t], [curr_y]
    while curr_t <= t_end:
        yxx = ifft(kxx * fft(curr_y)).real
        # make an explicit euler step
        curr_y = curr_y + dt * yxx * g(curr_t, x)
        curr_t += dt
        t.append(curr_t)
        y.append(curr_y)
    return t, x, y


def wave_solution(interval, grid_points, g, dt, t0, y0, t_end):

    def f(t, y):
        _, y = poisson_solution(interval, grid_points, lambda x: g(t, x))
        return y

    r = ode(f).set_integrator('zvode', method='adams')  # 'zvode' for complex valued problems, 'adams' for non-stiff
    r.set_initial_value(y0, t0)

    while r.successful() and r.t < t_end:
        r.integrate(r.t+dt)
        print("{} {}".format(r.t, r.y))

if __name__ == "__main__":
    test_poisson = False
    test_heat = True

    if test_heat:
        dt = 0.1
        n = 64
        result_t, result_x, result_y = heat_solution([-math.pi, math.pi], n, lambda t, x: 1, dt, 0,
                                                     np.sin,
                                                     5)
        for t in [0, 0.1, 0.5, 0.7, 0.8]:
            plt.plot(result_x, result_y[int(t / dt)], label="t=" + str(t))
        plt.legend()
        plt.title("Explicit euler with dt=" + str(dt) + " and N=" + str(n) + " and starting value sin(x)")
        plt.show()

    if test_poisson:
        bound = [-math.pi, math.pi]
        result_x, result_y = poisson_solution(bound, 64, lambda x: np.sin(x))  # true solution is -np.sin(x)
        # or lambda x: math.sqrt(2) / np.cosh(x) on [-5pi,5pi]

        x_highres = np.linspace(bound[0], bound[1], endpoint=False, num=512)  # higher resolution
        plt.plot(result_x, result_y, x_highres, -np.sin(x_highres))
        plt.legend(["Pseudospectral solution", "Original solution"])
        plt.show()
