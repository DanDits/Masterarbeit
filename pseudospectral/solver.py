import numpy as np
import math
from numpy.fft import ifft, fft
import matplotlib.pyplot as plt
from scipy.integrate import ode


def pseudospectral_factor(interval, grid_points, power):
    bound_left = interval[0]  # left border of interval
    bound_right = interval[1]  # right border of interval
    if bound_right <= bound_left or math.isinf(bound_left) or math.isinf(bound_right):
        raise ValueError("Left bound {} needs to be smaller than right bound {} and both be finite."
                         .format(bound_left, bound_right))
    N = grid_points  # should be a power of 2 for optimal performance

    scale = 2 * math.pi / (bound_right - bound_left)
    # the ordering of this numpy array is defined by the ordering of python's fft's result (see its documentation)
    kxx = (scale * np.append(np.arange(0, N / 2 + 1), np.arange(- N / 2 + 1, 0))) ** power
    if power % 4 == 2:  # for power in [2,6,10,...] avoid introducing complex numbers
        kxx *= -1
    elif power % 4 != 0:  # for power in [0,4,8,...] the factor would just be 1
        kxx *= 1j ** power

    # h = (bound_right - bound_left) / (N - 1)  # spatial step size
    x = np.linspace(interval[0], interval[1], endpoint=False, num=grid_points)
    return x, kxx


# returns (x,y) with x being the 1D grid points and y the pseudospectral derivative at
# these grid points: y = (d/dx)^power of u. Requires u to be periodical in the interval
def spectral_derivative(interval, grid_points, u, power):
    x, kxx = pseudospectral_factor(interval, grid_points, power)
    return x, ifft(kxx * fft(u(x)))


# u_t(t,x) = alpha * u_xx(t,x), u(t0,x)=u0(x), alpha > 0
def heat_solution(interval, grid_points, t0, u0, alpha, t_end, dt):
    assert alpha > 0.
    assert dt > 0.
    assert t_end >= t0
    # get grid points and the pseudospectral coefficients for the second order derivative in spatial domain
    x, kxx = pseudospectral_factor(interval, grid_points, 2)

    # variables ending in underscore note that the values are considered to be in fourier space

    y0 = u0(x)
    y0_ = fft(y0)  # starting condition in fourier space and evaluated at grid

    times, solutions = [t0], [y0]
    for t in np.arange(t0 + dt, t_end + dt, dt):
        # for all j solve d/dt u_hat(j; t) = -j*j*u_hat(j; t) and starting condition u_hat(j;0)=y0_(j)
        # here we are in the position to know the exact solution!
        u_hat_ = y0_ * np.exp(alpha * kxx * t)  # solution at time t with starting value y0_, all in fourier space

        y = ifft(u_hat_).real
        times.append(t)
        solutions.append(y)
    return x, times, solutions


def wave_solution(interval, grid_points, g, dt, t0, y0, t_end):
    def f(t, y):
        return y

    r = ode(f).set_integrator('zvode', method='adams')  # 'zvode' for complex valued problems, 'adams' for non-stiff
    r.set_initial_value(y0, t0)

    while r.successful() and r.t < t_end:
        r.integrate(r.t + dt)
        print("{} {}".format(r.t, r.y))


if __name__ == "__main__":
    test_derivative = False
    test_heat = True

    if test_heat:
        dt = 0.1
        n = 512
        alpha = 0.01

        def start_condition(x):
            # return np.sin(x)
            return 1 / np.cosh(10 * x / math.pi) ** 2
            # return np.where(-1 < x, np.where(x > 1, np.ones(shape=x.shape), 0), 1)  # discontinuous block
        x, t, y = heat_solution([-math.pi, math.pi], n, 0, start_condition, alpha, 5., dt)
        for time, sol in zip(t, y):
            if time in [0, 1, 2, 3, 4, 5]:
                plt.plot(x, sol, label="Solution at time=" + str(time))
        plt.legend()
        plt.title("Heat equation solution by pseudospectral spatial method and exact time solution\nwith N=" + str(n) + " gridpoints and starting value 1/cosh(10x/pi)^2")
        plt.show()

    if test_derivative:
        bound = [-1, 1]


        def testfun_solution(eks):  # second derivative of testfun
            var = math.pi * (eks + 1)
            return math.pi ** 2 * (np.cos(var) ** 2 * (np.sin(var) + 3) - np.sin(var) - 1) * np.exp(np.sin(var))


        testfun = lambda x: np.sin(math.pi * (x + 1)) * np.exp(np.sin(math.pi * (x + 1)))

        result_x, result_y = spectral_derivative(bound, 512, testfun, 2)
        x_highres = np.linspace(bound[0], bound[1], endpoint=False, num=512)  # higher resolution
        plt.plot(result_x, result_y, x_highres, testfun_solution(x_highres))
        plt.legend(["Pseudospectral solution", "Original solution"])
        plt.show()
