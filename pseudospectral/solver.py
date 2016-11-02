import numpy as np
import math
from numpy.fft import ifft, fft, fftn, ifftn
import matplotlib.pyplot as plt
from itertools import zip_longest, repeat
from matplotlib import animation
from scipy.integrate import quad


def pseudospectral_factor(interval, grid_points, power):
    bound_left = interval[0]  # left border of interval
    bound_right = interval[1]  # right border of interval
    if bound_right <= bound_left or math.isinf(bound_left) or math.isinf(bound_right):
        raise ValueError("Left bound {} needs to be smaller than right bound {} and both be finite."
                         .format(bound_left, bound_right))

    scale = 2 * math.pi / (bound_right - bound_left)
    # the ordering of this numpy array is defined by the ordering of python's fft's result (see its documentation)
    kxx = (1j * scale * np.append(np.arange(0, grid_points / 2 + 1), np.arange(- grid_points / 2 + 1, 0))) ** power

    # h = (bound_right - bound_left) / (N - 1)  # spatial step size
    x = np.linspace(interval[0], interval[1], endpoint=False, num=grid_points)
    return x, kxx


def pseudospectral_factor_multi(intervals, grid_points_list, power):
    # get grid points and the pseudospectral coefficients for the given derivative in spatial domain
    # in each dimension
    x, kxx = [], []
    for interval, grid_points in zip_longest(intervals, grid_points_list, fillvalue=grid_points_list[-1]):
        curr_x, curr_k = pseudospectral_factor(interval, grid_points, power)
        x.append(curr_x)
        kxx.append(curr_k)
    return x, np.meshgrid(*x, sparse=True), np.meshgrid(*kxx, sparse=True)


# returns (x,y) with x being the 1D grid points and y the pseudospectral derivative at
# these grid points: y = (d/dx)^power of u. Requires u to be periodical in the interval
def spectral_derivative(interval, grid_points, u, power):
    x, kxx = pseudospectral_factor(interval, grid_points, power)
    return x, ifft(kxx * fft(u(x)))


# u_t(t,x) = alpha * u_xx(t,x), u(t0,x)=u0(x), alpha > 0
def heat_solution(intervals, grid_points_list, t0, u0, alpha, wanted_times):
    assert alpha > 0.

    # plural 's' means a list, so list of x coordinates, list of meshgrid coordinates, list of fourier coefficients
    xs, xxs, ks = pseudospectral_factor_multi(intervals, grid_points_list, 2)

    # variables ending in underscore note that the values are considered to be in fourier space
    y0 = u0(xxs)
    y0_ = fftn(y0)  # starting condition in fourier space and evaluated at grid

    times = list(filter(lambda time_check: time_check >= t0, wanted_times))
    solutions = []
    for t in times:
        # for all j solve d/dt u_hat(j; t) = -j*j*u_hat(j; t) and starting condition u_hat(j;0)=y0_(j)
        # here we are in the position to know the exact solution!

        # solution at time t with starting value y0_, all in fourier space
        u_hat_ = y0_ * np.exp(alpha * sum(k for k in ks) * t)

        y = ifftn(u_hat_).real
        solutions.append(y)
    return xs, times, solutions


# u_tt(t,x) = alpha * u_xx(t,x), u(t0,x)=u0(x), u_t(t0,x)=u0_t(x), alpha > 0
def wave_solution(intervals, grid_points_list, t0, u0, u0t, alpha, wanted_times):
    assert alpha > 0.

    # plural 's' means a list, so list of x coordinates, list of meshgrid coordinates, list of fourier coefficients
    xs, xxs, ks = pseudospectral_factor_multi(intervals, grid_points_list, 1)

    # variables ending in underscore note that the values are considered to be in fourier space
    y0 = u0(xxs)
    y0_ = fftn(y0)  # starting condition in fourier space and evaluated at grid
    y0t_ = fftn(u0t(xxs))
    if abs(y0t_[0]) > 1e-9:
        print("Warning! Start velocity for wave solver not periodic, solution will be incorrect.")
        pass
    sum_ks = sum(k for k in ks)
    c1_ = y0_
    temp_ks = np.where(sum_ks != 0, sum_ks, math.inf)  # replace ks 0 with inf to ensure c2_ is zero at those points
    c2_ = y0t_ / temp_ks

    times = list(filter(lambda time_check: time_check >= t0, wanted_times))
    solutions = []
    for t in times:
        # for all j solve (d/dt)^2 u_hat(j; t) = -j*j*u_hat(j; t) and starting conditions u_hat(j;0)=y0_(j),
        # (d/dt)u_hat(j;0)=y0t_(j)
        # here we are in the position to know the exact solution for this linear ordinary differential equation!

        # solution at time t with starting value y0_ and y0t_, all in fourier space

        u_hat_ = c1_ * np.cosh(sum_ks * t) + c2_ * np.sinh(sum_ks * t)

        y = ifftn(u_hat_)
        solutions.append(y)
    return xs, times, solutions


def animate_1d(x, ys, animate_times, pause):
    fig = plt.figure()
    ax = plt.axes(xlim=(min(x), max(x)), ylim=(min(np.amin(vals.real) for vals in ys),
                                               max(np.amax(vals.real) for vals in ys)))
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(x, ys[i].real)
        time_text.set_text("Solution at time=" + str(animate_times[i]))
        return line, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(animate_times), interval=pause, blit=True)
    plt.show()


if __name__ == "__main__":
    test_derivative = False
    test_heat = True
    test_wave = False
    do_animate = True

    if test_wave:
        grid_n = 128  # power of 2 for best performance of fft
        wave_speed = 1  # > 0
        wave_speed_sq = wave_speed ** 2
        dimension = 1  # plotting only supported for one or two dimensional, higher dimension will require lower grid_n
        domain = list(repeat([-math.pi, math.pi], dimension))  # intervals with periodic boundary conditions, so a torus
        show_times = np.arange(0, 30, 0.1)  # times to evaluate solution for and plot it
        #show_times = [0, 1, 5, 10]

        def start_position(xs, delta=0):
            return np.sin(sum(x for x in xs) + delta)

        def start_velocity(xs):  # Needs to be periodic! (so zeroth fourier coefficient must be zero!)
            return np.cos(sum(x for x in xs)) + np.sin(sum(x for x in xs))
            # return np.ones(shape=sum(x for x in xs).shape)
            # return sum(np.zeros(shape=x.shape) for x in xs)

        def reference_1d(xs, t):
            # this is the d'Alembert reference solution if the indefinite integral of the start_velocity is known

            def start_velocity_integral(inner_xs, delta):
                sum_x = sum(x for x in inner_xs)
                # return np.sin(sum_x + delta)  # for start_velocity=cos(sum_x)
                return (sum_x + delta + np.sin(sum_x + delta) * np.cos(sum_x + delta)) / 2  # for start_velocity = cos^2
            return (start_position(xs, wave_speed * t) / 2
                    + start_position(xs, -wave_speed * t) / 2
                    + (start_velocity_integral(xs, wave_speed * t)
                       - start_velocity_integral(xs, -wave_speed * t)) / (2 * wave_speed))


        x_result, t_result, y_result = wave_solution(domain, [grid_n],
                                                     0, start_position, start_velocity, wave_speed_sq, show_times)
        if len(x_result) == 1:  # 1D plots
            if do_animate:
                animate_1d(x_result[0], y_result, show_times, 100)  # pause between frames in ms
            else:
                # all times in one figure

                plt.plot(*x_result, reference_1d(x_result, t_result[1]) + 0.05,
                         label="Reference solution at time=" + str(t_result[1]))
                plt.plot(*x_result, reference_1d(x_result, t_result[2]) + 0.05,
                         label="Reference solution at time=" + str(t_result[2]))
                for time, sol in zip(t_result, y_result):
                    plt.plot(*x_result, sol.real, label="Solution at time=" + str(time))
                plt.legend()
                plt.title("Wave equation solution by pseudospectral spatial method and exact time solution\nwith N="
                          + str(grid_n) + " grid points")
                plt.show()
        # TODO test and visualize wave solution in 2D
    if test_heat:
        grid_n = 128  # power of 2 for best performance of fft
        thermal_diffusivity = 0.1  # > 0
        dimension = 2  # plotting only supported for one or two dimensional, higher dimension will require lower grid_n
        domain = list(repeat([-math.pi, math.pi], dimension))  # intervals with periodic boundary conditions, so a torus
        show_times = np.arange(0, 20, 0.1)  # times to evaluate solution for and plot it

        # starting condition for homogeneous heat equation with periodic boundary equation in given domain
        def start_condition(xs):
            return -np.sin(sum(x ** 2 for x in xs))
            # return 1 / np.cosh(10 * sum(x for x in xs) / math.pi) ** 2
            # return np.where(-1 < x, np.where(x > 1, np.ones(shape=x.shape), 0), 1)  # discontinuous block in 1D


        x_result, t_result, y_result = heat_solution(domain, [grid_n],
                                                     0, start_condition, thermal_diffusivity, show_times)

        if len(x_result) == 1:  # 1D plots
            if do_animate:
                animate_1d(x_result[0], y_result, show_times, 100)  # pause between frames in ms
            else:
                # all times in one figure
                for time, sol in zip(t_result, y_result):
                    plt.plot(*x_result, sol, label="Solution at time=" + str(time))
                plt.legend()
                plt.title("Heat equation solution by pseudospectral spatial method and exact time solution\nwith N="
                          + str(grid_n) + " grid points")
                plt.show()
        elif len(x_result) == 2:  # 2D plots
            # one figure per time
            for time, sol in zip(t_result, y_result):
                plt.figure()
                plt.pcolormesh(*x_result, sol, vmin=-1, vmax=1)  # fix color bar range over different figures
                plt.colorbar()
                plt.show()

    if test_derivative:
        bound = [-1, 1]


        def testfun_solution(eks):  # second derivative of testfun
            var = math.pi * (eks + 1)
            return math.pi ** 2 * (np.cos(var) ** 2 * (np.sin(var) + 3) - np.sin(var) - 1) * np.exp(np.sin(var))


        def testfun(x):
            return np.sin(math.pi * (x + 1)) * np.exp(np.sin(math.pi * (x + 1)))

        result_x, result_y = spectral_derivative(bound, 512, testfun, 2)
        x_highres = np.linspace(bound[0], bound[1], endpoint=False, num=512)  # higher resolution
        plt.plot(result_x, result_y, x_highres, testfun_solution(x_highres))
        plt.legend(["Pseudospectral solution", "Original solution"])
        plt.show()
