import numpy as np
import math
from numpy.fft import ifft, fft, fftn, ifftn
from itertools import zip_longest


def pseudospectral_factor(interval, grid_points, power):
    bound_left = interval[0]  # left border of interval
    bound_right = interval[1]  # right border of interval
    if bound_right <= bound_left or math.isinf(bound_left) or math.isinf(bound_right):
        raise ValueError("Left bound {} needs to be smaller than right bound {} and both be finite."
                         .format(bound_left, bound_right))

    scale = 2 * math.pi / (bound_right - bound_left)
    # the ordering of this numpy array is defined by the ordering of python's fft's result (see its documentation)
    kxx = (1j * scale * np.append(np.arange(0, grid_points / 2 + 1), np.arange(- grid_points / 2 + 1, 0))) ** power

    # important not to include endpoint, else solution will be slightly incorrect because
    # the point -pi is equivalent to pi and appears twice, which will make the fourier coefficients wrong
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
        u_hat_ = y0_ * np.exp(alpha * sum(ks) * t)

        y = ifftn(u_hat_).real
        solutions.append(y)
    return xs, times, solutions


# u_tt(t,x) = (wave_speed ** 2) * u_xx(t,x), u(t0,x)=u0(x), u_t(t0,x)=u0_t(x), wave_speed > 0
def wave_solution(intervals, grid_points_list, t0, u0, u0t, wave_speed, wanted_times):
    assert wave_speed > 0.

    # calculations only depending on interval and grid points

    # plural 's' means a list, so list of x coordinates, list of meshgrid coordinates, list of fourier coefficients
    xs, xxs, ks = pseudospectral_factor_multi(intervals, grid_points_list, 2)

    # as the pseudospectral factors of order 2 are negative integers, negate sum!
    # numbers are real (+0j) anyways, but using '.real' saves some calculation time and storage space
    norm2_ks = np.sqrt(-sum(ks).real)

    # pre calculations depending on starting values, wave speed,...

    # variables ending in underscore note that the values are considered to be in fourier space
    # we accept u0 and u0t to be callable functions and evaluate them ourselves or as already evaluated discrete values
    y0 = u0(xxs) if callable(u0) else u0
    y0_ = fftn(y0)  # starting condition in fourier space and evaluated at grid
    y0t = u0t(xxs) if callable(u0t) else u0t
    y0t_ = fftn(y0t)

    # calculate factors c1_ and c2_
    c1_ = y0_

    zero_index = (0,) * len(norm2_ks.shape)  # top left corner of norm2_ks contains a zero, temporarily replace it!
    assert norm2_ks[zero_index] == 0
    if abs(y0t_[zero_index]) > norm2_ks.size * 1e-9:  # as the fourier coefficients are unscaled
        # the zeroth fourier coefficient is the (unscaled) discrete integral over the function (multiplied by 1=e^(i*0))
        # and since the pseudospectral method conserves the energy we require the starting energy to be zero
        # so positive and negative parts sum away (is there a better explanation?)
        print("Warning! Start velocity for wave solver not possible, solution will be incorrect: "
              + "0th fourier coefficient not zero but", y0t_[zero_index])

    norm2_ks[zero_index] = np.inf  # this makes sure that c2_[zero_index] is 0 and no warning is triggered
    c2_ = y0t_ / norm2_ks
    norm2_ks[zero_index] = 0  # revert temporal change
    c2_ *= 1 / wave_speed

    # for each wanted time that is actually in the future, calculate a solution
    times = list(filter(lambda time_check: time_check >= t0, wanted_times))
    solutions = []
    for t in times:
        # for all j solve (d/dt)^2 u_hat(j; t) = -j*j*u_hat(j; t) and starting conditions u_hat(j;0)=y0_(j),
        # (d/dt)u_hat(j;0)=y0t_(j)
        # here we are in the position to know the exact solution for this linear ordinary differential equation!

        # solution at time t with starting value y0_ and y0t_, all in fourier space

        u_hat_ = c1_ * np.cos(wave_speed * norm2_ks * (t - t0)) + c2_ * np.sin(wave_speed * norm2_ks * (t - t0))

        y = ifftn(u_hat_)
        solutions.append(y)
    return xs, times, solutions
