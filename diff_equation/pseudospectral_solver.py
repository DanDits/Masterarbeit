import numpy as np
from numpy.fft import ifft, fft, fftn, ifftn
from diff_equation.solver_config import SolverConfig


# returns (x,y) with x being the 1D grid points and y the pseudospectral derivative at
# these grid points: y = (d/dx)^power of u. Requires u to be periodical in the interval
def spectral_derivative(interval, grid_points, u, power):
    config = SolverConfig([interval], [grid_points], pseudospectral_power=power)
    x = config.xs[0]
    return x, ifft(config.pseudospectral_factors_mesh[0] * fft(u(x)))


# u_t(t,x) = alpha * u_xx(t,x), u(t0,x)=u0(x), alpha > 0
def heat_solution(intervals, grid_points_list, t0, u0, alpha, wanted_times):
    assert alpha > 0.

    config = SolverConfig(intervals, grid_points_list, pseudospectral_power=2)
    config.init_initial_values(t0, u0, None)

    # variables ending in underscore note that the values are considered to be in fourier space
    y0_ = fftn(config.start_position)  # starting condition in fourier space and evaluated at grid

    times = list(filter(lambda time_check: time_check >= t0, wanted_times))
    solutions = []
    for t in times:
        # for all j solve d/dt u_hat(j; t) = -j*j*u_hat(j; t) and starting condition u_hat(j;0)=y0_(j)
        # here we are in the position to know the exact solution!

        # solution at time t with starting value y0_, all in fourier space
        u_hat_ = y0_ * np.exp(alpha * sum(config.pseudospectral_factors_mesh) * t)

        y = ifftn(u_hat_).real
        solutions.append(y)
    return config.xs, times, solutions


# u_tt(t,x) = (wave_speed ** 2) * u_xx(t,x), u(t0,x)=u0(x), u_t(t0,x)=u0_t(x), wave_speed > 0
class WaveSolverConfig(SolverConfig):

    def __init__(self, intervals, grid_points_list, wave_speed):
        super().__init__(intervals, grid_points_list, pseudospectral_power=2)
        assert wave_speed > 0.
        self.wave_speed = wave_speed

        # as the pseudospectral factors of order 2 are negative integers, negate sum!
        # numbers are real (+0j) anyways, but using '.real' saves some calculation time and storage space
        self.norm2_factors = np.sqrt(-sum(self.pseudospectral_factors_mesh).real)

        # top left corner of norm2_ks contains a zero, temporarily replace it
        self.zero_index = (0,) * len(self.norm2_factors.shape)
        assert self.norm2_factors[self.zero_index] == 0

    def init_solver(self, t0, u0, u0t):
        # pre calculations depending on starting values, wave speed,...

        self.init_initial_values(t0, u0, u0t)
        # variables ending in underscore note that the values are considered to be in fourier space

        y0_ = fftn(self.start_position)  # starting condition in fourier space and evaluated at grid
        y0t_ = fftn(self.start_velocity)

        # calculate factors c1_ and c2_
        c1_ = y0_
        if abs(y0t_[self.zero_index]) > self.norm2_factors.size * 1e-9:  # as the fourier coefficients are unscaled
            # the zeroth fourier coefficient is the (unscaled) discrete integral
            # over the function (multiplied by 1=e^(i*0))
            # and since the pseudospectral method conserves the energy we require the starting energy to be zero
            # so positive and negative parts sum away (is there a better explanation?)
            print("Warning! Start velocity for wave solver not possible, solution will be incorrect: "
                  + "0th fourier coefficient not zero but", y0t_[self.zero_index], "for size",
                  self.norm2_factors.size)
        # this makes sure that c2_[zero_index] is 0 and no warning is triggered
        self.norm2_factors[self.zero_index] = np.inf
        c2_ = y0t_ / self.norm2_factors
        self.norm2_factors[self.zero_index] = 0  # revert temporal change
        c2_ *= 1 / self.wave_speed

        def solution_at(time):
            # for all j solve (d/dt)^2 u_hat(j; t) = -j*j*u_hat(j; t) and starting conditions u_hat(j;0)=y0_(j),
            # (d/dt)u_hat(j;0)=y0t_(j)
            # here we are in the position to know the exact solution for this linear ordinary differential equation!

            # solution at time t with starting value y0_ and y0t_, all in fourier space

            u_hat_ = c1_ * np.cos(self.wave_speed * self.norm2_factors * (time - t0)) \
                     + c2_ * np.sin(self.wave_speed * self.norm2_factors * (time - t0))

            return ifftn(u_hat_)

        self.solver = solution_at
