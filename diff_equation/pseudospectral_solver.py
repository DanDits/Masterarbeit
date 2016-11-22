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


class VelocityConfig(SolverConfig):
    def init_solver(self, t0, u0, u0t):
        self.init_initial_values(t0, u0, u0t)
        self.solver = (lambda time: [self.start_position + (time - self.start_time) * self.start_velocity,
                                     self.start_velocity])


class KleinGordonMomentConfig(SolverConfig):

    def __init__(self, intervals, grid_points_list, alpha, beta):
        super().__init__(intervals, grid_points_list, pseudospectral_power=2)
        self.pseudospectral_mesh_sum = sum(self.pseudospectral_factors_mesh)
        self.moment = None
        self.alpha = alpha()
        self.beta = beta(self.xs_mesh)

    def init_solver(self, t0, u0, u0t):
        self.init_initial_values(t0, u0, u0t)
        uxx = ifftn(self.pseudospectral_mesh_sum * fftn(self.start_position))
        # saves the moment alpha*u_xx -beta(x)u at the given start time to estimate a new velocity
        self.moment = (self.alpha * uxx - self.beta * self.start_position)

        def solution_at(time):
            return [self.start_position,
                    self.start_velocity + self.moment * (time - self.start_time)]
        self.solver = solution_at


class OffsetWaveSolverConfig(SolverConfig):
    def __init__(self, intervals, grid_points_list, wave_speed, offset):
        super().__init__(intervals, grid_points_list, pseudospectral_power=2)
        assert wave_speed > 0.
        assert offset > 0.
        self.wave_speed = wave_speed
        self.c = np.sqrt(offset) / wave_speed
        self.offset = offset

        # as the pseudospectral factors of order 2 are negative integers, negate sum!
        # numbers are real (+0j) anyways, but using '.real' saves some calculation time and storage space
        # in literature sometimes referred to as "japanese symbol"...
        self.japan = np.sqrt(-sum(self.pseudospectral_factors_mesh).real + self.c ** 2)

        # the name for this constant is a secret and has nothing to do with my ability to calculate stuff..
        self.magic = 1. / self.wave_speed

    def init_solver(self, t0, u0, u0t):

        self.init_initial_values(t0, u0, u0t)
        # variables ending in underscore note that the values are considered to be in fourier space

        y0_ = fftn(self.start_position)  # starting condition in fourier space and evaluated at grid
        y0t_ = fftn(self.start_velocity)
        v_start_ = y0_ - (1j / self.japan) * (y0t_ * self.magic)

        def solution_at(time):
            v_hat_ = np.exp(1j * self.japan / self.magic * (time - self.start_time)) * v_start_

            v = ifftn(v_hat_)
            u = v.real
            return [u,
                    ifftn(1j * self.japan / self.magic * (v_hat_ - fftn(u)))]

        self.solver = solution_at


# u_tt(t,x) = (wave_speed ** 2) * u_xx(t,x), u(t0,x)=u0(x), u_t(t0,x)=u0_t(x), wave_speed > 0
class WaveSolverConfig(SolverConfig):

    def __init__(self, intervals, grid_points_list, wave_speed):
        super().__init__(intervals, grid_points_list, pseudospectral_power=2)
        assert wave_speed > 0.
        self.wave_speed = wave_speed

        # as the pseudospectral factors of order 2 are negative integers, negate sum!
        # numbers are real (+0j) anyways, but using '.real' might save some calculation time and storage space
        self.norm2_factors = np.sqrt(-sum(self.pseudospectral_factors_mesh).real)
        # top left corner of norm2_ks contains a zero
        self.zero_index = (0,) * len(self.norm2_factors.shape)
        assert self.norm2_factors[self.zero_index] == 0
        # but when we calculate sin(j)/j for j->0 (so, the zeroth fourier coefficient) this will lead to errors
        # therefore we handle this linear growth of the zeroth fourier coefficient explicitly

        # more or less a copy, but zeroth factor changed to avoid division by zero
        self.norm2_factors_special = np.array(self.norm2_factors)
        self.norm2_factors_special[self.zero_index] = 1 / self.wave_speed

    def start_momentum(self):
        return (self.wave_speed ** 2) * ifftn(sum(self.pseudospectral_factors_mesh) * fftn(self.start_position))

    def init_solver(self, t0, u0, u0t):
        # pre calculations depending on starting values, wave speed,...

        self.init_initial_values(t0, u0, u0t)
        # variables ending in underscore note that the values are considered to be in fourier space

        y0_ = fftn(self.start_position)  # starting condition in fourier space and evaluated at grid
        y0t_ = fftn(self.start_velocity)

        # calculate factors c1_ and c2_
        c1_ = y0_
        c2_ = y0t_ / (self.wave_speed * self.norm2_factors_special)

        # c1_derivative_ = -self.norm2_factors * self.wave_speed * y0_
        # c2_derivative_ = y0t_

        def solution_at(time):
            # for all j solve (d/dt)^2 u_hat(j; t) = -j*j*u_hat(j; t) and starting conditions u_hat(j;0)=y0_(j),
            # (d/dt)u_hat(j;0)=y0t_(j)
            # here we are in the position to know the exact solution for this linear ordinary differential equation!

            # solution at time t with starting value y0_ and y0t_, all in fourier space
            sin_part = np.sin(self.wave_speed * self.norm2_factors_special * (time - t0))
            sin_part[self.zero_index] = (time - t0)  # special treatment for zeroth index as sin(j)/j ->1 (j->0)
            u_hat_ = (c1_ * np.cos(self.wave_speed * self.norm2_factors * (time - t0))
                      + c2_ * sin_part)
            # ut_hat_ = (c1_derivative_ * np.sin(self.wave_speed * self.norm2_factors * (time - t0))
            #           + c2_derivative_ * np.cos(self.wave_speed * self.norm2_factors * (time - t0)))

            return [ifftn(u_hat_),
                    self.start_velocity + self.start_momentum() * (time - t0)]
                    # ifftn(ut_hat_)]  # TODO incorrect, where did I miscalculate?

        self.solver = solution_at
