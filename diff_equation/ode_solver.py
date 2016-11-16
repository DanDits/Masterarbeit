import numpy as np
from diff_equation.solver_config import SolverConfig


# calculates the solution of the linear hyperbolic differential equation at grid points in intervals
# u_tt(t,x)=-beta(x)u(t,x), beta(x)>0 for all x
class LinhypSolverConfig(SolverConfig):

    def __init__(self, intervals, grid_points_list, beta):
        super().__init__(intervals, grid_points_list)
        self.beta = beta(self.xs_mesh)
        self.beta_sqrt = np.sqrt(self.beta)

    def start_momentum(self):
        return -self.beta * self.start_position

    def init_solver(self, t0, u0, u0t):
        self.init_initial_values(t0, u0, u0t)

        c1 = self.start_position
        # just ignore where beta is zero as sin(beta) will also be  zero
        c2 = np.nan_to_num(self.start_velocity / self.beta_sqrt)

        def solution_at(time):
            return [c1 * np.cos(self.beta_sqrt * (time - self.start_time))
                    + c2 * np.sin(self.beta_sqrt * (time - self.start_time)),
                    self.start_velocity + self.start_momentum() * (time - self.start_time)]

        self.solver = solution_at


# TODO incorrect and WIP
class OffsetLinhypSolver(SolverConfig):
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

    def init_solver(self, t0, u0, u0t):

        self.init_initial_values(t0, u0, u0t)
        v_start = self.start_velocity - (1j / self.japan) * (self.start_velocity * self.c / (self.wave_speed ** 2))
        v_ascend = 1  # TODO not correct, solver needs to return derivative as well

        def solution_at(time):
            v = v_start + (time - self.start_time) * v_ascend
            return [(v + np.conj(v)) / 2]

        self.solver = solution_at


""" This is not required for our case, but for quick lookup on general ode case this is the framework for solving:
from scipy.integrate import ode
def f(t, y):
    return y

r = ode(f).set_integrator('zvode', method='adams')  # 'zvode' for complex valued problems, 'adams' for non-stiff
r.set_initial_value(y0, t0)

while r.successful() and r.t < t_end:
    r.integrate(r.t + dt)
    print("{} {}".format(r.t, r.y))"""