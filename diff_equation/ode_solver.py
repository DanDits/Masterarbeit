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


class LinhypMomentConfig(SolverConfig):

    def __init__(self, intervals, grid_points_list, beta):
        super().__init__(intervals, grid_points_list)
        self.beta = beta(self.xs_mesh)
        self.start_moment = None

    def init_solver(self, t0, u0, u0t):
        self.init_initial_values(t0, u0, u0t)

        self.start_moment = -self.beta * self.start_position

        def solution_at(time):
            return [self.start_position,
                    self.start_velocity + self.start_moment * (time - self.start_time)]

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