import numpy as np
from diff_equation.solver_config import SolverConfig


# calculates the solution of the linear hyperbolic differential equation at grid points in intervals
# u_tt(t,x)=-beta(x)u(t,x), beta(x)>0 for all x
class LinhypSolverConfig(SolverConfig):

    # splitting_factor=0. is the old LinhypMomentConfig, though not quite as fast due to redundant calculations
    def __init__(self, intervals, grid_points_list, beta, splitting_factor=1.):
        super().__init__(intervals, grid_points_list)
        self.beta = beta(self.xs_mesh)
        self.beta_sqrt = np.sqrt(self.beta)
        self.splitting_factor = splitting_factor

    def start_momentum(self):
        return -self.beta * self.start_position

    def init_solver(self, t0, u0, u0t):
        self.init_initial_values(t0, u0, u0t)

        c2 = self.start_velocity * self.splitting_factor

        c1_derivative = -self.beta * self.start_position

        def solution_at(time):
            argument = self.beta_sqrt * (time - self.start_time) * np.sqrt(self.splitting_factor)
            cos_arg = np.cos(argument)
            # motivation to use np.sinc over np.sin is to avoid handling splitting_factor=0. explicitly
            sinc_arg = np.sinc(argument / np.pi) * (time - self.start_time)
            return [self.start_position * cos_arg + c2 * sinc_arg,
                    c1_derivative * sinc_arg + self.start_velocity * cos_arg]

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