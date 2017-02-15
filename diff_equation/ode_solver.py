import numpy as np
from diff_equation.solver_config import SolverConfig


# calculates the solution of the linear hyperbolic differential equation at grid points in intervals
# u_tt(t,x)=-beta(x)u(t,x), beta(x)>0 for all x
class LinhypSolverConfig(SolverConfig):

    # splitting_factor=0. is the old LinhypMomentConfig, though not quite as fast due to redundant calculations
    def __init__(self, intervals, grid_points_list, beta, splitting_factor=1.):
        super().__init__(intervals, grid_points_list)
        if callable(beta):
            self.beta = beta(self.xs_mesh)
        else:
            self.beta = beta  # if we cannot call it then it has to be already correct
        self.sinc_arg = None
        self.cos_arg = None
        self.beta_sqrt = np.sqrt(self.beta)
        self.splitting_factor = splitting_factor

    def start_momentum(self):
        return -self.beta * self.start_position

    def update_cached(self, delta_time):
        self.last_delta_time = delta_time
        argument = self.beta_sqrt * (delta_time * np.sqrt(self.splitting_factor))

        # np.sinc multiplies arguments by pi, so revert this
        # motivation to use np.sinc over np.sin is to avoid handling splitting_factor=0. explicitly
        self.sinc_arg = np.sinc(argument / np.pi) * delta_time
        self.cos_arg = np.cos(argument)

    def init_solver(self, t0, u0, u0t):
        self.init_initial_values(t0, u0, u0t)

        c2 = self.start_velocity * self.splitting_factor

        c1_derivative = -self.beta * self.start_position

        def solution_at(time):
            if self.is_new_delta_time(time - self.start_time):
                self.update_cached(time - self.start_time)
            return [self.start_position * self.cos_arg + c2 * self.sinc_arg,
                    c1_derivative * self.sinc_arg + self.start_velocity * self.cos_arg]

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