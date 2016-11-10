import numpy as np
from diff_equation.solver_config import SolverConfig


# TODO we could refactor this and the wave solver into a subclass of SolverConfig some time
# calculates the solution of the linear hyperbolic differential equation at grid points in intervals
# u_tt(t,x)=-beta(x)u(t,x), beta(x)>0 for all x
def make_linhyp_config(intervals, grid_points_list, beta):
    config = SolverConfig(intervals, grid_points_list)
    config.beta_sqrt = np.sqrt(beta(config.xs_mesh))
    return config


def init_linhyp_solver(config, t0, u0, u0t):
    config.init_initial_values(t0, u0, u0t)

    c1 = config.start_position
    # just ignore where beta is zero as sin(beta) will also be  zero
    c2 = np.nan_to_num(config.start_velocity / config.beta_sqrt)

    def solution_at(time):
        return c1 * np.cos(config.beta_sqrt * (time - t0)) + c2 * np.sin(config.beta_sqrt * (time - t0))
    config.solver = solution_at


""" This is not required for our case, but for quick lookup on general ode case this is the framework for solving:
from scipy.integrate import ode
def f(t, y):
    return y

r = ode(f).set_integrator('zvode', method='adams')  # 'zvode' for complex valued problems, 'adams' for non-stiff
r.set_initial_value(y0, t0)

while r.successful() and r.t < t_end:
    r.integrate(r.t + dt)
    print("{} {}".format(r.t, r.y))"""