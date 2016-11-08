import numpy as np
from itertools import zip_longest

# Splitting resources: https://www.math.ntnu.no/~holden/operatorsplitting/

# calculates the solution of the linear hyperbolic differential equation at grid points in intervals
# u_tt(t,x)=-beta(x)u(t,x), beta(x)>0 for all x
def linhyp_solution(intervals, grid_points_list, t0, u0, u0t, beta, wanted_times):
    xs = []
    for interval, grid_points in zip_longest(intervals, grid_points_list, fillvalue=grid_points_list[-1]):
        x = np.linspace(interval[0], interval[1], endpoint=False, num=grid_points)
        xs.append(x)
    xxs = np.meshgrid(*xs, sparse=True)
    beta_sqrt = np.sqrt(beta(xxs))
    y0 = u0(xxs) if callable(u0) else u0
    c1 = y0
    y0t = u0t(xxs) if callable(u0t) else u0t
    c2 = np.nan_to_num(y0t / beta_sqrt)  # just ignore where beta is zero as sin(beta) will also be zero

    times = list(filter(lambda time_check: time_check >= t0, wanted_times))
    solutions = []
    for time in times:
        y = c1 * np.cos(beta_sqrt * (time - t0)) + c2 * np.sin(beta_sqrt * (time - t0))
        solutions.append(y)

    return xs, times, solutions


""" This is not required for our case, but for quick lookup on general ode case this is the framework for solving:
from scipy.integrate import ode
def f(t, y):
    return y

r = ode(f).set_integrator('zvode', method='adams')  # 'zvode' for complex valued problems, 'adams' for non-stiff
r.set_initial_value(y0, t0)

while r.successful() and r.t < t_end:
    r.integrate(r.t + dt)
    print("{} {}".format(r.t, r.y))"""