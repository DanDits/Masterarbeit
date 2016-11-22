import numpy as np
from itertools import repeat
import polynomial_chaos.distributions as distributions
from stochastic_equations.stochastic_trial import StochasticTrial
from stochastic_equations.collocation.interpolation import matrix_inversion_expectancy
import matplotlib.pyplot as plt
from util.analysis import error_l2

# y[0] > 1
left_1, right_1 = 2, 3
trial_1 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: 2 * np.sin(sum(xs)),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: 2 * np.cos(t * ys[0]) * np.sin(sum(xs)),
                          # from U(-1,1) to U(left_1, right_1)
                          random_variables=[lambda y: (right_1 - left_1) / 2 * (y + 1) + left_1]) \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    "expectancy", lambda xs, t: (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                 * (np.sin(right_1 * t) - np.sin(left_1 * t))))
# y[0] in (0,1)
left_2, right_2 = 0.1, 0.9
trial_2 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                          # from U(-1,1) to U(left_2, right_2)
                          random_variables=[lambda y: (right_2 - left_2) / 2 * (y + 1) + left_2])
trial_2.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                       "alpha", lambda ys: 1 / ys[0])
# y[0] in (0,1), is enforced by random variable which can take any real value!
trial_2_1 = StochasticTrial([distributions.gaussian],
                            lambda xs, ys: np.zeros(shape=sum(xs).shape),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2])
trial_2_1.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])
left_3, right_3 = 10, 50  # y[0] bigger than 2
trial_3 = StochasticTrial([distributions.make_uniform(-1, 1)],  # y[0] bigger than 2 enforced by random variable
                          lambda xs, ys: 1 / (np.sin(sum(xs)) + ys[0]),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: np.cos(t) / (np.sin(sum(xs)) + ys[0]),
                          # from U(-1,1) to U(left_3, right_3)
                          random_variables=[lambda y: (right_3 - left_3) / 2 * (y + 1) + left_3]) \
    .add_parameters("beta", lambda xs, ys: 1 + (ys[0] - 2) * (np.sin(sum(xs)) / (np.sin(sum(xs)) + ys[0])
                                                              + 2 * np.cos(sum(xs)) ** 2
                                                              / (np.sin(sum(xs)) + ys[0]) ** 2),
                    "alpha", lambda ys: ys[0] - 2,
                    "expectancy", lambda xs, t: np.cos(t) / (right_3 - left_3)
                                                * (np.log(np.sin(sum(xs)) + right_3)
                                                   - np.log(np.sin(sum(xs)) + left_3)))

trial = trial_2_1
N = list(range(40))  # maximum degree of the polynomial, so N+1 polynomials
# from n+1 to n+10 notably difference for most examples
M = [n + 1 for n in N]  # number of nodes in random space, >= N+1, the higher the more accuracy (for higher polys)
spatial_dimension = 1
grid_size = 128
spatial_domain = list(repeat([-np.pi, np.pi], spatial_dimension))
start_time = 0
stop_time = 0.5
delta_time = 0.001

expectancies = []
for n, m in zip(N, M):
    result_xs, result_xs_mesh, expectancy = matrix_inversion_expectancy(trial, n, m, spatial_domain, grid_size,
                                                                        start_time, stop_time, delta_time)
    expectancies.append((n, m, expectancy))
# TODO try plotting error in dependence of N (and M?), seems to rise again if N gets too big
# TODO try other choice of random grid points
# TODO try other distributions
print("Plotting:")
if trial.has_parameter("expectancy"):
    trial_expectancy = trial.expectancy(result_xs_mesh, stop_time)
else:
    trial_expectancy = trial.calculate_expectancy(result_xs, stop_time, trial.raw_reference)
plt.figure()
plt.title("Expectancies by matrix inversion coll. in spatial grid size={}".format(grid_size))
for n, m, expectancy in expectancies:
    error = error_l2(trial_expectancy, expectancy)
    print("Error", n, "=", error)
    plt.plot(result_xs[0], expectancy, "o" if n == 2 else ".", label="deg={}, error={:.5E}"
             .format(n, error))
plt.plot(result_xs[0], trial_expectancy, label="Exact expectancy")
# plt.ylim((0, 1))
plt.legend()
plt.show()
