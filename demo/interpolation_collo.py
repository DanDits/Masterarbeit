import numpy as np
from itertools import repeat
import polynomial_chaos.distributions as distributions
from polynomial_chaos.multivariation import multi_index_bounded_sum_length
from stochastic_equations.collocation.discrete_projection import discrete_projection_expectancy
from stochastic_equations.stochastic_trial import StochasticTrial
from stochastic_equations.collocation.interpolation import matrix_inversion_expectancy_1d, matrix_inversion_expectancy
import matplotlib.pyplot as plt
from util.analysis import error_l2
from util.storage import save_fig

# y[0] > 1
left_1, right_1 = 2, 3
trial_1 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: 2 * np.sin(sum(xs)),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: 2 * np.cos(t * ys[0]) * np.sin(sum(xs)),
                          # from U(-1,1) to U(left_1, right_1)
                          random_variables=[lambda y: (right_1 - left_1) / 2 * (y + 1) + left_1],
                          name="Trial1") \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    "expectancy", lambda xs, t: (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                 * (np.sin(right_1 * t) - np.sin(left_1 * t))),
                    "variance", lambda xs, t: (1 / (t * (right_1 - left_1)) * np.sin(sum(xs)) ** 2
                                               * (2 * t * right_1 + np.sin(2 * t * right_1)
                                                  - 2 * t * left_1 - np.sin(2 * t * left_1))
                                               - (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                  * (np.sin(right_1 * t) - np.sin(left_1 * t))) ** 2))
# y[0] in (0,1)
left_2, right_2 = 0.1, 0.9
trial_2 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                          # from U(-1,1) to U(left_2, right_2)
                          random_variables=[lambda y: (right_2 - left_2) / 2 * (y + 1) + left_2],
                          name="Trial2")
trial_2.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                       "alpha", lambda ys: 1 / ys[0])
# y[0] in (0,1), is enforced by random variable which can take any real value!
trial_2_1 = StochasticTrial([distributions.gaussian],
                            lambda xs, ys: np.zeros(shape=sum(xs).shape),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_1")
trial_2_1.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])
trial_2_2 = StochasticTrial([distributions.make_gamma(2.5, 1)],
                            lambda xs, ys: np.zeros(shape=sum(xs).shape),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_2")
trial_2_2.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])
trial_2_3 = StochasticTrial([distributions.make_beta(-0.5, 0.7)],
                            lambda xs, ys: np.zeros(shape=sum(xs).shape),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_3")
trial_2_3.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])
left_3, right_3 = 10, 50  # y[0] bigger than 2
trial_3 = StochasticTrial([distributions.make_uniform(-1, 1)],  # y[0] bigger than 2 enforced by random variable
                          lambda xs, ys: 1 / (np.sin(sum(xs)) + ys[0]),
                          lambda xs, ys: np.zeros(shape=sum(xs).shape),
                          lambda xs, t, ys: np.cos(t) / (np.sin(sum(xs)) + ys[0]),
                          # from U(-1,1) to U(left_3, right_3)
                          random_variables=[lambda y: (right_3 - left_3) / 2 * (y + 1) + left_3],
                          name="Trial3") \
    .add_parameters("beta", lambda xs, ys: 1 + (ys[0] - 2) * (np.sin(sum(xs)) / (np.sin(sum(xs)) + ys[0])
                                                              + 2 * np.cos(sum(xs)) ** 2
                                                              / (np.sin(sum(xs)) + ys[0]) ** 2),
                    "alpha", lambda ys: ys[0] - 2,
                    "expectancy", lambda xs, t: (np.cos(t) / (right_3 - left_3)
                                                 * (np.log(np.sin(sum(xs)) + right_3)
                                                    - np.log(np.sin(sum(xs)) + left_3))))
# equal to mc trial_4
trial_mc4 = StochasticTrial([distributions.gaussian, distributions.make_uniform(-1, 1),
                             distributions.make_beta(-0.5, 2.5), distributions.make_uniform(-1, 1)],
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, ys: np.sin(sum(xs)) ** 2,
                            random_variables=[lambda y: np.exp(y), lambda y: (y + 1) / 2,
                                              lambda y: y, lambda y: y * 4 + 2],
                            name="Trialmc4") \
    .add_parameters("beta", lambda xs, ys: 3 + np.sin(xs[0] + ys[2]) + np.sin(xs[0] + ys[3]),
                    "alpha", lambda ys: 1 + 0.5 * ys[0] + 3 * ys[1],
                    "expectancy_data", "../data/mc_100000, Trial4, 0.5, 128.npy")
# equal to mc trial_5, we have saved simulation data: (-> dt=0.0001 or else unstable after degree 6)
trial_mc5 = StochasticTrial([distributions.gaussian],
                            lambda xs, ys: np.cos(sum(xs)),
                            lambda xs, ys: np.sin(sum([x ** 2 for x in xs])),
                            name="Trialmc5") \
    .add_parameters("beta", lambda xs, ys: 3 + np.sin(xs[0] * ys[0]) + np.sin(xs[0] + ys[0]),
                    "alpha", lambda ys: 1 + np.exp(ys[0]),
                    "expectancy_data", "../data/mc_100000, Trial5, 0.5, 512.npy")
trial = trial_mc4

# "High order is not the same as high accuracy. High order translates to high accuracy only when the integrand
# is very smooth" (http://apps.nrbook.com/empanel/index.html?pg=179#)
N = list(range(20))  # maximum degree of the polynomial, so N+1 polynomials
# from n+1 to n+10 notably difference for most examples

# number of nodes in random space, >= N+1, higher CAN give more accuracy (for higher polys)
# M = [(n + 1,) * len(trial.variable_distributions) for n in N]

# minimum possible value that the full tensor product of nodes is still bigger than the number of basis polynomials
# if minimal: rank of vandermonde decreases by 10-30% and solution does not improve; very fast!
# if minimal+1:
M = [(int(np.ceil(multi_index_bounded_sum_length(len(trial.variable_distributions), n)
              ** (1 / len(trial.variable_distributions)))),) * len(trial.variable_distributions)
     for n in N]
Q = [15] * len(N)  # number of nodes and weights used for discrete projection's quadrature formula
spatial_dimension = 1
grid_size = 128
spatial_domain = list(repeat([-np.pi, np.pi], spatial_dimension))
start_time = 0
stop_time = 0.5
delta_time = 0.001  # if grid_size is bigger this needs to be smaller, especially for higher poly degrees
use_matrix_inversion = True

rank_frac = None
exp_var_results, rank_fracs = [], []
for n, m, q in zip(N, M, Q):
    print("n,m,q=", n, m, q)
    if use_matrix_inversion:
        result_xs, result_xs_mesh, expectancy, variance, rank_frac = matrix_inversion_expectancy(trial, n, m,
                                                                                                 spatial_domain,
                                                                                                 grid_size,
                                                                                                 start_time, stop_time,
                                                                                                 delta_time)
    else:
        result_xs, result_xs_mesh, expectancy, variance = discrete_projection_expectancy(trial, n, q,
                                                                                         spatial_domain, grid_size,
                                                                                         start_time, stop_time,
                                                                                         delta_time)
    exp_var_results.append((n, m, expectancy, variance))
    if rank_frac is not None:
        rank_fracs.append(rank_frac)
rank_fractions = list(map(lambda frac: 10 ** ((frac - 1) * 10),
                          rank_fracs))  # rescale to make visible in logarithmic scale

print("Plotting:")
trial_expectancy = None
if trial.has_parameter("expectancy"):
    trial_expectancy = trial.expectancy(result_xs_mesh, stop_time)
elif trial.raw_reference is not None:
    print("Calculating expectancy")
    trial_expectancy = trial.calculate_expectancy(result_xs, stop_time, trial.raw_reference)
elif trial.has_parameter("expectancy_data"):
    try:
        trial_expectancy = np.load(trial.expectancy_data)
    except FileNotFoundError:
        print("No expectancy data found, should be here!?")
trial_variance = None
if trial.has_parameter("variance"):
    trial_variance = trial.variance(result_xs_mesh, stop_time)

plt.figure()
plt.title("Expectancies, spatial grid size={}, {}".format(grid_size, trial.name))
errors, errors_variance = [], []
for n, m, expectancy, variance in exp_var_results:
    error = -1
    if trial_expectancy is not None:
        error = error_l2(trial_expectancy, expectancy)
        errors.append(error)
        print("Error", n, "=", error)
    if trial_variance is not None:
        error_var = error_l2(trial_variance, variance)
        errors_variance.append(error_var)
        print("Error variance", n, "=", error_var)
    plt.plot(result_xs[0], expectancy, "o" if n < 7 else ".", label="deg={}, error={:.5E}"
             .format(n, error))
ref = trial_expectancy
if ref is not None:
    plt.plot(result_xs[0], ref, label="Exact reference")
# plt.ylim((0, 1))
plt.legend()
# save_fig(plt.axes(), "../data/interpol_invmat_trial5_512_0.00005.pickle")
if len(errors) > 0:
    plt.figure()
    plt.title("Collocation interpolation by {} for {}".format(("Matrix inversion" if use_matrix_inversion
                                                               else "Discrete projection"),
                                                              trial.name))
    plt.plot(N, errors, label="Errors to expectancy")
    if len(errors_variance) > 0:
        plt.plot(N, errors_variance, label="Errors to variance")
    if len(rank_fractions) > 0:
        plt.plot(N, rank_fractions, label="Vandermonde rank deficiency")
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Maximum polynom degree')
plt.show()
