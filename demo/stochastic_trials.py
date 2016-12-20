import numpy as np
from stochastic_equations.stochastic_trial import StochasticTrial
from polynomial_chaos import distributions


# y[0] > 1
left_1, right_1 = 2, 3
trial_1 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: 2 * np.sin(sum(xs)),
                          lambda xs, ys: 0 * sum(xs),
                          lambda xs, t, ys: 2 * np.cos(t * ys[0]) * np.sin(sum(xs)),
                          # from U(-1,1) to U(left_1, right_1)
                          random_variables=[lambda y: (right_1 - left_1) / 2 * (y + 1) + left_1],
                          name="Trial1") \
    .add_parameters("beta", lambda xs, ys: ys[0] ** 2 - ys[0],  # y^2 - alpha(y)
                    "alpha", lambda ys: ys[0],
                    # at t=0.5: 0.624096 sin(x)
                    "expectancy", lambda xs, t: (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                 * (np.sin(right_1 * t) - np.sin(left_1 * t))),
                    "variance", lambda xs, t: (1 / (t * (right_1 - left_1)) * np.sin(sum(xs)) ** 2
                                               * (2 * t * right_1 + np.sin(2 * t * right_1)
                                                  - 2 * t * left_1 - np.sin(2 * t * left_1))
                                               - (2 / (t * (right_1 - left_1)) * np.sin(sum(xs))
                                                  * (np.sin(right_1 * t) - np.sin(left_1 * t))) ** 2),
                    # at t=0.5: 0.630645 sin(x), the solution evaluated at the expectancy of y[0]
                    "orientation_func", lambda xs, t: np.sin(sum(xs) + t * 2.5) + np.sin(sum(xs) - t * 2.5))
# y[0] in (0,1)
left_2, right_2 = 0.1, 0.9
trial_2 = StochasticTrial([distributions.make_uniform(-1, 1)],
                          lambda xs, ys: 0 * sum(xs),
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                          # from U(-1,1) to U(left_2, right_2)
                          random_variables=[lambda y: (right_2 - left_2) / 2 * (y + 1) + left_2],
                          name="Trial2")
trial_2.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                       "alpha", lambda ys: 1 / ys[0])
# y[0] in (0,1), is enforced by random variable which can take any real value!
trial_2_1 = StochasticTrial([distributions.gaussian],
                            lambda xs, ys: 0 * sum(xs),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_1")
trial_2_1.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])

trial_2_2 = StochasticTrial([distributions.make_gamma(2.5, 1)],
                            lambda xs, ys: 0 * sum(xs),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_2")
trial_2_2.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])
trial_2_3 = StochasticTrial([distributions.make_beta(-0.5, 0.7)],
                            lambda xs, ys: 0 * sum(xs),
                            lambda xs, ys: np.sin(sum(xs)),
                            lambda xs, t, ys: np.sin(sum(xs)) * np.sin(t / ys[0]) * ys[0],
                            random_variables=[lambda y: 0.5 + 0.2 * np.sin(y) ** 2],
                            name="Trial2_3")
trial_2_3.add_parameters("beta", lambda xs, ys: 1 / ys[0] ** 2 - 1 / ys[0],  # 1/y^2 - alpha(y)
                         "alpha", lambda ys: 1 / ys[0])

left_3, right_3 = 10, 50  # y[0] bigger than 2
trial_3 = StochasticTrial([distributions.make_uniform(-1, 1)],  # y[0] bigger than 2 enforced by random variable
                          lambda xs, ys: 1 / (np.sin(sum(xs)) + ys[0]),
                          lambda xs, ys: 0 * sum(xs),
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
trial_4 = StochasticTrial([distributions.gaussian, distributions.make_uniform(-1, 1),
                           distributions.make_beta(-0.5, 2.5), distributions.make_uniform(-1, 1)],
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, ys: np.sin(sum(xs)) ** 2,
                          random_variables=[lambda y: np.exp(y), lambda y: (y + 1) / 2,
                                            lambda y: y, lambda y: y * 4 + 2],
                          name="Trial4") \
    .add_parameters("beta", lambda xs, ys: 3 + np.sin(xs[0] + ys[2]) + np.sin(xs[0] + ys[3]),
                    "alpha", lambda ys: 1 + 0.5 * ys[0] + 3 * ys[1],
                    "expectancy_data", "../data/qmc_exp, 100000, Trial4, 0.5, 128.npy",
                    "variance_data", "../data/qmc_var, 100000, Trial4, 0.5, 128.npy",
                    "stop_time", 0.5,
                    "grid_size", 128)

# unstable for N=512,dt=0.001:  Degree 6
#              N=128,dt=0.001:  Degree 15
#              N=128,dt=0.0001: Degree>29
# accuracy of expectancy data for MI(dt=0.001): 4.27996312063e-05 at degree=37, high condition all the time, unstable after degree=40
trial_5 = StochasticTrial([distributions.gaussian],
                          lambda xs, ys: np.cos(sum(xs)),
                          lambda xs, ys: np.sin(sum([x ** 2 for x in xs])),
                          name="Trial5") \
    .add_parameters("beta", lambda xs, ys: 3 + np.sin(xs[0] * ys[0]) + np.sin(xs[0] + ys[0]),
                    "alpha", lambda ys: 1 + np.exp(ys[0]),
                    "expectancy_data", "../data/qmc_200000, Trial5, 0.5, 128.npy",
                    "delta_time", 0.0001,
                    "grid_size", 128,
                    "stop_time", 0.5)

trial_6 = StochasticTrial([distributions.make_beta(1.5, 4.5), distributions.make_uniform(-1, 1),
                           distributions.make_beta(-0.5, 2.5), distributions.make_uniform(-1, 1)],
                          lambda xs, ys: np.sin(sum(xs)),
                          lambda xs, ys: np.sin(sum(xs)) ** 2,
                          random_variables=[lambda y: np.exp(y), lambda y: (y + 1) / 2,
                                            lambda y: y, lambda y: y * 4 + 2],
                          name="Trial6") \
    .add_parameters("beta", lambda xs, ys: 3 + np.sin(xs[0] + ys[2]) + np.sin(xs[0] + ys[3]),
                    "alpha", lambda ys: 1 + 0.5 * ys[0] + 3 * ys[1],
                    "expectancy_data", "../data/qmc_exp, 100000, Trial6, 0.5, 128.npy",
                    "variance_data", "../data/qmc_var, 100000, Trial6, 0.5, 128.npy",
                    "grid_size", 128,
                    "stop_time", 0.5)

# accuracy of expectancy data for MI(dt=0.001) around degree=8: 6.3483685515377584e-06, for Galerkin(?): 6.2370461164611651e-06
# accuracy of variance data for MI(dt=0.001) around degree=8: 5.86310961747e-07
trial_7 = StochasticTrial([distributions.make_beta(0.5, 0.5)],
                          lambda xs, ys: np.cos(sum(xs)),
                          lambda xs, ys: np.sin(sum([x ** 2 for x in xs])),
                          name="Trial7") \
    .add_parameters("beta", lambda xs, ys: 4 + np.sin(2 * xs[0] + ys[0]) + 2 * np.sin(xs[0] + ys[0]),
                    "alpha", lambda ys: 1 + np.exp(ys[0]),
                    "grid_size", 128,
                    "stop_time", 0.5,
                    "expectancy_data", "../data/qmc_exp, 100000, Trial7, 0.5, 128.npy",
                    "variance_data", "../data/qmc_var, 100000, Trial7, 0.5, 128.npy")

# accuracy of expectancy data for Galerkin: 0.006854368887610221, too many QP decrease again?! for MI: 1.1360616011928265e-05, zigzac, convergent for uneven poly degrees, unstable before fully converged around degree 60
# accuracy of variance data for MI(dt=0.001): 7.38374747346e-05, zigzac, convergent for uneven degrees, same behavior as expectancy
trial_discont = StochasticTrial([distributions.make_uniform(-1, 1)],
                                lambda xs, ys: np.cos(sum(xs)),
                                lambda xs, ys: np.sin(sum([x ** 2 for x in xs])),
                                name="TrialDiscont") \
    .add_parameters("beta", lambda xs, ys: 100 * np.cos(ys[0]) ** 2 + 1 if ys[0] > 0. else 1.5 + np.sin(3 * ys[0]),
                    "alpha", lambda ys: 2 + ys[0] if ys[0] > 0. else 0.5,
                    "expectancy_data", "../data/qmc_exp, 100000, TrialDiscont, 0.5, 128.npy",
                    "variance_data", "../data/qmc_var, 100000, TrialDiscont, 0.5, 128.npy")

trial_discont_simple = StochasticTrial([distributions.make_uniform(-1, 1)],
                                       lambda xs, ys: np.cos(sum(xs)),
                                       lambda xs, ys: np.sin(sum([x ** 2 for x in xs])),
                                       name="TrialDiscontSimple") \
    .add_parameters("beta", lambda xs, ys: 2 + np.sin(xs[0] + ys[0]) if ys[0] > 0. else 2 + np.cos(xs[0] + ys[0]),
                    "alpha", lambda ys: 2. if ys[0] > 0. else 1.)
