# Collocation using discrete projection. Uses a lot of caching of the collocation points to increase performance
import demo.stochastic_trials as st
from itertools import repeat
import numpy as np
from polynomial_chaos.poly_chaos_distributions import get_chaos_name_by_distribution
import util.quadrature.nesting as nst
from stochastic_equations.collocation.discrete_projection import discrete_projection
from util.analysis import error_l2_relative
import matplotlib.pyplot as plt

# EXPERIMENT OBSERVATIONS:
# for 1d: max(P)=15 (e.g. [0,1,5,15]) and max_level=5 is fine and only takes several seconds
# trial_discont_simple: for expectancy sparse is way worse (even though its 1d), variance converges very slowly in p
# trial_7: sparse and full_tensor same (its 1d), one observes that if p is higher needs more quad points as polynomials are higher
# for 4d: max_level=5 is ok (this is 9065 coll. points) but takes some time, max_level=4 is 2309
#           time increase in p is linear due to caching but for p=15 already takes like 30 minutes for lvl 4

# COMMON CONFIGURATION HERE:
trial = st.trial_4  # which trial to calculate expectancy and variance for
only_expectancy = False  # if only expectancy is to be calculated, this allows only using the zeroth polynomial!

# Which values to choose for P and max_level greatly depend on the dimension of the trial's random space
# the amount of basis polynomials in one dimension, is by one higher than values of P (= the sum bound)
P = [0, 1, 5, 15] if not only_expectancy else [0]  # should not use more than 4 different values for P
max_level = 5  # from 0 to max_level (inclusive) this is the setting for the sparse quadrature

# PROBABLY DOES NOT NEED TO BE CHANGED:
spatial_dimension = 1
grid_size = trial.get_parameter("grid_size", 128)
spatial_domain = tuple(repeat(tuple([-np.pi, np.pi]), spatial_dimension))  # tuple to make it hashable
start_time = 0
stop_time = trial.get_parameter("stop_time", 0.5)
# if grid_size is bigger this needs to be smaller, especially for higher poly degrees
delta_time = trial.get_parameter("delta_time", 0.001)

# degree of freedom for the quadrature method and its parameter (like level, amount of nodes per dimension,...)

chaos_names = [get_chaos_name_by_distribution(distr) for distr in trial.variable_distributions]
random_dim = len(trial.variable_distributions)
nesting = nst.get_nesting_for_multiple_names(chaos_names)
wanted_levels = list(range(max_level + 1))
wanted_quad_points = [nesting.calculate_point_num(random_dim, level) for level in wanted_levels]
print("Levels:", wanted_levels)
print("Wanted quad points:", wanted_quad_points)


def minimum_full_tensor_param_for_count(count):
    return (int(np.ceil(count ** (1 / random_dim))),) * random_dim
quadrature_methods = {"full_tensor": list(map(minimum_full_tensor_param_for_count, wanted_quad_points[:-1])),
                      "sparse": wanted_levels}
trial_expectancy, trial_variance = None, None

plt.figure()
plt.title("Collocation durch Diskrete Projektion, $T={}$, $\\tau={}$".format(stop_time, delta_time))
plt.yscale('log')
plt.xlabel('Anzahl Quadraturpunkte')
plt.ylabel("Fehler in diskreter L2-Norm")
expectancy_already_plotted_by_method = set()  # plot expectancy only once per method as it does not depend on p
# colors red to green, then shades of grey
variance_colors = ["#FF0000", "#FF6200", "#F5C711", "#54C41B", "#00FF00", "#000000", "#AAAAAA"]
if len(P) > len(variance_colors):
    raise ValueError("Too many values for P... reduce or add more colors.")

# for each quadrature method do projections
for (method, method_params), marker in zip(quadrature_methods.items(), ["-o", "-D"]):
    print("Starting method", method, "with params", method_params)
    exp_var_results = dict()
    quad_points = dict()

    # for each sum bound for the polynomial basis use all possible quadratures
    for p in P:
        print("P=", p, "of", P)
        exp_var_results[p] = []
        quad_points[p] = []
        # for each parameter of the method do a discrete projection and save it
        for param in method_params:
            print("Current param:", param)
            result_xs, result_xs_mesh, dp_expectancy, dp_variance, points, poly_count = \
                discrete_projection(trial, p, method, param, spatial_domain, grid_size, start_time, stop_time,
                                    delta_time)
            exp_var_results[p].append((param, dp_expectancy, dp_variance))
            quad_points[p].append(points)

    # get the references (if available and not yet calculated)
    if trial_expectancy is None:
        trial_expectancy = trial.obtain_evaluated_expectancy(result_xs, result_xs_mesh, stop_time)
    if trial_variance is None:
        trial_variance = trial.obtain_evaluated_variance(result_xs, result_xs_mesh, stop_time)

    # calculate and plot errors depending on the amount of quadrature points
    for p, variance_color in zip(P, variance_colors):
        errors_exp = []
        errors_var = []
        for param, expectancy, variance in exp_var_results[p]:
            if method not in expectancy_already_plotted_by_method and trial_expectancy is not None:
                error = error_l2_relative(expectancy, trial_expectancy)
                errors_exp.append(error)
                print("Error expectancy dp", method, param, "=", error)
            if not only_expectancy and trial_variance is not None:
                error = error_l2_relative(variance, trial_variance)
                errors_var.append(error)
                print("Error variance dp", method, param, "=", error)

        x_values = quad_points[p]
        if len(errors_exp) > 0 and len(errors_exp) == len(x_values):
            expectancy_already_plotted_by_method.add(method)
            plt.plot(x_values, errors_exp, marker, color="b", label="Erwartungswert P={} ({})".format(p, method))
        if len(errors_var) > 0 and len(errors_var) == len(x_values):
            plt.plot(x_values, errors_var, marker, color=variance_color, label="Varianz P={} ({})".format(p, method))
plt.legend(loc="best")
plt.tight_layout()
plt.show()
