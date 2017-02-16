import numpy as np
import demo.stochastic_trials as st
import matplotlib.pyplot as plt
import diff_equation.solver_config as config
trial = st.trial_3
stop_time = 2.

plt.figure()
xs, xs_mesh = config.SolverConfig.make_spatial_discretization([[-np.pi, np.pi]], [128])
expectancy = trial.obtain_evaluated_expectancy(xs, xs_mesh, stop_time)
exact_variance = None
if trial.has_parameter("variance"):
    exact_variance = trial.variance(xs_mesh, stop_time)
calculated_variance = trial.calculate_variance(xs, stop_time, trial.raw_reference, expectancy)

plt.plot(xs[0], trial.raw_reference(xs_mesh, stop_time, [0]), label="Reference at expected y")
plt.plot(xs[0], expectancy, label="obtained Expectancy")
if exact_variance is not None:
    plt.plot(xs[0], exact_variance, label="exact Variance")
plt.plot(xs[0], calculated_variance, label="calculated Variance")
if exact_variance is not None:
    from util.analysis import error_l2
    print("Error in variances:", error_l2(calculated_variance, exact_variance))
plt.legend()
plt.show()
