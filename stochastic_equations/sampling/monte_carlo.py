from diff_equation.splitting import Splitting
import diff_equation.klein_gordon as kg
from util.analysis import error_l2
from collections import deque
import numpy as np
from util.quasi_randomness.halton import halton
from util.quasi_randomness.sobol_lib import i4_sobol


def simulate(stochastic_trial, simulations_count, keep_solutions_at_steps,
             domain, grid_size_N, start_time, stop_time, delta_time, eval_time=None, heartbeat=100,
             do_calculate_expectancy=True, order_factor=2,
             quasi_monte_carlo=False, wave_weight=0.5):
    """
    Monte Carlo simulation for the stochastic Klein-Gordon equation using a splitting method to solve the equation
    for randomly generated values of the trial's distributions. Calculates the expectancy by calculating the mean
    value of the simulations. Takes quite some time for high simulations_count. Convergence is of
    order 1/sqrt(simulations_count), though the random character makes this not perfectly visible. Offers
    an estimation of the convergence order (should be 0.5), but as the convergence might switch from coming from above
    or from below, this is mostly not super accurate if no reference solution is available for the trial.
    If a setting of randomized values fails for some reason (result blows up, NaN or Inf appears), the simulation is
    repeated, but in total only the double amount of simulations will be done.
    :param stochastic_trial: The stochastic trial to get starting values and data from.
    :param simulations_count: (int) The amount (>0) of simulations to use. 1000 is ok and fast, 100,000 offers ca.
    error in range of O(10^-3) to O(10^-4) for most examples.
    :param keep_solutions_at_steps: A list of integers of solutions to keep (if intermediate estimates of expectancy
    are wanted).
    :param domain: The domain to solve the Klein-Gordon equation on. List of intervals.
    :param grid_size_N: The grid size(s) to discretize each spatial dimension into. Should be power of 2.
    For spatial dimension 1, this can be higher (like 512), for higher dimensions should be low. The higher this is,
    the lower lower the delta_time parameter should be (CFL-condition).
    :param start_time: The starting time to use. Does not really matter, mostly 0.
    :param stop_time: The stop time. Will stop when the splitting time is bigger than or equal to this time for the
    first time, so after (stop_time-start_time) / delta_time steps.
    :param delta_time: The delta time increment to use for splitting.
    :param eval_time: The time to evaluate the solution at. So the expectancy is calculated at this time only. If None,
    then stop_time is used.
    :param heartbeat: (int) if positive then every heartbeat steps a small print will be made to console.
    :param do_calculate_expectancy: If there is no expectancy set for the trial but it has a reference set, uses this
    reference to calculate the exact expectancy and later returns it. Can take some time but useful for error analysis.
    :param order_factor: Factor between the steps of the simulations used to approximate the order. Can be 2, default is
    10. Should not be changed though.
    :return: xs: A list of 1d nd-arrays of the spatial discretization
             xs_mesh: A list of the sparse mesh grids belong to xs.
             expectancy: The trial's evaluated expectancy, the calculated expectancy or None
             errors: 1d nd-array how to error evolved over the simulations
             solutions: The solutions that we should save which are approximations to the expectancy
             solutions_for_order_estimate: The 3 solutions for the order estimate (if possible)
    """
    xs, xs_mesh = None, None
    eval_solution_index = None
    expectancy = None
    if eval_time is None:
        eval_time = stop_time
    summed_solutions = None
    solutions = []
    errors = []
    order_factor = order_factor
    last_solutions_count_max = 50
    last_solutions = deque()
    last_solution_weights = [1 / (i + 1) for i in range(last_solutions_count_max)]  # not normalized
    steps_for_order_estimate = [int(simulations_count / (order_factor ** i)) for i in range(3)]
    solutions_for_order_estimate = []
    random_dimension = len(stochastic_trial.variable_distributions)
    actual_solutions_count = 0
    fail_count = 0

    # HINT: To estimate sample variance, you would need to sum the squares of the simulation solution minus the
    # estimated expectancy and then divide this by either: simulations_count, or simulations_count-1 (to eliminate bias)
    # see: https://en.wikipedia.org/wiki/Variance#Population_variance_and_sample_variance
    while actual_solutions_count < simulations_count:
        if heartbeat > 0 and actual_solutions_count % heartbeat == 0:
            print("Simulation",
                  actual_solutions_count)
        if not quasi_monte_carlo:
            stochastic_trial.randomize()
        else:
            # do not use the sequence's first element as this all zeros
            if random_dimension <= 6:  # by recommendation (of whom again...?)
                quasi_uniform = halton(actual_solutions_count + 1, random_dimension)
            else:
                quasi_uniform, _ = i4_sobol(random_dimension, actual_solutions_count + 1)
            quasi_random = [distribution.inverse_distribution(value) for value, distribution
                            in zip(quasi_uniform, stochastic_trial.variable_distributions)]
            stochastic_trial.set_random_values(quasi_random)

        # If the time step size is too small leapfrog is unstable and we should use wave_linhyp configs
        # but they are slower about a factor 3-4
        # configs = kg.make_klein_gordon_wave_linhyp_configs(domain, [grid_size_N], stochastic_trial.alpha,
        #                                                    stochastic_trial.beta, wave_weight)
        configs = kg.make_klein_gordon_leapfrog_configs(domain, [grid_size_N], stochastic_trial.alpha,
                                                        stochastic_trial.beta)
        splitting = Splitting.make_fast_strang(*configs, "FastStrang", start_time, stochastic_trial.start_position,
                                               stochastic_trial.start_velocity, delta_time)
        splitting.progress(stop_time, delta_time, 0)
        if xs is None:
            xs = splitting.get_xs()
            xs_mesh = splitting.get_xs_mesh()
            eval_solution_index = next((i for i, t in enumerate(splitting.times()) if t >= eval_time),
                                       -1)

        solution = splitting.solutions()[eval_solution_index]
        test_summed = np.sum(solution)
        if np.isnan(test_summed) or np.isinf(test_summed) or np.abs(test_summed) > 1E5:
            fail_count += 1
            print("Simulation", actual_solutions_count,
                  "got a invalid result, NaN or Inf or too big:", test_summed, "Skipping",
                  "Random parameters:", stochastic_trial.rvalues, stochastic_trial.name)
            if not quasi_monte_carlo and fail_count < simulations_count:
                continue
            else:
                break  # to make simulation stop at maximum of twice the runtime or if it is not (pseudo)random
        actual_solutions_count += 1
        if summed_solutions is not None:
            summed_solutions += solution
        else:
            summed_solutions = solution
        if expectancy is None and stochastic_trial.has_parameter("expectancy"):
            expectancy = stochastic_trial.expectancy(xs_mesh, eval_time)
        elif expectancy is None and stochastic_trial.raw_reference is not None and do_calculate_expectancy:
            expectancy = stochastic_trial.calculate_expectancy(xs, eval_time,
                                                               stochastic_trial.raw_reference)
        last_solution = summed_solutions / actual_solutions_count
        if len(last_solutions) >= last_solutions_count_max:
            last_solutions.popleft()
        last_solutions.append(last_solution)

        if actual_solutions_count in keep_solutions_at_steps:
            solutions.append((actual_solutions_count, last_solution))
        if actual_solutions_count in steps_for_order_estimate:
            # do not only use exactly the solution for step i but use weighted arithmetic mean of some beforehand
            count = min(len(last_solutions), len(last_solution_weights))
            averaged = (1 / sum(last_solution_weights[:count]) * sum(sol * weight for sol, weight
                                                                     in zip(last_solutions, last_solution_weights)))
            solutions_for_order_estimate.append(averaged)
        if expectancy is not None:
            errors.append(error_l2(expectancy, last_solution))
    return xs, xs_mesh, expectancy, errors, solutions, solutions_for_order_estimate
