from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from util.analysis import error_l2
from collections import deque
import numpy as np

def simulate(stochastic_trial, simulations_count, keep_solutions_at_steps,
             domain, grid_size_N, start_time, stop_time, delta_time, eval_time=None, heartbeat=100,
             do_calculate_expectancy=True, order_factor=2):
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

    actual_solutions_count = 0
    for i in range(1, simulations_count + 1):
        if heartbeat > 0 and i % heartbeat == 0:
            print("Simulation",
                  i)  # about 7s for 100 simulations with leapfrog (1min with lie trotter), 512, 0.001, [0,.5]
        stochastic_trial.randomize()

        splitting = make_klein_gordon_leapfrog_splitting(domain, [grid_size_N], start_time,
                                                         stochastic_trial.start_position,
                                                         stochastic_trial.start_velocity, stochastic_trial.alpha,
                                                         stochastic_trial.beta)
        splitting.progress(stop_time, delta_time)
        if xs is None:
            xs = splitting.get_xs()
            xs_mesh = splitting.get_xs_mesh()
            eval_solution_index = next((i for i, t in enumerate(splitting.times()) if t >= eval_time),
                                       -1)

        solution = splitting.solutions()[eval_solution_index]
        test_summed = np.sum(solution)
        if np.isnan(test_summed) or np.isinf(test_summed) or np.abs(test_summed) > 1E10:
            print("Simulation", i, "got a invalid result, NaN or Inf or too big:", test_summed, "Skipping",
                  "Random parameters:", stochastic_trial.rvalues, stochastic_trial.name)
            continue
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
