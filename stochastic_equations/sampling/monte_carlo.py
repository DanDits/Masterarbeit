from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from util.analysis import error_l2


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
    steps_for_order_estimate = [int(simulations_count / (order_factor ** i)) for i in range(3)]
    solutions_for_order_estimate = []

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
        if summed_solutions is not None:
            summed_solutions += solution
        else:
            summed_solutions = solution
        if expectancy is None and stochastic_trial.has_parameter("expectancy"):
            expectancy = stochastic_trial.expectancy(xs_mesh, eval_time)
        elif expectancy is None and stochastic_trial.raw_reference is not None and do_calculate_expectancy:
            expectancy = stochastic_trial.calculate_expectancy(xs, eval_time,
                                                               stochastic_trial.raw_reference)
        if i in keep_solutions_at_steps:
            solutions.append((i, summed_solutions / i))
        if i in steps_for_order_estimate:
            solutions_for_order_estimate.append(summed_solutions / i)
        if expectancy is not None:
            errors.append(error_l2(expectancy, summed_solutions / i))
    return xs, xs_mesh, expectancy, errors, solutions, solutions_for_order_estimate
