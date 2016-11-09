from diff_equation.pseudospectral_solver import make_wave_config, init_wave_solver
from diff_equation.ode_solver import init_linhyp_solver, make_linhyp_config
import numpy as np
from itertools import repeat
from math import pi
import matplotlib.pyplot as plt
from util.trial import Trial
from itertools import cycle, islice


# Klein Gordon equation: u_tt=alpha*u_xx -beta(x)*u, alpha>0, beta(x)>0


# Strang splitting:
# from starting values u(t0,x)=g(x), u_t(t0,x)=h(x)
# solve wave equation to v(t0+dt/2,x), calculate v_t(t0+dt/2,x)
# with these as starting values solve linear hyperbolic ode with mol to w(t0+dt,x), calculate w_t(t0+dt,x)
# using these as starting values finally solve wave equation again to u(t0+dt,x)

# TODO how to get v_t and w_t? We want second order!? central differences (so also calculate v(t0+dt,x),...)?
# TODO instead of solving wave twice, is it faster and equally accurate to solve linhyp twice?

def get_derivative(previous_derivative, previous_value, current_value, next_value, time_step_size):
    # Use Taylor expansion of the first derivative f'(x)=f'(x-h)+h*f''(x-h)
    # and then use forward differences of second order to estimate f''(x-h)=(f(x+h)-2f(x)+f(x-h))/(h*h)
    return previous_derivative + (previous_value + next_value - 2 * current_value) / time_step_size


class Splitting:
    def __init__(self, configs, step_fractions, config_initializers):
        self.solver_configs = configs
        self.solver_step_fractions = step_fractions
        self.config_initializers = config_initializers
        self.timed_solutions = []
        assert len(step_fractions) == len(configs)

    def progress(self, end_time, time_step_size, save_solution_step=1):
        # zeroth config is assumed to be properly initialized with starting values and solver
        assert len(self.solver_configs[0].timed_solutions) == 0  # without any solutions yet
        save_solution_counter = save_solution_step
        for counter, step_fraction, config, \
            next_config, next_initializer in zip(cycle(range(len(self.solver_configs))),
                                                 cycle(self.solver_step_fractions),
                                                 cycle(self.solver_configs),
                                                 islice(cycle(self.solver_configs), 1, None),
                                                 islice(cycle(self.config_initializers), 1, None)):
            time = config.start_time
            config.solve([time + time_step_size * step_fraction, time + 2 * time_step_size * step_fraction])

            next_position = config.timed_solutions[0][1]
            next_velocity = get_derivative(config.start_velocity, config.start_position,
                                           next_position, config.timed_solutions[1][1],
                                           time_step_size * step_fraction)
            splitting_step_completed = counter == len(self.solver_configs) - 1
            if splitting_step_completed:
                time += time_step_size  # when one splitting step is complete, progress time (for book keeping)
                save_solution_counter -= 1
                if save_solution_counter <= 0:
                    save_solution_counter = save_solution_step
                    self.timed_solutions.append((time, next_position))
            next_initializer(next_config, time, next_position, next_velocity)
            if splitting_step_completed and time >= end_time:
                break

    def get_xs(self):
        return self.solver_configs[0].xs

    def get_xs_mesh(self):
        return self.solver_configs[0].xs_mesh

    def times(self):
        return [time for time, _ in self.timed_solutions]


def make_lie_trotter_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    # due to the second order time derivative alpha and beta are getting multiplied by 1/2
    wave_config = make_wave_config(intervals, grid_points_list, np.sqrt(0.5 * alpha))
    linhyp_config = make_linhyp_config(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    init_wave_solver(wave_config, t0, u0, u0t)

    # due to the splitting into two operators and having a second order time derivative u_tt=...
    # there is a factor 1/2 introduced. We only need to apply it once initially
    # as it cancels out with further calls.
    wave_config.start_velocity *= 0.5
    return Splitting([wave_config, linhyp_config], [1., 1.], [init_wave_solver, init_linhyp_solver])


def make_strang_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    # see lie_trotter splitting for an explanation of the 1/2 factors appearing
    wave_config = make_wave_config(intervals, grid_points_list, np.sqrt(0.5 * alpha))
    linhyp_config = make_linhyp_config(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    init_wave_solver(wave_config, t0, u0, u0t)
    wave_config.start_velocity *= 0.5
    return Splitting([wave_config, linhyp_config, wave_config], [0.5, 1., 0.5],
                     [init_wave_solver, init_linhyp_solver, init_wave_solver])

if __name__ == "__main__":
    dimension = 1
    grid_size_N = 512
    domain = list(repeat([-pi, pi], dimension))
    delta_time = 0.001
    save_every_x_solution = 1
    plot_solutions_count = 5
    start_time = 0.
    stop_time = 4
    show_errors = True
    show_reference = True

    param_g1 = 3  # some parameter greater than one
    trial_1 = Trial(lambda xs: np.sin(sum(xs)),
                    lambda xs: param_g1 * np.cos(sum(xs)),
                    lambda xs, t: np.sin(sum(xs) + param_g1 * t)) \
        .set_config("beta", lambda xs: param_g1 ** 2 - 1) \
        .set_config("alpha", 1)

    # still invalid example since the time derivatives 0th fourier is now only zero at the start (t=0), but only there
    offset = 15.9098530420256905490264393 / (2 * pi)  # is (normalized) integral from -pi to pi over exp(-cos(x))
    trial_2 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                    lambda xs: 2 * np.exp(-np.cos(sum(xs))) - offset,
                    lambda xs, t: np.sin(2 * t) * np.exp(-np.cos(sum(xs))) - offset * t) \
        .set_config("beta", lambda xs: 4 + np.cos(sum(xs)) + np.sin(sum(xs)) ** 2) \
        .set_config("alpha", 1)

    param_1, param_2, param_n1, param_3, alpha_g0 = 0.3, 0.5, 2, 1.2, 0.3
    assert param_n1 * alpha_g0 < param_3  # to ensure beta > 0
    trial_3 = Trial(lambda xs: param_1 * np.cos(param_n1 * sum(xs)),
                    lambda xs: param_2 * param_3 * np.cos(param_n1 * sum(xs)),
                    lambda xs, t: np.cos(param_n1 * sum(xs)) * (param_1 * np.cos(param_3 * t)
                                                                + param_2 * np.sin(param_3 * t))) \
        .set_config("beta", lambda xs: -alpha_g0 * (param_n1 ** 2) + param_3 ** 2) \
        .set_config("alpha", alpha_g0)

    param_g1 = 2  # some parameter greater than one
    trial_4 = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                    lambda xs: param_g1 * np.sin(sum(xs)),
                    lambda xs, t: np.sin(sum(xs)) * np.sin(param_g1 * t)) \
        .set_config("beta", lambda xs: param_g1 ** 2 - 1) \
        .set_config("alpha", 1)

    trial = trial_4

    plt.figure()

    lie_splitting = make_lie_trotter_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                               trial.start_velocity, trial.config["alpha"], trial.config["beta"])
    lie_splitting.progress(stop_time, delta_time, save_every_x_solution)
    strang_splitting = make_strang_splitting(domain, [grid_size_N], start_time, trial.start_position,
                                             trial.start_velocity, trial.config["alpha"], trial.config["beta"])
    strang_splitting.progress(stop_time, delta_time, save_every_x_solution)
    xs = lie_splitting.get_xs()
    xs_mesh = lie_splitting.get_xs_mesh()

    plot_counter = 0
    plot_every_x_solution = ((stop_time - start_time) / delta_time) / plot_solutions_count
    plt.plot(*xs, trial.start_position(xs_mesh), label="Start position")
    for (t, solution_lie), (_, solution_strang), color in zip(lie_splitting.timed_solutions,
                                                              strang_splitting.timed_solutions,
                                                              cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y'])):
        plot_counter += 1
        if plot_counter == plot_every_x_solution:
            plot_counter = 0
            plt.plot(*xs, solution_lie, "o", color=color, label="Lie solution at {}".format(t))
            plt.plot(*xs, solution_strang, "+", color=color, label="Strang solution at {}".format(t))
            if show_reference:
                plt.plot(*xs, trial.reference(xs_mesh, t), color=color, label="Reference at {}".format(t))
    plt.legend()
    plt.title("Splitting methods for Klein Gordon equation, dt={}".format(delta_time))
    if show_errors:
        errors_lie = [trial.error(xs_mesh, t, y) for t, y in lie_splitting.timed_solutions]
        errors_strang = [trial.error(xs_mesh, t, y) for t, y in strang_splitting.timed_solutions]
        plt.figure()
        plt.plot(lie_splitting.times(), errors_lie, label="Errors of lie in discrete L2 norm")
        plt.plot(strang_splitting.times(), errors_strang, label="Errors of strang in discrete L2 norm")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.yscale('log')
        plt.legend()

    plt.show()
