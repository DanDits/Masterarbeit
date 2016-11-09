from diff_equation.pseudospectral_solver import make_wave_config, init_wave_solver
from diff_equation.ode_solver import init_linhyp_solver, make_linhyp_config
import numpy as np
from itertools import cycle, islice


# Klein Gordon equation: u_tt=alpha*u_xx -beta(x)*u, alpha>0, beta(x)>0


# Strang splitting:
# from starting values u(t0,x)=g(x), u_t(t0,x)=h(x)
# solve wave equation to v(t0+dt/2,x), calculate v_t(t0+dt/2,x)
# with these as starting values solve linear hyperbolic ode with mol to w(t0+dt,x), calculate w_t(t0+dt,x)
# using these as starting values finally solve wave equation again to u(t0+dt,x)

# TODO instead of solving wave twice, is it faster and equally accurate to solve linhyp twice?
# TODO what about accumulating two consecutive 0.5 steps to one full?

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


def make_klein_gordon_lie_trotter_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    # due to the second order time derivative alpha and beta are getting multiplied by 1/2
    wave_config = make_wave_config(intervals, grid_points_list, np.sqrt(0.5 * alpha))
    linhyp_config = make_linhyp_config(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    init_wave_solver(wave_config, t0, u0, u0t)

    # due to the splitting into two operators and having a second order time derivative u_tt=...
    # there is a factor 1/2 introduced. We only need to apply it once initially
    # as it cancels out with further calls.
    wave_config.start_velocity *= 0.5
    return Splitting([wave_config, linhyp_config], [1., 1.], [init_wave_solver, init_linhyp_solver])


def make_klein_gordon_strang_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    # see lie_trotter splitting for an explanation of the 1/2 factors appearing
    wave_config = make_wave_config(intervals, grid_points_list, np.sqrt(0.5 * alpha))
    linhyp_config = make_linhyp_config(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    init_wave_solver(wave_config, t0, u0, u0t)
    wave_config.start_velocity *= 0.5
    return Splitting([wave_config, linhyp_config, wave_config], [0.5, 1., 0.5],
                     [init_wave_solver, init_linhyp_solver, init_wave_solver])
