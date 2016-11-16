from diff_equation.pseudospectral_solver import WaveSolverConfig, KleinGordonMomentConfig, VelocityConfig
from diff_equation.ode_solver import LinhypSolverConfig
import numpy as np
from itertools import cycle, islice


# Klein Gordon equation: u_tt=alpha()*u_xx -beta(x)*u, alpha()>0, beta(x)>0


# Strang splitting:
# from starting values u(t0,x)=g(x), u_t(t0,x)=h(x)
# solve wave equation to v(t0+dt/2,x), calculate v_t(t0+dt/2,x)
# with these as starting values solve linear hyperbolic ode with mol to w(t0+dt,x), calculate w_t(t0+dt,x)
# using these as starting values finally solve wave equation again to u(t0+dt,x)


class Splitting:
    def __init__(self, configs, step_fractions, name, on_end_callback=None):
        self.solver_configs = configs
        self.name = name
        self.solver_step_fractions = step_fractions
        self.timed_positions = []
        self.on_end_callback = on_end_callback
        assert len(step_fractions) == len(configs)

    @classmethod
    def sub_step(cls, config, time, time_step_size, step_fraction):
        # TODO can we implicitly make a leapfrog step for every splitting? Does this improve performance?
        config.solve([time + 1 * time_step_size * step_fraction])
        positions = config.solutions()
        next_position = positions[0]
        velocities = config.velocities()
        next_velocity = velocities[0]
        return next_position, next_velocity

    def progress(self, end_time, time_step_size, save_solution_step=1):
        # zeroth config is assumed to be properly initialized with starting values and solver
        assert len(self.solver_configs[0].timed_solutions()) == 0  # without any solutions yet
        save_solution_counter = save_solution_step
        for counter, step_fraction, config, next_config \
                in zip(cycle(range(len(self.solver_configs))),
                       cycle(self.solver_step_fractions),
                       cycle(self.solver_configs),
                       islice(cycle(self.solver_configs), 1, None)):
            time = config.start_time
            next_position, next_velocity = self.sub_step(config, time, time_step_size, step_fraction)

            splitting_step_completed = counter == len(self.solver_configs) - 1
            if splitting_step_completed:
                time += time_step_size  # when one splitting step is complete, progress time (for book keeping)
                save_solution_counter -= 1
                if save_solution_counter <= 0:
                    save_solution_counter = save_solution_step
                    self.timed_positions.append((time, next_position))
            next_config.init_solver(time, next_position, next_velocity)
            if splitting_step_completed and time >= end_time:
                if self.on_end_callback:
                    self.on_end_callback()
                break

    def get_xs(self):
        return self.solver_configs[0].xs

    def get_xs_mesh(self):
        return self.solver_configs[0].xs_mesh

    def solutions(self):
        return [solution for _, solution in self.timed_positions]

    def timed_solutions(self):
        return self.timed_positions

    def times(self):
        return [time for time, _ in self.timed_positions]


def make_klein_gordon_lie_trotter_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    # due to the second order time derivative alpha and beta are getting multiplied by 1/2
    wave_config = WaveSolverConfig(intervals, grid_points_list, np.sqrt(0.5 * alpha()))
    linhyp_config = LinhypSolverConfig(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    wave_config.init_solver(t0, u0, u0t)

    # due to the splitting into two operators and having a second order time derivative u_tt=...
    # there is a factor 1/2 introduced. We only need to apply it once initially
    # as it cancels out with further calls.
    wave_config.start_velocity *= 0.5
    return Splitting([wave_config, linhyp_config], [1., 1.], name="Lie")


def make_klein_gordon_leapfrog_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    moment_config = KleinGordonMomentConfig(intervals, grid_points_list, alpha, beta)
    velocity_config = VelocityConfig(intervals, grid_points_list)

    velocity_config.init_solver(t0, u0, u0t)
    return Splitting([velocity_config, moment_config, velocity_config], [0.5, 1., 0.5],
                             name="Leapfrog")


def make_klein_gordon_leapfrog_reversed_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    moment_config = KleinGordonMomentConfig(intervals, grid_points_list, alpha, beta)
    velocity_config = VelocityConfig(intervals, grid_points_list)

    moment_config.init_solver(t0, u0, u0t)
    return Splitting([moment_config, velocity_config, moment_config], [0.5, 1., 0.5],
                             name="RLeapfrog")


def make_klein_gordon_lie_trotter_reversed_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    wave_config = WaveSolverConfig(intervals, grid_points_list, np.sqrt(0.5 * alpha()))
    linhyp_config = LinhypSolverConfig(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    linhyp_config.init_solver(t0, u0, u0t)
    linhyp_config.start_velocity *= 0.5

    return Splitting([linhyp_config, wave_config], [1., 1.], name="RLie")


def make_klein_gordon_strang_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    # see lie_trotter splitting for an explanation of the 1/2 factors appearing
    wave_config = WaveSolverConfig(intervals, grid_points_list, np.sqrt(0.5 * alpha()))
    linhyp_config = LinhypSolverConfig(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    wave_config.init_solver(t0, u0, u0t)
    wave_config.start_velocity *= 0.5
    return Splitting([wave_config, linhyp_config, wave_config], [0.5, 1., 0.5], name="Strang")


def make_klein_gordon_strang_reversed_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta):
    wave_config = WaveSolverConfig(intervals, grid_points_list, np.sqrt(0.5 * alpha()))
    linhyp_config = LinhypSolverConfig(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    linhyp_config.init_solver(t0, u0, u0t)
    linhyp_config.start_velocity *= 0.5

    return Splitting([linhyp_config, wave_config, linhyp_config], [0.5, 1., 0.5], name="RStrang")


# this is as fast as lie splitting and (theoretically) equivalent to strang, but has the drawback that
# getting intermediate results would require additional computation and the error is somewhere between lie and strang
def make_klein_gordon_fast_strang_splitting(intervals, grid_points_list, t0, u0, u0t, alpha, beta, time_step_size):
    # instead of doing: (wave/2 -> linhyp -> wave/2) -> (wave/2 -> linhyp -> wave/2) -> ...
    # we do: wave/2 -> linhyp -> (wave -> linhyp) -> (wave -> linhyp) -> ... -> wave/2
    wave_config = WaveSolverConfig(intervals, grid_points_list, np.sqrt(0.5 * alpha()))
    linhyp_config = LinhypSolverConfig(intervals, grid_points_list, lambda *params: 0.5 * beta(*params))

    wave_config.init_solver(t0, u0, u0t)
    wave_config.start_velocity *= 0.5

    next_position, next_velocity = Splitting.sub_step(wave_config, wave_config.start_time,
                                                      time_step_size, 0.5)
    linhyp_config.init_solver(wave_config.start_time, next_position, next_velocity)
    next_position, next_velocity = Splitting.sub_step(linhyp_config, wave_config.start_time,
                                                      time_step_size, 1.)
    # we advance time here, but keep in mind that intermediate results for this splitting are not useful as it
    # would require one additional half step of wave, so at the end discard all but the last
    wave_config.init_solver(t0 + time_step_size, next_position, next_velocity)

    def on_progress_end():
        # do one more half step for wave, discard intermediate solutions
        last_position, _ = Splitting.sub_step(wave_config, wave_config.start_time, time_step_size, 0.5)
        splitting.timed_positions = [(wave_config.start_time, last_position)]

    splitting = Splitting([wave_config, linhyp_config], [1., 1.], name="FStrang", on_end_callback=on_progress_end)
    return splitting
