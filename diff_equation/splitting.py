from itertools import cycle, islice


class Splitting:
    def __init__(self, configs, step_fractions, name, on_end_callback=None, on_start_callback=None):
        self.solver_configs = configs
        self.name = name
        self.solver_step_fractions = step_fractions
        self.timed_positions = []
        self.on_start_callback = on_start_callback
        self.first_progress_start = True
        self.on_end_callback = on_end_callback
        self.steps_already_made = 0
        assert len(step_fractions) == len(configs)

    @classmethod
    def sub_step(cls, config, time_step_size, step_fraction):
        config.solve([config.start_time + time_step_size * step_fraction])
        result_time, result = config.pop_last_solved()
        next_position = result[0]
        next_velocity = result[1]

        return next_position, next_velocity

    def get_current_time(self):
        return self.solver_configs[0].start_time

    def approx_steps_to_end_time(self, end_time, time_step_size):  # can be inaccurate due to floating errors
        return int((end_time - self.get_current_time()) / time_step_size)

    def progress(self, steps, time_step_size, save_solution_step=1):
        # zeroth config is assumed to be properly initialized with starting values and solver
        # if there are previous solutions these will be ignored
        if self.on_start_callback:
            self.on_start_callback(self)
        save_solution_counter = save_solution_step
        total_start_time = self.get_current_time()
        steps -= self.steps_already_made
        self.steps_already_made = 0
        steps_completed_count = 0
        time = total_start_time
        assert type(steps) == int

        for counter, step_fraction, config, next_config \
                in zip(cycle(range(len(self.solver_configs))),
                       cycle(self.solver_step_fractions),
                       cycle(self.solver_configs),
                       islice(cycle(self.solver_configs), 1, None)):
            next_position, next_velocity = self.sub_step(config, time_step_size, step_fraction)

            splitting_step_completed = counter == len(self.solver_configs) - 1
            if splitting_step_completed:
                steps_completed_count += 1
                # when one splitting step is complete, progress time (for book keeping)
                time = total_start_time + time_step_size * steps_completed_count
                save_solution_counter -= 1
                if save_solution_counter == 0 or steps_completed_count >= steps:
                    # either if we want to save or if its the last solution anyways: save it
                    save_solution_counter = save_solution_step
                    self.timed_positions.append((time, next_position))
            next_config.init_solver(time, next_position, next_velocity)
            if splitting_step_completed and steps_completed_count >= steps:
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

    @staticmethod
    def make_lie(config1, config2, name, t0, u0, u0t):
        config1.init_solver(t0, u0, u0t)
        return Splitting([config1, config2], [1., 1.], name=name)

    @staticmethod
    def make_strang(config1, config2, name, t0, u0, u0t):
        config1.init_solver(t0, u0, u0t)
        return Splitting([config1, config2, config1], [0.5, 1., 0.5], name=name)

    @staticmethod
    def make_fast_strang(config1, config2, name, t0, u0, u0t, time_step_size):
        # this is as fast as lie splitting and (theoretically) equivalent to strang, but has the drawback that
        # getting intermediate results would require additional computation.
        # If stable the error is exactly (up to rounding errors) equal to the strang's error

        # instead of doing: (config1/2 -> config2 -> config1/2) -> (config1/2 -> config2 -> config1/2) -> ...
        # we do: config1/2 -> config2 -> (config1 -> config2) -> (config1 -> config2) -> ... -> config1/2

        def on_progress_start(for_splitting):
            if for_splitting.first_progress_start:
                config1.init_solver(t0, u0, u0t)
                for_splitting.first_progress_start = False

            next_position, next_velocity = Splitting.sub_step(config1, time_step_size, 0.5)
            config2.init_solver(config1.start_time, next_position, next_velocity)
            next_position, next_velocity = Splitting.sub_step(config2, time_step_size, 1.)
            # we advance time here, but keep in mind that intermediate results for this splitting are not useful as it
            # would require one additional half step of wave, so at the end discard all but the last
            config1.init_solver(config1.start_time + time_step_size, next_position, next_velocity)
            for_splitting.steps_already_made = 1

        def on_progress_end():
            # do one more half step for wave, discard intermediate solutions
            last_position, last_velocity = Splitting.sub_step(config1, time_step_size, 0.5)
            config1.init_solver(config1.start_time, last_position, last_velocity)  # needed when progress restarted
            splitting.timed_positions = [(config1.start_time, last_position)]

        splitting = Splitting([config1, config2], [1., 1.], name=name, on_start_callback=on_progress_start,
                              on_end_callback=on_progress_end)
        return splitting
