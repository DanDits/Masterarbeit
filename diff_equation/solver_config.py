from itertools import zip_longest
from math import isinf, pi
import numpy as np


class SolverConfig:

    def __init__(self, intervals, grid_points_list, pseudospectral_power=None):
        self.last_delta_time = 0
        self.param = {}
        self.intervals, self.grid_points_list = intervals, grid_points_list
        self.xs, self.xs_mesh = SolverConfig.make_spatial_discretization(intervals, grid_points_list)
        self.pseudospectral_factors_power = None
        self.pseudospectral_factors_mesh = None
        if pseudospectral_power:
            self.init_pseudospectral_factors(pseudospectral_power)
        self.start_time = None
        self.solver = None
        self.start_position, self.start_velocity = None, None
        self.timed_solved = []

    def is_new_delta_time(self, delta_time):
        return abs(self.last_delta_time - delta_time) > 1E-15

    def init_pseudospectral_factors(self, power):
        factors = []
        for interval, grid_points in zip_longest(self.intervals, self.grid_points_list,
                                                 fillvalue=self.grid_points_list[-1]):
            scale = 2 * pi / (interval[1] - interval[0])
            # the ordering of this numpy array is defined by the ordering of python's fft's result
            # (see its documentation)
            factor = (1j * scale * np.append(np.arange(0, grid_points / 2 + 1),
                                             np.arange(- grid_points / 2 + 1, 0))) ** power
            factors.append(factor)
        self.pseudospectral_factors_power = power
        self.pseudospectral_factors_mesh = np.meshgrid(*factors, sparse=True)
        return self.pseudospectral_factors_mesh

    def init_initial_values(self, start_time, start_position, start_velocity):
        self.timed_solved = []
        self.start_time = start_time
        # we accept starting values to be callable functions
        # and evaluate them ourselves, making a promise to only keep them evaluated for easier arithmetic manipulation
        self.start_position = start_position
        self.start_velocity = start_velocity
        if callable(start_position):
            self.start_position = start_position(self.xs_mesh)
        if callable(start_velocity):
            self.start_velocity = start_velocity(self.xs_mesh)

    @staticmethod
    def make_spatial_discretization(intervals, grid_points_list):
        assert len(intervals) >= len(grid_points_list)
        xs = []
        for interval, grid_points in zip_longest(intervals, grid_points_list, fillvalue=grid_points_list[-1]):
            if interval[1] <= interval[0] or isinf(interval[0]) or isinf(interval[1]):
                raise ValueError("Left bound {} needs to be smaller than right bound {} and both be finite."
                                 .format(interval[0], interval[1]))
            # important not to include endpoint, else solution will be slightly incorrect because
            # the point -pi is equivalent to pi and appears twice, which will make the fourier coefficients wrong
            x = np.linspace(interval[0], interval[1], endpoint=False, num=grid_points)
            xs.append(x)
        return xs, np.meshgrid(*xs, sparse=True)

    def solve(self, wanted_times):
        if self.solver is None:
            raise ValueError("Solver not yet initialized!")
        for time in wanted_times:
            self.timed_solved.append((time, self.solver(time)))

    def timed_solutions(self):
        return [(time, solution[0]) for time, solution in self.timed_solved]

    def times(self):
        return [time for time, _ in self.timed_solved]

    def solutions(self):
        return [solution[0] for _, solution in self.timed_solved]

    def velocities(self):
        return [solution[1] for _, solution in self.timed_solved]

    def pop_last_solved(self):
        return self.timed_solved.pop()
