from util.trial import Trial
from functools import partial


class StochasticTrial(Trial):

    def __init__(self, variable_distributions, start_position, start_velocity, reference=None):
        super().__init__(start_position, start_velocity, reference)
        self.variable_distributions = variable_distributions
        self.rvars = None
        self.randomize()

    def randomize(self):
        self.rvars = [distr.generate() for distr in self.variable_distributions]

    def __getattribute__(self, item):
        if item in ["start_position", "start_velocity", "reference", "alpha", "beta"]:
            return partial(super().__getattribute__(item), ys=self.rvars)
        return super().__getattribute__(item)
