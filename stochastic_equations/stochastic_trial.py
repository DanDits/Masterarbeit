from util.trial import Trial
from functools import partial


class StochasticTrial(Trial):
    """
    Trial extension with the main purpose to generate randomized trials for monte carlo simulation.
    The attributes "start_position", "start_velocity", "reference", "alpha", "beta" are (if existent)
    functions that take an additional list of random parameters (named 'ys'). Getting these attributes from this trial
    will assign them the currently randomized values.
    To generate a new trial according to the given distributions, use the randomize method after all required attributes
    have been referenced.
    """
    def __init__(self, variable_distributions, start_position, start_velocity, reference=None):
        super().__init__(start_position, start_velocity, reference)
        self.variable_distributions = variable_distributions
        self.rvars = None
        self.randomize()

    def randomize(self):
        """
        Re randomizes the list of random variables by generating new values of the given distributions.
        Make sure to obtain a reference of every required attribute that uses this randomization before randomizing
        again
        :return: None
        """
        self.rvars = [distr.generate() for distr in self.variable_distributions]

    def __getattribute__(self, item):
        if item in ["start_position", "start_velocity", "reference", "alpha", "beta"]:
            return partial(super().__getattribute__(item), ys=self.rvars)
        return super().__getattribute__(item)
