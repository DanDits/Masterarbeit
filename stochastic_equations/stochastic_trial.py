from util.trial import Trial
from functools import partial
from scipy.integrate import nquad
import operator
from functools import reduce
from itertools import product
import numpy as np


def mul_prod(factors):
    return reduce(operator.mul, factors, 1)


class StochasticTrial(Trial):
    """
    Trial extension with the main purpose to generate randomized trials for monte carlo simulation.
    The attributes "start_position", "start_velocity", "reference", "alpha", "beta" are (if existent)
    functions that take an additional list of random parameters (named 'ys'). Getting these attributes from this trial
    will assign them the currently randomized values.
    To generate a new trial according to the given distributions, use the randomize method after all required attributes
    have been referenced.
    """

    def __init__(self, variable_distributions, start_position, start_velocity, reference=None,
                 random_variables=None):
        super().__init__(start_position, start_velocity, reference)
        self.variable_distributions = variable_distributions
        self.rvalues = None

        def identity(*p):
            if len(p) == 1:
                return p[0]
            return p
        random_variables = [] if random_variables is None else random_variables
        self.rvars = ([(identity if variable is None else variable)
                       for variable, _ in zip(random_variables, variable_distributions)]  # trim to shorter
                      + [identity] * (len(variable_distributions) - len(random_variables)))  # make sure lengths fit
        assert len(self.rvars) == len(self.variable_distributions)
        self.randomize()

    def randomize(self):
        """
        Re randomizes the list of random variables by generating new values of the given distributions.
        Make sure to obtain a reference of every required attribute that uses this randomization before randomizing
        again
        :return: None
        """
        self.rvalues = [rvar(distr.generate()) for rvar, distr in zip(self.rvars, self.variable_distributions)]

    # TODO does not yet include a dependency on x, this probably needs to be done point wise for each x
    def calculate_expectancy(self, xs_lines, t, function):
        sizes = tuple(map(len, xs_lines))
        result = np.zeros(shape=sizes)
        for index, x_coords in zip(product(map(range, sizes)), product(*xs_lines)):

            def func_in_ys(*ys):
                transformed_ys = [rvar(value) for rvar, value in zip(self.rvars, ys)]
                return (function(x_coords, t, transformed_ys)  # TODO contains an error still
                        * mul_prod(distr.weight(y) for y, distr in zip(ys, self.variable_distributions)))

            point_value = nquad(func_in_ys, [distr.support for distr in self.variable_distributions])[0]
            result[index] = point_value
        return result

    def __getattribute__(self, item):
        if item in ["start_position", "start_velocity", "reference", "alpha", "beta"]:
            return partial(super().__getattribute__(item), ys=self.rvalues)
        return super().__getattribute__(item)
