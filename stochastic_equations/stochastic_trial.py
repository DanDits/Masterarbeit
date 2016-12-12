from util.trial import Trial
from functools import partial
from scipy.integrate import nquad
from util.analysis import mul_prod
from itertools import product
import numpy as np


class StochasticTrial(Trial):
    """
    Trial extension with the main purpose to generate randomized trials for monte carlo simulation.
    The attributes "start_position", "start_velocity", "reference", "alpha", "beta" are (if existent)
    functions that take an additional list of random parameters (named 'ys'). Getting these attributes from this trial
    will assign them the currently randomized values.
    To generate a new trial according to the given distributions, use the randomize method after all required attributes
    have been referenced. Can be overwritten by set_random_values(values) to arbitrary values.
    """

    def __init__(self, variable_distributions, start_position, start_velocity, reference=None,
                 random_variables=None, name=None, flag_raw_attributes=False):
        """
        Creates a new Stochastic Trial. See Trial's constructor for parameters: start_position, start_velocity,
        reference, name for basic usage, remember they take an additional parameter ys.
        If you need to get the reference which is not seeded by the random values, use attribute
        raw_reference instead.
        :param variable_distributions: A list of distributions to use to generate random values. List length determines
         the dimension of the random space of this trial.
        :param start_position: See Trial.
        :param start_velocity: See Trial.
        :param reference: (Optional) See Trial
        :param random_variables: (Optional) A list of random variables. These are functions taking a single y and
        arbitrarily mapping it to another y. Useful if the distributions are expected to have some default parameters
        and do not repeat oneself all the time when using y for reference,...
        If none or not enough given will be filled by identity function.
        :param name: See Trial.
        :param flag_raw_attributes: If set to True, will NOT set the random parameters ys but always return the raw
        ones which were given to this object.
        """
        super().__init__(start_position, start_velocity, reference, name=name)
        self.raw_reference = reference
        self.variable_distributions = variable_distributions
        self.rvalues = None
        self.flag_raw_attributes = flag_raw_attributes

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

    def transform_values(self, values):
        return [rvar(value) for rvar, value in zip(self.rvars, values)]

    def set_random_values(self, values):
        """
        Allows to overwrite the randomized values by a list of given values. If these values are correct and meaningful
        for the trial's distributions is not validated. They will get transformed by the set random variables though.
        :param values: A list of values for each random value corresponding to a distribution.
        :return: The new random values.
        """
        assert len(values) == len(self.rvalues)
        self.rvalues = self.transform_values(values)
        return self.rvalues

    def randomize(self):
        """
        Re randomizes the list of random variables by generating new values of the given distributions.
        Make sure to obtain a reference of every required attribute that uses this randomization before randomizing
        again
        :return: None
        """
        self.rvalues = [rvar(distr.generate()) for rvar, distr in zip(self.rvars, self.variable_distributions)]


# TODO compare nquad with custom gauss QF, maybe instead use custom QF with >50 nodes
    def calculate_expectancy(self, xs_lines, t, function):
        """
        This very far from efficient method calculates the expectancy of a given function(x-coordinates, time, ys) at
        the every point on the grid defined by the axes "xs_lines" and the given time t.
        Basically uses calculate_expectancy_simple on every grid point.
        :param xs_lines: List of one dimensional numpy arrays containing the grid coordinates for the given dimension.
        :param t: The time to get expectancy at.
        :param function: The function as described above.
        :return: A nd-array with shape defined by the lengths of the given xs_lines.
        """
        sizes = tuple(map(len, xs_lines))
        result = np.zeros(shape=sizes)
        for index, x_coords in zip(product(*map(range, sizes)), product(*xs_lines)):
            # iterate over every coordinate, calculating the expectancy for this point by calculating the integral
            # over the variables ys weighted by the distribution's weights
            def func_in_ys(*ys):
                transformed_ys = [rvar(value) for rvar, value in zip(self.rvars, ys)]
                return (function(x_coords, t, transformed_ys)
                        * mul_prod(distr.weight(y) for y, distr in zip(ys, self.variable_distributions)))

            point_value = nquad(func_in_ys, [distr.support for distr in self.variable_distributions])[0]
            result[index] = point_value
        return result

    def calculate_expectancy_simple(self, function):
        """
        Calculates the expectancy of a function(ys) which takes a list of values, one for each distribution and returns
        a value. This does NOT use the random variables to transform the input ys!
        This uses nquad to perform the integration which is very costly and does not consider the special distribution
        weight implicitly!
        :param function: The function as described above
        :return: The expectancy, a float.
        """

        # nquad expects multiple arguments
        def function_transformed(*ys):
            return (function(ys)
                    * mul_prod(distr.weight(y) for y, distr in zip(ys, self.variable_distributions)))

        return nquad(function_transformed, [distr.support for distr in self.variable_distributions])[0]

    def __getattribute__(self, item):
        if (item != "flag_raw_attributes" and not self.flag_raw_attributes
                and item in ["start_position", "start_velocity", "reference", "alpha", "beta"]):
            return partial(super().__getattribute__(item), ys=self.rvalues)
        return super().__getattribute__(item)
