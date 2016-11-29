from util.analysis import error_l2
from itertools import islice


class Trial:
    """ This class is merely a structure that holds a bundle of functions and parameters that belong together.
        This allows fast changes to a test and demo case and optionally evaluation of a reference solution and the error
        to an approximation. By default the error_function is the discrete normalized l2 error.
    """
    def __init__(self, start_position, start_velocity, reference=None, name=None):
        """
        Creates a new Trial. Takes a start position and a start velocity which take a list of sparse mesh grid
        x coordinates and return nd-arrays of the same shape as the sum of the x coordinates.
        These will be given to the differential equation solver as starting values.
        :param start_position: The starting position of the solution of the differential equation.
        :param start_velocity: The starting velocity, that is the time derivative of the solution at start time.
        :param reference: (Optional) Reference solution taking an additional time parameter returning an nd-array.
        :param name: (Optional) Name of this Trial for referencing.
        """
        self.start_position = start_position
        self.start_velocity = start_velocity
        self.reference = reference
        self.error_function = error_l2
        self.name = name

    def add_parameters(self, *key_values):
        """
        Add the key value pairs as attributes to this instance. This format requires the amount of parameters to be
        a multiple of 2, the first one being the immutable key name of the pair, the second being the value.
        The parameters can be accessed as attributes of this instance. To check if there is a parameter use
        the has_parameter(key) method.
        :param key_values: (key1, value1, key2, value2,...)
        :return: The trial itself for easier chaining.
        """
        for key, value in zip(islice(key_values, 0, None, 2), islice(key_values, 1, None, 2)):
            self.__setattr__(key, value)
        return self

    def has_parameter(self, key):
        """
        Checks if this Trial has the given parameter identified by key.
        :param key: The key to check.
        :return: True if the parameter can be accessed as an attribute.
        """
        return hasattr(self, key)

    def error(self, xs, t, approximation):
        """
        Calculates the error of the given approximation (nd-array of same shape as sum of xs) to the
        Trial's reference evaluated at xs and t. If there is no reference returns None. Set
        error_function (default discrete l2-error) to desired error function.
        :param xs: A list of sparse mesh grid coordinates used to evaluate the reference if needed.
        :param t: The time used to evaluate the reference.
        :param approximation: (nd-array) The approximation to use.
        :return: The error >= 0. or None if no reference is set.
        """
        if self.reference is None:
            return
        ref = self.reference(xs, t) if callable(self.reference) else self.reference
        return self.error_function(approximation, ref)
