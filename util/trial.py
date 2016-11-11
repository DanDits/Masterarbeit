from util.analysis import error_l2
from itertools import islice


class Trial:
    """ This class is merely a structure that holds a bundle of functions and parameters that belong together.
        This allows fast changes to a test and demo case and optionally evaluation of a reference solution and the error
        to an approximation. By default the error_function is the discrete normalized l2 error.
    """
    def __init__(self, start_position, start_velocity, reference=None):
        self.start_position = start_position
        self.start_velocity = start_velocity
        self.reference = reference
        self.error_function = error_l2
        self.param = {}

    def add_parameters(self, *key_values):
        """
        Add the key value pairs to the param dict. This format requires the amount of parameters to be
        a multiple of 2, the first one being the immutable key of the pair, the second being the value.
        :param key_values: (key1, value1, key2, value2,...)
        :return: The trial itself for easier chaining.
        """
        for key, value in zip(islice(key_values, 0, None, 2), islice(key_values, 1, None, 2)):
            self.param[key] = value
        return self

    def has_parameter(self, key):
        return key in self.param

    def error(self, xs, t, approximation):
        if self.reference is None:
            return
        ref = self.reference(xs, t) if callable(self.reference) else self.reference
        return self.error_function(approximation, ref)

