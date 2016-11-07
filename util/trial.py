from util.analysis import error_l2


class Trial:

    def __init__(self, start_position, start_velocity, reference=None):
        self.start_position = start_position
        self.start_velocity = start_velocity
        self.reference = reference
        self.error_function = error_l2
        self.params = []
        self.config = {}

    def param(self, index):
        return self.params[index]

    def set_config(self, key, value):
        self.config[key] = value
        return self

    def error(self, xs, t, approximation):
        if self.reference is None:
            return
        ref = self.reference(xs, t) if callable(self.reference) else self.reference
        return self.error_function(approximation, ref)

