import numpy as np
from util.trial import Trial


param_g0 = 2  # some parameter greater than one
alpha = 0.5  # smaller than param_g2 ** 2 / dimension to ensure beta>0
trial = Trial(lambda xs: np.zeros(shape=sum(xs).shape),
                lambda xs: param_g0 * np.sin(sum(xs)),
                lambda xs, t: np.sin(sum(xs)) * np.sin(param_g0 * t)) \
    .add_parameters("beta", lambda xs: param_g0 ** 2 - len(xs) * alpha,
                    "alpha", alpha)
# TODO maybe Trial subclass to generate functions that depend on variables and those can be given a distribution?
# TODO how to enforce constraints for alpha, beta>0 ? implicitly? discard if invalid?
# TODO do we find an example where start pos and vel do not depend on random variable (like in the thesis description)
