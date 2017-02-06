from functools import lru_cache
from diff_equation.splitting import Splitting
import diff_equation.klein_gordon as kg
import numpy as np


def check_distribution_assertions(distr):
    """
    The assertions made are necessary as the polynomials are only an orthogonal basis for special
    distributions. This is not necessarily a weakness, as the distributions aka. the random variables can
    be transformed arbitrarily.
    :param distr: The distribution to check assertion for by using its name attribute for identification.
    :return: None
    """
    distr_assertions = {"Uniform": distr.parameters == (-1, 1),
                        "Gaussian": distr.parameters == (0, 1),
                        "Gamma": distr.parameters[1] == 1,
                        "Beta": distr.support == (-1, 1)}
    assert distr_assertions[distr.name]


@lru_cache(maxsize=20000)
def cached_collocation_point(spatial_domain, grid_size, trial, wave_weight, start_time, stop_time, delta_time,
                             nodes, real_only=False, flatten=False, ensure_is_finite=False):
    trial.set_random_values(nodes)
    configs = kg.make_klein_gordon_wave_linhyp_configs(spatial_domain, [grid_size], trial.alpha,
                                                       trial.beta, wave_weight)
    splitting = Splitting.make_fast_strang(*configs, "FastStrang",
                                           start_time, trial.start_position, trial.start_velocity, delta_time)
    splitting.progress(stop_time, delta_time, 0)
    last_solution = splitting.solutions()[-1]
    if real_only:
        last_solution = last_solution.real
    if ensure_is_finite and not np.all(np.isfinite(last_solution)):
        last_solution = np.nan_to_num(last_solution)
    if not flatten:
        return last_solution, splitting
    else:
        actual_shape = last_solution.shape
        return last_solution.flatten(), splitting, actual_shape
