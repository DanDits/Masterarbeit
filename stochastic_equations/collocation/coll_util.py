from functools import lru_cache
from diff_equation.splitting import Splitting
import diff_equation.klein_gordon as kg


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


@lru_cache(maxsize=15000)
def cached_collocation_point(spatial_domain, grid_size, trial, wave_weight, start_time, stop_time, delta_time,
                             nodes):
    trial.set_random_values(nodes)
    configs = kg.make_klein_gordon_wave_linhyp_configs(spatial_domain, [grid_size], trial.alpha,
                                                       trial.beta, wave_weight)
    splitting = Splitting.make_fast_strang(*configs, "FastStrang",
                                           start_time, trial.start_position, trial.start_velocity, delta_time)
    splitting.progress(stop_time, delta_time, 0)
    last_solution = splitting.solutions()[-1]
    return last_solution, splitting
