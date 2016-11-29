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
