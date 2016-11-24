def check_distribution_assertions(distr):
    distr_assertions = {"Uniform": distr.parameters == (-1, 1),
                        "Gaussian": distr.parameters == (0, 1),
                        "Gamma": distr.parameters[1] == 1}
    assert distr_assertions[distr.name]