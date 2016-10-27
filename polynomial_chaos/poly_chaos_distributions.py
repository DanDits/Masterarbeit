import math
import polynomial_chaos.poly as poly
import polynomial_chaos.distributions as distr

attribute_normalization_gamma = "normalization_gamma"


# TODO make this not a dict but a class
def _make_poly_chaos_distribution(poly_name, poly_basis, distribution_name, distribution):
    return {"poly_name": poly_name,
            "poly_basis": poly_basis,
            "distribution_name": distribution_name,
            "distribution": distribution}


chaos = [_make_poly_chaos_distribution(*params) for params in
         [("Hermite", poly.hermite_basis(), "Gaussian", distr.gaussian),
          # ("Gamma", True, "Laguerre", [0, math.inf]),
          # ("Beta", True, "Jacobi", [interval_left, interval_right]),
          ("Legendre", poly.legendre_basis(), "Uniform", distr.make_uniform(-1, 1)),
          # ("Poisson", False, "Charlier", support_Naturals),
          # ("Binomial", False, "Krawtchouk", support_NaturalsFinite),
          # ("Negative Binomial", False, "Meixner", support_Naturals),
          # ("Hypergeometric", False, "Hahn", support_NaturalsFinite)
          ]
         ]


def get_chaos_by_poly(poly_name):
    for curr in chaos:
        if curr["poly_name"] == poly_name:
            return curr
    raise ValueError("No polynomial chaos distribution for polynomial name", poly_name)


def get_chaos_by_distribution(distribution_name):
    for curr in chaos:
        if curr["distribution_name"] == distribution_name:
            return curr
    raise ValueError("No polynomial chaos distribution for distribution name", distribution_name)


get_chaos_by_distribution("Gaussian")[attribute_normalization_gamma] = lambda n: math.factorial(n)
get_chaos_by_distribution("Uniform")[attribute_normalization_gamma] = lambda n: 2 / (2 * n + 1)
