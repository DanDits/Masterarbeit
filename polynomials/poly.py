import math
from functools import lru_cache


def _make_distribution(name, is_continuous, gpc_basis_polynomial_name, support):
    return {"name": name,
            "continuous": is_continuous,
            "polybasis_name": gpc_basis_polynomial_name,
            "support": support}


support_R = "R"
support_HalfR = "HalfR"
support_Interval = "Interval"
support_Naturals = "Naturals"
support_NaturalsFinite = "NaturalsFinite"

distributions = {name: _make_distribution(name, *params) for (name, *params) in
                 [("Gaussian", True, "Hermite", support_R),
                  ("Gamma", True, "Laguerre", support_HalfR),
                  ("Beta", True, "Jacobi", support_Interval),
                  ("Uniform", True, "Legendre", support_Interval),
                  ("Poisson", False, "Charlier", support_Naturals),
                  ("Binomial", False, "Krawtchouk", support_NaturalsFinite),
                  ("Negative Binomial", False, "Meixner", support_Naturals),
                  ("Hypergeometric", False, "Hahn", support_NaturalsFinite)
                  ]
                 }

attribute_weight = "weight"
attribute_polybasis = "polybasis"
attribute_polybasis_normalization_gamma = "polybasis_normalization_gamma"


# Pretty general implementation for a recursively defined polynomial basis, but not optimized as terms that
# cancel out are still calculated and may lead to inaccuracies. For higher accuracy (for low order terms) use more hand
# calculated polynomials for "polys_start"
def _poly_basis_recursive(polys_start, recursion_factors):
    """
    Factory function for a basis of polynomials p(x) for a single double variable x.
    The returned function takes a non negative integer n and returns a polynomial of degree n
    which is the n-th basis polynomial. The first basis polynomials are given as function by polys_start.
    The following polynomials are defined by the recursion factors f as follows:
    p_n(x) = f_0(n,x) * p_(n-1)(x) + f_1(n,x) * p_(n-2)(x) + ...
    :param polys_start: The m starting polynomials.
    :param recursion_factors: factors as functions from degree n and position x,
    defining the recursion for higher order polynomials.
    :return: A function that takes an order n and returns a polynomial p mapping a double to a double.
    Polynomials are cached internally but not mathematically simplified and not optimized.
    """

    @lru_cache(maxsize=None)
    def poly(n):
        if 0 <= n < len(polys_start):
            return polys_start[n]
        elif n >= len(polys_start):
            return lambda x: sum(f_a(n, x) * poly(n - 1 - i)(x) for i, f_a in enumerate(recursion_factors))
        else:
            raise ValueError("Illegal degree " + n + " for polynomial basis function.")

    return poly


# Gaussian - additional attributes, recursion p_n(x)=x*p_(n-1)(x)-(n-1)*p_(n-2), p_0(x)=1, p_1(x)=x
distributions["Gaussian"][attribute_weight] = lambda x: math.exp(-x * x / 2.) / math.sqrt(2. * math.pi)
distributions["Gaussian"][attribute_polybasis_normalization_gamma] = lambda n: math.factorial(n)
distributions["Gaussian"][attribute_polybasis] = _poly_basis_recursive((lambda x: 1, lambda x: x),  # TODO more polys
                                                                       (lambda n, x: x, lambda n, x: 1 - n))

# Uniform - additional attributes,
# interval assumed to be [-1,1], recursion p_n(x)=x(2n-1)/n * p_(n-1)(x)-(n-1)/n*p_(n-2)(x)
distributions["Uniform"][attribute_weight] = lambda x: 0.5
distributions["Uniform"][attribute_polybasis_normalization_gamma] = lambda n: 2 / (2 * n + 1)
distributions["Uniform"][attribute_polybasis] = _poly_basis_recursive((lambda x: 1, lambda x: x),  # TODO more polys
                                                                      (lambda n, x: (2. * n - 1) / n * x,
                                                                           lambda n, x: (1. - n) / n))
