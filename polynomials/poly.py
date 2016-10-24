import math


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


# Gaussian - additional attributes
distributions["Gaussian"][attribute_weight] = lambda x: math.exp(-x*x/2.) / math.sqrt(2.*math.pi)
distributions["Gaussian"][attribute_polybasis_normalization_gamma] = lambda n: math.factorial(n)


def hermite_poly(n):
    if n == 0:
        return lambda z: 1
    elif n == 1:
        return lambda z: z
    elif n == 2:
        return lambda z: z*z-1
    elif n == 3:
        return lambda z: z*z*z - 3*z
    elif n >= 2:  # General case (would also apply for n==2 and n==3)
        prev = hermite_poly(n-1)
        pre_prev = hermite_poly(n-2)
        return lambda z: z*prev(z) - (n-1)*pre_prev(z)
    else:
        raise ValueError("Illegal degree " + n + " for polynomial basis function.")

distributions["Gaussian"][attribute_polybasis] = hermite_poly


# Uniform - additional attributes, interval assumed to be [-1,1]
distributions["Uniform"][attribute_weight] = lambda x: 0.5
distributions["Uniform"][attribute_polybasis_normalization_gamma] = lambda n: 2/(2*n+1)


def legendre_poly(n):
    if n == 0:
        return lambda z: 1
    elif n == 1:
        return lambda z: z
    elif n == 2:
        return lambda z: 0.5*(3*z*z-1)
    elif n == 3:
        return lambda z: 0.5*(5*z*z*z-3*z)
    elif n >= 2:  # General case (would also apply for n==2 and n==3)
        prev = hermite_poly(n - 1)
        pre_prev = hermite_poly(n - 2)
        return lambda z: ((2.*n-1)*z*prev(z) - (n-1)*pre_prev(z)) / n
    else:
        raise ValueError("Illegal degree " + n + " for polynomial basis function.")

distributions["Gaussian"][attribute_polybasis] = legendre_poly
