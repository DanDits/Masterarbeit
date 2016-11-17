from functools import lru_cache
from numpy import array
import numpy.polynomial.polynomial as npoly


# Pretty general implementation for a recursively defined polynomial basis in function form, so it is not optimized
# as terms that cancel out are still calculated and may lead to inaccuracies.
# For higher accuracy (for low order terms) use more hand
# calculated polynomials for "polys_start"
def _poly_function_basis_recursive(polys_start, recursion_factors):
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
            raise ValueError("Illegal degree n={} for polynomial basis function.".format(n))

    return poly


# Pretty general implementation for a recursively defined polynomial basis in numpy's polynomial coefficient form,
# which is optimized to use Horner's scheme for evaluation and can be extended more easily for multi dimensionality.
def _poly_basis_recursive(polys_start_coeff, recursive_poly_functions):
    """
    :param polys_start_coeff: List of coefficients as 1d numpy arrays for starting polynomials.
    :param recursive_poly_functions: List of tuples of an integer and a function. The integer indicates which previous
    polynomials coefficients to apply on, 0 for the previous, 1 for the one before the previous,...
    The function takes the degree n and the previous coefficients and returns new coefficients which are summed up to
    make the new polynomial's coefficients.
    :return: A function taking a degree n and returning a function which evaluates the n-th polynomial at the point x.
    """
    @lru_cache(maxsize=None)
    def poly_coeff(n):
        if 0 <= n < len(polys_start_coeff):
            return polys_start_coeff[n]
        elif n >= len(polys_start_coeff):
            coeff = array([0.])
            for i, func in recursive_poly_functions:
                if i < 0 or i >= n:
                    raise ValueError("Can't apply on not yet calculated polynomial! i={}, n={}".format(i, n))
                coeff = npoly.polyadd(coeff, func(n, poly_coeff(n - i - 1)))
            return coeff
        else:
            raise ValueError("Illegal degree n={} for polynomial coefficients.".format(n))

    @lru_cache(maxsize=None)
    def poly(n):
        print("N=", n)
        coeff = poly_coeff(n)
        return lambda x: npoly.polyval(x, coeff)
    return poly


# Helper method that just ignores the required parameter n for the polynomial recursion functions
# noinspection PyUnusedLocal
def _polymulx(n, coeff):
    return npoly.polymulx(coeff)


# Hermite polynomials, recursion p_n(x)=x*p_(n-1)(x)-(n-1)*p_(n-2), p_0(x)=1, p_1(x)=x
# _poly_function_basis_recursive((lambda x: 1, lambda x: x),  (lambda n, x: x, lambda n, x: 1 - n)) # as example
def hermite_basis():
    return _poly_basis_recursive([array([1.]), array([0., 1.])],  # starting values
                                 [(0, _polymulx),
                                  (1, lambda n, c: npoly.polymul(c, array([1. - n])))])


# Legendre polynomials, interval assumed to be [-1,1], recursion p_n(x)=x(2n-1)/n * p_(n-1)(x)-(n-1)/n*p_(n-2)(x)
def legendre_basis():
    return _poly_basis_recursive([array([1.]), array([0., 1.])],  # starting values
                                 [(0, lambda n, c: (2. * n - 1) / n * npoly.polymulx(c),
                                  (1, lambda n, c: (1. - n) / n))])
