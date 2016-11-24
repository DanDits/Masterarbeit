from functools import lru_cache
import numpy as np
import numpy.polynomial.polynomial as npoly
from scipy.special import gamma as gamma_func


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
            coeff = np.array([0.])
            for i, func in recursive_poly_functions:
                if i < 0 or i >= n:
                    raise ValueError("Can't apply on not yet calculated polynomial! i={}, n={}".format(i, n))
                coeff = npoly.polyadd(coeff, func(n, poly_coeff(n - i - 1)))
            return coeff
        else:
            raise ValueError("Illegal degree n={} for polynomial coefficients.".format(n))

    @lru_cache(maxsize=None)
    def poly(n):
        coeff = poly_coeff(n)
        return lambda x: npoly.polyval(x, coeff)

    return poly


# Helper method that just ignores the required parameter n for the polynomial recursion functions
# noinspection PyUnusedLocal
def _polymulx(n, coeff):
    return npoly.polymulx(coeff)


# TODO I dont understand why I need to scale the weights depending on the polybasis (but not on the degree)
# weights_scle is tested for laguerre to be gamma(alpha), for legendre to be 2, the rest is currently untested
def calculate_nodes_and_weights(alphas, betas, weights_scale=1):
    # The Golub-Welsch algorithm in symmetrized form
    # see https://en.wikipedia.org/wiki/Gaussian_quadrature#Computation_of_Gaussian_quadrature_rules
    # or see http://dlmf.nist.gov/3.5#vi  for calculation of nodes = zeros of polynomial
    # p_k(x)=(x-alpha_(k-1))*p_(k-1)(x)-beta_(k-1)p_(k-2)(x)
    beta_sqrt = np.sqrt(betas)
    trimat = (np.diag(alphas)
              + np.diag(beta_sqrt, 1) + np.diag(beta_sqrt, -1))
    nodes, vectors = np.linalg.eigh(trimat)
    return nodes, weights_scale * np.reshape(vectors[0, :] ** 2, (len(nodes),))

# Polynomial basis: http://dlmf.nist.gov/18.3
# Recurrence correlations: http://dlmf.nist.gov/18.9#i


# Hermite polynomials, recursion p_n(x)=x*p_(n-1)(x)-(n-1)*p_(n-2), p_0(x)=1, p_1(x)=x
# _poly_function_basis_recursive((lambda x: 1, lambda x: x),  (lambda n, x: x, lambda n, x: 1 - n)) # as example
def hermite_basis():
    return _poly_basis_recursive([np.array([1.]), np.array([0., 1.])],  # starting values
                                 [(0, _polymulx),
                                  (1, lambda n, c: c * (1. - n))])


def hermite_nodes_and_weights(degree):
    return calculate_nodes_and_weights(np.zeros(degree), np.array(range(1, degree)), np.sqrt(2 * np.pi))


def laguerre_basis(alpha):
    return _poly_basis_recursive([np.array([1.]), np.array([alpha, -1.])],
                                 [(0, lambda n, c: npoly.polyadd(c * (2 * (n - 1) + alpha) / n,
                                                                 - npoly.polymulx(c) / n)),
                                  (1, lambda n, c: -(n - 1 + alpha - 1) / n * c)])


def laguerre_nodes_and_weights(degree, alpha):
    # normalized recurrence relation: q_n=xq_(n-1) - (2(n-1) + alpha)q_(n-1) - (n-1)(n - 2 + alpha)q_(n-2)
    # to obtain the regular polynomials multiply by (-1)^n / n!
    return calculate_nodes_and_weights(2 * np.array(range(0, degree)) + alpha,
                                       np.array(range(1, degree)) * (np.array(range(1, degree)) - 1 + alpha),
                                       gamma_func(alpha))


# Legendre polynomials, interval assumed to be [-1,1], recursion p_n(x)=x(2n-1)/n * p_(n-1)(x)-(n-1)/n*p_(n-2)(x)
def legendre_basis():
    return _poly_basis_recursive([np.array([1.]), np.array([0., 1.])],  # starting values
                                 [(0, lambda n, c: (2. * n - 1) / n * npoly.polymulx(c)),
                                  (1, lambda n, c: (1. - n) / n * c)])


# http://math.stackexchange.com/questions/12160/roots-of-legendre-polynomial gives the monic version of the legendre
# polynomials: p_n(x)=x*p_(n-1)(x)-(n-1)^2/(4(n-1)^2-1)p_(n-2), to get the normal polynomial divide by
# (n!)^2 * 2^n / (2n)!
def legendre_nodes_and_weights(degree):
    nm1 = np.array(range(1, degree))
    return calculate_nodes_and_weights(np.zeros(degree), nm1 ** 2 / (4 * (nm1 ** 2) - 1), 2)


def legendre_nodes_fast(degree):
    # when returned 'amount' nodes is fixed (no matter the degree),
    # this using 'amount' nodes gives stable and converging results up to degree<2*amount
    if degree <= 30:
        # Taken from http://keisan.casio.com/exec/system/1281195844 the roots of the legendre polynomial P_30
        return [-0.9968934840746495402716, -0.98366812327974720997, -0.9600218649683075122169,
                -0.9262000474292743258793,
                -0.8825605357920526815431, -0.8295657623827683974429, -0.767777432104826194918,
                -0.6978504947933157969323,
                -0.6205261829892428611405, -0.5366241481420198992642, -0.4470337695380891767806,
                -0.352704725530878113471,
                -0.2546369261678898464398, -0.1538699136085835469638, -0.051471842555317695833, 0.051471842555317695833,
                0.1538699136085835469638, 0.2546369261678898464398, 0.352704725530878113471, 0.4470337695380891767806,
                0.5366241481420198992642, 0.6205261829892428611405, 0.6978504947933157969323, 0.767777432104826194918,
                0.8295657623827683974429, 0.8825605357920526815431, 0.9262000474292743258793, 0.9600218649683075122169,
                0.98366812327974720997, 0.9968934840746495402716]
    else:
        # approximate the roots of higher degree polynomials
        # http://math.stackexchange.com/questions/12160/roots-of-legendre-polynomial by  Francesco Tricomi
        # http://naturalunits.blogspot.de/2013/10/zeros-of-legendre-polynomials.html
        n = degree
        k = np.array(range(1, n + 1))
        sigma = np.pi * (n - k + 3 / 4) / (n + 1 / 2)
        return (1 - 1 / (8 * n ** 2) + 1 / (8 * n ** 3) - (39 - 28 / (np.sin(sigma) ** 2)) / (384 * n ** 4)) * np.cos(
            sigma)  # O(n^(-5))
        # return (1-1/(8*n*n)+1/(8*n*n*n))*np.cos(np.pi*(4*k-1)/(4*n+2)) # O(n^(-4))


# jacobi polynomial basis (belongs to beta distribution on (-1,1))
def jacobi_basis(alpha, beta):
    def get_factor(n):
        return (2 * n + alpha + beta - 1) * (2 * n + alpha + beta) / (2 * n * (n + alpha + beta))
    # this doesn't even look nice on paper
    return _poly_basis_recursive([np.array([1.]),
                                  np.array([0.5 * (alpha - beta), 0.5 * (alpha + beta + 2)])],  # starting values
                                 [(0, lambda n, c: npoly.polyadd(npoly.polymulx(c) * get_factor(n),
                                                                 -c * get_factor(n) * (beta ** 2 - alpha ** 2) / ((2 * n + alpha + beta - 2) * (2 * n + alpha + beta)))),
                                  (1, lambda n, c: (-2 * get_factor(n) * (n + alpha - 1) * (n + beta - 1) / ((2 * n + alpha + beta - 2) * (2 * n + alpha + beta - 1))))])


def jacobi_nodes_and_weights(degree, alpha, beta):
    # TODO jacobi basis, nodes, weights and beta distribution all untested!
    temp = np.array(range(1, degree + 1))
    return calculate_nodes_and_weights((beta ** 2 - alpha ** 2) / ((2 * np.array(range(degree)) + alpha + beta)
                                                                   * (2 * np.array(range(degree)) + 2 + alpha + beta)),
                                       4 * temp * (temp + alpha) * (temp + beta) * (temp + alpha + beta)
                                       / ((2 * temp + alpha + beta - 1) * ((2 * temp + alpha + beta) ** 2)
                                          * (2 * temp + alpha + beta + 1)),
                                       1)
