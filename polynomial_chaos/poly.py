from functools import lru_cache
import numpy as np
import numpy.polynomial.polynomial as npoly
import math


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


def poly_by_roots(roots, leading_coefficient):
    """
    Returns the polynom that is defined by:
    p(x)=leading_coefficient * (x-r1)*(x-r2)*...*(x-r_n)
    if n roots are given. The leading factor is the factor between the monic version and the normal
    version of the polynomials.
    :param roots: The roots of the polynomial.
    :param leading_coefficient: The leading coefficient in front of the highest power.
    :return: A polynomial function which can be evaluated at an array-like parameter.
    """

    def poly(x):
        prod = leading_coefficient
        for root in roots:
            prod *= (x - root)
        return prod

    return poly


# to get the traditional weights for gauss-laguerre multiply by gamma(alpha),
# to get the traditional weights for gauss legendre multiply by 2
def calculate_nodes_and_weights(alphas, betas):
    if len(alphas) <= 0 and len(betas) <= 0:
        return [], []
    # The Golub-Welsch algorithm in symmetrized form
    # see https://en.wikipedia.org/wiki/Gaussian_quadrature#Computation_of_Gaussian_quadrature_rules
    # or see http://dlmf.nist.gov/3.5#vi  for calculation of nodes = zeros of polynomial
    # p_k(x)=(x-alpha_(k-1))*p_(k-1)(x)-beta_(k-1)p_(k-2)(x)  # the monic recurrence correlation of the polynomials
    beta_sqrt = np.sqrt(betas)
    trimat = (np.diag(alphas)
              + np.diag(beta_sqrt, 1) + np.diag(beta_sqrt, -1))
    nodes, vectors = np.linalg.eigh(trimat)
    # nodes are the roots of the n-th polynom which are the eigenvalues of this matrix
    # the weights are the squares of the first entry of the corresponding eigenvectors
    return nodes, np.reshape(vectors[0, :] ** 2, (len(nodes),))


# Polynomial basis: http://dlmf.nist.gov/18.3
# Recurrence correlations: http://dlmf.nist.gov/18.9#i

# TODO maybe save calculated nodes/weights on disk, especially for higher degrees


# when returned 'amount' of nodes is fixed (no matter the degree),
# then using 'amount' nodes gives stable and converging results up to degree<2*amount for collocation
class PolyBasis:
    def __init__(self, name, polys, nodes_and_weights):
        self.name = name
        self.polys = polys
        self.nodes_and_weights = lru_cache(maxsize=None)(nodes_and_weights)


def make_hermite():
    # Hermite polynomials, recursion p_n(x)=x*p_(n-1)(x)-(n-1)*p_(n-2), p_0(x)=1, p_1(x)=x
    # _poly_function_basis_recursive((lambda x: 1, lambda x: x),  (lambda n, x: x, lambda n, x: 1 - n)) # as example
    basis = PolyBasis("Hermite",
                      _poly_basis_recursive([np.array([1.]), np.array([0., 1.])],  # starting values
                                            [(0, lambda n, c: npoly.polymulx(c)),
                                             (1, lambda n, c: c * (1. - n))]),
                      lambda degree: calculate_nodes_and_weights(np.zeros(degree), np.arange(1, degree)))
    basis.polys = lambda degree: poly_by_roots(basis.nodes_and_weights(degree)[0], 1)
    return basis


def make_laguerre(alpha):
    # normalized recurrence relation: q_n=xq_(n-1) - (2(n-1) + alpha)q_(n-1) - (n-1)(n - 2 + alpha)q_(n-2)
    # to obtain the regular polynomials multiply by (-1)^n / n!
    basis = PolyBasis("Laguerre",
                      _poly_basis_recursive([np.array([1.]), np.array([alpha, -1.])],
                                            [(0, lambda n, c: npoly.polyadd(c * (2 * (n - 1) + alpha) / n,
                                                                            - npoly.polymulx(c) / n)),
                                             (1, lambda n, c: -(n - 1 + alpha - 1) / n * c)]),
                      lambda degree: calculate_nodes_and_weights(2 * np.arange(0, degree) + alpha,
                                                                 np.arange(1, degree) * (
                                                                 np.arange(1, degree) - 1 + alpha)))
    basis.polys = lambda degree: poly_by_roots(basis.nodes_and_weights(degree)[0],
                                               (1, -1)[degree % 2] / math.factorial(degree))  # (-1)^n/n!
    return basis


def make_legendre():
    # Legendre polynomials, interval assumed to be [-1,1], recursion p_n(x)=x(2n-1)/n * p_(n-1)(x)-(n-1)/n*p_(n-2)(x)
    # http://math.stackexchange.com/questions/12160/roots-of-legendre-polynomial gives the monic version of the legendre
    # polynomials: p_n(x)=x*p_(n-1)(x)-(n-1)^2/(4(n-1)^2-1)p_(n-2), to get the normal polynomial divide by
    # (n!)^2 * 2^n / (2n)!
    basis = PolyBasis("Legendre",
                      _poly_basis_recursive([np.array([1.]), np.array([0., 1.])],  # starting values
                                 [(0, lambda n, c: (2. * n - 1) / n * npoly.polymulx(c)),
                                  (1, lambda n, c: (1. - n) / n * c)]),
                      lambda degree: calculate_nodes_and_weights(np.zeros(degree), np.arange(1, degree) ** 2
                                                                 / (4 * (np.arange(1, degree) ** 2) - 1)))
    return basis


def make_jacobi(alpha, beta):
    # jacobi polynomial basis (belongs to beta distribution on (-1,1))
    def jacobi_basis():
        def get_factor(n):
            return (2 * n + alpha + beta - 1) * (2 * n + alpha + beta) / (2 * n * (n + alpha + beta))

        # this doesn't even look nice on paper
        return _poly_basis_recursive([np.array([1.]),
                                      np.array([0.5 * (alpha - beta), 0.5 * (alpha + beta + 2)])],  # starting values
                                     [(0, lambda n, c: npoly.polyadd(npoly.polymulx(c) * get_factor(n),
                                                                     -c * get_factor(n) * (beta ** 2 - alpha ** 2)
                                                                     / ((2 * n + alpha + beta - 2)
                                                                        * (2 * n + alpha + beta)))),
                                      (1, lambda n, c: c * (-2 * get_factor(n) * (n + alpha - 1) * (n + beta - 1) / (
                                          (2 * n + alpha + beta - 2) * (2 * n + alpha + beta - 1))))])

    def jacobi_nodes_and_weights(degree):
        temp = np.arange(1, degree)
        return calculate_nodes_and_weights((beta ** 2 - alpha ** 2) / ((2 * np.arange(degree) + alpha + beta)
                                                                       * (2 * np.arange(degree) + 2 + alpha + beta)),
                                           4 * temp * (temp + alpha) * (temp + beta) * (temp + alpha + beta)
                                           / ((2 * temp + alpha + beta - 1) * ((2 * temp + alpha + beta) ** 2)
                                              * (2 * temp + alpha + beta + 1)))
    basis = PolyBasis("Jacobi",
                      jacobi_basis(),
                      jacobi_nodes_and_weights)
    return basis


if __name__ == "__main__":
    a, b = 0.5, 3.7
    test_degree = 25
    # HINT: hermite (so hermite-gauss chaos) and laguerre (so laguerre-gamma chaos)
    # nodes are becoming wrong for degree >= 15 when using the recurrence correlation
    # as the image becomes very big (but also if normalized very small (O(10^-16))), orthonormal basis are correct!
    # the nodes are also correct, the problem is that the polynomial evaluation becomes increasingly bad for these
    # types of basis because of cancellation and round off errors. Therefore use definition by roots.
    poly_basis = make_hermite()
    test_nodes = poly_basis.nodes_and_weights(test_degree)[0]
    # nodes are the roots of the corresponding polynom
    test_poly = poly_basis.polys(test_degree)
    compare_poly = poly_by_roots(test_nodes, 1)  # for hermite
    print("Nodes:", np.array(test_nodes))
    # multiply test_nodes by np.sqrt(0.5) when comparing to http://keisan.casio.com/exec/system/1281195844
    print("Should all be ~zero:", np.vectorize(test_poly)(test_nodes))
    print("Should all be ~zero:", compare_poly(test_nodes))
    import matplotlib.pyplot as plt

    plt.figure()
    x_data = np.arange(-5, 5, 0.01)
    plt.plot(x_data, np.vectorize(test_poly)(x_data), label="test_poly")
    plt.legend()
    plt.show()
