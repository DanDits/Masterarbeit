import math
import numpy as np


# Source of Chebyshev class: http://www.excamera.com/sphinx/article-chebyshev.html
class Chebyshev:
    """
    Chebyshev(a, b, n, func)
    Given a function func, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.
    Method eval(x) yields the approximated function value.
    """
    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                  for k in range(n)]) for j in range(n)]

    def eval(self, x):
        a,b = self.a, self.b
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)             # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]   # Last step is different

eps = np.finfo(float).eps


# here would be an implementation of the Remez algorithm to calculate the best approximation (aka minimax)
# with respect to the maximumsnorm. I didn't find a python implementation for this.. in the scipy.signal package
# they use the remez algorithm, but only indirectly and signal related, not for pure function approximation
# But there is the chebfun project for matlab which does exactly what I want here, so we just import the data
# generated by matlab (see Masterarbeit/Code/Remez/bestapprox_demo.m)


if __name__ == "__main__":
    def func(x):
        return np.abs(x - 0.25)
    N = 100
    ch = Chebyshev(-1, 1, N, func)
    import matplotlib.pyplot as plt

    x_cheb = np.arange(-1, 1, 0.001)
    y_cheb = np.vectorize(ch.eval)(x_cheb)
    y_exact = func(x_cheb)
    plt.figure()
    plt.plot(x_cheb, y_cheb, label="Cheby. Approx")
    plt.plot(x_cheb, y_exact, label="Exact")
    plt.legend()

    rel_size = 'x-large'
    plt.figure()
    plt.title('Polynomapproximation von $f(x)=|x-1/4|$ mit $p\in\mathbb{P}_{100}$', fontsize=rel_size)
    plt.xlabel('x', fontsize=rel_size)
    plt.ylabel('Punktweiser Fehler bzgl. $|\cdot|_{\infty}$', fontsize=rel_size)
    plt.plot(x_cheb, np.abs(y_cheb - y_exact), label='Chebyshev Interpolation')
    x_remez = np.genfromtxt('../data/remez_xdata', delimiter=',')
    error_remez = np.genfromtxt('../data/remez_ydata', delimiter=',')
    plt.plot(x_remez, error_remez, label='Bestapproximation bzgl. $|\cdot |_{\infty}$')
    print("Error cheb=", max(np.abs(y_cheb - y_exact)))
    print("Error best approx=", max(np.nan_to_num(error_remez)))
    plt.legend(fontsize=rel_size)
    plt.show()
