import unittest
from util.quadrature.rules import SparseQuadrature


class PolyTestCase(unittest.TestCase):

    def testSparseQuadrature(self):
        from polynomial_chaos.poly import make_hermite, make_laguerre, make_legendre
        hermite = make_hermite()
        laguerre = make_laguerre(0.3)
        legendre = make_legendre()
        import util.quadrature.open_weakly_nested as own
        dim = 2
        quad = SparseQuadrature(5, own.OpenWeaklyNesting(), dim * [hermite.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 6), 45., places=10,
                               msg="OWN Hermite")

        dim = 2
        quad = SparseQuadrature(5, own.OpenNonNesting(), dim * [laguerre.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 1 * xs[1] ** 2), 0.117, places=10,
                               msg="ONN Laguerre")

        dim = 1
        quad = SparseQuadrature(2, own.OpenWeaklyNesting(), dim * [legendre.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 12), 0.0769230769231, places=10,
                               msg="OWN Legendre")

        quad = SparseQuadrature(3, own.OpenWeaklyNesting(), [hermite.nodes_and_weights, legendre.nodes_and_weights])
        result_own_hl = quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 12)
        self.assertAlmostEqual(result_own_hl, 0.2307692307693, places=10,
                               msg="OWN Hermite Legendre")

        quad = SparseQuadrature(3, own.OpenWeaklyNesting(), [legendre.nodes_and_weights, hermite.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 12 * xs[1] ** 4), result_own_hl, places=10,
                               msg="OWN Legendre Hermite = Hermite Legendre!")

        quad = SparseQuadrature(4, own.OpenNonNesting(), [hermite.nodes_and_weights, legendre.nodes_and_weights,
                                                          laguerre.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 12 * xs[2] ** 2), 0.090000000000027,
                               places = 10, msg="ONN Hermite Legendre Laguerre")

        # TODO also test for OWN/ ONN jacobi
