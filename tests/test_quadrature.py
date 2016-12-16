import unittest
from util.quadrature.rules import SparseQuadrature, FullTensorQuadrature
import util.quadrature.nesting as nst
from polynomial_chaos.poly import make_hermite, make_laguerre, make_legendre, make_jacobi
import numpy as np
from polynomial_chaos.poly_chaos_distributions import legendreChaos


class QuadratureTestCase(unittest.TestCase):

    def testSparseQuadrature(self):
        hermite = make_hermite()
        laguerre = make_laguerre(0.3)
        legendre = make_legendre()
        jacobi = make_jacobi(0.5, 1.5)
        dim = 2
        quad = SparseQuadrature(5, nst.get_nesting_for_name("Hermite"), dim * [hermite.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 6), 45., places=12,
                               msg="OWN Hermite")

        dim = 2
        quad = SparseQuadrature(5, nst.get_nesting_for_name("Laguerre"), dim * [laguerre.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 1 * xs[1] ** 2), 0.117, places=12,
                               msg="ONN Laguerre")

        dim = 1
        quad = SparseQuadrature(2, nst.get_nesting_for_name("Legendre"), dim * [legendre.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 12), 0.0769230769231, places=12,
                               msg="OWN Legendre")

        quad = SparseQuadrature(3, nst.get_nesting_for_multiple_names(["Hermite", "Legendre"]),
                                [hermite.nodes_and_weights, legendre.nodes_and_weights])
        result_own_hl = quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 12)
        self.assertAlmostEqual(result_own_hl, 0.2307692307693, places=12,
                               msg="OWN Hermite Legendre")

        quad = SparseQuadrature(3, nst.get_nesting_for_multiple_names(["Legendre", "Hermite"]),
                                [legendre.nodes_and_weights, hermite.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 12 * xs[1] ** 4), result_own_hl, places=12,
                               msg="OWN Legendre Hermite = Hermite Legendre!")

        quad = SparseQuadrature(4, nst.get_nesting_for_multiple_names(["Hermite", "Legendre", "Laguerre"]),
                                [hermite.nodes_and_weights, legendre.nodes_and_weights, laguerre.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 12 * xs[2] ** 2), 0.090000000000027,
                               places=12, msg="ONN Hermite Legendre Laguerre")

        dim = 1
        quad = SparseQuadrature(3, nst.get_nesting_for_name("Jacobi"), dim * [jacobi.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 2), 0.25, places=12, msg="Jacobi")

        dim = 2
        quad = SparseQuadrature(3, nst.get_nesting_for_name("Jacobi"), dim * [jacobi.nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 2 * xs[1] ** 2), 0.25 * 0.25, places=12, msg="Jacobi 2d")

    def testFullTensorQuadrature(self):

        chaos_list = [legendreChaos, legendreChaos, legendreChaos]
        nodes_and_weights_funcs = [chaos.poly_basis.nodes_and_weights for chaos in chaos_list]
        quad = FullTensorQuadrature([4, 4, 3], nodes_and_weights_funcs)
        self.assertAlmostEqual(quad.apply(lambda xs: np.sin(xs[0] + 0.5) * np.cos(xs[1] + 1) * xs[2] ** 4),
                               0.0366831, places=7, msg="Legendre^3 simple")

        quad = FullTensorQuadrature([7, 7, 7], nodes_and_weights_funcs)
        self.assertAlmostEqual(quad.apply(lambda xs: np.sin(xs[0] + 0.5) * np.cos(xs[1] + 1) * xs[2] ** 4),
                               0.0366831, places=7, msg="Legendre^3 simple big")

        quad = FullTensorQuadrature([7, 7, 7], nodes_and_weights_funcs)
        self.assertAlmostEqual(quad.apply(lambda xs: np.sin(xs[0] + xs[1] + xs[2] + 0.5) * xs[2] ** 4),
                               0.0451753, places=7, msg="Legendre^3 connected")

if __name__ == "__main__":
    tc = QuadratureTestCase()
    tc.testSparseQuadrature()
