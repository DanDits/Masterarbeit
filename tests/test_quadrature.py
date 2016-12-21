import unittest
from util.quadrature.rules import SparseQuadrature, FullTensorQuadrature
import util.quadrature.nesting as nst
from polynomial_chaos.poly import make_hermite, make_laguerre, make_legendre, make_jacobi
import numpy as np
from polynomial_chaos.poly_chaos_distributions import legendreChaos
from itertools import product


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

    def testGlenshawCurtisQuadrature(self):
        import util.quadrature.glenshaw_curtis as gc
        nodes_and_weights = gc.nodes_and_weights
        from util.quadrature.closed_fully_nested import ClosedFullNesting
        nesting = ClosedFullNesting()

        # 1d polynomial exactness
        count = 10  # exact if polynomials are degree < count
        quad = FullTensorQuadrature([count], [nodes_and_weights])
        for n in range(count):
            result = quad.apply(lambda xs: xs[0] ** n)
            wanted = 2. / (n + 1) if n % 2 == 0 else 0.
            self.assertAlmostEqual(result, wanted, places=10, msg="GC quad with count = {}, n={} polynomial exactness"
                                   .format(count, n))

        # 2d test
        count = 8  # exact if univariate polynomials of degree < count
        for n, m in product(range(count), repeat=2):
            quad = FullTensorQuadrature([count, count], [nodes_and_weights] * 2)
            result = quad.apply(lambda xs: xs[0] ** n * xs[1] ** m)
            if n % 2 == 1 or m % 2 == 1:
                wanted = 0.
            else:
                wanted = 2./(n+1) * 2./(m+1)
            self.assertAlmostEqual(result, wanted, places=10,
                                   msg="GC quad with count = {}, 2d poly exactness, n={}, m={}".format(count, n, m))

        # sparse test 1d
        for level in range(4):
            quad = SparseQuadrature(level, nesting, [nodes_and_weights])
            count = quad.get_nodes_count()
            for n in range(count):
                result = quad.apply(lambda xs: xs[0] ** n)
                wanted = 2. / (n + 1) if n % 2 == 0 else 0.
                self.assertAlmostEqual(result, wanted, places=10,
                                       msg="GC sparse quad on level={} with count = {}, n={} polynomial exactness"
                                       .format(level, count, n))


        # sparse test nd
        from util.quadrature.helpers import multi_index_bounded_sum
        from util.analysis import mul_prod
        for level in range(5):
            for dim in range(1, 4):
                quad = SparseQuadrature(level, nesting, [nodes_and_weights] * dim)
                for ind in multi_index_bounded_sum(dim, level + 1):
                    result = quad.apply(lambda xs: mul_prod(x ** i for x, i in zip(xs, ind)))
                    if any(i % 2 == 1 for i in ind):
                        wanted = 0.
                    else:
                        wanted = mul_prod(2./(i+1) for i in ind)
                    self.assertAlmostEqual(result, wanted, places=10,
                                           msg="CG sparse {}d on level={}, ind={}, quadpoints={}"
                                           .format(dim, level, ind, quad.get_nodes_count()))

    def testTransformedGlenshawCurtisQuadrature(self):
        import util.quadrature.glenshaw_curtis as gc
        from util.quadrature.closed_fully_nested import ClosedFullNesting
        nesting = ClosedFullNesting()
        from polynomial_chaos.poly_chaos_distributions import make_jacobiChaos
        jacobi = make_jacobiChaos(0.5, 1.5)
        nodes_and_weights = gc.calculate_transformed_nodes_and_weights(jacobi.distribution)
        count = 50  # as the polynomial exactness is now not implicitly true as we just transform the unweighted
        # integral, we need much higher order to ensure convergence as the weight is not a polynomial
        quad = FullTensorQuadrature([count], [nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 2), 0.25, places=5, msg="CG transformed Jacobi")

        # sparse 1d
        quad = SparseQuadrature(6, nesting, [nodes_and_weights])  # level 5 would be 2**5+1=33 nodes, but we need 50 for 5 digit accuracy
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 2), 0.25, places=5, msg="CG transformed Jacobi sparse")

        # sparse 2d
        quad = SparseQuadrature(6, nesting, [nodes_and_weights] * 2)
        result = quad.apply(lambda xs: 1)
        wanted = 1.
        self.assertAlmostEqual(result, wanted, places=3, msg="CG transformed Jacobi sparse 2d")

    # test gaussian on transformed support (-Inf,Inf)
    def testTransformedGaussian(self):
        import util.quadrature.glenshaw_curtis as gc
        from polynomial_chaos.poly_chaos_distributions import hermiteChaos
        from util.quadrature.closed_fully_nested import ClosedFullNesting
        nesting = ClosedFullNesting()
        count = 30
        gc_hermite_nodes_and_weights = gc.calculate_transformed_nodes_and_weights(hermiteChaos.distribution)
        quad = FullTensorQuadrature([count], [gc_hermite_nodes_and_weights])
        result = quad.apply(lambda xs: 1)
        wanted = 1.
        self.assertAlmostEqual(result, wanted, places=5, msg="GC hermite simple")

        # 1d sparse
        quad = SparseQuadrature(6, nesting, [gc_hermite_nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: np.sin(xs[0]) ** 2), 0.432332, places=5,
                               msg="CG Hermite sparse 1d")

        # 2d sparse
        quad = SparseQuadrature(8, nesting, 2 * [gc_hermite_nodes_and_weights])
        self.assertAlmostEqual(quad.apply(lambda xs: xs[0] ** 4 * xs[1] ** 2), 3., places=1,
                               msg="CG Hermite sparse 2d high poly")

    # test gamma on transformed support (0,Inf)
    def testTransformedLaguerre(self):
        from polynomial_chaos.poly_chaos_distributions import make_laguerreChaos
        laguerre = make_laguerreChaos(0.9)
        import util.quadrature.glenshaw_curtis as gc

        # this highly depends on laguerre's alpha parameter: around 1. it is fine, towards 0. or above 2.
        # the error increases rapidly. For lower than 1. this is because we implicitly drop the node at -1 which
        # would result in evaluating the weight function at 0. which is not possible due to 0.^(alpha-1) in weight func
        count = 100
        gc_laguerre_nodes_and_weights = gc.calculate_transformed_nodes_and_weights(laguerre.distribution)
        quad = FullTensorQuadrature([count], [gc_laguerre_nodes_and_weights])
        result = quad.apply(lambda xs: 1)
        wanted = 1.
        self.assertAlmostEqual(result, wanted, places=3, msg="GC laguerre simple")

        laguerre = make_laguerreChaos(0.3)
        # it seems like integrating polynomials this way performs much worse than something periodic
        gc_laguerre_nodes_and_weights = gc.calculate_transformed_nodes_and_weights(laguerre.distribution)
        count = 50
        quad = FullTensorQuadrature([count], [gc_laguerre_nodes_and_weights])
        result = quad.apply(lambda xs: np.sin(xs[0]) ** 2)
        wanted = 0.128709
        self.assertAlmostEqual(result, wanted, places=5)

    def testSciQuadVersusQuadrature(self):
        from scipy.integrate import nquad
        from util.quadrature.open_weakly_nested import OpenWeaklyNesting
        # use domain (-Inf,Inf) and Gaussian weight
        from polynomial_chaos.poly_chaos_distributions import hermiteChaos

        # continuous function
        def to_integrate_weighted(x, y):
            return np.sin(x + 3) * (np.cos(y) ** 2) * hermiteChaos.distribution.weight(x) * hermiteChaos.distribution.weight(y)

        def to_integrate(xs):
            return np.sin(xs[0] + 3) * (np.cos(xs[1]) ** 2)

        result_sciquad = nquad(to_integrate_weighted, [(-np.Inf, np.Inf)] * 2)[0]
        count = 15
        quad = FullTensorQuadrature([count] * 2, hermiteChaos.get_nodes_and_weights() * 2)
        result_full_quad = quad.apply(to_integrate)
        nesting = OpenWeaklyNesting()
        quad = SparseQuadrature(4, nesting, hermiteChaos.get_nodes_and_weights() * 2)
        result_sparse_quad = quad.apply(to_integrate)

        from polynomial_chaos.multivariation import chaos_multify
        chaos = chaos_multify([hermiteChaos] * 2, 1)
        chaos.init_quadrature_rule("sparse_gc", 8)
        result_sparse_gc_quad = chaos.integrate(to_integrate)
        self.assertAlmostEqual(result_sciquad, result_full_quad, places=11)
        self.assertAlmostEqual(result_sciquad, result_sparse_quad, places=5)
        self.assertAlmostEqual(result_sciquad, result_sparse_gc_quad, places=4)

    def testSciQuadVersusQuadratureBeta(self):
        from scipy.integrate import nquad
        from util.quadrature.open_weakly_nested import OpenNonNesting
        # use domain (-1,1) and Beta weight
        from polynomial_chaos.poly_chaos_distributions import make_jacobiChaos, legendreChaos

        chaos = make_jacobiChaos(0., 0.)  # theoretically almost equivalent to legendreChaos
        # except the gamma distribution can't handle the edge points -1 and 1 for alpha or beta <= 0., which results
        # in worse results for sparse_gc then we observe for higher values of alpha and beta

        # continuous function
        def to_integrate_weighted(x, y):
            return np.sin(x + 3) * (np.cos(y) ** 2) * chaos.distribution.weight(x) * chaos.distribution.weight(y)

        def to_integrate(xs):
            return np.sin(xs[0] + 3) * (np.cos(xs[1]) ** 2)

        result_sciquad = nquad(to_integrate_weighted, [(-1, 1)] * 2)[0]
        count = 15
        quad = FullTensorQuadrature([count] * 2, chaos.get_nodes_and_weights() * 2)
        result_full_quad = quad.apply(to_integrate)
        nesting = OpenNonNesting()
        quad = SparseQuadrature(3, nesting, chaos.get_nodes_and_weights() * 2)
        # level 4: 225=15**2 nodes, 12 places; level 3: 95 nodes, 9 places for jacobi(0.2,3.)
        result_sparse_quad = quad.apply(to_integrate)

        from polynomial_chaos.multivariation import chaos_multify
        chaos = chaos_multify([chaos] * 2, 1)
        chaos.init_quadrature_rule("sparse_gc", 6)  # level 6: 312 nodes, still worse by a lot
        result_sparse_gc_quad = chaos.integrate(to_integrate)
        # print(result_sciquad, result_full_quad, result_sparse_quad, result_sparse_gc_quad, sep='\n')
        self.assertAlmostEqual(result_sciquad, result_full_quad, places=12)
        self.assertAlmostEqual(result_sciquad, result_sparse_quad, places=8)
        self.assertAlmostEqual(result_sciquad, result_sparse_gc_quad, places=2)

if __name__ == "__main__":
    tc = QuadratureTestCase()
    tc.testSciQuadVersusQuadrature()
