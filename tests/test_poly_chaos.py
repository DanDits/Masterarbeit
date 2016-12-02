import unittest
import polynomial_chaos.poly as poly
import numpy as np
import polynomial_chaos.poly_chaos_distributions as pcd
from scipy.integrate import quad


class PolyTestCase(unittest.TestCase):
    def setUp(self):
        laguerre_params = [0.1, 0.5, 1., 1.5, 2.5]
        jacobi_params = [(-0.9, -0.9), (-0.9, 2), (0, 0), (-0.5, -0.5), (0.5, 0.5), (2, 3), (2, 2)]
        self.polys = []
        self.polys.extend(poly.make_laguerre(alpha) for alpha in laguerre_params)
        self.polys.append(poly.make_hermite())
        self.polys.append(poly.make_legendre())
        self.polys.extend(poly.make_jacobi(alpha, beta) for alpha, beta in jacobi_params)

        self.chaos_list = []
        self.chaos_list.extend(pcd.make_laguerreChaos(alpha) for alpha in laguerre_params)
        self.chaos_list.append(pcd.hermiteChaos)
        self.chaos_list.append(pcd.legendreChaos)
        self.chaos_list.extend(pcd.make_jacobiChaos(alpha, beta) for alpha, beta in jacobi_params)

    def testNodesAreRoots(self):
        for p in self.polys:
            name = p.name
            params = p.params
            for degree in range(1, 25):  # higher degrees would require lower absolute tolerance (especially for jacobi)
                test_nodes = p.nodes_and_weights(degree)[0]
                test_poly = p.polys(degree)
                result = test_poly(test_nodes)
                self.assertTrue(np.allclose(result, np.zeros(test_nodes.shape), atol=1E-7),  # rtol will have no effect
                                "Nodes are not zeros for poly {} with params {} and degree {}\nResult: {}\nNodes: {}"
                                .format(name, params, degree, result, test_nodes))

    def testNodesInDistributionSupport(self):
        for degree in [1, 3, 4, 6, 10]:
            for chaos in self.chaos_list:
                nodes = chaos.poly_basis.nodes_and_weights(degree)[0]
                for node in nodes:
                    self.assertGreater(node, chaos.distribution.support[0])
                    self.assertGreater(chaos.distribution.support[1], node)
                    
    def testOrthonormality(self):
        poly_count = 20
        for chaos in self.chaos_list:
            basis = [chaos.normalized_basis(i) for i in range(poly_count)]
            mat = []
            for b1 in basis:
                row = []
                for b2 in basis:
                    result = quad(lambda x: b1(x) * b2(x) * chaos.distribution.weight(x),
                                  chaos.distribution.support[0], chaos.distribution.support[1])[0]
                    row.append(result)
                mat.append(row)
            self.assertTrue(np.allclose(mat, np.eye(poly_count, poly_count)),
                            "No orthonormal basis for chaos {}, distr. params {}\nResult: {}"
                            .format(chaos.poly_basis.name, chaos.distribution.parameters, mat))

    def testQuadratureWeights(self):
        for degree in [1, 2, 3, 4, 6, 10, 13]:
            for p in self.polys:
                # "integrate" the constant function 1 over the support with the pdf belonging to the
                # polynomial: as it's a pdf, this will be 1, so in discrete form the sum of the weights
                weights = p.nodes_and_weights(degree)[1]
                self.assertAlmostEqual(1., sum(weights))
