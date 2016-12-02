import unittest
import polynomial_chaos.distributions as distr
from scipy.integrate import quad


class DistributionTestCase(unittest.TestCase):
    def setUp(self):
        self.distrs = [distr.make_uniform(-3.7, 2),
                       distr.make_uniform(-1, 1),
                       distr.make_uniform(0, 1),
                       distr.make_exponential(2.),
                       distr.make_gamma(1.5, 0.7),
                       distr.make_gamma(1, 1),
                       distr.make_beta(-0.5, -0.5),
                       distr.make_beta(0.5, 0.5),
                       distr.make_beta(0.3, 0.7),
                       distr.make_beta(0, 0),
                       distr.gaussian]

    def testWeight(self):
        # weight function of distribution needs to be a PDF, so the integral over the support normalized to 1
        for d in self.distrs:
            result = quad(d.weight, d.support[0], d.support[1])[0]
            self.assertAlmostEqual(result, 1.)
