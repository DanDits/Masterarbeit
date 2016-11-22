import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly as poly
import polynomial_chaos.poly_chaos_distributions as ch
import polynomial_chaos.distributions as dst
from scipy.integrate import quad

alpha = 0.1
chaos = ch.make_laguerreChaos(alpha)
basis = [chaos.normalized_basis(i) for i in range(5)]
for i in range(5):
    print(chaos.normalization_gamma(i))

for b1 in basis:
    print("New")
    for b2 in basis:
        result = quad(lambda x: b1(x) * b2(x) * chaos.distribution.weight(x),
                      chaos.distribution.support[0], chaos.distribution.support[1])[0]
        print(result)