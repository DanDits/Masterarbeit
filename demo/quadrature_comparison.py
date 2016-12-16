import numpy as np
import polynomial_chaos.poly_chaos_distributions as pcd
import polynomial_chaos.multivariation as mv
from util.analysis import mul_prod

setting1 = (lambda xs: np.sin(xs[0] + 0.5), 0.403423, 1, pcd.legendreChaos)
setting2 = (lambda xs: 1./(1 + 25*xs[0]**2), 0.27468, 1, pcd.legendreChaos)  # runge function
chaos = mv.chaos_multify([pcd.hermiteChaos, pcd.legendreChaos, pcd.make_laguerreChaos(0.3)], 1)
setting3 = (lambda xs: xs[0] ** 4 * xs[1] ** 12 * xs[2] ** 2, 0.090000000000027, 3, chaos)
chaos = mv.chaos_multify([pcd.hermiteChaos, pcd.legendreChaos], 3)
setting4 = (lambda xs: xs[0] ** 4 * xs[1] ** 12, 0.2307692307693, 2, chaos)
chaos = mv.chaos_multify([pcd.hermiteChaos] * 2, 3)
setting5 = (lambda xs: xs[0] ** 4 * xs[1] ** 6, 45., 2, chaos)
chaos = mv.chaos_multify([pcd.legendreChaos] * 2, 1)
setting6 = (lambda xs: 1./(1 + 25*xs[0]**2)/(1 + 225*xs[1]**2), 0.27468 * 0.10028, 2, chaos)  # runge function
chaos = mv.chaos_multify([pcd.legendreChaos] * 2, 1)
setting7 = (lambda xs: np.sin(3 * xs[0] + np.cos(xs[1])), 0.0347458, 2, chaos)

setting = setting6
visualize_quad = False
nodes_counts_sparse, nodes_counts_full = [], []
error_sparse, error_full = [], []
for level in range(10):
    chaos = setting[3]
    chaos.init_quadrature_rule("sparse", level)
    nodes_count = chaos.quadrature_rule.get_nodes_count()
    nodes_counts_sparse.append(nodes_count)
    sparse_result = chaos.integrate(setting[0])
    error_sparse.append(abs(sparse_result - setting[1]))
    dim = setting[2]
    nodes_count_full = [int(nodes_count ** (1./dim))] * dim
    nodes_counts_full.append(mul_prod(nodes_count_full))
    chaos.init_quadrature_rule("full_tensor", nodes_count_full)
    full_result = chaos.integrate(setting[0])
    error_full.append(abs(setting[1] - full_result))

    nodes_count_full = [count + 1 for count in nodes_count_full]
    nodes_counts_full.append(mul_prod(nodes_count_full))
    chaos.init_quadrature_rule("full_tensor", nodes_count_full)
    full_result = chaos.integrate(setting[0])
    error_full.append(abs(setting[1] - full_result))

import matplotlib.pyplot as plt
print(nodes_counts_sparse)
plt.figure()
plt.plot(nodes_counts_sparse, error_sparse, "o-", label="Sparse")
plt.plot(nodes_counts_full, error_full, "o-", label="FullTensor")
plt.legend(loc='best')
plt.yscale('log')


nodes = chaos.quadrature_rule.get_nodes()
weights = chaos.quadrature_rule.get_weights()
if nodes.shape[1] == 2 and visualize_quad:
    plt.figure()
    plt.plot(nodes[:, 0], nodes[:, 1], ".")
    plt.figure()
    print("Weights=", len(weights))
    sq = int(np.sqrt(len(weights)))
    plt.pcolormesh(weights.reshape((sq, sq)))
plt.show()
