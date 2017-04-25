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
setting6 = (lambda xs: 1./(1 + 25*xs[0]**2)/(1 + 225*xs[1]**2), 0.27468 * 0.10028, 2, chaos)  # runge function 2d
chaos = mv.chaos_multify([pcd.legendreChaos] * 2, 1)
setting7 = (lambda xs: np.sin(3 * xs[0] + np.cos(xs[1])), 0.0347458, 2, chaos)
chaos = mv.chaos_multify([pcd.make_laguerreChaos(1.), pcd.make_laguerreChaos(1.)], 1)
setting8 = (lambda xs: 0, 1., 2, chaos)  # not useful for testing but only visualization

setting = setting4
visualize_quad = True
nodes_counts_sparse, nodes_counts_full, nodes_counts_sparse_gc = [], [], []
error_sparse, error_full, error_sparse_gc = [], [], []
level = 6
for level in range(level + 1):
    print("current level=", level)
    chaos = setting[3]
    chaos.init_quadrature_rule("sparse_gc", level)
    nodes_count = chaos.quadrature_rule.get_nodes_count()
    nodes_counts_sparse_gc.append(nodes_count)
    sparse_gc_result = chaos.integrate(setting[0])
    error_sparse_gc.append(abs(sparse_gc_result - setting[1]))

    dim = setting[2]
    current_count = [int(nodes_count ** (1./dim))] * dim
    chaos.init_quadrature_rule("full_tensor", current_count)
    nodes_counts_full.append(chaos.quadrature_rule.get_nodes_count())
    full_result = chaos.integrate(setting[0])
    error_full.append(abs(setting[1] - full_result))

    current_count = [count + 1 for count in current_count]
    chaos.init_quadrature_rule("full_tensor", current_count)
    nodes_counts_full.append(chaos.quadrature_rule.get_nodes_count())
    full_result = chaos.integrate(setting[0])
    error_full.append(abs(setting[1] - full_result))

    chaos.init_quadrature_rule("sparse", level)
    nodes_count = chaos.quadrature_rule.get_nodes_count()
    nodes_counts_sparse.append(nodes_count)
    sparse_result = chaos.integrate(setting[0])
    error_sparse.append(abs(sparse_result - setting[1]))

print(nodes_counts_full)
print(nodes_counts_sparse)
import matplotlib.pyplot as plt
print(nodes_counts_sparse_gc)
plt.figure()
plt.xlabel("Anzahl an Quadraturpunkten")
plt.ylabel("Fehler der Quadratur")
plt.plot(nodes_counts_sparse_gc, error_sparse_gc, "o-", label="sparse GC")
plt.plot(nodes_counts_sparse, error_sparse, "o-", label="sparse")
plt.plot(nodes_counts_full, error_full, "o-", label="full_tensor")
plt.legend(loc='best')
plt.yscale('log')
plt.xscale('log')

nodes = chaos.quadrature_rule.get_nodes()
weights = chaos.quadrature_rule.get_weights()
if nodes.shape[1] == 2 and visualize_quad:
    plt.figure()
    plt.title("Dünne Gitter Collocationspunkte $N=2$, $\\ell={}$, Anzahl: {}"
              .format(level, chaos.quadrature_rule.get_nodes_count()))
    print("Nodes=", nodes)
    plt.plot(nodes[:, 0], nodes[:, 1], ".")
    plt.xlabel("Punkte von den {}-Polynomen".format(chaos.chaos_list[0].poly_basis.name))
    plt.ylabel("Punkte von den {}-Polynomen".format(chaos.chaos_list[1].poly_basis.name))
    print("Weights=", len(weights))
plt.show()
