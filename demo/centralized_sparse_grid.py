from util.quadrature.rules import CentralizedDiamondQuadrature
import polynomial_chaos.poly as p
import matplotlib.pyplot as plt


legendre = p.make_legendre()
hermite = p.make_hermite()

rule = CentralizedDiamondQuadrature([legendre.nodes_and_weights, hermite.nodes_and_weights], 14, False)

plt.figure()
plt.title("Interpolationspunkte des centralized-Ansatzes, $N=2$, $P=14$")
plt.xlabel("Punkte der Nullstellen von Legendre-Polynomen")
plt.ylabel("Punkte der Nullstellen von Hermite-Polynomen")
plt.plot(*zip(*rule.get_nodes()), ".", color="black", markersize=15)
plt.show()
