from util.quadrature.rules import CentralizedDiamondQuadrature, PseudoSparseDiamond, FullTensorQuadrature
import polynomial_chaos.poly as p
import matplotlib.pyplot as plt


chaos1 = p.make_legendre()
chaos2 = p.make_laguerre(3)

P = 14
ruleCentralized = CentralizedDiamondQuadrature([chaos1.nodes_and_weights, chaos2.nodes_and_weights], P, False)
ruleSparseDiamond = PseudoSparseDiamond([chaos1.nodes_and_weights, chaos2.nodes_and_weights],
                                        [chaos1.name, chaos2.name], P)
ruleFullTensor = FullTensorQuadrature([P, P], [chaos1.nodes_and_weights, chaos2.nodes_and_weights])

rule, name = ruleFullTensor, "full_tensor"

plt.figure()
plt.title("Interpolationspunkte des {}-Ansatzes, $N=2$, $P={}$"
          .format(name, P))
plt.xlabel("Punkte der Nullstellen von {}-Polynomen".format(chaos1.name))
plt.ylabel("Punkte der Nullstellen von {}-Polynomen".format(chaos2.name))
plt.plot(*zip(*rule.get_nodes()), ".", color="black", markersize=15)
plt.tight_layout()
plt.show()
