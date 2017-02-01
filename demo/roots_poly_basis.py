# Visualize the roots of the polynomials of ascending degree
import polynomial_chaos.poly as p
import matplotlib.pyplot as plt

plt.figure()
polybasis = p.make_legendre()
max_n = 15

for n in range(max_n):
    roots = polybasis.nodes_and_weights(n)[0]
    print(roots)
    plt.plot(roots, (n,) * len(roots), ".", color="black", markersize=15)
plt.title("Nullstellen der {}-Polynome".format(polybasis.name))
plt.ylim((0, max_n+1))
plt.ylabel("Polynomgrad")
plt.show()

