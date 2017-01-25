# When calculating statistics for higher moments k>2 there appears a tensor in the gPC approximation when
# calculating E[f^k(Y)]
from polynomial_chaos.poly_chaos_distributions import hermiteChaos
from scipy.integrate import quad
import numpy as np

chaos = hermiteChaos
poly_count = 5
basis = [chaos.normalized_basis(i) for i in range(poly_count)]
tensor = []
for b1 in basis:
    mat = []
    for b2 in basis:
        row = []
        for b3 in basis:
            result = quad(lambda x: b1(x) * b2(x) * b3(x) * chaos.distribution.weight(x),
                          chaos.distribution.support[0], chaos.distribution.support[1])[0]
            row.append(result)
        mat.append(row)
    tensor.append(mat)
tensor = np.array(tensor)
tensor = np.where(np.abs(tensor) < 1e-14, 0, tensor)
print(tensor)
