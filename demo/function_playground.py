import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 100)


def function(var1, var2):
    return np.sin(2 * var1 + var2) * np.cos(var1 + var2)

plt.plot(x, function(x, 1))
plt.show()
