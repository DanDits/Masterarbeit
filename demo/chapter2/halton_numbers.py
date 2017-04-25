from util.quasi_randomness import halton
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

plt.figure()
plt.subplot(121)
plt.title("Halton Sequenz in 2D", fontsize="x-large")
halton_values = halton.halton_sequence(1, 201, 2)
plt.plot(halton_values[0, :], halton_values[1, :], "o")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.subplot(122)
plt.title("Gleichverteilte Pseudozufallszahlen in 2D", fontsize="x-large")
random_values = rnd.uniform(0., 1., (2, 200))
plt.plot(random_values[0, :], random_values[1, :], "o")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()

