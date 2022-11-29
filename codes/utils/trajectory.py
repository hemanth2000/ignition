import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 50, 0.1)
dy1 = 4.05
dy2 = 5.7
dx1 = 25
dx2 = 21.95
z1 = (2.4 / 25) * (X - 27.19) - 1.2
z2 = (2.4 / 21.95) * (X - 56.46) - 1

Y_ref = np.arctan(dy1 * (1 / np.cosh(z1)) ** 2 * (1.2 / dx1) - dy2 * (1 / np.cosh(z2)) ** 2 * (1.2 / dx2))

phi_ref = dy1 / 2 * (1 + np.tanh(z1)) - dy2 / 2 * (1 + np.tanh(z2))

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(X, Y_ref)
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(X, phi_ref)
plt.grid()
plt.show()
