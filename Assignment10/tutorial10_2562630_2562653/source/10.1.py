from __future__ import division

import numpy as np
import math
import matplotlib.pyplot as plt

# input

print("Part a) --------------------------------------------------")
x = np.array([[-2, -2, 3], [-10, -1, 6], [10, -2, -9]])

# eigenvalues an eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(x)

print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
eigenvectors = eigenvectors.transpose()
print eigenvectors

# power method
v = np.array([[-1.0/3, -2.0/3, 2.0/3]]).transpose()

print "v ="
print v

u = np.random.rand(3, 1)

p = []
while (True):
    u = x.dot(u)
    l = math.sqrt(u[0][0] * u[0][0] + u[1][0] * u[1][0] + u[2][0] * u[2][0])
    u /= l # normalize u

    t = math.sqrt(abs(u[0][0] * v[0][0] + u[1][0] * v[1][0] + u[2][0] * v[2][0]))

    p.append(t)
    if abs(t - 1) <= 1e-4:
        break

plt.plot(p, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('|<u, v>|')
plt.title("power method")

plt.show()
