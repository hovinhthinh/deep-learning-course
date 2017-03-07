from __future__ import division

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# f(x, y) = 3 x^2 -  y^2
# d/dx = 6x
# d/dy = -2y

# d2/dxdx = 6
# d2/dxdy = 0
# d2/dydx = 0
# d2/dydy = -2

H = np.array([[6.0, 0.0], [0.0, -2.0]])
H_inv = inv(H)

def get_f(p):
    return 3 * p[0] * p[0] - p[1] * p[1]

def get_move_direction(p):
    return [6 * p[0], -2 * p[1]]


p = [5, -1]

x = []
y = []
z = []
x.append(p[0])
y.append(p[1])
z.append(get_f(p))

iteration = 0
while True:
    m = get_move_direction(p)
    iteration += 1
    if iteration > 5:
        print("Stop!")
        break
    m = np.array([m])
    m = np.dot(H_inv, m.transpose())
    m = m.transpose()[0]
    p_new = p - m

    print 'Old_(x,y) = (%.4f, %.4f) | Gradient = (%.4f, %.4f) | new_(x,y) = (%.4f, %.4f) | loss = %.4f' % (p[0], p[1], m[0], m[1], p_new[0], p_new[1], get_f(p_new))
    p = p_new
    x.append(p[0])
    y.append(p[1])
    z.append(get_f(p))

print
print 'Final_(x,y) = (%.4f, %.4f) | loss = %.4f' % (p[0], p[1], get_f(p))

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-4, 6, 0.1)
Y = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(X, Y)
R = (3 * (X**2) - (Y**2))

surf = ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(x, y, z, linewidth=2, label='newton method')
ax.set_zlim(0, 100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()
plt.show()