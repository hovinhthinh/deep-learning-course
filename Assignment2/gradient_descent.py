from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# f(x, y) = 20 x^2 + 1/4 y^2
# d/dx = 40x
# d/dy = 1/2 y
EPS = 1e-9
learning_rate = 0.04

def can_stop(p):
    return abs(p[0]) <= EPS and abs(p[1]) <= EPS

def get_f(p):
    return 20 * p[0] * p[0] + 0.25 * p[1] * p[1]

def get_move_direction(p):
    return [40 * p[0], 0.5 * p[1]]


p = [-2, 4]

x = []
y = []
z = []
x.append(p[0])
y.append(p[1])
z.append(get_f(p))
while True:
    m = get_move_direction(p)
    if can_stop(p):
        print("Stop!")
        break
    p_new = [p[0] - m[0] * learning_rate, p[1] - m[1] * learning_rate]

    print 'Old_X = (%.12f, %.12f) | Gradient = (%.12f, %.12f) | new_X = (%.12f, %.12f) | F = %.12f\\\\' % (p[0], p[1], m[0], m[1], p_new[0], p_new[1], get_f(p_new))
    p = p_new
    x.append(p[0])
    y.append(p[1])
    z.append(get_f(p))

print
print 'Final_X = (%.12f, %.12f) | F = %.12f' % (p[0], p[1], get_f(p))

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-2, 1.5, 0.1)
Y = np.arange(-1.5, 4.5, 0.1)
X, Y = np.meshgrid(X, Y)
R = (20 * (X**2) + 0.25 * (Y**2))

surf = ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(x, y, z, label='parametric curve')
ax.set_zlim(0, 100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()