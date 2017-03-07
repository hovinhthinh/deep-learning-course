from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# f(x, y) = 0.001 x^2 - 0.001 y^2
# d/dx = 0.002 x
# d/dy = -0.002 y
learning_rate = 0.1
delta = 1e-9

def get_f(p):
    return 0.001 * p[0] * p[0] - 0.001 * p[1] * p[1]


def get_move_direction(p):
    return [0.002 * p[0], -0.002 * p[1]]

########## AdaGrad
print '----------AdaGrad----------'
p = [3, -1]

x = []
y = []
z = []
x.append(p[0])
y.append(p[1])
z.append(get_f(p))

iteration = 0
r = 0
while True:
    m = get_move_direction(p)
    iteration += 1

    if iteration > 300:
        print("Stop!")
        break
    r += m[0] * m[0] + m[1] * m[1]
    p_new = [p[0] - learning_rate / (delta + math.sqrt(r)) * m[0], p[1] - learning_rate / (delta + math.sqrt(r)) * m[1]]

    print 'Old_X = (%.12f, %.12f) | Gradient = (%.12f, %.12f) | new_X = (%.12f, %.12f) | F = %.12f' % (
        p[0], p[1], m[0], m[1], p_new[0], p_new[1], get_f(p_new))
    p = p_new
    x.append(p[0])
    y.append(p[1])
    z.append(get_f(p))

print
print 'Final_X = (%.12f, %.12f) | F = %.12f' % (p[0], p[1], get_f(p))
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0.5, 3.5, 0.05)
Y = np.arange(-4, -0.5, 0.05)
X, Y = np.meshgrid(X, Y)
R = (0.001 * (X ** 2) - 0.001 * (Y ** 2))

surf = ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(x, y, z, 'b-', linewidth=2, label='AdaGrad')
ax.set_zlim(-0.025,0.025)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)


########## GD

print '----------Gradient Descent----------'
p = [3, -1]

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

    if iteration > 300:
        print("Stop!")
        break
    p_new = [p[0] - m[0] * learning_rate, p[1] - m[1] * learning_rate]

    print 'Old_X = (%.12f, %.12f) | Gradient = (%.12f, %.12f) | new_X = (%.12f, %.12f) | F = %.12f' % (
        p[0], p[1], m[0], m[1], p_new[0], p_new[1], get_f(p_new))
    p = p_new
    x.append(p[0])
    y.append(p[1])
    z.append(get_f(p))

print
print 'Final_X = (%.12f, %.12f) | F = %.12f' % (p[0], p[1], get_f(p))

ax.plot(x, y, z, 'r-', linewidth=2, label='gradient descent')

########## show plot
plt.legend()
plt.show()
