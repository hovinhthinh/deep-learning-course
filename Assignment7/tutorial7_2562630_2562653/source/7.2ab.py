from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# f(x, y) = 3 x^2 - y^2
# d/dx = 6x
# d/dy = -2 y
learning_rate = 0.01
monument = 0.7


def get_f(p):
    return 3 * p[0] * p[0] - p[1] * p[1]


def get_move_direction(p):
    return [6 * p[0], -2 * p[1]]


########## GD with monument

print '----------Gradient Descent with Monument----------'
p = [5, -1]

x = []
y = []
z = []
x.append(p[0])
y.append(p[1])
z.append(get_f(p))

iteration = 0
velocity = [0, 0]
while True:
    m = get_move_direction(p)
    iteration += 1

    if iteration > 30:
        print("Stop!")
        break
    velocity[0] = velocity[0] * monument - learning_rate * m[0];
    velocity[1] = velocity[1] * monument - learning_rate * m[1];
    p_new = [p[0] + velocity[0], p[1] + velocity[1]]

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
X = np.arange(-4, 6, 0.1)
Y = np.arange(-6, 3, 0.1)
X, Y = np.meshgrid(X, Y)
R = (3 * (X ** 2) - (Y ** 2))

surf = ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.plot(x, y, z, 'r-', linewidth=2, label='gradient descent with monument')

########## normal GD
print '----------normal Gradient Descent----------'
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

    if iteration > 30:
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


ax.plot(x, y, z, 'b-', linewidth=2, label='gradient descent')
ax.set_zlim(0, 100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

########## show plot
plt.legend()
plt.show()
