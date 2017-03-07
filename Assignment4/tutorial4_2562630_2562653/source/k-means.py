from __future__ import division

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random

def dist(x1, y1, x2, y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

x = []; y = []; flag = []
with open('data_kmeans.txt') as f:
    for line in f:
        args = line.split()
        x.append(float(args[0])); y.append(float(args[1]))
        flag.append(-1)

plt.plot(x, y, 'yo')
plt.legend()
plt.xlabel('X'), plt.ylabel('Y')
plt.show()

p1 = random.randrange(len(x))
p2 = random.randrange(len(x))

x1 = x[p1]; y1 = y[p1]; x2 = x[p2]; y2 = y[p2];

loss = 1e18


while (True):
    sx1 = 0; sy1 = 0; sx2 = 0; sy2 = 0
    n1 = 0; n2 = 0
    new_loss = 0
    for i in range(len(x)):
        d1 = dist(x1, y1, x[i], y[i])
        d2 = dist(x2, y2, x[i], y[i])
        if d1 < d2:
            sx1 += x[i]; sy1 += y[i]; n1 += 1
            flag[i] = 1
            new_loss += d1
        else:
            sx2 += x[i]; sy2 += y[i]; n2 += 1
            flag[i] = 2
            new_loss += d2
    if new_loss == loss:
        break
    loss = new_loss
    x1 = sx1 / n1; y1 = sy1 / n1
    x2 = sx2 / n2; y2 = sy2 / n2

x1 = []; y1 = []
x2 = []; y2 = []

for i in range(len(x)):
    if flag[i] == 1:
        x1.append(x[i]); y1.append(y[i])
    else:
        x2.append(x[i]); y2.append(y[i])

plt.plot(x1, y1, 'ro')
plt.plot(x2, y2, 'bo')
plt.legend()
plt.xlabel('X'), plt.ylabel('Y')
plt.show()







