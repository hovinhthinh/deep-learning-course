import numpy as np
import math
import random
import matplotlib.pyplot as plt


# -12
# -0.333333
# -0.666666
# 0.666666

def normalize(vector):
    length = 0
    for i in range(len(vector)):
        length += vector[i][0] * vector[i][0]
    length = math.sqrt(length)
    for i in range(len(vector)):
        vector[i][0] /= length
    return vector


def distance(x, y):
    length = 0
    for i in range(len(x)):
        length += x[i][0] * y[i][0]
    return math.sqrt(abs(length))


M = np.array(
    [[-2, -2, 3],
     [-10, -1, 6],
     [10, -2, -9]])

# init v
v = np.array([[-0.333333], [-0.666666], [0.666666]])

# init u
u = np.zeros([3, 1])
for i in range(3):
    u[i][0] = random.random()

logger = []

while (True):
    u = normalize(M.dot(u))
    d = distance(u, v)
    logger.append(d)
    if abs(d - 1) <= 0.00001:
        break

plt.xlabel('iters')
plt.ylabel('distance |<u,v>|')
plt.plot(logger)
plt.show()
