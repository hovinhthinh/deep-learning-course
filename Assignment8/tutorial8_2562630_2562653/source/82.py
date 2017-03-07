from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def min_max_rescale(matrix, l, r, to_int=True):
    m = len(matrix)
    n = len(matrix[0])
    lowest = 1e9
    greatest = -1e9
    for i in range(m):
        for j in range(n):
            lowest = min(lowest, matrix[i][j])
            greatest = max(greatest, matrix[i][j])
    for i in range(m):
        for j in range(n):
            matrix[i][j] = l + (matrix[i][j] - lowest) / (greatest - lowest) * (r - l)
            if to_int:
                matrix[i][j] = int(matrix[i][j] + 0.5)

def get_value_at_pos(imm, i, j):
    if i < 0 or j < 0 or i >= len(imm) or j >= len(imm[0]):
        return int(0)
    return int(imm[i][j])


# 82b
im = misc.imread('clock_noise.png')

m = len(im)
n = len(im[0])

matrix = np.empty([m, n])

for i in range(m):
    for j in range(n):
        matrix[i][j] = 0.0
        for x in range(-1, 2):
            for y in range(-1, 2):
                matrix[i][j] += get_value_at_pos(im, x + i, y + j) / 9.0
min_max_rescale(matrix, 0, 255)

plt.figure("8.2b")
plt.subplot(121)
plt.title("8.2b - original")
plt.imshow(im, cmap=plt.cm.gray)
plt.subplot(122)
plt.title("8.2b - kernel applied")
plt.imshow(matrix, cmap=plt.cm.gray)
plt.show()

# 82c
im = misc.imread('clock.png')

m = len(im)
n = len(im[0])

matrix = np.empty([m, n])
for x in range(m):
    for y in range(n):
        matrix[x][y] = get_value_at_pos(im, x, y - 1) + get_value_at_pos(im, x, y + 1) - 2 * get_value_at_pos(im, x, y)

min_max_rescale(matrix, 0, 255)

plt.figure("8.2c")
plt.subplot(121)
plt.title("8.2c - original")
plt.imshow(im, cmap=plt.cm.gray)
plt.subplot(122)
plt.title("8.2c - kernel applied")
plt.imshow(matrix, cmap=plt.cm.gray)
plt.show()
