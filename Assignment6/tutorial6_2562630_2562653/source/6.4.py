from __future__ import division

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

u1 = [];
v1 = [];
u0 = [];
v0 = [];

# plot data
with open('logReg.csv') as f:
    lines = f.readlines()
    for line in range(1, len(lines)):
        p = lines[line].split(',')
        if (p[2][0] == '0'):
            u0.append(float(p[0]));
            v0.append(float(p[1]));
        else:
            u1.append(float(p[0]));
            v1.append(float(p[1]));

plt.plot(u0, v0, 'b.', label='class 0')
plt.plot(u1, v1, 'r.', label='class 1')
plt.legend()
plt.xlabel('X'), plt.ylabel('Y')
plt.show()

# generate features
x = []
y = []


def get_feature(ui, vi):
    xp = []
    for degree in range(1, 7):  # loop degree 0->6
        for deg_u in range(0, degree + 1):  # loop degree of u
            deg_v = degree - deg_u
            xp.append(pow(ui, deg_u) * pow(vi, deg_v))
    return xp


for p in range(len(u1)):  # loop class 1
    xp = get_feature(u1[p], v1[p])
    x.append(xp)
    y.append(1)
for p in range(len(u0)):  # loop class 0
    xp = get_feature(u0[p], v0[p])
    x.append(xp)
    y.append(0)

# newton
N = len(x[0])  # dimension of parameter


def h(P, X):
    z = 0
    for i in range(len(P)):
        z += P[i] * X[i]
    if z < -100:
        return 1e-12
    if z > 100:
        return 1 - (1e-12)
    return 1 / (1 + math.exp(-z))


def get_move_direction(p, lamda):
    g = []
    for i in range(N):
        sum = 0
        for j in range(len(x)):
            sum += (h(p, x[j]) - y[j]) * x[j][i]
        sum += lamda * p[i]
        g.append(sum / len(x))
    return g


EPS = 1e-9

def can_stop(p):
    for i in range(len(p)):
        if abs(p[i]) > EPS:
            return False;
    return True;


def get_hessian(p, lamda):
    H = []
    for i in range(N):
        row = []
        for j in range(N):
            hij = 0
            for k in range(len(x)):
                h_theta = h(p, x[k])
                hij += h_theta * (1 - h_theta) * x[k][i] * x[k][j]
            if i == j:
                hij += lamda
            row.append(hij / len(x))
        H.append(row)
    return np.array(H)


def get_f(p, lamda):
    sum = 0;
    for i in range(len(x)):
        hi = h(p, x[i]);
        if hi < 1 and hi > 0:
            sum -= y[i] * math.log(hi) + (1 - y[i]) * math.log(1 - hi);
        else:
            sum = 1e12

    for i in range(N):
        sum += lamda / 2 * p[i] * p[i]
    return sum / len(x)


def process(lamda):
    print 'process with lambda = ', lamda
    # init parameter
    p = []
    for i in range(N):
        # p.append(random.random())
        p.append(0)
        # for j in range(len(x)):
        #     p[i] += x[j][i] / len(x);

    # learn
    test = 0
    while True:
        test += 1
        m = get_move_direction(p, lamda)
        print 'gradient:', m
        print 'loss:', get_f(p, lamda)
        if can_stop(m) or test > 10:
            print("Plotting for lambda:", lamda)
            break
        m = np.array([m])
        H_inv = inv(get_hessian(p, lamda))
        m = np.dot(H_inv, m.transpose())
        m = m.transpose()[0]
        p -= m

    boundx = []
    boundy = []
    rx = np.arange(-1, 1.25, 0.005)
    ry = np.arange(-1, 1.25, 0.005)
    for i in range(len(rx)):
        for k in range(len(ry)):
            xp = get_feature(rx[i], ry[k])
            sum = 0
            for j in range(N):
                sum += xp[j] * p[j];
            if abs(sum) <= 0.1:
                boundx.append(rx[i]);
                boundy.append(ry[k]);
    plt.plot(u0, v0, 'b.', label='class 0')
    plt.plot(u1, v1, 'r.', label='class 1')
    plt.plot(boundx, boundy, 'g.', label=('boundary with lambda = ' + str(lamda)))
    plt.legend()
    plt.xlabel('X'), plt.ylabel('Y')
    plt.show()

process(0)
process(1)
process(10)
process(0.01)
