from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


# input
x = None
def compress(output):
    print("X before normalized")
    print(x)

    # normalize
    sum = np.array([0.0,0.0])
    for i in range(0, len(x)):
        sum += x[i]
    sum /= len(x)
    for i in range(0, len(x)):
        x[i] -= sum
    print("\nX after normalized")
    print(x)

    # X^TX
    print("\n(X^T)X = ")
    xtx = np.dot(x.transpose(), x)
    print(xtx)

    # eigenvalues an eigenvectors
    w, v = np.linalg.eig(xtx)

    print("\nEigenvalues:")
    print(w)
    print("Corresponding eigenvectors:")
    print(v)

    best_eigenvalue_position = -1
    for i in range(0,len(w)):
        if best_eigenvalue_position == -1 or w[i] > w[best_eigenvalue_position]:
            best_eigenvalue_position = i

    D = np.array([v[best_eigenvalue_position]]).transpose()
    print "Chosen eigenvectors:", v[best_eigenvalue_position]

    print("\nProjected values of X:")
    result = np.dot(D.transpose(), x.transpose())
    print(result)

    reconstructed = np.dot(D, result)

    print("Reconstructed vector:")
    print(reconstructed)
    print("\n")

    ax = []
    ay = []
    for i in range(len(x)):
        ax.append(x[i][0])
        ay.append(x[i][1])
    plt.figure(figsize=(7, 9))
    plt.axis([-3, 3, -2, 2])
    plt.plot(ax, ay, 'ro', label='original mean normalized data')
    plt.plot(reconstructed[0], reconstructed[1], 'bs', label='reconstructed data using mean normalized X')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


print("Part a) --------------------------------------------------")
print("----------------------------------------------------------\n")
x = np.array([[1.0, 1.0], [2.0,2.0], [3.0,1.0], [4.0,1.0]])
compress('pca_a')
print("Part b) --------------------------------------------------")
print("----------------------------------------------------------\n")
x = np.array([[-1.0,1.0], [-2.0,2.0], [-1.0,3.0], [-1.0,4.0]])
compress('pca_b')
