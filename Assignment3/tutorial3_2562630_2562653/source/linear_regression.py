import numpy as np
import matplotlib.pyplot as plt

# Exercise 3.1 a
show_plot = True
X, Y = np.loadtxt('auto-mpg.data', usecols=(0, 2), unpack=True)
T = np.arange(5, 35, 0.01)

def prep_data(x, y, deg):
    return np.vander(x, deg+1), y.reshape(-1, 1)

# Exercise 3.1 b
if show_plot:
    plt.plot(X[0:50], Y[0:50], 'bo', label='training data')
    plt.legend()
    plt.xlabel('X'), plt.ylabel('Y')
    plt.show()

# Exercise 3.1 c/f
def cost(w, x, y):
    n_inv = 1/float(y.shape[0])
    diff = np.dot(x, w) - y
    loss = n_inv * np.dot(diff.T, diff) * 0.5
    gradient = np.dot(x.T, diff) * n_inv
    return loss, gradient

# Exercise 3.1 d/g
def gradient_descent(w, x, y, alpha, num_iter):
    epochs = np.zeros(num_iter)
    for i in range(num_iter):
        loss, gradient = cost(w, x, y)
        w -= alpha * gradient
        epochs[i] = loss
        print 'Epoch = %d | loss = %.12f | MSE = %.12f | W = ' % (i+1, loss, loss * 2) + repr(w.ravel())
    return w, epochs

def ex3a_g(x, y, degree, learning_rate, iterations, show_plot=True):
    # Exercise 3.1 d/f/g
    train_x, train_y = prep_data(x, y, degree)
    W = np.random.rand(train_x.shape[1], train_y.shape[1])
    W, loss_history = gradient_descent(W, train_x, train_y, learning_rate, iterations)
    # Exercise 3.1 d/e/g
    if show_plot:
        plt.subplot(211)
        plt.plot(loss_history, label='loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.subplot(212)
        plt.plot(T, np.polyval(W, T), 'r', label='prediction')
        plt.plot(x, y, 'bo', label='training data')
        plt.legend()
        plt.xlabel('X'), plt.ylabel('Y')
        plt.show()
    return W

# Exercise 3.1 h
def ex3h(x, y, degree, show_plot=True):
    W = np.polyfit(x, y, degree)
    h = np.polyval(W, x) - y
    MSE = np.dot(h.T, h) / float(y.shape[0])
    print "MSE(polyfit, deg = 9) = " + repr(MSE)
    if show_plot:
        plt.plot(T, np.polyval(W, T), 'r', label='prediction')
        plt.plot(x, y, 'bo', label='training data')
        plt.legend()
        plt.xlabel('X'), plt.ylabel('Y')
        plt.ylim([0, 500])
        plt.show()
    return W

def print_MSE(h, y, text=""):
    print text + repr(sum((h - y)**2) / float(y.shape[0]))

if __name__ == "__main__":
    train_X = X[0:50]; test_X = X[50:100]
    train_Y = Y[0:50]; test_Y = Y[50:100]
    '''Exercise 3.1 c/d/e'''
    W1 = ex3a_g(train_X, train_Y, 1, 0.0028, 10000)
    '''Exercise 3.1 f/g'''
    W2 = ex3a_g(train_X, train_Y, 2, 0.000012, 12000)
    '''Exercise 3.1 h'''
    W9 = ex3h(train_X, train_Y, 9)
    '''Exercise 3.1 i'''
    # plt.plot(train_X, train_Y, 'bo', label='training data')
    plt.plot(test_X, test_Y, 'ro', label='test_data')
    plt.plot(T, np.polyval(W1, T), 'g', label='prediction degree 1 with lin_reg')
    plt.plot(T, np.polyval(W2, T), 'y', label='prediction degree 2 with lin_reg')
    plt.plot(T, np.polyval(W9, T), 'k', label='prediction degree 9 with polyfit')
    plt.legend()
    plt.xlabel('X'), plt.ylabel('Y')
    plt.ylim([0, 500])
    print_MSE(np.polyval(W1, test_X), test_Y, "MSE(test, deg = 1): ")
    print_MSE(np.polyval(W2, test_X), test_Y, "MSE(test, deg = 2): ")
    print_MSE(np.polyval(W9, test_X), test_Y, "MSE(test, deg = 9): ")
    plt.show()
