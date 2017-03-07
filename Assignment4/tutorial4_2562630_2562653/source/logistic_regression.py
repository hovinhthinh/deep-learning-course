#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

trX = np.loadtxt('iris.data', delimiter=',', usecols=(0, 3))
teX = trX
# Setosa = [0, 1], Virginica = [1, 0]
trY = np.zeros((100, 2))
trY[50:100, 0] = 1
trY[0:50, 1] = 1
teY = trY

X = tf.placeholder("float", [None, trX.shape[1]]) # create symbolic variables
Y = tf.placeholder("float", [None, trY.shape[1]])

w = init_weights([trX.shape[1], trY.shape[1]]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in range(20):
        for start, end in zip(range(0, len(trX), 1), range(1, len(trX)+1, 1)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
    prediction = sess.run(predict_op, feed_dict={X: teX})

# plot Setosa
plt.plot(np.where(prediction == 1)[0], prediction[prediction == 1], 'bo')
# plot Virginica
plt.plot(np.where(prediction == 0)[0], prediction[prediction == 0], 'ro')
plt.show()