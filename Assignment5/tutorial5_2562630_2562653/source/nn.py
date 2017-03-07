from __future__ import division

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Network Parameters
n_hidden = 256  # hidden layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'h': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def predict(x):
    # Hidden layer with sigmoid activation
    hidden_layer = tf.add(tf.matmul(x, weights['h']), biases['h'])
    hidden_layer = tf.nn.sigmoid(hidden_layer)
    # Output layer with linear activation
    out_layer = tf.matmul(hidden_layer, weights['out']) + biases['out']
    return out_layer


# predict model
predict_model = predict(x)


# Train model

def train(X, epochs, learning_rate):
    BATCH_SIZE = 100

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.square(predict_model - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()
    eps = []
    losses = []
    accs = []

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(X.num_examples / BATCH_SIZE)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = X.next_batch(BATCH_SIZE)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Test accuracy
            correct_prediction = tf.equal(tf.argmax(predict_model, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")).eval(
                {x: mnist.test.images, y: mnist.test.labels})

            # log
            print("Epoch:", '%d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), 'accuracy=',
                  "{:.3f}%".format(accuracy * 100))

            eps.append(epoch + 1)
            losses.append(avg_cost)
            accs.append(accuracy)

        print("Training finished!")

    # plot
    plt.subplot(211)
    plt.plot(eps, losses, 'r-', label='MSE after n-epochs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.subplot(212)
    plt.plot(eps, accs, 'b-', label='Accuracy after n-epochs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

# train
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
EPOCHS = 30
LEARNING_RATE = 0.05

train(mnist.train, EPOCHS, LEARNING_RATE)
