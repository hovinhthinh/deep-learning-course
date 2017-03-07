from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt

# Parameters
learning_rate = 1e-4
training_batches = 550
batch_size = 50

display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# build the network
def conv_net(x, weights, biases):
    _x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer
    conv = tf.nn.bias_add(tf.nn.conv2d(_x, weights['w1'], strides=[1, 1, 1, 1],
                                       padding='SAME'), biases['b1'])
    conv = tf.nn.relu(conv)
    conv = tf.nn.avg_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Fully connected layer
    # Reshape conv output to fit dense layer input
    dense = tf.reshape(conv, [-1, weights['w2'].get_shape().as_list()[0]])
    dense = tf.add(tf.matmul(dense, weights['w2']), biases['b2'])
    dense = tf.nn.relu(dense)

    # Output, class prediction
    out = tf.add(tf.matmul(dense, weights['out']), biases['out'])
    return out


length = []
acc = []


# Launch the graph with size of hidden layer and return accuracy
def run(n_hidden_2):
    length.append(n_hidden_2)
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'w1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # fully connected, 14*14*32 inputs, 1024 outputs
        'w2': tf.Variable(tf.random_normal([14 * 14 * 32, n_hidden_2])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[32])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Construct model
    pred = conv_net(x, weights, biases)
    # Define loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()

    print("Train with size of second hidden layer:", n_hidden_2)
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        batch = 0
        stop = False
        while (True):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)

            # Loop over all batches
            for i in range(total_batch):
                batch += 1
                if batch % display_step == 0:
                    print("Training batch:", batch, "/", training_batches)
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})

                if batch == training_batches:
                    stop = True
                    break
            if stop:
                break

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        result = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = result.eval({x: mnist.test.images, y: mnist.test.labels});
        print("Accuracy:", accuracy)
        acc.append(accuracy)


##########

run(128)
run(256)
run(512)
run(1024)

plt.plot(length, acc, 'b-', linewidth=2)
plt.title("model accuracy with respect to second hidden layer size")
plt.xlabel('second layer size')
plt.ylabel('accuracy')
plt.show()
