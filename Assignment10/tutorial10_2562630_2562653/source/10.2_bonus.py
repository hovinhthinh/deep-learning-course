from __future__ import division

import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# a) read data
str = ""

with open("data2.txt") as f:
    for line in f.readlines():
        for s in line.split():
            str += s


# b) random chunk
def random_chunk(str, k):
    p = random.randint(0, len(str) - k)
    return str[p: p + k]


# c) encode chunk
def get_pos(c):
    if c == 'a':
        return 0
    if c == 'c':
        return 1
    if c == 'g':
        return 2
    if c == 't':
        return 3
    return -1


def encode(str):
    a = np.zeros([len(str) + 1, 4])
    for i in range(1, len(str) + 1):
        pos = get_pos(str[i - 1])
        if pos != -1:
            a[i][pos] = 1
    return a


# d)

# params
num_steps = 5  # depth of training (length of training chunks)
batch_size = 1
num_classes = 4  # a c g t
state_size = 4  # dimension of hidden state
learning_rate = 1e-4
total_training_length = len(str)

# Placeholders

x = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes], name='labels_placeholder')

# for perplexity in bonus part
total_x = tf.placeholder(tf.float32, [batch_size, total_training_length, num_classes], name='input_placeholder')
total_y = tf.placeholder(tf.float32, [batch_size, total_training_length, num_classes], name='labels_placeholder')

init_state = tf.zeros([batch_size, state_size])

# RNN Inputs

rnn_inputs = tf.unpack(x, axis=1)
# for perplexity in bonus part
rnn_inputs_total = tf.unpack(total_x, axis=1)

# rnn_cell (hidden cell)
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [state_size, state_size])
    U = tf.get_variable('U', [num_classes, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [state_size, state_size])
        U = tf.get_variable('U', [num_classes, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(state, W) + tf.matmul(rnn_input, U) + b)


# adding rnn_cells to graph and compute output cells
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)

# for perplexity in bonus part
state = init_state
rnn_outputs_total = []
for rnn_input in rnn_inputs_total:
    state = rnn_cell(rnn_input, state)
    rnn_outputs_total.append(state)

# logits and predictions
with tf.variable_scope('softmax'):
    C = tf.get_variable('C', [state_size, num_classes])
    v = tf.get_variable('v', [num_classes], initializer=tf.constant_initializer(0.0))

# outputs
logits = [tf.matmul(rnn_output, C) + v for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# for perplexity in bonus part
logits_total = [tf.matmul(rnn_output, C) + v for rnn_output in rnn_outputs_total]
predictions_total = [tf.nn.softmax(logit) for logit in logits_total]

# Turn our y placeholder into a list labels
y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

# losses and train_step
losses = [tf.nn.softmax_cross_entropy_with_logits(logit, label) for logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# e) train
# train func

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    training_losses = []
    for step1 in range(500):
        training_loss = 0
        for step2 in range(100):
            training_state = np.zeros((batch_size, state_size))
            chunk = random_chunk(str, num_steps)
            s = encode(chunk)
            X = [s[0:num_steps, :]]
            Y = [s[1:num_steps + 1, :]]

            _, training_loss_ = sess.run([optimizer,
                                          total_loss,
                                          ],
                                         feed_dict={x: X, y: Y,
                                                    init_state: training_state})
            training_loss += training_loss_ / 100
        training_losses.append(training_loss)
        print "average lost last 100 steps:", training_loss

    # f) plot
    plt.plot(np.arange(0, 50000, 100), training_losses)
    plt.title("Training RNN")
    plt.xlabel('Iteration')
    plt.ylabel('average loss last 100 steps')

    plt.show()

    # bonus, a) compute complexity of trained model on whole sequence
    print ("compute perplexity of whole sequence")
    training_state = np.zeros((batch_size, state_size))
    s = encode(str)
    X = [s[0:total_training_length, :]]
    Y = [s[1:total_training_length + 1, :]]
    pred = sess.run([predictions_total],
                    feed_dict={total_x: X, total_y: Y,
                               init_state: training_state})

    negative_log_likelihood = 0
    for i in range(total_training_length):
        pos = get_pos(str[i])
        print "softmax at x[", i, "] =", pred[0][i][0]
        negative_log_likelihood += math.log(pred[0][i][0][pos], 2)
    perplexity = math.pow(2, -1.0 / total_training_length * negative_log_likelihood)

    print "perplexity  =", perplexity
