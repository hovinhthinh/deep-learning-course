'''
This code is doing PCA on a purely numerical basis using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Dietrich Klakow based on the logistic regression by Aymeric Damien
'''
#Project: https://repos.lsv.uni-saarland.de/dietrich/Neural_Networks_Implementation_and_Application/tree/master


# Import MINST data
import input_data
import numpy as np
import re
mnist = input_data.read_data_sets("../../data/mnist", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.02
training_epochs = 20
batch_size = 1000
display_step = 1

# Dimension of hidden space
l = 2

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
D = 0.04* (tf.Variable(tf.random_uniform([l,784]))+1.0)
mu = 0.04* (tf.Variable(tf.random_uniform([784]))+1.0)

# Construct model
c = tf.matmul( (x-mu) , tf.transpose(D) ) 
xr = tf.matmul( c , D) + mu

# Minimize L2 error  
cost =  tf.reduce_sum(tf.pow(xr-x, 2))/batch_size
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)  


    print "Optimization Finished!"
    cs=sess.run(c, feed_dict={x: batch_xs, y: batch_ys})
#    csy = np.append(cs, batch_ys, axis=1)
    print cs.shape[0]
#    np.set_printoptions(threshold=np.inf)
    for i in range(0,9):
        log = open("digit"+str(i)+".pca","w")
	for j in range(0,cs.shape[0]):
		if batch_ys[j,i] == 1:
			log.write(re.sub(r"[\[\]]", "",np.array_str(cs[j,:]))+"\n")
	log.close
#    print(re.sub('[\[\]]', '', np.array_str(csy)))
#	if ( batch_ys[
#        log.write(np.array_str(csy))
#        


 
