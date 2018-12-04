# Playground for variables in tf
import tensorflow as tf
import numpy as np

g1 = tf.Graph()

# Creating a variable object from np.array
# dtype is automatically assumed frmo the input
with g1.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]]))
    print()
    print(w)
    print()

# Variables contain no values until they are initialized
# We must initialize variables before execution
# Initialization == allocating memory for associated tensors

# GVI Returns an operator for initializing all the variables in the graph
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))

# Can also store GVI operator in object such as init_op
# ie
'''
import tensorflow as tf
'''

g2 = tf.Graph()

with g2.as_default():
    w1 = tf.Variable(1, name='w1')
    init_op = tf.global_variables_initializer()
    w2 = tf.Variable(2, name='w2')


with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print()
    print('w1: ', sess.run(w1))
    print()


# Should give an error as w2 was not initialized
# (init_op was defined before adding w2 to graph)
with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print()
    print('w2: ', sess.run(w2))
    print()
