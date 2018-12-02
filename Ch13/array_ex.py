# Working with array structures

import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name='input_x')
    x2 = tf.reshape(x, shape=(-1, 6), name='x2')

    ## Calculate the sum of each column
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')

    ## Calculate the mean of each column
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')


with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)

    print('input shape: ', x_array.shape)
    print('Reshaped: \n', sess.run(x2, feed_dict={x:x_array}))
    print()
    print('Column Sums: \n', sess.run(xsum, feed_dict={x:x_array}))
    print()
    print('Column Means: \n', sess.run(xmean, feed_dict={x:x_array}))

