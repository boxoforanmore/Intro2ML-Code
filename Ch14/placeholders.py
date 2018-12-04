# Playing with Placeholders in TensorFlow

import tensorflow as tf

g = tf.Graph()

with g.as_default():
    # Place holders need shape, type, and what data should
    # be fed through to them upon execution
    tf_a = tf.placeholder(tf.int32, shape=[], name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[], name='tf_b')
    tf_c = tf.placeholder(tf.int32, shape=[], name='tf_c')

    r1 = tf_a - tf_b
    r2 = 2*r1
    z  = r2 + tf_c


# Feeding placeholders
with tf.Session(graph=g) as sess:
    feed = {tf_a : 1,
            tf_b : 2,
            tf_c : 3}

    print()
    print('Scalar placeholders -> z:', sess.run(z, feed_dict=feed))
    print()


# Defining placeholders for data arrays with varying batchsizes
# Can specify None for dimension when unknown or may vary
g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(tf.float32,
                          shape=[None, 2],
                          name='tf_x')

    x_mean = tf.reduce_mean(tf_x,
                            axis=0,
                            name='mean')


# Evaluate x_mean with two different inputs
import numpy as np

np.random.seed(123)
np.set_printoptions(precision=2)

with tf.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, size=(5, 2))

    print()
    print('Feeding data with shape ', x1.shape)
    print('Result: ', sess.run(x_mean, feed_dict={tf_x : x1}))
    print()

    x2 = np.random.uniform(low=0, high=1, size=(10, 2))

    print()
    print('Feeding data with shape', x2.shape)
    print('Result:', sess.run(x_mean, feed_dict={tf_x : x2}))
    print()

print(tf_x)
