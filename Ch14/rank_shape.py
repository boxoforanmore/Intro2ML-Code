## Get the rank and shape of a tensor
import tensorflow as tf
import numpy as np

g = tf.Graph()

# Define the computation graph
with g.as_default():
    # Define three tensors
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2],
                      [3, 4]])

    # Get their ranks
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)

    # Get their shapes
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()

    print()
    print('Shapes: ', s1, s2, s3)
    print()

with tf.Session(graph=g) as sess:
    print()
    print('Ranks: ', r1.eval(),
                     r2.eval(),
                     r3.eval())
    print()
