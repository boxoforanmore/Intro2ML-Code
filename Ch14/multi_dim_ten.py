# Trainsforming Tensors as multidimensional arrays
import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    arr = np.array([[1., 2., 3., 3.5],
                    [4., 5., 6., 6.5],
                    [7., 8., 9., 9.5]])

    T1 = tf.constant(arr, name='T1')

    print()
    print('T1: ', T1)
    print()

    s = T1.get_shape()
    print('Shape of T1 is: ', s)
    print()

    T2 = tf.Variable(tf.random_normal(shape=s))
    print('T2: ', T2)
    print()

    T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0],)))
    print('T3: ', T3)
    print()


# Reshape T1 into T4 and T5
# T4 and T5 should both have rank 3 after reshape
with g.as_default():
    T4 = tf.reshape(T1, shape=[1, 1, -1],
                    name='T4')
    print('T4: ', T4)
    print()

    T5 = tf.reshape(T1, shape=[1, 3, -1],
                    name='T5')
    print('T5: ', T5)
    print()


# Print Elements of T4 and T5
with tf.Session(graph=g) as sess:
    print('T4 Elements: \n', sess.run(T4))
    print()
    print('T5 Elements: \n', sess.run(T5))
    print()


# Transposing T5 into T6 and T7
with g.as_default():
    T6 = tf.transpose(T5, perm=[2, 1, 0],
                      name='T6')
    print('T6: ', T6)
    print()

    T7 = tf.transpose(T5, perm=[0, 2, 1],
                      name='T7')
    print('T7: ', T7)
    print()

# Print Elements of T6 and T7
with tf.Session(graph=g) as sess:
    print('T6 Elements: \n', sess.run(T6))
    print()
    print('T7 Elements: \n', sess.run(T7))
    print()


# Split a tensor into a list of subtensors using the tf.split function
with g.as_default():
    t5_split = tf.split(T5,
                        num_or_size_splits=2,
                        axis=2, name='T8')
    print(t5_split)
    print()


# Concatenation of multiple tensors
print()
print('------------------------------------')
print()

g = tf.Graph()

with g.as_default():
    t1 = tf.ones(shape=(5, 1), dtype=tf.float32, name='t1')
    t2 = tf.zeros(shape=(5, 1), dtype=tf.float32, name='t2')

    print('t1: ', t1)
    print('t2: ', t2)
    print()

with g.as_default():
    t3 = tf.concat([t1, t2], axis=0, name='t3')
    t4 = tf.concat([t1, t2], axis=1, name='t4')

    print('t3 (concat of t1, t2; axis=0): ', t3)
    print('t4 (concat of t1, t2; axis=1): ', t4)
    print()


# Print values of these concattenated tensors
with tf.Session(graph=g) as sess:
    print('t3 elements: \n', t3.eval())
    print()
    print('t4 elements: \n', t4.eval())


