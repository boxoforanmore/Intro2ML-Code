# Variable scoping

import tensorflow as tf

g = tf.Graph()

with g.as_default():
    # Defining two scopes, layers are nested
    with tf.variable_scope('net_A'):
        with tf.variable_scope('layer-1'):
            w1 = tf.Variable(tf.random_normal(shape=(10,4)), name='weights')
        with tf.variable_scope('layer-2'):
            w2 = tf.Variable(tf.random_normal(shape=(20,10)), name='weights')
    with tf.variable_scope('net_B'):
        with tf.variable_scope('layer-1'):
            w3 = tf.Variable(tf.random_normal(shape=(10, 4)), name='weights')


    # Variable names should be prefixed with their nested scopes
    print()
    print('w1: ', w1)
    print('w2: ', w2)
    print('w3: ', w3)
    print()


