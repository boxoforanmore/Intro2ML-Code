import tensorflow as tf

## create a graph
g = tf.Graph()

# Computes net input z of a sample point x in 

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')
    z = w*x + b

    init = tf.global_variables_initializer()

## Create a session and pass in graph g
with tf.Session(graph=g) as sess:
    ## Initialize w and b:
    sess.run(init)

    ## Evaluate z:
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z=%4.1f' % (t, sess.run(z, feed_dict={x:t})))

## The following allows feeding in elements as batch of input data
with tf.Session(graph=g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x:[1., 2., 3.]}))
