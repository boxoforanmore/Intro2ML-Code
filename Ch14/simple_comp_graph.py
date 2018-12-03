import tensorflow as tf

# Create a graph to evaluate z=2*(a-b)+c

g = tf.Graph()

# If we do not create a graph, there is always a default graph
# All nodes are added to the default graph

with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')

    z = 2*(a-b) + c

with tf.Session(graph=g) as sess:
    print('2*(a-b)+c => ', sess.run(z))

