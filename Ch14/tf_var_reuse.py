# What if we want to feed different data through the same classifier?
# Variable Reuse

import tensorflow as tf

## Helper Functions

# Add a classifier
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights', 
                              shape=(data_shape[1], n_classes),
                              dtype=tf.float32)

    bias = tf.get_variable(name='bias',
                           initializer=tf.zeros(shape=n_classes))

    logits = tf.add(tf.matmul(data, weights), bias, name='logits')

    return logits, tf.nn.softmax(logits)


# Build the generator network
def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()

    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1], n_hidden)),
                     name='w1')

    b1 = tf.Variable(tf.zeros(shape=n_hidden),
                     name='b1')

    hidden = tf.add(tf.matmul(data, w1), b1,
                    name='hidden_pre-activation')

    w2 = tf.Variable(tf.random_normal(shape=(n_hidden, data_shape[1])),
                     name='w2')

    b2 = tf.Variable(tf.zeros(shape=data_shape[1]),
                     name='b2')

    output = tf.add(tf.matmul(hidden, w2), b2,
                    name = 'output')

    return output, tf.nn.sigmoid(output)


## Build the graph
batch_size = 64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100),
                          dtype=tf.float32,
                          name='tf_X')

    ## Build the generator
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X,
                                   n_hidden=50)

    ## Build the classifier
    with tf.variable_scope('classifier') as scope:
        ## Classifier for the original data:
        cls_out1 = build_classifier(data=tf_X,
                                    labels=tf.ones(shape=batch_size))
        
        ## Reuse the classifier for generated data
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(shape=batch_size))




## Can also reuse by specifying parameters as such
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100),
                          dtype=tf.float32,
                          name='tf_X')

    ## Build the generator
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)

    ## Build the classifier
    with tf. variable_scope('classifier'):
        cls_out1 = build_classifier(data=tf_X,
                                   labels=tf.ones(shape=batch_size))

    with tf.variable_scope('classifier', reuse=True):
        ## Reuse the classifier for generated data
        cls_out2 = build_classifier(data=tf_X,
                                    labels=tf.zeros(shape=batch_size))
