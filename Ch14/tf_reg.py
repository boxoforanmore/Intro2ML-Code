#### Building a regression model
# We will use MSE here for cost

'''
Input x : tf_x : defined as a placeholder
Input y : tf_y : defined as a placeholder

Model parameter w : weight :  defined as a variable
Model parameter b : bias   :  defined as a variable

Model output yhat : y_hat : returned by the TensorFlow operations
                            to compute the prediction using the 
                            regressoin model
'''

import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(123)

    ## Placeholders
    tf_x = tf.placeholder(shape=(None), dtype=tf.float32,
                          name='tf_x')

    tf_y = tf.placeholder(shape=(None), dtype=tf.float32,
                          name='tf_y')

    ## Define the variable (model parameters)
    weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25),
                         name='weight')
    bias = tf.Variable(0.0, name='bias')

    ## Build the model
    y_hat = tf.add(weight * tf_x, bias,
                   name='y_hat')

    ## Compute the cost
    cost = tf.reduce_mean(tf.square(tf_y - y_hat),
                          name='cost')

    ## Train the model
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')


# Create a session to launch the graph and train a model

# Create a random toy dataset for regression
import numpy
import matplotlib.pyplot as plt

np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []

    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t*t/3), size=None)
        y.append(r)

    return x, 1.726*x - 0.84 + np.array(y)

x, y = make_random_data()

# Plot random regression data
plt.plot(x, y, 'o')
plt.show()


## Train/Test splits
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]


n_epochs = 500
training_costs = []

print()

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    ## train the model for n_epochs
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op],
                        feed_dict={'tf_x:0': x_train,
                                   'tf_y:0': y_train})

        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))

print()

plt.plot(training_costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()


# Executing the nodes by their name
n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:
    # First run variables initializer
    sess.run(tf.global_variables_initializer())

    # Train the model for n_epochs
    for e in range(n_epochs):
        # Evaluating cost and train_op by name
        c, _ = sess.run(['cost:0', 'train_op'],
                        feed_dict={'tf_x:0' : x_train,
                                  'tf_y:0' : y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))

