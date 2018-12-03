# Load data from MNIST dataset of handwritten digits
import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`

       * Returns two arrays:
       -images (nxm) => n = # of samples, m = # of features (pixels)
       -labels => contains corresponding target variable, the class labels 
                  (integers 0-9) of the handwritten digits

       * Images here are 28x28 pixels => converted to 1-D row vectors
                                         784 per row or image

    """

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)

    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        # magic number = description of file protocol 
        # n = number of items form the file buffer
        # >II =>
        #     '>' big endian
        #     'I' unsigned integer
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magix, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        # Normalize pixel values from MNIST to range of -1 to 1 (fairly common)
        # Can also use other feature scaling, through this tends to work fine
        # Batch normalization is also common and very effective
        images = ((images / 255.) - 0.5) * 2 

    return images, labels


X_train, y_train = load_mnist('./mnist/', kind='train')

print()
print('Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))
print()

X_test, y_test = load_mnist('./mnist/', kind='t10k')
print('Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))
print()

# Mean centering and normalization
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print()
print(X_test_centered.shape, y_test.shape)
print()


# Build a multilayer perceptron w/ three fully connected layers
import tensorflow as tf

n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='tf_x')

    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')

    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
    
    h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name='layer1')
    h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name='layer2')
    logits = tf.layers.dense(inputs=h2, units=10, activation=None, name='layer3')

    predictions = {
        'classes' : tf.argmax(logits, axis=1, name='predicted_classes'), 
        'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')
    }

    # Define cost function and optimizer for initializing model variables
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    train_op = optimizer.minimize(loss=cost)

    init_op = tf.global_variables_initializer()


# Generator for batches of data
def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])


## Create a new TF session
sess = tf.Session(graph=g)

# Run inisialization op
sess.run(init_op)

# 50 epochs of training 
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size=64)

    for batch_X, batch_y, in batch_generator:
        # Prepare a dict to feed data to our nn
        feed = {tf_x:batch_X, tf_y:batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(' -- Epoch %2d  Average Training Loss: %.4f' % (epoch+1, np.mean(training_costs)))


# Do prediction on the test set with trained model
feed = {tf_x : X_test_centered}

y_pred = sess.run(predictions['classes'], feed_dict=feed)

print()
print()
print('Test Accuracy %.2f%%' % (100*np.sum(y_pred == y_test)/y_test.shape[0]))
