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


# Set the random seed
import tensorflow as tf
import tensorflow.keras as keras

np.random.seed(123)
tf.set_random_seed(123)

y_train_onehot = keras.utils.to_categorical(y_train)

print()
print('First 3 labels: ', y_train[:3])
print()
print('First 3 labels (one-hot): \n', y_train_onehot[:3])
print()




# Implement nn
# 1) Use 3 layers (first 2 have 50 hidden units)

# Sequential implements a feedforward network
model = keras.models.Sequential()

# Input layer; input dimensions must match number of features in the training set
# Number of output and input units in two consecutive layers must also match

model.add(keras.layers.Dense(units=50, input_dim=X_train_centered.shape[1],
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='tanh'))

model.add(keras.layers.Dense(units=50, input_dim=50,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='tanh'))

model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=50, 
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='softmax'))

# Must define an optimizer before compiling
sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)

# Crossentropy is the generalization of logistic regression for
# multiclass predictions via softmax
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')


# Train with fit method
history = model.fit(X_train_centered, y_train_onehot,
                    batch_size=64, epochs=50, verbose=1,
                    validation_split=0.1)


# predict class labels (return class labels as integers
y_train_pred = model.predict_classes(X_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

print()
print('FIrst 3 predictions: ', y_train_pred[:3])
print()
print('Training accuracy: %.2f%%' % (train_acc * 100))
print()

y_test_pred = model.predict_classes(X_test_centered, verbose=0)

correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

print('Test accuracy: %.2f%%' % (test_acc * 100))
print()
