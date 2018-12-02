# MLP working on MNIST dataset of handwritten digits

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



# Load 60,000 training instances and 10,000 test samples
X_train, y_train = load_mnist('', kind='train')
print()
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('', kind='t10k')
print()
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
print()


# Visualize examples of the digits 0-9 after reshaping the 784-pixel vectors
# from dfeature matrix into original 28x28 image
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()

for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# Plot multiple examples of the same digit to see variance
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()

for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# Now with 2s
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()

for i in range(25):
    img = X_train[y_train == 2][i].reshape(28, 28) 
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
