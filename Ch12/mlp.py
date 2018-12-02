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
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magix, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        images = ((images / 255.) - 0.5) * 2

    return images, labels


