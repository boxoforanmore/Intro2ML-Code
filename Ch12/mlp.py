# Running an MLP nn
import numpy as np
from neuralnet import NeuralNetMLP


mnist = np.load('mnist_scaled.npz')

X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

# l2=0.01 overfits the data
nn = NeuralNetMLP(n_hidden=100, l2=0.1, epochs=250, eta=0.0005,
                  minibatch_size=100, shuffle=True, seed=1)

print(nn.fit(X_train=X_train[:55000], y_train=y_train[:55000],
             X_valid=X_train[55000:], y_valid=y_train[55000:]))

import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_['cost'])

# Plots cost over 200 epochs
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()


# Plot training and validation accuracy
plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Evaluate generalization performance by calculating prediction accuracy on test set
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])

print()
print('Training accuracy: %.2f%%' % (acc * 100))
print()


# Looking at the images the MLP struggles with:
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()

for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
