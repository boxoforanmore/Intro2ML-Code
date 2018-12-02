# Running an MLP nn
from neuralnet import NeuralNetMLP

mnist = np.load('mnist_scaled.npz')

X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005,
                  minibatch_size=100, shuffle=True, seed=1)

print(nn.fit(X_train=X_train[:55000], y_train=y_train[:55000],
             X_test=X_test[55000:], y_test=y_test[55000:]))

import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_['cost'])

plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
