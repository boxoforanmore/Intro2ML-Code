# Choosing an activation function for Multilayer Networks

# Logistic Func recap
# To model the probability that a sample belongs in a
# given class
import numpy as np

X = np.array([1, 1.4, 2.5])  # First value must be 1
w = np.array([0.4, 0.3, 0.5])

def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print()
print('P(y=1|x) = %.3f' % logistic_activation(X, w))
print()

#### An output layer consisting of multiple logistic activation
#### units does not produce meaningful, interpretable 
#### probability values

# W : array with shape(n_output_units, n_hidden_units + 1)
#     note that the first column are the bias units
W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])


# A : data array with shape = (n_hidden_units + 1, n_samples)
#     note that the first column of this array must be 1

A = np.array([[1, 0.1, 0.4, 0.6]])

Z = np.dot(W, A[0])

y_probas = logistic(Z)

print()
print('Net Input: \n', Z)
print()
print('Output Units: \n', y_probas)
print()

# Cannot be used as probabilities (they do not sum up to 1)
# If we only are predicting class labels (not class membership
# probabilities), this is not a problem
# Can predict class label from maximum value
y_class = np.argmax(Z, axis=0)
print()
print('Predicted class label %d' % y_class)
print()


#### Estimating class probabilities in multiclass classification 
#### via the softmax function
## Softmax is form of argmax
## --> The probability of a particular sample with net input z 
##     belonging to the ith class can be computed with a 
##     normalization term in the denominator, that is, the sum 
##     of all M linear functions

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)

print()
print('Probabilities (softmax): \n', y_probas)
print()
print('Sum of Probs: %.f' % np.sum(y_probas))
print()


#### Broadening the output spectrum using a hyperbolic tangent (tanh)
## Advantages over logistic function
## --> Broader output spectrum
## --> Ranges in open interval (-1, 1)
## --> Can improve convergence in the back propagation algorithm
import matplotlib.pyplot as plt

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

# Compare logistic and tanh sigmoid functions
plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='grey', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='grey', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')

plt.plot(z, tanh_act, linewidth=3, linestyle='--', label='tanh')
plt.plot(z, log_act, linewidth=3, label='logistic')

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


#### Alternatives to manual implementation:
## TANH:
##     --> tanh_act = np.tanh(z)
##
## Logistic:
##     --> from scipy.special import expit
##         log_act = expit(z)



#### Rectified Linear Unit Activation (ReLU)
## Common in deep neural networks
## Addresses the issue of the vanishing gradient problem
##  --> Derivative of activations with respect to net 
##      input diminishes as z becomes large
##
##  --> Definition:
##       phi(z) = max(0, z)
##
## The derivative of ReLU (with respect to its input) is 
## always 1 for positive input values
