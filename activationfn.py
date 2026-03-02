import numpy as np

X = np.array([[1, 2],
              [-1, -2],
              [0.5, -0.5],
              [3, -1]])
np.random.seed(0)
W = np.random.randn(2, 1)
b = np.random.randn(1)
def step(x):
    return np.where(x >= 0, 1, 0)
def sign(x):
    return np.sign(x)
def linear(x):
    return x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
z = np.dot(X, W) + b
print("Raw Output (z):\n", z)
print("\nStep Activation:\n", step(z))
print("\nSign Activation:\n", sign(z))
print("\nLinear Activation:\n", linear(z))
print("\nSigmoid Activation:\n", sigmoid(z))
print("\nTanh Activation:\n", tanh(z))
print("\nReLU Activation:\n", relu(z))
