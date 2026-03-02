import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = make_classification(n_samples=500, n_features=4, random_state=1)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
np.random.seed(1)
W1 = np.random.randn(4, 5)
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 1)
b2 = np.zeros((1, 1))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)
lr = 0.01
epochs = 1000
for i in range(epochs):
    z1 = X_train.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    error = y_train - a2
    loss = np.mean(error ** 2)
    d2 = error * sigmoid_deriv(a2)
    d1 = d2.dot(W2.T) * sigmoid_deriv(a1)
    W2 += a1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += X_train.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr
    if i % 200 == 0:
        print("Epoch:", i, "Loss:", loss)
z1 = X_test.dot(W1) + b1
a1 = sigmoid(z1)
z2 = a1.dot(W2) + b2
a2 = sigmoid(z2)
pred = (a2 > 0.5).astype(int)
accuracy = np.mean(pred == y_test)

print("Accuracy:", accuracy)
