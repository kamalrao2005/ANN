from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
filename = next(iter(uploaded))
df = pd.read_csv(filename)
print("\nDataset Loaded:")
print(df.head())
class Perceptron:
    def __init__(self, input_dim, lr=0.1, n_epochs=50):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr
        self.n_epochs = n_epochs
    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)
    def fit(self, X, y):
        for _ in range(self.n_epochs):
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                update = self.lr * (target - y_pred)
                self.weights += update * xi
                self.bias += update
input_columns = df.columns[:-1]
target_column = df.columns[-1]
X = df[input_columns].values
y = df[target_column].values
model = Perceptron(input_dim=X.shape[1], lr=0.1, n_epochs=50)
model.fit(X, y)
pred = model.predict(X)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
acc = accuracy(y, pred)
print("\n=== Perceptron Results ===")
print("Weights :", model.weights)
print("Bias    :", model.bias)
print("Targets :", y)
print("Predicted:", pred)
print(f"Accuracy: {acc*100:.2f}%")
