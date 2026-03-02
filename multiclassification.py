from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_classes=3,
    n_informative=5
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=15, verbose=0)
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print("Accuracy:", accuracy)
