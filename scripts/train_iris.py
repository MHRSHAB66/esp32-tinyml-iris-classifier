import os
import random
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)

# 1) Load Iris dataset
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)
class_names = iris.target_names

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# 3) Standardization: x_scaled = (x - mean) / scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# 4) Build Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5) Train
model.fit(
    X_train_scaled,
    y_train,
    epochs=120,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# 6) Evaluate
probs = model.predict(X_test_scaled)
preds = np.argmax(probs, axis=1)
acc = accuracy_score(y_test, preds)

print("\nTest accuracy:", acc)
print("Class names:", class_names)
print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_)

# 7) Save model and normalization parameters
model.save("models/iris_model.keras")

np.savez(
    "models/iris_scaler_params.npz",
    mean=scaler.mean_.astype(np.float32),
    scale=scaler.scale_.astype(np.float32),
    class_names=class_names
)

print("\nSaved:")
print("models/iris_model.keras")
print("models/iris_scaler_params.npz")