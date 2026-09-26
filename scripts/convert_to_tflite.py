import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/iris_model.keras"
TFLITE_PATH = "models/iris_model.tflite"
SCALER_PATH = "models/iris_scaler_params.npz"

class_names = ["setosa", "versicolor", "virginica"]

# 1) Load Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# 2) Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3) Save TFLite model
os.makedirs("models", exist_ok=True)
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved: {TFLITE_PATH}")
print(f"TFLite model size: {len(tflite_model)} bytes")

# 4) Quick TFLite test on required samples
params = np.load(SCALER_PATH)
mean = params["mean"].astype(np.float32)
scale = params["scale"].astype(np.float32)

test_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.0, 2.2, 4.0, 1.0],  # versicolor
    [6.9, 3.1, 5.4, 2.1],  # virginica
], dtype=np.float32)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput details:", input_details)
print("Output details:", output_details)

print("\nTFLite test results:")
for sample in test_samples:
    x_scaled = ((sample - mean) / scale).astype(np.float32)
    x_scaled = np.expand_dims(x_scaled, axis=0)

    interpreter.set_tensor(input_details[0]["index"], x_scaled)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_index = int(np.argmax(output))
    predicted_class = class_names[predicted_index]

    print("-------------------------")
    print("Raw input:", sample)
    print("Probabilities:", output)
    print("Predicted class:", predicted_class)