import os
import numpy as np

TFLITE_PATH = "models/iris_model.tflite"
SCALER_PATH = "models/iris_scaler_params.npz"
HEADER_PATH = "include/iris_model.h"

os.makedirs("include", exist_ok=True)

with open(TFLITE_PATH, "rb") as f:
    model_bytes = f.read()

params = np.load(SCALER_PATH)
mean = params["mean"].astype(np.float32)
scale = params["scale"].astype(np.float32)

def bytes_to_c_array(data, values_per_line=12):
    lines = []
    for i in range(0, len(data), values_per_line):
        chunk = data[i:i + values_per_line]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append("  " + line)
    return ",\n".join(lines)

def floats_to_c_array(values):
    return ", ".join(f"{float(v):.8f}f" for v in values)

header = f"""#ifndef IRIS_MODEL_H
#define IRIS_MODEL_H

const unsigned char tflite_model_iris[] = {{
{bytes_to_c_array(model_bytes)}
}};

const unsigned int tflite_model_iris_len = {len(model_bytes)};

const float IRIS_MEAN[4] = {{ {floats_to_c_array(mean)} }};
const float IRIS_SCALE[4] = {{ {floats_to_c_array(scale)} }};

// Aliases, in case main.cpp uses these names
#define MEAN_IRIS IRIS_MEAN
#define SCALE_IRIS IRIS_SCALE

#endif
"""

with open(HEADER_PATH, "w", encoding="utf-8") as f:
    f.write(header)

print(f"Saved: {HEADER_PATH}")
print(f"Model bytes: {len(model_bytes)}")
print("IRIS_MEAN:", mean)
print("IRIS_SCALE:", scale)