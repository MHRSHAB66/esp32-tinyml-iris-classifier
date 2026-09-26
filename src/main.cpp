#include <Arduino.h>
#include "iris_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

HardwareSerial IO(1);

namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 16 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  const char* class_names[] = {
    "setosa",
    "versicolor",
    "virginica"
  };
}

bool parseInput(String line, float values[4]) {
  line.trim();

  char buffer[80];
  line.toCharArray(buffer, sizeof(buffer));

  int parsed = sscanf(
    buffer,
    "%f,%f,%f,%f",
    &values[0],
    &values[1],
    &values[2],
    &values[3]
  );

  return parsed == 4;
}

void printHelp() {
  IO.println();
  IO.println("TinyML Iris Classification on ESP32");
  IO.println();
  IO.println("Enter Iris features like this:");
  IO.println("5.1,3.5,1.4,0.2");
  IO.println();
}

void setup() {
  IO.begin(115200, SERIAL_8N1, 16, 17);
  delay(1000);

  IO.println("BOOT OK");
  IO.println("Serial1 started on GPIO16/GPIO17");

  model = tflite::GetModel(tflite_model_iris);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    IO.print("Model schema version mismatch. Model: ");
    IO.print(model->version());
    IO.print(", Supported: ");
    IO.println(TFLITE_SCHEMA_VERSION);
    while (true) {
      delay(1000);
    }
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter
  );

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    IO.println("AllocateTensors failed.");
    while (true) {
      delay(1000);
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  IO.println("Model loaded successfully.");
  IO.print("Input tensor type: ");
  IO.println(input->type);
  IO.print("Output tensor type: ");
  IO.println(output->type);

  printHelp();
}

void loop() {
  if (!IO.available()) {
    delay(10);
    return;
  }

  String line = IO.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) {
    return;
  }

  IO.print("Received: ");
  IO.println(line);
  IO.println("-------------------------");

  float raw_input[4];

  if (!parseInput(line, raw_input)) {
    IO.println("Invalid input format.");
    IO.println("Use this format:");
    IO.println("5.1,3.5,1.4,0.2");
    IO.println("-------------------------");
    IO.println();
    return;
  }

  IO.print("Raw input: ");
  for (int i = 0; i < 4; i++) {
    IO.print(raw_input[i], 4);
    if (i < 3) {
      IO.print(", ");
    }
  }
  IO.println();

  for (int i = 0; i < 4; i++) {
    input->data.f[i] = (raw_input[i] - IRIS_MEAN[i]) / IRIS_SCALE[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    IO.println("Inference failed.");
    IO.println("-------------------------");
    IO.println();
    return;
  }

  int predicted_index = 0;
  float max_prob = output->data.f[0];

  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > max_prob) {
      max_prob = output->data.f[i];
      predicted_index = i;
    }
  }

  IO.print("Probabilities: ");
  IO.print("setosa=");
  IO.print(output->data.f[0], 4);
  IO.print(", versicolor=");
  IO.print(output->data.f[1], 4);
  IO.print(", virginica=");
  IO.println(output->data.f[2], 4);

  IO.print("Predicted class: ");
  IO.println(class_names[predicted_index]);

  IO.println("-------------------------");
  IO.println();
  IO.println("Enter another sample:");
}