/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Copyright 2026 Arm Limited and/or its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TensorFlow Lite Micro "Hello World": runs the sine model embedded by the AI
// layer (Model/model_float.c, Model/model_int8.c), first the float model on
// the CPU, then the int8 model -- on the Ethos-U when the AI layer was
// generated for an NPU (create_ai_layer.py compiled it with Vela), on the CPU
// otherwise. The board layer provides main(), stdout and the Ethos-U driver
// init and then calls app_main().

#include <math.h>
#include <stdio.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const unsigned char model_int8_tflite[];
extern "C" const unsigned char model_float_tflite[];

namespace {

// The sine model needs one operator on the CPU (FullyConnected). A model
// compiled with Vela consists of a single Ethos-U custom operator instead;
// AddEthosU() registers it when the Ethos-U kernel variant is built and is a
// no-op otherwise, so one resolver serves every AI layer variant.
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<2>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEthosU());
  return kTfLiteOk;
}

// One arena, used by one interpreter at a time. Static and 16-byte aligned:
// the Ethos-U reads its inputs and scratch from here, so it must not live on
// the stack of a core-local memory.
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int kNumTestValues = 4;
constexpr float kTolerance = 0.25f;  // |sin(x) - prediction| accepted

// The int8 model is either the trained model or its Vela compilation; the
// latter is one custom operator that only the Ethos-U kernel implements.
const char* Int8ModelRunsOn() {
  return tflite::Register_ETHOSU() != nullptr ? "Ethos-U" : "CPU";
}

}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  // MicroProfiler has room for 4096 events (~80 KB): keep it off the stack.
  static tflite::MicroProfiler profiler;
  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  constexpr int kNumResourceVariables = 24;
  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(model_float_tflite), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference(float& max_delta) {
  const tflite::Model* model = ::tflite::GetModel(model_float_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  const float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};
  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter.input(0)->data.f[0] = golden_inputs[i];
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    const float y_pred = interpreter.output(0)->data.f[0];
    const float delta = fabsf(sinf(golden_inputs[i]) - y_pred);
    if (delta > max_delta) max_delta = delta;
    printf("Input [%.3f] = %.3f / Delta %.3f\n", golden_inputs[i], y_pred, delta);
  }
  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference(float& max_delta) {
  const tflite::Model* model = ::tflite::GetModel(model_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  // Quantization parameters come from the model, so retraining it (which
  // changes the scales) needs no change here.
  const float input_scale = input->params.scale;
  const int input_zero_point = input->params.zero_point;
  const float output_scale = output->params.scale;
  const int output_zero_point = output->params.zero_point;

  const float golden_inputs[kNumTestValues] = {0.77f, 1.57f, 2.3f, 3.14f};
  for (int i = 0; i < kNumTestValues; ++i) {
    const float q = roundf(golden_inputs[i] / input_scale) + input_zero_point;
    input->data.int8[0] = static_cast<int8_t>(q < -128.f ? -128.f : (q > 127.f ? 127.f : q));
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    const float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
    const float delta = fabsf(sinf(golden_inputs[i]) - y_pred);
    if (delta > max_delta) max_delta = delta;
    printf("Input [%.3f] = %.3f / Delta %.3f\n", golden_inputs[i], y_pred, delta);
  }
  return kTfLiteOk;
}

extern "C" int app_main(void) {
  printf("Tensorflow LiteRT Hello World!\n");
  tflite::InitializeTarget();
  // Without a callback TFLM's own messages (allocation failures, kernel
  // errors, the profile below) are dropped on Cortex-M.
  RegisterDebugLogCallback([](const char* s) { printf("%s", s); });

  float max_delta = 0.f;

  printf("(INFO) Profile Memory and Latency\n");
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());

  printf("(INFO) Load Float Model and Perform Inference (CPU)\n");
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference(max_delta));

  printf("(INFO) Load Quantized Model and Perform Inference (%s)\n", Int8ModelRunsOn());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference(max_delta));

  if (max_delta <= kTolerance) {
    printf("~~~ALL TESTS PASSED~~~\n");
  } else {
    printf("~~~TESTS FAILED~~~ (max delta %.3f > %.2f)\n", max_delta, kTolerance);
  }
  printf("\x04");  // EOT: ends the FVP run (semihosting exit); a board just sees a 0x04
  fflush(stdout);
  return max_delta <= kTolerance ? 0 : 1;
}
