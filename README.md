# TensorFlow Lite Micro "Hello World" on Ethos-U

This example runs the TensorFlow Lite Micro (LiteRT) "Hello World" sine model on
the Arm **Ethos-U85** NPU of the **Corstone-320 (SSE-320)** FVP. It is the
[Hello World reference application](https://github.com/MDK-Packs/tensorflow-pack/tree/main/tensorflow-build/add/examples/TFLiteRT_HelloWorld)
of the [`tensorflow::tensorflow-lite-micro`](https://www.keil.arm.com/packs/tensorflow-lite-micro-tensorflow/overview/)
CMSIS pack, rebuilt around a three-step MLOps flow: the CMSIS-Toolbox describes
the target, a script turns that into the AI layer, and the toolbox builds the
application.

## What the example demonstrates

- The NPU and its Vela configuration live in the CMSIS solution (`mlops:` node), not in Python.
- `cbuild setup` hands them to the MLOps side as `cmsis-litert.cbuild-mlops.yml`.
- `create_ai_layer.py` compiles the int8 model with Vela for exactly that NPU and writes the
  complete AI layer: the TensorFlow Lite Micro kernel variant for the target and the model data.
- The same application source runs the int8 model on the NPU or, for a target without one, on the
  CMSIS-NN kernels: `AddEthosU()` registers the Ethos-U operator only when the Ethos-U kernels are built.
- Training is a separate, scripted step that produces committed `.tflite` files.

## Prerequisites

- Python `>=3.10` for the build (Vela); Python `3.9` to `3.12` only if you want to retrain (TensorFlow 2.17).
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- Tools listed in [`vcpkg-configuration.json`](./vcpkg-configuration.json) (CMSIS-Toolbox 2.14.1, Arm Compiler 6, Corstone-320 FVP).
- Keil Studio manages the required license; the free Keil MDK Community edition can be used for evaluation.

## Quick start

1. Install [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack).
2. Clone this repository and open its folder in VS Code.
3. Before using the example for the first time, select **Terminal > Run Task >
   Setup Python virtual environment**. It creates `.venv` with Vela.
4. Select **Terminal > Run Task > Create AI layer**. This compiles the model for
   the NPU of the active target and writes `Model/`. (The repository ships a
   generated layer, so this is only needed after retraining or retargeting.)
5. Use the CMSIS action buttons to build the application, then select **Run**.
   Keil Studio starts the Corstone-320 FVP automatically.

A successful run prints the Ethos-U configuration, the profile of the float
model, both inferences and a pass result:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
Tensorflow LiteRT Hello World!
(INFO) Profile Memory and Latency
...
(INFO) Load Float Model and Perform Inference (CPU)
Input [0.000] = 0.036 / Delta 0.036
...
(INFO) Load Quantized Model and Perform Inference (Ethos-U)
Input [0.770] = 0.731 / Delta 0.034
...
~~~ALL TESTS PASSED~~~
```

### Command-line build

The same workflow from a shell with the tools from `vcpkg-configuration.json` on
the path is three commands plus the one-time venv setup.

#### 0. Create the Python environment (once)

```bash
./setup_venv.sh          # Linux/macOS
.\setup_venv.bat         # Windows
```

This creates `.venv/` with Vela and PyYAML, all that `create_ai_layer.py`
needs. It is safe to run again; `--recreate` starts from scratch.

#### 1. Generate the MLOps information

```bash
cbuild setup cmsis-litert.csolution.yml --active SSE-320-U85 --packs --update-rte
```

This resolves the packs and the active target and writes
`cmsis-litert.cbuild-mlops.yml`: the processor, NPU and Vela options of the
target and the location of the AI layer. (`--packs` and `--update-rte` are only
needed on a fresh checkout. On a checkout without a generated `Model/` layer the
command reports the missing layer but still writes the file.)

#### 2. Create the AI layer

```bash
python3 create_ai_layer.py cmsis-litert.cbuild-mlops.yml
```

This is the MLOps step. The script reads the NPU and Vela settings from the
file, compiles `Model/model_int8.tflite` with Vela for that NPU, and writes the
complete AI layer into `Model/`: the kernel selection and both models as C
arrays. It runs itself in `.venv` when started with another interpreter (use
`python` on Windows).

#### 3. Build the application

```bash
cbuild cmsis-litert.csolution.yml --active SSE-320-U85
```

A plain CMSIS build; no Python is involved. The resulting image is:

```text
out/cmsis-litert/SSE-320-U85/Debug/cmsis-litert.axf
```

#### 4. Run on the FVP

```bash
FVP_Corstone_SSE-320 \
    -f board/Corstone-320/fvp_config.txt \
    -a out/cmsis-litert/SSE-320-U85/Debug/cmsis-litert.axf
```

The application ends the simulation when it is done.

## How model generation works

The target is described by the `mlops:` node in
`cmsis-litert.csolution.yml`:

```yaml
mlops:
  npu:
    type: Ethos-U85
    macs: 256
  vela:
    system: Ethos_U85_SYS_DRAM_Mid
    memory: Shared_Sram
  model:
    clayer: ./Model/model.clayer.yml
    name: HelloWorld
```

`cbuild setup --active SSE-320-U85` resolves it into
`cmsis-litert.cbuild-mlops.yml`, which carries the Vela command line
(`--accelerator-config ethos-u85-256 --system-config ... --memory-mode ...`).
`create_ai_layer.py` passes that to Vela, so the target configuration is never
duplicated in Python. The script then writes:

- `Model/model.clayer.yml`: the TensorFlow Lite Micro components for the target,
  `Kernel&Ethos-U` when the file names an NPU, `Kernel&CMSIS-NN` otherwise.
- `Model/model_int8.c`: the int8 model as a C array, Vela-compiled for the NPU.
- `Model/model_float.c`: the float model as a C array (runs on the CPU).

See [the MLOps flow](documentation/mlops-flow.md) for a detailed walkthrough.

## Training the model

The models in `Model/*.tflite` are committed. To retrain:

```bash
PYTHON=python3.12 ./setup_venv.sh --training   # Training/.venv with TensorFlow 2.17
Training/.venv/bin/python Training/train_model.py
```

`Training/train_model.py` is the pack notebook's training as a script: it
trains the sine model with a fixed seed, converts it to a float and a
fully-integer int8 `.tflite`, and writes both into `Model/`. Then re-run steps 2
and 3.

## Adapting the example

To use a different model, change `Training/train_model.py` (or drop your own
`Model/model_int8.tflite` and `Model/model_float.tflite` in place), adapt the
operator registration in `Source/hello_world_test.cpp`, and re-run steps 2 and 3.

To target another Ethos-U configuration, update the target and `mlops:`
settings in the CMSIS solution and re-run all three steps. The Vela options then
follow that configuration automatically. Moving to a different board also
requires the corresponding device pack, board layer with the matching Ethos-U
driver, and FVP configuration.

## Project layout

| Path | Purpose |
|------|---------|
| `cmsis-litert.csolution.yml` | Solution, target, and MLOps configuration |
| `cmsis-litert.cproject.yml` | Application project: source plus the Board and AI layers |
| `create_ai_layer.py` | Compiles the model for the target and writes the AI layer |
| `Model/` | The trained `.tflite` models and the generated AI layer |
| `Training/train_model.py` | Trains and exports the sine model |
| `setup_venv.py` (`.sh` / `.bat`) | Creates the Python environments |
| `board/Corstone-320/` | Corstone-320 board support, Ethos-U driver setup and FVP configuration |
| `Source/hello_world_test.cpp` | Runs both models and prints the result |
| `documentation/` | Detailed MLOps and cross-platform notes |

## Known limitations

- The supplied platform configuration targets Corstone-320 with Ethos-U85;
  another target needs its corresponding platform integration.
- On silicon with a data cache, the tensor arena needs cache maintenance around
  the NPU invocation; the FVP is cache-transparent.

## License

The example code is licensed under Apache-2.0; see `LICENSE`. TensorFlow uses
the Apache-2.0 license.

## References

- [TensorFlow Lite Micro CMSIS Pack](https://www.keil.arm.com/packs/tensorflow-lite-micro-tensorflow/overview/)
- [TensorFlow Lite for Microcontrollers](https://ai.google.dev/edge/litert/microcontrollers/overview)
- [Vela compiler](https://pypi.org/project/ethos-u-vela/)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
