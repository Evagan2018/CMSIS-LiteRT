# LiteRT for Microcontrollers "Hello World" on Ethos-U

This CMSIS reference application runs the LiteRT for Microcontrollers
(formerly TensorFlow Lite Micro) "Hello World" sine model: once as a float
model on the CPU, once as an int8 model on the Arm **Ethos-U** NPU when the
target has one, on the CMSIS-NN kernels otherwise. It is the
[Hello World example](https://github.com/MDK-Packs/tensorflow-pack/tree/main/tensorflow-build/add/examples/TFLiteRT_HelloWorld)
of the [`tensorflow::tensorflow-lite-micro`](https://www.keil.arm.com/packs/tensorflow-lite-micro-tensorflow/overview/)
CMSIS pack, rebuilt around a three-step MLOps flow: the CMSIS-Toolbox describes
the target, a script turns that into the AI layer, and the toolbox builds the
application. The tested target is the **Corstone-320 (SSE-320)** FVP with its
**Ethos-U85**, whose board layer ships in `board/Corstone-320/`.

## What the example demonstrates

- The NPU and its Vela configuration live in the CMSIS solution (`mlops:` node), not in Python.
- `cbuild setup` hands them to the MLOps side as `cmsis-litert.cbuild-mlops.yml`.
- `create_ai_layer.py` compiles the int8 model with Vela for exactly that NPU and writes the
  complete AI layer: the TensorFlow Lite Micro kernel variant for the target and the model data.
- The same application source runs the int8 model on the NPU or, for a target without one, on the
  CMSIS-NN kernels: `AddEthosU()` registers the Ethos-U operator only when the Ethos-U kernels are built.
- Training is a separate, scripted step that produces committed `.tflite` files.

## Boards

The application consumes `STDOUT` (and 64 KB of heap) from a board layer, as
declared in `cmsis-litert.cproject.yml`:

- **Corstone-320 FVP** (`SSE-320-U85`): shipped, tested, built by CI. The layer
  in `board/Corstone-320/` provides the console over semihosting, the Ethos-U85
  driver and its initialisation.
- **Other boards**: in the CMSIS view open **Manage Solution**, add a
  target-type for the board and pick one of the board layers the installed
  packs offer for it (the extension lists the layers that provide `STDOUT`).
  On a board without an NPU the int8 model runs on the CMSIS-NN kernels; leave
  `npu:` and `vela:` out of the `mlops:` node, or let them default from a
  device pack that describes its NPU. A board with an Ethos-U needs a layer
  that also selects the Ethos-U driver and initialises the NPU before
  `app_main()`, as `board/Corstone-320/ethos_setup.c` does. The Alif Ensemble
  pack ships one for the DevKit-E8 (`Boards/DevKit-e8/Layers/M55_HP/
  Board_HP-U85.clayer.yml`: STDOUT, Ethos-U85 driver, `ethos_setup.c`); it is
  untested with this application.

## Prerequisites

- Python `3.10` to `3.14` for the build (Vela); Python `3.9` to `3.12` only if you want to retrain (TensorFlow 2.17).
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- Tools listed in [`vcpkg-configuration.json`](./vcpkg-configuration.json) (CMSIS-Toolbox 2.14.1, Arm Compiler 6, Corstone-320 FVP); the Arm Tools Environment Manager installs them when the project is opened.
- Keil Studio manages the required license; the free Keil MDK Community edition can be used for evaluation.

## Quick start

1. Install [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack).
2. Clone this repository and open its folder in VS Code.
3. Before using the example for the first time, select **Terminal > Run Task >
   Setup Python virtual environment**. It creates `.venv` with Vela. (The
   **(uv)** variant of the task uses [uv](https://docs.astral.sh/uv/) and can
   download the Python version it asks for.)
4. Select **Terminal > Run Task > Create AI layer**. This compiles the model for
   the NPU of the active target and writes `Model/`. (The repository ships a
   generated layer, so this is only needed after retraining or retargeting.)
5. Use the CMSIS action buttons to build the application, then select **Run**
   or **Debug**. Keil Studio starts the Corstone-320 FVP automatically. On
   macOS, where Arm ships no FVP build, `.vscode/fvp.sh` runs the model in
   Docker: Docker Desktop must be running, and the first Run or Debug builds
   the container image (about 100 MB download). On Windows, set `model:` in
   the csolution's target-set back to `FVP_Corstone_SSE-320` (the shim is a
   bash script).

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
Input [0.000] = 0.001 / Delta 0.001
Input [1.000] = 0.830 / Delta 0.012
Input [3.000] = 0.096 / Delta 0.045
Input [5.000] = -0.955 / Delta 0.004
(INFO) Load Quantized Model and Perform Inference (Ethos-U)
Input [0.770] = 0.715 / Delta 0.018
Input [1.570] = 0.987 / Delta 0.013
Input [2.300] = 0.722 / Delta 0.023
Input [3.140] = -0.023 / Delta 0.025
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
needs. It is safe to run again; `--recreate` starts from scratch. The wrappers
use `python3` (`python` on Windows); point them at another interpreter with
`PYTHON=python3.12 ./setup_venv.sh`. With
[uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH`,
`./setup_venv.sh --uv --python 3.12` creates the environment with `uv venv`
for that Python version, downloading the interpreter if needed, and installs
with `uv pip`.

#### 1. Generate the MLOps information

```bash
cbuild setup cmsis-litert.csolution.yml --active SSE-320-U85 --packs
```

This resolves the packs and the active target and writes
`cmsis-litert.cbuild-mlops.yml`: the processor, NPU and Vela options of the
target and the location of the AI layer. (`--packs` installs missing packs and
is only needed on a fresh checkout; the layers' RTE configuration is committed,
so no `--update-rte` is required. On a checkout without a generated `Model/`
layer the command reports the missing layer but still writes the file.)

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
.vscode/fvp.sh \
    -f board/Corstone-320/fvp_config.txt --simlimit 60 \
    -a out/cmsis-litert/SSE-320-U85/Debug/cmsis-litert.axf
```

`.vscode/fvp.sh` is the model command the Run and Debug buttons use too; on
Linux and Windows `FVP_Corstone_SSE-320` can be called directly with the same
arguments. The application ends the simulation when it is done, and the model
exits with status 0 either way, so check the verdict in the output as CI does:

```bash
.vscode/fvp.sh -f board/Corstone-320/fvp_config.txt --simlimit 60 \
    -a out/cmsis-litert/SSE-320-U85/Debug/cmsis-litert.axf | tee fvp_stdout.log
grep "~~~ALL TESTS PASSED~~~" fvp_stdout.log
```

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
trains the sine model with fixed seeds (Python, NumPy and TensorFlow, so two
runs give the same models), converts it to a float and a fully-integer int8
`.tflite`, checks both against `sin(x)` at the inputs the firmware tests with
the firmware's tolerance, and only then writes them into `Model/`. A run whose
models would fail on the target exits with status 1 and leaves `Model/` as it
was. Then re-run steps 2 and 3.

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
| `.vscode.d/tasks.json` | The VS Code tasks (venv setup, training, Create AI layer) merged by the CMSIS Solution extension |
| `.vscode/fvp.sh`, `.vscode/fvp.Dockerfile` | The FVP model command used by Run and Debug; runs the model in Docker on macOS |
| `board/Corstone-320/` | Corstone-320 board support, Ethos-U driver setup and FVP configuration |
| `Source/hello_world_test.cpp` | Runs both models and prints the result |
| `tests/` | The CPU-target fixture CI uses to check the CMSIS-NN variant of the layer |
| `documentation/mlops-flow.md` | The MLOps flow in detail |

## Known limitations

- Only the Corstone-320 target ships with a board layer; an Ethos-U board
  needs a layer with the NPU driver and initialisation (see Boards).
- On silicon with a data cache, the tensor arena needs cache maintenance around
  the NPU invocation; the FVP is cache-transparent.
- The solution selects CMSIS-NN 8.0.0 while `tensorflow-lite-micro@1.26.2`
  declares CMSIS-NN 7.x, so every build prints a pack-version warning. The
  combination is tested here; 8.0.0 needs the `ARM_NN_ENABLE_F32`/`F16`
  defines in the generated layer to compile (see issue #1).

## License

The example code is licensed under Apache-2.0; see `LICENSE`. TensorFlow and
LiteRT use the Apache-2.0 license.

## References

- [TensorFlow Lite Micro CMSIS Pack](https://www.keil.arm.com/packs/tensorflow-lite-micro-tensorflow/overview/)
- [TensorFlow Lite for Microcontrollers](https://ai.google.dev/edge/litert/microcontrollers/overview)
- [Vela compiler](https://pypi.org/project/ethos-u-vela/)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
