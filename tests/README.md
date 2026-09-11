# CPU-path check for `create_ai_layer.py`

`cpu.cbuild-mlops.yml` mimics the MLOps information of a target without an NPU.
Running the script against it must select the CMSIS-NN kernel variant and embed
the int8 model as trained:

```bash
mkdir -p tests/layer
cp Model/model_int8.tflite Model/model_float.tflite tests/layer/
python3 create_ai_layer.py tests/cpu.cbuild-mlops.yml
grep 'Kernel&CMSIS-NN' tests/layer/model.clayer.yml
```

`tests/layer/` is generated and not committed.
