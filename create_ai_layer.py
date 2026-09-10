#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Create the AI layer of the CMSIS solution from its MLOps information.

    python create_ai_layer.py <solution>.cbuild-mlops.yml

Step 2 of the three-step flow:

    cbuild setup <solution>.csolution.yml --active <target>   # writes *.cbuild-mlops.yml
    python create_ai_layer.py <solution>.cbuild-mlops.yml     # this script
    cbuild <solution>.csolution.yml --active <target>         # compile and link

The *.cbuild-mlops.yml is what CMSIS-Toolbox generates from the `mlops:` node
of the csolution. This script reads the NPU and Vela settings from it and
writes the complete AI layer into the directory of the clayer named under
`model.clayer` (Model/ in this example):

    model.clayer.yml   the TensorFlow Lite Micro kernels for the target: the
                       Ethos-U variant when the target has an NPU, CMSIS-NN
                       otherwise
    model_int8.c       the int8 model as a C array; compiled with Vela for the
                       NPU named in the file, or as trained when there is none
    model_float.c      the float model as a C array (runs on the CPU)

The models come from Model/model_int8.tflite and Model/model_float.tflite,
which Training/train_model.py produces; the extra keys `int8-model` and
`float-model` of the `model:` node (CMSIS-Toolbox 2.14.1+p88 passes them
through) name other files, relative to the layer directory.

The script runs itself in the solution's .venv (see setup_venv.py) when it is
started with an interpreter that has no Vela.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TFLM_PACKS = ["tensorflow-lite-micro", "flatbuffers", "gemmlowp", "kissfft", "ruy"]


def run_in_venv() -> None:
    """Re-run under .venv when Vela is not importable from this interpreter."""
    try:
        import ethosu.vela  # noqa: F401
    except ImportError:
        venv = HERE / ".venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # sys.prefix is the venv directory when running inside it (comparing
        # interpreter paths does not work: venv symlinks resolve to the base).
        if not python.is_file() or Path(sys.prefix).resolve() == venv.resolve():
            sys.exit(
                f"Vela (ethos-u-vela) is not installed for {sys.executable}.\n"
                "Create the venv first: ./setup_venv.sh (Linux/macOS) or setup_venv.bat (Windows)"
            )
        sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


def pack_versions(mlops_file: Path) -> dict[str, str]:
    """The tensorflow pack versions cbuild setup resolved, from <solution>.cbuild-pack.yml.

    The generated layer pins them so that `cbuild --packs` on a fresh checkout
    installs exactly these versions and not whatever the public index lists
    as the newest. Without the file (the CPU fixture in tests/) the packs
    are listed unversioned.
    """
    import yaml

    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    if not pack_file.is_file():
        return {}
    versions = {}
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        vendor, _, pack = name.partition("::")
        if vendor == "tensorflow":
            versions[pack] = version
    return versions


def vela_command(mlops: dict, mlops_dir: Path, model: Path, out_dir: Path) -> list[str]:
    """The Vela command line for the NPU and options named in the mlops file."""
    npu, vela = mlops["npu"], mlops.get("vela", {})
    options = vela.get("options", "")
    if "--accelerator-config" not in options:
        if "macs" not in npu:
            sys.exit(
                f"the mlops file names the NPU {npu['type']} without its MAC count; "
                "add `macs:` to the csolution's mlops.npu node"
            )
        options = f"--accelerator-config {npu['type'].lower()}-{npu['macs']} {options}"
    if vela.get("ini"):
        options += f" --config {mlops_dir / vela['ini']}"
    return [sys.executable, "-m", "ethosu.vela", str(model), "--output-dir", str(out_dir), *shlex.split(options)]


def model_files(mlops: dict, layer_dir: Path) -> tuple[Path, Path]:
    """The int8 and float model files: from the model: node's extra keys, else the defaults."""
    params = mlops["model"]
    for key in params:
        if key not in ("clayer", "name", "int8-model", "float-model"):
            print(f"[ai_layer] warning: ignoring unknown model key {key!r}", file=sys.stderr)
    int8 = layer_dir / params.get("int8-model", "model_int8.tflite")
    float_ = layer_dir / params.get("float-model", "model_float.tflite")
    return int8, float_


def compile_int8(mlops: dict, mlops_dir: Path, layer_dir: Path, model: Path) -> tuple[bytes, str]:
    """The int8 model: Vela-compiled for the NPU, or as trained without one."""
    if "npu" not in mlops:
        return model.read_bytes(), "plain int8 (the target has no NPU)"

    with tempfile.TemporaryDirectory() as tmp:
        cmd = vela_command(mlops, mlops_dir, model, Path(tmp))
        print(f"[ai_layer] + {' '.join(cmd[2:])}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"[ai_layer] Vela failed:\n{result.stdout}\n{result.stderr}")
        counts = dict(re.findall(r"(CPU|NPU) operators = (\d+)", result.stdout))
        summary = f"{counts.get('NPU', '?')} NPU / {counts.get('CPU', '?')} CPU operators"
        if counts.get("CPU", "0") != "0":
            sys.exit(
                f"[ai_layer] Vela left {counts['CPU']} operator(s) on the CPU; the "
                "application only registers the Ethos-U operator for the int8 model.\n"
                + result.stdout
            )
        compiled = Path(tmp) / f"{model.stem}_vela.tflite"
        (layer_dir / compiled.name).write_bytes(compiled.read_bytes())  # kept for inspection
    return compiled.read_bytes() if compiled.exists() else (layer_dir / compiled.name).read_bytes(), (
        f"Vela {cmd[cmd.index('--accelerator-config') + 1]} ({summary})"
    )


def c_array(data: bytes, symbol: str, provenance: str) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in data[i : i + 16]) for i in range(0, len(data), 16)]
    return (
        f"// Generated by create_ai_layer.py -- do not edit. {provenance}.\n"
        f"__attribute__((aligned(16))) const unsigned char {symbol}[] = {{\n  "
        + ",\n  ".join(rows)
        + f"\n}};\nconst unsigned int {symbol}_len = {len(data)};\n"
    )


def clayer(mlops: dict, mlops_file: Path) -> str:
    kernel = "Ethos-U" if "npu" in mlops else "CMSIS-NN"
    versions = pack_versions(mlops_file)
    lines = [
        f"# Generated by create_ai_layer.py from {mlops_file.name} -- do not edit.",
        f"# Re-run `python create_ai_layer.py {mlops_file.name}` after retraining the",
        "# model or changing the csolution's mlops: node.",
        "layer:",
        "  type: AI",
        f"  description: {mlops.get('description', mlops['model'].get('name', 'AI layer'))}",
        "",
        "  packs:",
        *[f"    - pack: tensorflow::{p}" + (f"@{versions[p]}" if p in versions else "") for p in TFLM_PACKS],
        "",
        "  define:",
        "    # CMSIS-NN 8.0.0 compiles its float kernels only with these set, but",
        "    # its NN Lib component lists the sources unconditionally.",
        "    - ARM_NN_ENABLE_F32: 1",
        "    - ARM_NN_ENABLE_F16: 1",
        "",
        "  components:",
        "    - component: tensorflow::Data Exchange:Serialization:flatbuffers",
        "    - component: tensorflow::Data Processing:Math:gemmlowp fixed-point",
        "    - component: tensorflow::Data Processing:Math:kissfft",
        "    - component: tensorflow::Data Processing:Math:ruy",
        f"    - component: tensorflow::Machine Learning:TensorFlow:Kernel&{kernel}",
        "    - component: tensorflow::Machine Learning:TensorFlow:Kernel Utils",
        "    # Both kernel variants compile the CMSIS-NN kernels; the Ethos-U one",
        "    # does not declare the dependency, so select the library explicitly.",
        "    - component: ARM::CMSIS:NN Lib",
        "",
        "  groups:",
        f"    - group: {mlops['model'].get('name', 'Model')}",
        "      files:",
        "        - file: ./model_int8.c",
        "        - file: ./model_float.c",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".cbuild-mlops.yml"):
        sys.exit(f"usage: {Path(__file__).name} <solution>.cbuild-mlops.yml")
    run_in_venv()

    import yaml

    mlops_file = Path(sys.argv[1]).resolve()
    text = mlops_file.read_text()
    if unresolved := sorted(set(re.findall(r"\$[A-Za-z0-9_-]+\$", text))):
        sys.exit(
            f"{mlops_file.name}: unresolved variables {unresolved}.\n"
            "The CMSIS-Toolbox that wrote this file does not expand variables in the "
            "mlops: node (needs a version newer than 2.14.1); update the toolbox or "
            "write the values literally in the csolution."
        )
    mlops = yaml.safe_load(text)["cbuild-mlops"]
    if not mlops.get("model", {}).get("clayer"):
        sys.exit(f"{mlops_file.name}: no model.clayer; set mlops.model.clayer in the csolution")
    layer_file = mlops_file.parent / mlops["model"]["clayer"]
    layer_dir = layer_file.parent

    int8_model, float_model = model_files(mlops, layer_dir)
    for path in (int8_model, float_model):
        if not path.is_file():
            sys.exit(f"{path} is missing; run Training/train_model.py first")
    print(f"[ai_layer] models: {int8_model.name}, {float_model.name}")

    int8, provenance = compile_int8(mlops, mlops_file.parent, layer_dir, int8_model)
    (layer_dir / "model_int8.c").write_text(c_array(int8, "model_int8_tflite", provenance), newline="\n")
    (layer_dir / "model_float.c").write_text(
        c_array(float_model.read_bytes(), "model_float_tflite", "float model as trained"),
        newline="\n",
    )
    layer_file.write_text(clayer(mlops, mlops_file), newline="\n")

    print(f"[ai_layer] int8 model: {len(int8)} bytes, {provenance}")
    print(f"[ai_layer] wrote {layer_file}")


if __name__ == "__main__":
    main()
