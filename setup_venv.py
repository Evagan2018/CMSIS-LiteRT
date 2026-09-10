#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Create the Python venvs: .venv (Vela, for create_ai_layer.py) and, with
# --training, Training/.venv (TensorFlow, for Training/train_model.py).
#
# Runs on Linux, macOS and Windows. The thin wrappers setup_venv.sh and
# setup_venv.bat just delegate here; everything OS-specific lives in this file.
"""Create (or repair) the venv used to compile the model (or, with --training, to train it)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The two environments. Vela runs on any current Python; TensorFlow 2.17 ships
# wheels for 3.9 to 3.12 only, which is why training gets its own venv.
VENVS = {
    "vela": (HERE / ".venv", HERE / "requirements.txt", (3, 10), (3, 15)),
    "training": (HERE / "Training" / ".venv", HERE / "Training" / "requirements.txt", (3, 9), (3, 13)),
}


def venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside a venv, on any host OS.

    Windows puts it in Scripts/python.exe, everyone else in bin/python. This is
    the one place that difference is encoded; run_in_venv() in
    create_ai_layer.py makes the same choice.
    """
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def check_host_python(what: str, lo: tuple[int, int], hi: tuple[int, int]) -> None:
    if not (lo <= sys.version_info[:2] < hi):
        have = ".".join(str(n) for n in sys.version_info[:3])
        sys.exit(
            f"error: {what} needs Python >={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]}; this is {have}\n"
            f"  ({sys.executable})\n"
            "Re-run with a supported interpreter, e.g.\n"
            "  PYTHON=python3.12 ./setup_venv.sh        (Linux/macOS)\n"
            "  py -3.12 setup_venv.py                   (Windows)"
        )


def venv_is_usable(venv_dir: Path) -> bool:
    """True if the venv exists and its interpreter still runs.

    A venv outlives the interpreter it was created from (a Python upgrade, a
    removed pyenv version): the directory is there but the symlinks and
    lib/pythonX.Y paths are stale.
    """
    python = venv_python(venv_dir)
    if not python.is_file():
        return False
    try:
        subprocess.run(
            [str(python), "-c", "import sys"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def python_version(value: str) -> tuple[int, ...]:
    """Accept a major.minor version, optionally with a patch version; the range is checked later."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value):
        raise argparse.ArgumentTypeError("expected a Python version such as 3.12 or 3.12.10")
    return tuple(int(n) for n in value.split("."))


def check_venv_python(python: Path, what: str, lo: tuple[int, int], hi: tuple[int, int], requested) -> None:
    """The venv's interpreter must be in range and, with --python, the one asked for."""
    result = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    )
    have = result.stdout.strip()
    version = tuple(int(n) for n in have.split("."))
    if not (lo <= version[:2] < hi):
        sys.exit(
            f"error: {python} uses Python {have}; {what} needs >={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]}. "
            "Re-run with --recreate and a supported Python version."
        )
    if requested and version[: len(requested)] != requested:
        want = ".".join(map(str, requested))
        sys.exit(
            f"error: the venv uses Python {have}, but --python {want} was requested. "
            "Re-run with --recreate to change the environment's Python version."
        )


def pip(python: Path, *args: str, uv: str | None = None, env: dict[str, str] | None = None) -> None:
    if uv:
        cmd = [uv, "pip", *args, "--python", str(python)]
    else:
        cmd = [str(python), "-m", "pip", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def smoke_test(python: Path, what: str) -> None:
    """Run the tool the venv is for, so a broken install shows up here."""
    if what == "vela":
        subprocess.run([str(python), "-m", "ethosu.vela", "--version"], check=True)
    else:
        subprocess.run([str(python), "-c", "import tensorflow as tf; print('TensorFlow', tf.__version__)"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training",
        action="store_true",
        help="create Training/.venv (TensorFlow) for Training/train_model.py instead of .venv (Vela)",
    )
    parser.add_argument(
        "--uv",
        action="store_true",
        help="create the environment with `uv venv` and install with `uv pip` (needs uv on PATH)",
    )
    parser.add_argument(
        "--python",
        metavar="VERSION",
        type=python_version,
        help=(
            "Python version for uv, e.g. 3.12 or 3.12.10 (needs --uv; default: the "
            "interpreter running this script, or uv's own pick through the wrappers)"
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete and rebuild the venv even if it looks usable",
    )
    args = parser.parse_args()

    what = "training" if args.training else "vela"
    tool = "TensorFlow" if args.training else "Vela"
    venv_dir, requirements, lo, hi = VENVS[what]
    if args.python and not args.uv:
        parser.error("--python requires --uv")
    if args.python and not (lo <= args.python[:2] < hi):
        parser.error(f"{tool} needs Python >={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]}")
    uv = shutil.which("uv") if args.uv else None
    if args.uv and not uv:
        parser.error(
            "--uv requires uv on PATH; install it from "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    if not args.python:
        check_host_python(tool, lo, hi)

    if args.recreate and venv_dir.exists():
        print(f"Removing {venv_dir}")
        shutil.rmtree(venv_dir)

    if venv_dir.exists() and not venv_is_usable(venv_dir):
        print(f"{venv_dir} exists but its interpreter does not run; recreating.")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"Creating venv at {venv_dir}")
        if uv:
            # The wrapper may run this script in uv's temporary isolated
            # environment: request a version, not a path inside that environment.
            requested = ".".join(map(str, args.python or sys.version_info[:3]))
            cmd = [uv, "venv", "--python", requested, str(venv_dir)]
            print(f"+ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
        else:
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(venv_dir)

    python = venv_python(venv_dir)
    check_venv_python(python, tool, lo, hi, args.python)
    if not uv:
        pip(python, "install", "--upgrade", "pip")
    pip(python, "install", "-r", str(requirements), uv=uv)
    smoke_test(python, what)

    print()
    print(f"venv ready: {venv_dir}")
    if args.training:
        print(f"Train the model with:\n  {python} Training/train_model.py")
    else:
        print("create_ai_layer.py runs itself with this interpreter:")
        print(f"  {python}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
