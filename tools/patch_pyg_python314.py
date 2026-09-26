#!/usr/bin/env python3

"""Patch PyTorch Geometric 2.8.0.post1 for Python 3.14.

PyG 2.8.0.post1 unconditionally applies torch.jit.script() to
SelectOutput and ConnectOutput at import time. PyTorch does not support
torch.jit.script() on Python 3.14+.

This patch preserves the original behavior on Python < 3.14 and leaves
the output dataclasses unscripted on Python 3.14+.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path


SUPPORTED_PYG_VERSION = "2.8.0.post1"

PATCHES = (
    (
        "nn/pool/select/base.py",
        "SelectOutput = torch.jit.script(SelectOutput)",
    ),
    (
        "nn/pool/connect/base.py",
        "ConnectOutput = torch.jit.script(ConnectOutput)",
    ),
)


def get_pyg_root() -> Path:
    spec = importlib.util.find_spec("torch_geometric")

    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("torch_geometric is not installed")

    return Path(next(iter(spec.submodule_search_locations))).resolve()


def patch_file(path: Path, expression: str) -> str:
    text = path.read_text()

    guarded = (
        "if sys.version_info < (3, 14):\n"
        f"    {expression}"
    )

    if guarded in text:
        return "already patched"

    if expression not in text:
        raise RuntimeError(
            f"Expected PyG source expression not found in {path}:\n"
            f"    {expression}\n"
            "Refusing to patch an unknown source layout."
        )

    if "import sys\n" not in text:
        text = "import sys\n" + text

    text = text.replace(expression, guarded, 1)
    path.write_text(text)

    return "patched"


def main() -> int:
    if sys.version_info < (3, 14):
        print(
            "Python < 3.14: PyG compatibility patch is not required."
        )
        return 0

    version = importlib.metadata.version("torch-geometric")

    if version != SUPPORTED_PYG_VERSION:
        raise RuntimeError(
            "This compatibility patch is validated only for "
            f"torch-geometric {SUPPORTED_PYG_VERSION}; "
            f"installed version is {version}."
        )

    root = get_pyg_root()

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyG:    {version}")
    print(f"Root:   {root}")

    for relative_path, expression in PATCHES:
        path = root / relative_path

        if not path.is_file():
            raise RuntimeError(f"Expected PyG source file not found: {path}")

        result = patch_file(path, expression)
        print(f"{result}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
