"""Compatibility entry point; use the adjacent locomanilab module."""

from importlib import import_module
from pathlib import Path
import sys

_DIRECTORY = str(Path(__file__).resolve().parent)
if _DIRECTORY not in sys.path:
    sys.path.insert(0, _DIRECTORY)
_module = import_module("locomanilab_heft_to_psi_lerobot")
globals().update({key: value for key, value in vars(_module).items() if not key.startswith("__")})

if __name__ == "__main__":
    print(f"Deprecated entry point: use {_module.__name__}.py", file=sys.stderr)
    _module.main()
