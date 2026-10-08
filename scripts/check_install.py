"""Post-installation check: torch, jax and conduit.Node must import from the active virtual environment."""

import importlib
import sys
from pathlib import Path


def in_venv(path: str) -> bool:
    return Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())


def check(name: str) -> bool:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        return False
    location = getattr(module, "__file__", None) or ""
    where = "venv" if in_venv(location) else "OUTSIDE venv"
    print(f"{'ok  ' if where == 'venv' else 'WARN'}  {name} {getattr(module, '__version__', '')} ({where}: {location})")
    return where == "venv"


def main() -> int:
    print(f"venv: {sys.prefix}")
    if sys.prefix == sys.base_prefix:
        print("FAIL  no virtual environment is active")
        return 1
    ok = all([check("torch"), check("jax"), check("conduit")])
    try:
        import conduit

        conduit.Node()
        print("ok    conduit.Node()")
    except Exception as exc:
        print(f"FAIL  conduit.Node: {type(exc).__name__}: {exc}")
        ok = False
    try:
        import torch

        print(f"info  torch CUDA available: {torch.cuda.is_available()}")
        import jax

        print(f"info  jax devices: {jax.devices()}")
    except Exception as exc:
        print(f"info  device query failed: {exc}")
    print("All checks passed." if ok else "Some checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
