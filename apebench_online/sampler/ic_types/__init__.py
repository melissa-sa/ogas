import importlib
import logging

from .registry import register_ic_maker, get_ic_maker_class, available_ic_types

logger = logging.getLogger("melissa")

_BUILTIN_MODULES = (
    "sine",
    "fourier",
    "gaussian_blobs",
)
_BUILTINS_LOADED = False


def ensure_builtin_ic_types_loaded():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for module in _BUILTIN_MODULES:
        module_name = f"{__name__}.{module}"
        importlib.import_module(module_name)

    _BUILTINS_LOADED = True


__all__ = [
    "register_ic_maker",
    "get_ic_maker_class",
    "available_ic_types",
    "ensure_builtin_ic_types_loaded",
]
