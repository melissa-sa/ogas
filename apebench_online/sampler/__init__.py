from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from apebench_online.core.constants import VALIDATION_DIR, VALIDATION_INPUT_PARAM_FILE


__all__ = [
    "scenarios_utils",
    "dl_utils",
    "plot_utils",
    "ic_generation",
    "ic_sampler",
    "monitoring_utils",
    "VALIDATION_DIR",
    "VALIDATION_INPUT_PARAM_FILE",
]

_lazy_modules = {
    "scenarios_utils": "apebench_online.scenarios.scenarios_utils",
    "dl_utils": "apebench_online.core.dl_utils",
    "ic_generation": "apebench_online.sampler.ic_generation",
    "ic_sampler": "apebench_online.sampler.ic_sampler",
}


if TYPE_CHECKING:
    import apebench_online.scenarios.scenarios_utils as scenarios_utils
    import apebench_online.core.dl_utils as dl_utils
    import apebench_online.sampler.ic_generation as ic_generation
    import apebench_online.sampler.ic_sampler as ic_sampler


def __getattr__(name: str):
    """
    Lazily import heavy modules to speed up solver start-up.
    """
    if name in _lazy_modules:
        module = import_module(_lazy_modules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'apebench_online.sampler' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
