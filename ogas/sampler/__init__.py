from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ogas.core.constants import VALIDATION_DIR, VALIDATION_INPUT_PARAM_FILE


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
    "scenarios_utils": "ogas.scenarios.scenarios_utils",
    "dl_utils": "ogas.core.dl_utils",
    "ic_generation": "ogas.sampler.ic_generation",
    "ic_sampler": "ogas.sampler.ic_sampler",
}


if TYPE_CHECKING:
    import ogas.scenarios.scenarios_utils as scenarios_utils
    import ogas.core.dl_utils as dl_utils
    import ogas.sampler.ic_generation as ic_generation
    import ogas.sampler.ic_sampler as ic_sampler


def __getattr__(name: str):
    """
    Lazily import heavy modules to speed up solver start-up.
    """
    if name in _lazy_modules:
        module = import_module(_lazy_modules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'ogas.sampler' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
