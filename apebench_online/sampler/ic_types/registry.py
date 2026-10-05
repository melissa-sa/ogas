from typing import Dict, Type

from .base import BaseICMaker

_CUSTOM_IC_MAKERS: Dict[str, Type[BaseICMaker]] = {}


def register_ic_maker(name: str):
    def decorator(cls: Type[BaseICMaker]):
        if not issubclass(cls, BaseICMaker):
            raise TypeError(
                f"Cannot register {cls.__name__} as IC maker '{name}': "
                "class must inherit from BaseICMaker."
            )
        _CUSTOM_IC_MAKERS[name] = cls
        return cls

    return decorator


def get_ic_maker_class(name: str) -> Type[BaseICMaker]:
    try:
        return _CUSTOM_IC_MAKERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown IC type '{name}'. Registered: {list(_CUSTOM_IC_MAKERS)}") from exc


def available_ic_types():
    return tuple(_CUSTOM_IC_MAKERS.keys())
