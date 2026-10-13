"""Entry point kept for backward compatibility.

The implementation now lives in ``ogas.validation.external_validation``.
"""

from ogas.validation.external_validation.validator import (
    OgasExternalValidator,
    build_parser,
    get_dir_if_absolute,
    main,
)

__all__ = [
    "OgasExternalValidator",
    "build_parser",
    "get_dir_if_absolute",
    "main",
]


if __name__ == "__main__":
    main()
