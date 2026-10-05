"""Entry point kept for backward compatibility.

The implementation now lives in ``apebench_online.validation.external_validation``.
"""

from apebench_online.validation.external_validation.validator import (
    APEBenchExternalValidator,
    build_parser,
    get_dir_if_absolute,
    main,
)

__all__ = [
    "APEBenchExternalValidator",
    "build_parser",
    "get_dir_if_absolute",
    "main",
]


if __name__ == "__main__":
    main()
