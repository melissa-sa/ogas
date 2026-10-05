#!/usr/bin/env bash

# Source with no argument to load an existing environment, or with `install`
# to create the uv environment and build both OGAS and Melissa. Melissa is
# installed as a regular wheel so client jobs never trigger an editable rebuild.
_LEO_INIT_FILE="${BASH_SOURCE[0]}"
_LEO_INIT_DIR="$(cd "$(dirname "$_LEO_INIT_FILE")" && pwd)"
export APEBENCH_ROOT="${APEBENCH_ROOT:-$_LEO_INIT_DIR}"
export PATH="${HOME}/.local/bin:${PATH:-}"

if ! declare -F module >/dev/null 2>&1; then
    echo "The Leonardo module command is unavailable." >&2
    return 1 2>/dev/null || exit 1
fi

unset _APEBENCH_ENVRC_LOADED
source "$APEBENCH_ROOT/.leo_envrc"
module purge
module use /leonardo_scratch/large/userinternal/amemmol1/python-spackenv/spack-0.22.2-06/modules
module load python/3.11.7--gcc--12.2.0
module load cmake/3.27.9 cuda/12.2 gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2

_leo_mode="${1:-load}"
case "$_leo_mode" in
    install)
        command -v uv >/dev/null 2>&1 || {
            echo "uv is required; install it under ${HOME}/.local/bin." >&2
            return 1 2>/dev/null || exit 1
        }
        if [[ -n "${UV_CACHE_DIR:-}" ]]; then mkdir -p "$UV_CACHE_DIR"; fi
        rm -rf -- "$MELISSA_ENV"
        uv venv --seed --python 3.11.7 "$MELISSA_ENV"
        uv pip sync --python "$MELISSA_ENV/bin/python" "$APEBENCH_ROOT/frozen_requirements.txt"
        source "$MELISSA_ENV/bin/activate"
        uv pip install --python "$MELISSA_ENV/bin/python" --no-deps -e "$APEBENCH_ROOT"
        uv pip install --python "$MELISSA_ENV/bin/python" 'scikit-build-core>=0.11' 'pybind11>=3'
        if [[ "${MELISSA_EDITABLE:-0}" == "1" ]]; then
            # Build once, then keep the source tree importable without a
            # per-client CMake rebuild or editable_rebuild.lock contention.
            uv pip install --python "$MELISSA_ENV/bin/python" --no-deps --no-build-isolation \
                -Ceditable.rebuild=false \
                -Ccmake.define.INSTALL_ZMQ=OFF \
                -Ccmake.define.INSTALL_CONDUIT=ON \
                -Ccmake.define.CMAKE_BUILD_PARALLEL_LEVEL=8 \
                -Cbuild-dir="$MELISSA_ROOT/build" \
                -e "$MELISSA_ROOT[launcher,server,torch]"
        else
            uv pip install --python "$MELISSA_ENV/bin/python" --no-deps --no-build-isolation \
                -Ccmake.define.INSTALL_ZMQ=OFF \
                -Ccmake.define.INSTALL_CONDUIT=ON \
                -Ccmake.define.CMAKE_BUILD_PARALLEL_LEVEL=8 \
                -Cbuild-dir="$MELISSA_ROOT/build" \
                "$MELISSA_ROOT[launcher,server,torch]"
        fi
        ;;
    load)
        [[ -f "$MELISSA_ENV/bin/activate" ]] || {
            echo "Melissa uv environment missing: $MELISSA_ENV; run: source leo_melissa_init.sh install" >&2
            return 1 2>/dev/null || exit 1
        }
        source "$MELISSA_ENV/bin/activate"
        ;;
    *)
        echo "usage: source leo_melissa_init.sh [install|load]" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

# Melissa 3.x deliberately has no melissa_set_env.sh.  The uv venv and the
# editable Python package are the runtime authority.
_leo_prepend_cuda_libs 2>/dev/null || true
export PYTHONPATH="$APEBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda}"
