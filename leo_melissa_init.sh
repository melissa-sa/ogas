#!/usr/bin/env bash

# Source with no argument to load an existing environment, or with `install`
# to create the uv environment (`uv sync`) and build both OGAS and Melissa.
# Melissa is installed as a regular wheel.
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
        # uv sync installs into $MELISSA_ENV; Melissa (INSTALL_CONDUIT/INSTALL_ZMQ=ON, see pyproject.toml)
        # is built in this venv, so its build requirements must be installed first.
        export UV_PROJECT_ENVIRONMENT="$MELISSA_ENV"
        export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
        uv venv --python 3.11 "$MELISSA_ENV"
        (cd "$APEBENCH_ROOT" \
            && uv pip install --python "$MELISSA_ENV/bin/python" --group build --no-binary mpi4py \
            && uv sync) || return 1 2>/dev/null || exit 1
        source "$MELISSA_ENV/bin/activate"
        python "$APEBENCH_ROOT/scripts/check_install.py" || echo "Post-install check failed." >&2
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
# installed packages are the runtime authority.
_leo_prepend_cuda_libs 2>/dev/null || true
export PYTHONPATH="$APEBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda}"
