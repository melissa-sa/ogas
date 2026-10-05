#!/usr/bin/env bash

#SBATCH --job-name=melissa-apebench-offline
#SBATCH --error=std/ma-off.%j.err
#SBATCH --output=std/ma-off.%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --gres=tmpfs:20g
#SBATCH --account=euhpc_d36_033_0
#SBATCH --partition=dcgp_usr_prod

#SBATCH hetjob
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --gres=tmpfs:20g
#SBATCH --account=euhpc_d36_033_0
#SBATCH --partition=dcgp_usr_prod

set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export APEBENCH_ROOT=${APEBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
  val=${EXP_DIR:-$APEBENCH_ROOT}/experiments/conditional_models_exp/val_data
  [ $# -gt 0 ] || set -- $(ls "$val"/config_offline_*.json | grep -v config_offline_detail_)
  mkdir -p std
  for cfg in "$@"; do sbatch "$0" "$(realpath "$cfg")"; done
  exit 0
fi
CONFIG_FILE=$1
source "$APEBENCH_ROOT/leo_melissa_init.sh" load
CONFIG_FILE=$(realpath "$CONFIG_FILE")
test -f "$CONFIG_FILE"
exec melissa-launcher --config "$CONFIG_FILE"
