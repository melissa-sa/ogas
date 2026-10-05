#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
exp=${EXP_DIR:-$repo}/experiments
regime=${1:?usage: submit_benchmark.sh REGIME [CONFIG_ROOT [OUT_ROOT]]}
src_root=$(realpath "${2:-$exp/test_ensemble}"); out=$(realpath -m "${3:-$exp/runs}")
mkdir -p "$out/configs" && cd "$out"

walltime() {
  case "$1:$2" in
    navier*:scot*) echo 14:00:00 ;; navier*) echo 11:00:00 ;;
    *:scot*) echo 10:00:00 ;; *) echo 07:00:00 ;;
  esac
}

for cfg in "$src_root"/*/*/"$regime"/{,seed_*/}config_online_*.json; do
  [ -e "$cfg" ] || continue
  seed=${cfg##*config_online_}; seed=${seed%.json}
  rel=${cfg#"$src_root"/}; pde=${rel%%/*}; rel=${rel#*/}; arch=${rel%%/*}
  name="${regime}_${pde%%_*}_${arch%%_*}_s$seed"
  group=$(python3 -c "import json, re, sys; print(json.loads(re.sub(r',(\s*[}\]])', r'\1', open(sys.argv[1]).read()))['dl_config'].get('wandb_group', ''))" "$cfg")
  python3 "$repo/scripts/configs/make_study_config.py" "$cfg" "configs/$name.json" "$out/$pde/$arch/$regime/seed_$seed" \
    --seed "$seed" --wandb-group "$group${WANDB_SUFFIX:-}"
  # shellcheck disable=SC2086
  "$repo/scripts/leo/job_study.sh" "configs/$name.json" "$name" --qos=normal --time="$(walltime "$pde" "$arch")" \
    --job-name="$name" ${SBATCH_EXTRA:-}
done
