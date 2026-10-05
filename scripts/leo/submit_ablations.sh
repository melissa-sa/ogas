#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
exp=${EXP_DIR:-$repo}/experiments
which=${1:-all}; src_root=$(realpath "${2:-$exp/test_ensemble}"); out=$(realpath -m "${3:-$exp/ablations}")
pde=kuramoto_sivashinsky_2d_low_res
mkdir -p "$out/configs" && cd "$out"

submit() {
  local name=$1 time=$2 src=$3 dir=$4; shift 4
  python3 "$repo/scripts/configs/make_study_config.py" "$src" "configs/$name.json" "$dir" "$@"
  "$repo/scripts/leo/job_study.sh" "configs/$name.json" "$name" --qos=normal --time="$time" --job-name="$name"
}
src() { ls "$src_root/$pde/$1/ddpm_conf_ratio_proportional_uncertainty"/{,seed_$2/}config_online_$2.json 2>/dev/null | head -1; }

for seed in 1 2 3; do
  if [[ $which == all || $which == ensemble ]]; then
    for m in 2 3 5; do
      submit "ens_M${m}_s$seed" 06:00:00 "$(src unet_cond $seed)" \
        "$out/ensemble_size/M$m/$pde/unet_cond/ddpm_conf_ratio_proportional_uncertainty/seed_$seed" --n-models "$m" --seed "$seed"
    done
  fi
  if [[ $which == all || $which == ecrps ]]; then
    for arch_time in ${ECRPS_ARCHS:-unet_cond:06:00:00 fno_cond:06:00:00 scot_t256:09:00:00}; do
      arch=${arch_time%%:*}
      submit "ecrps_${arch%%_*}_s$seed" "${arch_time#*:}" "$(src "$arch" $seed)" \
        "$out/ecrps/$pde/$arch/ddpm_conf_ratio_proportional_ecrps/seed_$seed" --signal ecrps --seed "$seed"
    done
  fi
done
