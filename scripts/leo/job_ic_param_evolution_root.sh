#!/bin/bash

#SBATCH --job-name=ic-param-evol
#SBATCH --output=std/ic-param-evol.%j.out
#SBATCH --error=std/ic-param-evol.%j.err

#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=lrd_all_serial

ROOT=$1
OUTPUT_ROOT=$2

if [ -z "$ROOT" ]; then
  echo "Usage: $0 <experiments_root> [output_root]"
  exit 1
fi


: "${APEBENCH_ROOT:?export APEBENCH_ROOT=<repository> before sbatch}"
source "$APEBENCH_ROOT/leo_melissa_init.sh" load

SCRIPT="$APEBENCH_ROOT/scripts/analysis/parameter_generation_evolution.py"
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found"
  exit 1
fi

if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$APEBENCH_ROOT/param_generation_evolution"
fi

DDPM_NAME="ddpm_conf_ratio_proportional_normalized"
SEED=1

for pde_dir in "$ROOT"/*; do
  [ -d "$pde_dir" ] || continue
  pde_name="$(basename "$pde_dir")"
  for arch_dir in "$pde_dir"/*; do
    [ -d "$arch_dir" ] || continue
    arch_name="$(basename "$arch_dir")"
    exp_dir="$arch_dir/$DDPM_NAME/seed_${SEED}"
    [ -d "$exp_dir" ] || continue
    param_source_dir="$arch_dir/ddpm_conf_ratio_proportional_normalized_sm/seed_${SEED}"
    exp_dir_use="$exp_dir"
    if [ ! -f "$exp_dir/checkpoints/sampled_parameters.npy" ] && [ ! -f "$exp_dir/trajectories/input_parameters.npy" ]; then
      if [ -f "$param_source_dir/checkpoints/sampled_parameters.npy" ] || [ -f "$param_source_dir/trajectories/input_parameters.npy" ]; then
        echo "[INFO] Using parameters from $param_source_dir (missing in $exp_dir)"
        exp_dir_use="$param_source_dir"
      fi
    fi

    case "$pde_name" in
      *navier*|*stokes*)
        params=(diffusivity cutoff)
        pairs="diffusivity,cutoff"
        ;;
      *kuramoto*|*sivashinsky*)
        params=(domain_extent cutoff)
        pairs="domain_extent,cutoff"
        ;;
      *gray_scott*|*gray-scott*)
        params=(domain_extent mean_blob_amplitude mean_blob_center_x mean_blob_center_y)
        pairs="domain_extent,mean_blob_amplitude;mean_blob_center_x,mean_blob_center_y"
        ;;
      *)
        params=(all)
        pairs=""
        ;;
    esac

    out_dir="$OUTPUT_ROOT/${pde_name}__${arch_name}"
    mkdir -p "$out_dir"

    python3 "$SCRIPT" \
      --exp-dir "$exp_dir_use" \
      --output "$out_dir" \
      --pde-name "$pde_name" \
      --arch-name "$arch_name" \
      --gen-size 56 \
      --params "${params[@]}" \
      --plot-2d \
      --pairs "$pairs" \
      --max-2d-points 20000 \
      --skip-missing
  done
done
