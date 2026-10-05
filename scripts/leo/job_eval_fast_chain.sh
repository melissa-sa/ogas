#!/bin/bash
#SBATCH --job-name=evalfast
#SBATCH --account=euhpc_d36_033
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=450000M
#SBATCH --output=std/eval-fast-chain.%j.out
#SBATCH --error=std/eval-fast-chain.%j.err
set -uo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export APEBENCH_ROOT=${APEBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
  exp=${EXP_DIR:-$APEBENCH_ROOT}/experiments
  runs=$(realpath "${1:-$exp/runs}"); export OUT=$(realpath -m "${2:-$exp/eval}"); shift $(( $# < 2 ? $# : 2 ))
  EVAL_ARGS="--root $runs $*"
  if [[ " $* " != *" --val "* ]]; then
    for v in "$exp"/conditional_models_exp/val_data/*/trajectories; do
      pde=$(basename "$(dirname "$v")")
      [ -n "$(find "$runs" -maxdepth 3 -type d -name "$pde" -print -quit)" ] && EVAL_ARGS="$EVAL_ARGS --val $pde=$v"
    done
  fi
  export EVAL_ARGS TOTAL=$(find "$runs" -path '*/seed_*/checkpoints/model_*.pt' | wc -l)
  echo "$TOTAL checkpoints -> $OUT/metrics_long.csv"
  mkdir -p "$OUT" std
  exec sbatch --nodes=4 "$0"
fi
: "${OUT:?}" "${TOTAL:?}" "${EVAL_ARGS:?}"
unset LD_LIBRARY_PATH PYTHONPATH PKG_CONFIG_PATH
source $APEBENCH_ROOT/leo_melissa_init.sh load
export LD_LIBRARY_PATH="$MELISSA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
export WANDB_MODE=disabled
PY=$MELISSA_ENV/bin/python
E=$APEBENCH_ROOT/scripts/analysis
ARGS="--output $OUT $EVAL_ARGS --deadline-min 24 --batch 64"
mkdir -p $OUT/state
ITER=$(( $(cat $OUT/state/iteration 2>/dev/null || echo 0) + 1 )); echo $ITER > $OUT/state/iteration
PREV=$(cat $OUT/state/prev_done 2>/dev/null || echo -1)
NOW=$(find $OUT/raw -name "*.npz" ! -name "*.tmp.npz" 2>/dev/null | wc -l)
echo "EVALFAST iter=$ITER done=$NOW prev=$PREV total=$TOTAL job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES $(date)"
if [ "$NOW" -ge "$TOTAL" ]; then
  $PY -u $E/aggregate_fast.py --output $OUT; echo "EVALFAST_AGGREGATE rc=$?"; exit 0
fi
if [ "$ITER" -gt 40 ]; then echo EVALFAST_ITER_LIMIT; exit 0; fi
if [ "$ITER" -gt 1 ] && [ "$NOW" -le "$PREV" ]; then
  echo EVALFAST_NO_PROGRESS_STOP; $PY -u $E/aggregate_fast.py --output $OUT; exit 0
fi
echo $NOW > $OUT/state/prev_done
sbatch --parsable --nodes=8 --dependency=afterany:$SLURM_JOB_ID $APEBENCH_ROOT/scripts/leo/job_eval_fast_chain.sh \
  || sbatch --parsable --nodes=4 --dependency=afterany:$SLURM_JOB_ID $APEBENCH_ROOT/scripts/leo/job_eval_fast_chain.sh \
  || echo EVALFAST_RESUBMIT_FAILED
srun --mpi=none --ntasks-per-node=1 --cpus-per-task=32 bash -c "
  for v in \$(env | grep -oE '^(PMIX|PMI|SLURM|OMPI_MCA_orte|OMPI_COMM)[A-Za-z0-9_]*'); do unset \$v; done
  for g in 0 1 2 3; do
    $PY -u $E/eval_fast.py $ARGS --gpu \$g > $OUT/state/log_${SLURM_JOB_ID}_\$(hostname -s)_gpu\$g.txt 2>&1 &
  done; wait"
END=$(find $OUT/raw -name "*.npz" ! -name "*.tmp.npz" | wc -l)
echo "EVALFAST_SEGMENT_END done=$END $(date)"
