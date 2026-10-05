#!/usr/bin/env bash
#SBATCH --job-name=ogas-study
#SBATCH --account=euhpc_d36_033
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output=std/study.%j.out
#SBATCH --error=std/study.%j.err
set -uo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  config=$(realpath "$1"); tag=$2; shift 2
  export APEBENCH_ROOT=${APEBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
  gpus=$(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); t = c['dl_config']['torch_model']
print(int(t.get('n_models', 2 if t.get('use_ensemble') else 1)) + int(bool(c.get('campaign_metadata', {}).get('sampler_gpu'))))" "$config")
  nodes=$(( (gpus + 3) / 4 )); gpn=$(( (gpus + nodes - 1) / nodes ))
  mkdir -p std
  exec sbatch --nodes=$nodes --ntasks-per-node=1 --gpus-per-node=$gpn --cpus-per-task=$((8 * gpn)) \
    --mem=$((115000 * gpn))M "$@" "$0" "$config" "$tag"
fi
config_file=$1; tag=$2
unset LD_LIBRARY_PATH PYTHONPATH PKG_CONFIG_PATH
export PATH="$HOME/.local/bin:$PATH"
source "$APEBENCH_ROOT/leo_melissa_init.sh" load
export LD_LIBRARY_PATH="$MELISSA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
out_dir=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$config_file")
ckpt=$out_dir/checkpoints
MON=${MON_DIR:-std}/study_${tag}_${SLURM_JOB_ID}
mkdir -p "$MON"
echo "start $(date +%s) $(date) nodes=$SLURM_JOB_NODELIST" > "$MON/timeline.txt"
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  gpn=${SLURM_GPUS_PER_NODE##*:}
  srv=(); for h in "${hosts[@]}"; do srv+=("$h:$gpn"); done
  srv[-1]="${hosts[-1]}:$((gpn + 1))"
  export MELISSA_SERVER_HOSTS=$(IFS=,; echo "${srv[*]}") MELISSA_CLIENT_HOSTS=$(IFS=,; echo "${hosts[*]}")
  export MELISSA_CLIENT_COUNTER=$(realpath "$MON")/client_counter
  export OMPI_MCA_plm_slurm_args="--overlap --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus-per-node=$gpn"
  echo "server hosts $MELISSA_SERVER_HOSTS" >> "$MON/timeline.txt"
fi
nvidia-smi dmon -s um -d 30 -o T > "$MON/gpu.log" 2>&1 &
( while true; do echo "$(date +%s) load=$(cut -d' ' -f1-3 /proc/loadavg) solvers=$(pgrep -fc 'core/solver.py') srv=$(pgrep -fc melissa-server)"; sleep 60; done ) > "$MON/cpu.log" 2>&1 &
( seen=0; gone=0; while sleep 60; do
    if pgrep -u "$USER" -f melissa-server > /dev/null; then seen=1; gone=0
    elif [ $seen = 1 ] && [ $((gone += 1)) -ge 10 ]; then
      echo "server gone for 10 min, stopping launcher $(date)" >> "$MON/timeline.txt"
      pkill -u "$USER" -TERM -f melissa-launcher; break
    fi
  done ) &
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %L | python -c "import sys
d, _, t = sys.stdin.read().strip().rpartition('-'); p = [0, 0, 0] + [int(x) for x in t.split(':')]
print(int(d or 0) * 86400 + p[-3] * 3600 + p[-2] * 60 + p[-1])")
timeout --signal=TERM --kill-after=120 $(( ${limit:-1800} - 600 )) melissa-launcher --config "$config_file" > "$MON/launcher.out" 2> "$MON/launcher.err"
echo "launcher rc=$? end $(date +%s) $(date)" >> "$MON/timeline.txt"
kill %1 %2 %3 2>/dev/null
python - "$ckpt" <<'PY'
import io, sys, cloudpickle, torch, torch.storage
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
ckpt = sys.argv[1]
d = cloudpickle.load(open(f"{ckpt}/sampler_metadata.pkl", "rb"))
keep = ("model_state", "ratio_mlp_state", "ratio_optimizer_state", "parameters", "parameters_is_bred",
        "current_generation", "batch", "R_value", "history_state")
torch.save({k: d[k] for k in keep if k in d}, f"{ckpt}/ddpm_final.pt")
print(f"ddpm_final.pt: generation {d.get('current_generation')}, breeder batch {d.get('batch')}, "
      f"{sum(v.numel() for v in d['model_state']['model_state_dict'].values())} DDPM parameters")
PY
L=$out_dir/melissa_server_0.log
echo "finished sims: $(grep -c 'has finished sending' "$L")  last step: $(grep -oh 'TRAINING:[0-9]*' "$L" | tail -1)"
awk '$1 !~ /^#/ && $3 ~ /^[0-9]+$/ {g[$2]+=$3; n[$2]++} END {for (k in g) print "gpu" k, "mean SM util", g[k]/n[k] "%"}' "$MON/gpu.log"
