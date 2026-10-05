#!/bin/bash
set -euo pipefail
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader,tcp}"
export OMPI_MCA_btl_vader_single_copy_mechanism="${OMPI_MCA_btl_vader_single_copy_mechanism:-none}"
export OMPI_MCA_hcoll="${OMPI_MCA_hcoll:-^hcoll}"
hosts="$(hostname -s):256"
args=(); server=0
for a in "$@"; do if [ "$a" = --server ]; then server=1; else args+=("$a"); fi; done
set -- "${args[@]}"
if [ $server = 1 ]; then
  hosts=${MELISSA_SERVER_HOSTS:-$hosts}
elif [ -n "${MELISSA_CLIENT_HOSTS:-}" ]; then
  IFS=, read -ra pool <<< "$MELISSA_CLIENT_HOSTS"
  n=$(flock "$MELISSA_CLIENT_COUNTER.lock" bash -c 'n=$(cat "$0" 2>/dev/null || echo 0); echo $((n + 1)) > "$0"; echo $n' "$MELISSA_CLIENT_COUNTER")
  hosts="${pool[$((n % ${#pool[@]}))]}:256"
fi
nice=(); [ $server = 1 ] || nice=(nice -n "${MELISSA_CLIENT_NICE:-10}")
exec "${nice[@]}" mpirun --bind-to none --oversubscribe --host "$hosts" "$@"
