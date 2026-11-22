#!/bin/bash
# Heralded-loss sweep: p in {0.003, 0.004, 0.005}, loss 2% per data qubit per cycle, X and Z memory.
# Each call adds one file per (basis, p, seed); existing files are skipped. Stops after BUDGET seconds.
cd "$(dirname "$0")/.."
BUDGET=${1:-4200}; START=$(date +%s); LOSS=${2:-0.02}
for round in 0 1 2 3 4 5; do
  for cfg in "0.003 600" "0.004 160" "0.005 80"; do
    set -- $cfg; p=$1; shots=$2
    for b in X Z; do
      seed=$((33 + round)); out=results/loss/${b}_${p}_l${LOSS}_s${seed}.npz
      [ -f "$out" ] && continue
      [ $(( $(date +%s) - START )) -gt $BUDGET ] && exit 0
      python3 scripts/eval_loss.py --basis $b --p $p --p_loss $LOSS --shots $shots --seed $seed --out $out
    done
  done
done
