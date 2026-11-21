#!/usr/bin/env bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for seed in $(seq 42 51); do
 python experiment.py --seed "$seed"
done
python verify.py
python summarize.py
