#!/bin/bash

export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "$(date) Running bootstrap..."
python scripts/run_nitrogen_bootstrap_10e8o_long.py
echo "$(date) Running bootstrap repeat backward..."
python scripts/run_nitrogen_bootstrap_repeat_backward_10e8o_long.py
echo "$(date) Running bootstrap repeat forward..."
python scripts/run_nitrogen_bootstrap_repeat_forward_10e8o_long.py