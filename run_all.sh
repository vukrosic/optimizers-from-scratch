#!/bin/sh
# Runs every step in order and prints what the video shows. About two minutes.
set -e
cd "$(dirname "$0")"
for f in step1_gradient_descent step2_sgd step3_momentum step4_rmsprop step5_adam step6_adamw step7_muon step8_race; do
  echo "== $f"
  python3 $f.py
done
