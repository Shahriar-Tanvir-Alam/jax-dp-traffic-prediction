#!/bin/bash

export PYTHONPATH=.
export JAX_RUN_MODE=paper_test
export JAX_PAPER_TEST_EPOCHS=2

SCRIPTS=(
scripts/train_private_jax_final.py
scripts/train_jax_multiprediction_5.py
scripts/train_jax_multiprediction_6.py
scripts/train_jax_multiprediction_7.py
scripts/train_jax_multiprediction_9.py
scripts/train_jax_multiprediction_11.py
scripts/train_jax_multiprediction_13.py
scripts/train_jax_multiprediction_17.py
scripts/train_jax_multiprediction_18.py
scripts/train_jax_multiprediction_21.py
scripts/train_jax_multiprediction_26_dp.py
scripts/train_jax_multiprediction_51.py
scripts/train_jax_multiprediction_52.py
scripts/train_jax_MP0_i.py
scripts/train_jax_MP0_o.py
scripts/train_jax_MP25_i.py
scripts/train_jax_MP25_o.py
scripts/train_jax_MP25PP_i.py
scripts/train_jax_MP25PP_o.py
scripts/train_jax_MA2_I3.py
scripts/train_jax_MA2_O3.py
scripts/train_jax_MA3_S_I41.py
scripts/train_jax_MA3_S_O41.py
scripts/train_jax_MA3_S_I42.py
scripts/train_jax_MA3_S_O42.py
scripts/train_jax_MA3_S_I54.py
scripts/train_jax_MA3_S_O54.py
scripts/train_jax_MA4_I1.py
scripts/train_jax_MA4_O1.py
scripts/train_jax_MA4_I2.py
scripts/train_jax_MA4_O2.py
scripts/train_jax_MA4_I1_L.py
scripts/train_jax_MA4_O1_L.py
scripts/train_jax_MA4_I2_L.py
scripts/train_jax_MA4_O2_L.py
scripts/train_jax_MA5_S_I12.py
scripts/train_jax_MA5_S_O12.py
scripts/train_jax_MA5_S_DP_I12.py
scripts/train_jax_MA5_S_DP_O12.py
)

echo "script,status" > jax_paper_test_logs/status.csv

for script in "${SCRIPTS[@]}"; do
  name=$(basename "$script" .py)
  log="jax_paper_test_logs/${name}.log"

  echo ""
  echo "============================================================"
  echo "Running $script"
  echo "Log: $log"
  echo "============================================================"

  python "$script" > "$log" 2>&1
  code=$?

  if [ $code -eq 0 ]; then
    echo "$script,OK" >> jax_paper_test_logs/status.csv
    echo "OK: $script"
  else
    echo "$script,FAILED" >> jax_paper_test_logs/status.csv
    echo "FAILED: $script"
  fi
done

echo ""
echo "Finished paper_test suite."
cat jax_paper_test_logs/status.csv
