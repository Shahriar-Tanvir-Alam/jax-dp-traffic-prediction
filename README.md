# JAX GPU Implementation for Private Traffic Prediction

This repository contains a JAX GPU implementation of the private traffic prediction code, adapted from the original PyTorch implementation.

## Main Files

- `scripts/train_private.py`: original PyTorch training code
- `scripts/run_pytorch_fixed.py`: fixed PyTorch runner without W&B sweep
- `scripts/train_private_jax_final.py`: JAX GPU implementation
- `scripts/slurm/run_fair_same_gpu_1000_a100_cudnn_fix.slurm`: fair same-GPU A100 comparison script
- `scripts/slurm/test_jax_a100_cudnn_fix.slurm`: JAX A100 test script

## Dataset

The experiment used the TLC NYC inflow dataset.

Dataset file used:

`data/tlc_nyc/fhvhv/clean_aggregate_tlc_inflow_2019-02_2019-03_B_Q_10min_c90.parquet`

The dataset is not uploaded to GitHub because it is large.

## Fair Same-GPU A100 Comparison

The fair comparison was done on USC CARC Discovery.

Experiment setup:

- Job ID: 9749567
- Node: b05-14.hpc.usc.edu
- GPU: NVIDIA A100 80GB PCIe
- Dataset: tlc_nyc_inflow
- Epochs: 1000
- Batch size: 4
- Private training: True
- Noise multiplier sigma: 1.0
- Max grad norm: 1.0
- Delta: 1e-05

## Results

| Implementation | Hardware | Runtime | Test MSE | Test MAE/L1 | Exit Code |
|---|---:|---:|---:|---:|---:|
| PyTorch GPU | A100 | 1092 sec | 0.1427 | 0.2557 | 0 |
| JAX GPU | A100 | 2425 sec | 0.2679 | 0.3185 | 0 |

Additional JAX results:

- Final test RMSE: 0.5176124572753906
- Final test MAPE: 162.00039672851562 %
- Final epsilon: 5.643493013110433
- Final delta: 1e-05

## Conclusion

The JAX GPU implementation runs successfully on GPU with DP-SGD and RDP privacy accounting. However, in this current implementation, PyTorch GPU is faster and achieves lower test error than JAX GPU on the same A100 GPU.

Further JAX optimization is needed before it can outperform PyTorch GPU.

## Reproducibility

Final logs and summaries are stored in:

`results/fair_same_gpu_a100_9749567/`

The Slurm scripts are stored in:

`scripts/slurm/`

## cuDNN Note

The JAX GPU run needed a compatible cuDNN version. The environment was fixed using:

`pip install -U "nvidia-cudnn-cu12>=9.8,<10"`

and this path was added in the Slurm script:

`LD_LIBRARY_PATH=/home1/alams/jax-dp-traffic-prediction/jax_gpu_env/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH`
