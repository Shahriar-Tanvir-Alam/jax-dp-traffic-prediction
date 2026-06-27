# JAX Implementation of a Differential Privacy Model for Traffic Prediction

This repository contains a JAX implementation of a differential privacy model for traffic prediction, converted from an original PyTorch-based implementation.

## Current Status

- JAX CPU implementation completed.
- 2-epoch paper-test completed for 39 architecture scripts.
- 50-epoch selected-model validation completed.
- 1000-epoch proposed/private model run completed.
- 10000-epoch proposed/private model run completed on CPU.

## Main JAX CPU 10000-Epoch Result

Model: `train_private_jax_final.py`

Final test MSE  = 0.12718874216079712  
Final test RMSE = 0.35663530230522156  
Final test MAE  = 0.24156597256660461  
Final test MAPE = 126.02887725830078 %  
Final epsilon   = 1603.7990078248047  

## Next Goal

The next goal is to run the converted JAX implementation on GPU using Google Colab and then on the USC CARC Discovery GPU cluster.

## Notes

Large raw data files are not included in this repository.
