import os

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from train_private import train

# Disable W&B logging for fixed local/offline comparison run
import wandb
wandb.log = lambda *args, **kwargs: None
wandb.finish = lambda *args, **kwargs: None
from traffic_prediction.utils import read_yaml, yaml_to_config

config_path = "../data/m_2_3_y_2019_optimals_check/config_tlc_nyc_inflow_optimal_c90.yaml"

config_dict = read_yaml(config_path)
config = yaml_to_config(**config_dict["parameters"])

# Force same fixed experiment settings
config.epochs = int(os.environ.get("PYTORCH_EPOCHS", "5"))

# For short tests, validation step must not exceed epochs
if getattr(config, "val_step", None) is None:
    config.val_step = 1
if config.val_step > config.epochs:
    config.val_step = 1

# Required by original PyTorch train_private.py
if getattr(config, "device", None) is None:
    config.device = "gpu"

if getattr(config, "use_private_training", None) is None:
    config.use_private_training = True

if getattr(config, "noise_multiplier", None) is None:
    config.noise_multiplier = 1.0

if getattr(config, "max_grad_norm", None) is None:
    config.max_grad_norm = 1.0

if getattr(config, "delta", None) is None:
    config.delta = 1e-5

print("Running original PyTorch training code without W&B sweep")
print("Config path:", config_path)
print("Epochs:", config.epochs)
print("Device:", config.device)
print("Private training:", config.use_private_training)
print("Noise multiplier:", config.noise_multiplier)
print("Max grad norm:", config.max_grad_norm)
print("Delta:", config.delta)

train(config=config)
