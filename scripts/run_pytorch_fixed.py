import os
import json
import time
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from train_private import train
from traffic_prediction.utils import read_yaml, yaml_to_config

# Save all wandb.log metrics locally
RESULT_DIR = Path("../Fair_Comparison_Same_GPU_1000/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PYTORCH_JSONL = RESULT_DIR / "pytorch_wandb_metrics.jsonl"

import wandb

def local_wandb_log(data=None, *args, **kwargs):
    if data is None:
        return
    record = {
        "time": time.time(),
        "data": data,
        "kwargs": kwargs,
    }
    with open(PYTORCH_JSONL, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

wandb.log = local_wandb_log
wandb.finish = lambda *args, **kwargs: None

config_path = "../data/m_2_3_y_2019_optimals_check/config_tlc_nyc_inflow_optimal_c90.yaml"

config_dict = read_yaml(config_path)
config = yaml_to_config(**config_dict["parameters"])

config.epochs = int(os.environ.get("PYTORCH_EPOCHS", "1000"))

# For 1000 epochs, keep validation every 50 epochs unless config says otherwise
if getattr(config, "val_step", None) is None:
    config.val_step = 50
if config.val_step > config.epochs:
    config.val_step = 50

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
print("Validation step:", config.val_step)
print("Device:", config.device)
print("Private training:", config.use_private_training)
print("Noise multiplier:", config.noise_multiplier)
print("Max grad norm:", config.max_grad_norm)
print("Delta:", config.delta)
print("Local metric file:", PYTORCH_JSONL)

train(config=config)
