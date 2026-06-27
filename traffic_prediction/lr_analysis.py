# automatically reload any changes to the traffic prediction pkg
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, '../')
import matplotlib.pyplot as plt
import numpy as np 
import math
import random 
from tqdm import tqdm
from traffic_prediction.linear_regression_analysis import *





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml_address', type=str, default='../data/config_multiprediction.yaml')
    args = parser.parse_args()
    
    wandb.login()
    sweep_configuration = read_yaml(args.yaml_address)
    sweep_id = wandb.sweep(sweep=sweep_configuration, entity='ghafelebashi', project='ptp-sweep-2')
    wandb.agent(sweep_id, function=train)