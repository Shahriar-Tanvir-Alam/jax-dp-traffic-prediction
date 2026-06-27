import os
import pickle
import random
import subprocess
from typing import Optional
from pathlib import Path
import yaml
import numpy as np
import torch


def set_seed(seed: int):
    """
    Sets the random seed for numpy, torch, and CUDA (if available) to ensure reproducibility.

    Args:
        seed (int): The random seed to be used. (np requires integer seed)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def write_obj(obj, filename):
    """
    Writes a Python object to a file using pickle.

    Args:
        obj: The object to be written.
        filename (str): The path to the file to write to.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def read_file(filename):
    """
    Reads a Python object from a file using pickle.

    Args:
        filename (str): The path to the file to read from.

    Returns:
        The Python object read from the file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def git_hash() -> str:
    """
    Get the git hash. Assumes being run in a git repo (otherwise error).

    Returns:
        A str representing current git hash.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def exists(x):
    return x is not None


def file_exists(file_path):
    """
    Check if the file exists

    Args:
        file_path (str): The path to the file to check.

    Returns:
        A bool indicating the file exists or not.
    """
    if os.path.exists(file_path):
        print("The file exists")
        return True
    else:
        print("The file does not exist")
        return False

    
def create_folder(directory_path):
    """
    Creates a directory if it does not already exist.

    Parameters:
    directory_path (str or Path): The path of the directory to create.

    Returns:
    None

    Raises:
    OSError: If the directory cannot be created for any reason other than it already existing.
    """
    # Ensure directory_path is a Path object
    directory_path = Path(directory_path)

    try:
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully or already exists.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")
        raise

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def read_yaml(yaml_address):
    """ A function to read YAML file"""
    with open(yaml_address) as f:
        config = yaml.safe_load(f)
 
    return config

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def yaml_to_config(**params):    # pass in variable numbers of args
    d = {}
    for key1, value1 in params.items():
        d[key1] = value1['value']
    d = dotdict(d)
    return d

