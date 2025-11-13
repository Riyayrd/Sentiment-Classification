# src/utils.py
import os
import time
import random
import pickle
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        info["ram_gb"] = "unknown"
    return info

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def time_elapsed(start):
    return time.time() - start
