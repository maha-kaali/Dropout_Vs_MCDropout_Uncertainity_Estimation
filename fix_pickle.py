import pickle
import torch
import numpy as np

# 1. Define the path to your problematic file
BAD_FILE_PATH = "pickled/mc_dropout_epoch100/dashboard_data.pkl"
# BAD_FILE_PATH = "pickled/mc_dropout_epoch100/dashboard_data_lite.pkl" # Uncomment if using the lite version

def to_cpu_numpy(obj):
    """Recursively convert Tensors to CPU/NumPy"""
    if isinstance(obj, torch.Tensor):
        # Move to CPU, detach from graph, convert to simple NumPy array
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: to_cpu_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu_numpy(v) for v in obj]
    else:
        return obj

print(f"Loading {BAD_FILE_PATH}...")
with open(BAD_FILE_PATH, "rb") as f:
    data = pickle.load(f)

print("Converting tensors to CPU/NumPy...")
clean_data = to_cpu_numpy(data)

print("Overwriting file with clean data...")
with open(BAD_FILE_PATH, "wb") as f:
    pickle.dump(clean_data, f)

print("Success! The file is now safe for Linux/Streamlit.")