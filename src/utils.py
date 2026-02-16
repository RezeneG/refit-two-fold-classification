# utils.py (optional)
import pandas as pd
import numpy as np

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    # Add other random seeds if needed
    return seed

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    import os
    os.makedirs(directory, exist_ok=True)
    return directory
