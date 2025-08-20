import random
import numpy as np
import os

DEFAULT_SEED = 42

def seed_everything(seed=DEFAULT_SEED):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")