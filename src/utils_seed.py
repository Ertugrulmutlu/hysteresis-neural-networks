# src/utils_seed.py
import random
import numpy as np
import torch

def set_seed(seed: int, strict: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print("[WARN] Deterministic algorithms not fully supported:", e)
