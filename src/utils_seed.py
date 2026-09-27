"""Random seed utilities."""
import random
import numpy as np
import torch


def set_seed(seed: int, strict: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as exc:
            raise RuntimeError("Failed to enable strict deterministic PyTorch algorithms") from exc
