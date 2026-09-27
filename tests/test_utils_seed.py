import pytest
import torch
from src.utils_seed import set_seed

def test_strict_determinism_failure_propagates(monkeypatch):
    def fail(_enabled):
        raise RuntimeError("unsupported")
    monkeypatch.setattr(torch, "use_deterministic_algorithms", fail)
    with pytest.raises(RuntimeError, match="Failed to enable strict deterministic"):
        set_seed(1, strict=True)
