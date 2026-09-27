"""Static validation for artifacts that claim to support exact training resume.

The current weight-only Tracker checkpoints intentionally do not satisfy this
schema. This module prevents them from being mislabeled as exact-resume state.
"""
from __future__ import annotations

from collections.abc import Mapping

EXACT_RESUME_KEYS = {
    "model_state",
    "optimizer_state",
    "relaxation_update",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state_all",
    "data_loader_generator_state",
    "sampler_state_or_batch_position",
    "tracker_state",
}


def missing_exact_resume_state(checkpoint: object, scheduler_used: bool = False) -> list[str]:
    """Return every state component missing from a purported exact checkpoint."""
    if not isinstance(checkpoint, Mapping):
        return sorted(EXACT_RESUME_KEYS | ({"scheduler_state"} if scheduler_used else set()))
    required = EXACT_RESUME_KEYS | ({"scheduler_state"} if scheduler_used else set())
    return sorted(key for key in required if key not in checkpoint or checkpoint[key] is None)


def validate_exact_resume_state(checkpoint: object, scheduler_used: bool = False) -> None:
    missing = missing_exact_resume_state(checkpoint, scheduler_used)
    if missing:
        raise ValueError("Not an exact-resume checkpoint; missing state: " + ", ".join(missing))
