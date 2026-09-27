"""Experiment configuration loading with optional deterministic base inheritance."""
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    base_reference = config.pop("base_config", None)
    if base_reference is None:
        return config
    base_path = (path.parent / str(base_reference)).resolve()
    return deep_merge(load_experiment_config(base_path), config)
