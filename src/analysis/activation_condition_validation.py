"""Validation for matched ReLU/LeakyReLU mechanism-control artifacts."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from src.analysis.common import load_checkpoint, load_yaml, validate_state_dicts

ALLOWED_CONDITION_DIFFERENCES = {
    "model.activation", "model.leaky_relu_negative_slope",
    "logging.run_name", "logging.save_dir", "train.scenario",
}


def flatten(value, prefix=""):
    if not isinstance(value, Mapping):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    return result


def validate_condition_configs(relu: Mapping, leaky: Mapping) -> None:
    a, b = flatten(relu), flatten(leaky)
    differences = {key: (a.get(key), b.get(key)) for key in sorted(set(a) | set(b))
                   if a.get(key) != b.get(key) and key not in ALLOWED_CONDITION_DIFFERENCES}
    if differences:
        raise ValueError(f"Unexpected cross-condition config differences: {differences}")
    if relu["model"]["activation"] != "relu" or leaky["model"]["activation"] != "leaky_relu":
        raise ValueError("Expected ReLU then LeakyReLU configs")
    if relu["train"]["scenario"] != leaky["train"]["scenario"]:
        raise ValueError("Cross-condition scenario identities differ")
    if relu["train"]["relaxation_checkpoints"] != leaky["train"]["relaxation_checkpoints"]:
        raise ValueError("Cross-condition checkpoint schedules differ")


def validate_bit_identical_initializations(relu_state: Mapping, leaky_state: Mapping) -> None:
    validate_state_dicts(relu_state, leaky_state, check_dtype=True)
    unequal = [key for key in relu_state if not torch.equal(relu_state[key], leaky_state[key])]
    if unequal:
        raise ValueError(f"Cross-condition initialization tensors differ: {unequal}")


def validate_condition_run_pair(relu_run: str | Path, leaky_run: str | Path) -> None:
    relu_run, leaky_run = Path(relu_run), Path(leaky_run)
    validate_condition_configs(load_yaml(relu_run / "config_resolved.yaml"),
                               load_yaml(leaky_run / "config_resolved.yaml"))
    validate_bit_identical_initializations(load_checkpoint(relu_run / "weights_epoch_000.pt"),
                                           load_checkpoint(leaky_run / "weights_epoch_000.pt"))


def _manifest_grid(path: str | Path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8")); grid = {}
    for entry in manifest["entries"]:
        key = (int(entry["seed"]), entry["scenario"])
        if key in grid:
            raise ValueError(f"Duplicate manifest entry: {key}")
        grid[key] = Path(entry["expected_run_directory"])
    seeds = set(int(seed) for seed in manifest["seeds"])
    expected = {(seed, scenario) for seed in seeds for scenario in ("SABC", "SBAC")}
    if set(grid) != expected:
        raise ValueError(f"Incomplete manifest grid: missing={sorted(expected-set(grid))}, extra={sorted(set(grid)-expected)}")
    return seeds, grid


def validate_condition_manifests(relu_manifest: str | Path, leaky_manifest: str | Path) -> None:
    relu_seeds, relu = _manifest_grid(relu_manifest); leaky_seeds, leaky = _manifest_grid(leaky_manifest)
    if relu_seeds != leaky_seeds:
        raise ValueError(f"Cross-condition manifest seeds differ: {sorted(relu_seeds)} != {sorted(leaky_seeds)}")
    for key in sorted(relu):
        validate_condition_run_pair(relu[key], leaky[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relu-manifest", required=True); parser.add_argument("--leaky-manifest", required=True)
    args = parser.parse_args(); validate_condition_manifests(args.relu_manifest, args.leaky_manifest)
    print("Cross-condition configs, schedules, pair grids, and initializations are exactly compatible.")


if __name__ == "__main__":
    main()
