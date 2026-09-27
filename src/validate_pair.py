"""Validate that SAB and SBA run directories form a comparable pair."""
import argparse
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.analysis.common import (load_checkpoint, load_yaml, relaxation_checkpoint_path,
                                 validate_state_dicts, write_json)

REQUIRED_METRICS = {"epoch", "phase", "phase_data", "train_loss", "test_loss_full", "test_acc_full",
                    "test_loss_A", "test_acc_A", "test_loss_B", "test_acc_B"}
REQUIRED_RELAXATION_METRICS = {"scenario", "relaxation_step", "train_loss_C_recent", "test_loss_full",
                               "test_acc_full", "test_loss_A", "test_acc_A", "test_loss_B", "test_acc_B"}
ALLOWED_CONFIG_PATHS = {"train.scenario", "logging.run_name", "logging.save_dir"}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    return result


def config_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    flat_a, flat_b = _flatten(a), _flatten(b)
    return {key: (flat_a.get(key), flat_b.get(key)) for key in sorted(set(flat_a) | set(flat_b))
            if flat_a.get(key) != flat_b.get(key)}


def validate_pair(run_a: Path | str, run_b: Path | str) -> dict[str, Any]:
    errors, checks = [], []
    runs = [Path(run_a), Path(run_b)]
    configs = []
    for run in runs:
        config_path = run / "config_resolved.yaml"
        if not config_path.exists():
            errors.append(f"Missing resolved config: {config_path}")
            configs.append(None)
        else:
            configs.append(load_yaml(config_path))
    if all(configs):
        scenarios = (configs[0].get("train", {}).get("scenario"), configs[1].get("train", {}).get("scenario"))
        expected_pair = ("SABC", "SBAC") if any(value in {"SABC", "SBAC"} for value in scenarios) else ("SAB", "SBA")
        if scenarios != expected_pair:
            errors.append(f"Runs must be ordered run_a={expected_pair[0]} and run_b={expected_pair[1]}, got {scenarios}")
        disallowed = {k: v for k, v in config_differences(configs[0], configs[1]).items() if k not in ALLOWED_CONFIG_PATHS}
        if disallowed:
            errors.append(f"Disallowed config differences: {disallowed}")
        else:
            checks.append("configs match outside the explicit allowlist")
    init_paths = [run / "weights_epoch_000.pt" for run in runs]
    if not all(path.exists() for path in init_paths):
        errors.extend(f"Missing initialization checkpoint: {path}" for path in init_paths if not path.exists())
    else:
        states = [load_checkpoint(path) for path in init_paths]
        try:
            validate_state_dicts(states[0], states[1], check_dtype=True)
        except ValueError as exc:
            errors.append(f"Initialization state dictionaries differ: {exc}")
        if set(states[0]) == set(states[1]) and all(
            states[0][key].shape == states[1][key].shape and
            states[0][key].dtype == states[1][key].dtype and
            torch.equal(states[0][key], states[1][key]) for key in states[0]
        ):
            checks.append("initialization checkpoints are exactly equal")
        elif not any(error.startswith("Initialization state dictionaries differ") for error in errors):
            errors.append("Initialization tensors are not exactly equal")
    frames = []
    for run, config in zip(runs, configs):
        metrics = run / "metrics.csv"
        if not metrics.exists():
            errors.append(f"Missing metrics file: {metrics}")
            frames.append(None)
        else:
            frame = pd.read_csv(metrics)
            frames.append(frame)
            missing = REQUIRED_METRICS - set(frame.columns)
            if missing:
                errors.append(f"{metrics} missing columns: {sorted(missing)}")
            if "epoch" in frame:
                epochs = pd.to_numeric(frame["epoch"], errors="coerce")
                expected_epochs = list(range(1, int(config["train"]["epochs_total"]) + 1)) if config else []
                actual_epochs = epochs.tolist()
                if epochs.isna().any() or actual_epochs != expected_epochs:
                    errors.append(f"Epochs must be exactly 1..epochs_total in {metrics}; got {actual_epochs}")
            if config:
                epochs_total = int(config["train"]["epochs_total"])
                if len(frame) != epochs_total:
                    errors.append(f"Metrics row count must equal epochs_total ({epochs_total}) in {metrics}; got {len(frame)}")
                scenario = config.get("train", {}).get("scenario")
                phase_epochs = int(config["train"].get("phase_epochs", epochs_total // 2))
                phase_numbers = pd.to_numeric(frame["phase"], errors="coerce") if "phase" in frame else pd.Series(dtype=float)
                if ({"epoch", "phase", "phase_data"}.issubset(frame.columns)
                        and scenario in {"SAB", "SBA", "SABC", "SBAC"}
                        and not epochs.isna().any()
                        and not phase_numbers.isna().any()):
                    expected_data = (("A", "B") if scenario in {"SAB", "SABC"} else ("B", "A"))
                    for row in frame.itertuples(index=False):
                        expected_phase = 1 if int(row.epoch) <= phase_epochs else 2
                        expected_phase_data = expected_data[expected_phase - 1]
                        if int(row.phase) != expected_phase or str(row.phase_data) != expected_phase_data:
                            errors.append(f"Incorrect phase semantics in {metrics} at epoch {row.epoch}: "
                                          f"expected phase {expected_phase}/{expected_phase_data}, "
                                          f"got {row.phase}/{row.phase_data}")
                            break
                expected = run / f"weights_epoch_{epochs_total:03d}.pt"
                if not expected.exists():
                    errors.append(f"Missing expected final checkpoint: {expected}")
    if all(frame is not None for frame in frames):
        epoch_sequences = [pd.to_numeric(frame["epoch"], errors="coerce").tolist() for frame in frames if "epoch" in frame]
        if len(epoch_sequences) == 2 and epoch_sequences[0] != epoch_sequences[1]:
            errors.append("Runs have different epoch sequences")
    if all(configs) and (configs[0]["train"].get("scenario"), configs[1]["train"].get("scenario")) == ("SABC", "SBAC"):
        configured_sequences = [[int(value) for value in config["train"].get("relaxation_checkpoints", [])]
                                for config in configs]
        if configured_sequences[0] != configured_sequences[1]:
            errors.append("Runs have different configured relaxation-step sequences")
        for run, config, expected_steps in zip(runs, configs, configured_sequences):
            total_step = int(config["train"].get("relaxation_steps", -1))
            if (not expected_steps or expected_steps[0] != 0 or expected_steps != sorted(expected_steps)
                    or len(expected_steps) != len(set(expected_steps))
                    or any(step < 0 or step > total_step for step in expected_steps)
                    or total_step not in expected_steps):
                errors.append(f"Invalid relaxation checkpoint schedule in {run}")
            for step in expected_steps:
                path = relaxation_checkpoint_path(run, step)
                if not path.exists():
                    errors.append(f"Missing relaxation checkpoint: {path}")
            metrics_path = run / "relaxation_metrics.csv"
            if not metrics_path.exists():
                errors.append(f"Missing relaxation metrics file: {metrics_path}")
                continue
            relaxation_frame = pd.read_csv(metrics_path)
            missing = REQUIRED_RELAXATION_METRICS - set(relaxation_frame.columns)
            if missing:
                errors.append(f"{metrics_path} missing columns: {sorted(missing)}")
            if "relaxation_step" in relaxation_frame:
                actual_steps = pd.to_numeric(relaxation_frame["relaxation_step"], errors="coerce").tolist()
                if actual_steps != expected_steps:
                    errors.append(f"Relaxation metrics steps in {metrics_path} must equal {expected_steps}; got {actual_steps}")
            if "scenario" in relaxation_frame and not relaxation_frame["scenario"].eq(config["train"]["scenario"]).all():
                errors.append(f"Relaxation metrics scenario mismatch in {metrics_path}")
        if configured_sequences[0] == configured_sequences[1] and configured_sequences[0]:
            final_step = configured_sequences[0][-1]
            final_paths = [relaxation_checkpoint_path(run, final_step) for run in runs]
            if all(path.exists() for path in final_paths):
                try:
                    validate_state_dicts(load_checkpoint(final_paths[0]), load_checkpoint(final_paths[1]), check_dtype=True)
                except ValueError as exc:
                    errors.append(f"Final relaxation state dictionaries differ: {exc}")
    if not errors:
        checks.extend(["metrics schemas and epochs are valid", "expected final checkpoints exist"])
    return {"valid": not errors, "run_a": str(runs[0]), "run_b": str(runs[1]), "checks": checks, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    report = validate_pair(args.run_a, args.run_b)
    print("PASS" if report["valid"] else "FAIL")
    for item in report["checks"]:
        print(f"  [ok] {item}")
    for item in report["errors"]:
        print(f"  [error] {item}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.json_out, report)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
