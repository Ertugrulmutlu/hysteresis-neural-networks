"""Shared, deterministic analysis utilities."""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import yaml

from src.data import balanced_rotated_indices, get_mnist_datasets, indices_by_digits, make_loader
from src.model import SimpleCNN

EPOCH_RE = re.compile(r"weights_epoch_(\d+)\.pt$")
PAIR_CONFIG_ALLOWLIST = {"train.scenario", "logging.run_name", "logging.save_dir"}
PROTOCOL_FIELDS = (
    "experiment.seed", "data.dataset", "data.batch_size", "data.normalize",
    "data.normalize_mean", "data.normalize_std", "data.augmentation",
    "split.A_digits", "split.B_digits", "model.arch", "model.norm",
    "model.group_norm_groups", "model.activation", "train.optimizer", "train.lr",
    "train.momentum", "train.weight_decay", "train.epochs_total", "train.phase_epochs",
)


@dataclass(frozen=True)
class AnalysisPair:
    """Validated SAB/SBA inputs shared by every Part 2 analysis."""

    config_sab: dict[str, Any]
    config_sba: dict[str, Any]
    init_sab: Path
    init_sba: Path
    final_sab: Path
    final_sba: Path


@dataclass(frozen=True)
class CommonRelaxationPair:
    """Validated SABC/SBAC relaxation artifacts indexed by update count."""

    config_sabc: dict[str, Any]
    config_sbac: dict[str, Any]
    init_sabc: Path
    init_sbac: Path
    checkpoints_sabc: dict[int, Path]
    checkpoints_sbac: dict[int, Path]
    relaxation_steps: tuple[int, ...]


def relaxation_checkpoint_path(run_dir: Path | str, step: int) -> Path:
    if step < 0:
        raise ValueError("relaxation step must be non-negative")
    return Path(run_dir) / f"weights_relax_step_{step:06d}.pt"


def validate_common_relaxation_pair(run_sabc: Path | str, run_sbac: Path | str) -> CommonRelaxationPair:
    """Validate the complete SABC/SBAC checkpoint pair before analysis output."""
    runs = (Path(run_sabc), Path(run_sbac))
    config_paths = tuple(run / "config_resolved.yaml" for run in runs)
    for path in config_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing resolved config: {path}")
    config_sabc, config_sbac = (load_yaml(path) for path in config_paths)
    scenarios = (config_sabc.get("train", {}).get("scenario"), config_sbac.get("train", {}).get("scenario"))
    if scenarios != ("SABC", "SBAC"):
        raise ValueError(f"Common-relaxation pair must be ordered SABC then SBAC, got {scenarios}")
    flat_a, flat_b = _flatten_mapping(config_sabc), _flatten_mapping(config_sbac)
    differences = {key: (flat_a.get(key), flat_b.get(key)) for key in sorted(set(flat_a) | set(flat_b))
                   if flat_a.get(key) != flat_b.get(key) and key not in PAIR_CONFIG_ALLOWLIST}
    if differences:
        raise ValueError(f"Disallowed config differences: {differences}")
    steps = tuple(int(value) for value in config_sabc["train"].get("relaxation_checkpoints", []))
    total = int(config_sabc["train"].get("relaxation_steps", -1))
    if total < 0 or not steps or steps != tuple(sorted(steps)) or len(steps) != len(set(steps)):
        raise ValueError("Invalid relaxation checkpoint schedule")
    if steps[0] != 0 or steps[-1] > total or total not in steps:
        raise ValueError("Relaxation checkpoints must begin at 0, stay in range, and include the final step")
    init_paths = tuple(run / "weights_epoch_000.pt" for run in runs)
    for path in init_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing initialization checkpoint: {path}")
    init_states = tuple(load_checkpoint(path) for path in init_paths)
    validate_state_dicts(*init_states, check_dtype=True)
    for key in init_states[0]:
        if not torch.equal(init_states[0][key], init_states[1][key]):
            raise ValueError(f"Initialization tensor differs for {key}")
    epochs_total = int(config_sabc["train"]["epochs_total"])
    phase_epochs = int(config_sabc["train"]["phase_epochs"])
    epoch_final_paths = tuple(run / f"weights_epoch_{epochs_total:03d}.pt" for run in runs)
    for path in epoch_final_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing post-history checkpoint: {path}")
    for run, scenario in zip(runs, ("SABC", "SBAC")):
        metrics_path = run / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing history metrics file: {metrics_path}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            history_rows = list(csv.DictReader(handle))
        expected_epochs = list(range(1, epochs_total + 1))
        actual_epochs = [int(row["epoch"]) for row in history_rows]
        if actual_epochs != expected_epochs:
            raise ValueError(f"History metrics epochs in {metrics_path} are {actual_epochs}, expected {expected_epochs}")
        phase_data = ("A", "B") if scenario == "SABC" else ("B", "A")
        for row in history_rows:
            expected_phase = 1 if int(row["epoch"]) <= phase_epochs else 2
            if int(row["phase"]) != expected_phase or row["phase_data"] != phase_data[expected_phase - 1]:
                raise ValueError(f"Invalid {scenario} history phase at epoch {row['epoch']} in {metrics_path}")
    maps = tuple({step: relaxation_checkpoint_path(run, step) for step in steps} for run in runs)
    for checkpoint_map in maps:
        for path in checkpoint_map.values():
            if not path.exists():
                raise FileNotFoundError(f"Missing relaxation checkpoint: {path}")
    for epoch_path, checkpoint_map in zip(epoch_final_paths, maps):
        post_history, relax_zero = load_checkpoint(epoch_path), load_checkpoint(checkpoint_map[0])
        validate_state_dicts(post_history, relax_zero, check_dtype=True)
        if any(not torch.equal(post_history[key], relax_zero[key]) for key in post_history):
            raise ValueError(f"Relaxation step 0 does not equal post-history checkpoint: {epoch_path}")
    for run, scenario in zip(runs, ("SABC", "SBAC")):
        metrics_path = run / "relaxation_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing relaxation metrics file: {metrics_path}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metric_rows = list(csv.DictReader(handle))
        metric_steps = [int(row["relaxation_step"]) for row in metric_rows]
        if metric_steps != list(steps):
            raise ValueError(f"Relaxation metrics steps in {metrics_path} are {metric_steps}, expected {list(steps)}")
        if any(row.get("scenario") != scenario for row in metric_rows):
            raise ValueError(f"Relaxation metrics scenario mismatch in {metrics_path}")
    final_states = tuple(load_checkpoint(checkpoint_map[total]) for checkpoint_map in maps)
    validate_state_dicts(*final_states, check_dtype=True)
    return CommonRelaxationPair(config_sabc, config_sbac, init_paths[0], init_paths[1],
                                maps[0], maps[1], steps)


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_mapping(child, path))
    return flattened


def validate_analysis_pair(run_sab: Path | str, run_sba: Path | str) -> AnalysisPair:
    """Validate scientific comparability and return reusable pair artifacts."""
    runs = (Path(run_sab), Path(run_sba))
    config_paths = tuple(run / "config_resolved.yaml" for run in runs)
    for path in config_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing resolved config: {path}")
    config_sab, config_sba = (load_yaml(path) for path in config_paths)
    scenarios = (config_sab.get("train", {}).get("scenario"), config_sba.get("train", {}).get("scenario"))
    if scenarios != ("SAB", "SBA"):
        raise ValueError(f"Analysis pair must be ordered SAB then SBA, got {scenarios}")

    flat_sab, flat_sba = _flatten_mapping(config_sab), _flatten_mapping(config_sba)
    differences = {
        key: (flat_sab.get(key), flat_sba.get(key))
        for key in sorted(set(flat_sab) | set(flat_sba))
        if flat_sab.get(key) != flat_sba.get(key) and key not in PAIR_CONFIG_ALLOWLIST
    }
    if differences:
        raise ValueError(f"Disallowed config differences: {differences}")
    missing_protocol = [key for key in PROTOCOL_FIELDS if key not in flat_sab or key not in flat_sba]
    if missing_protocol:
        raise ValueError(f"Missing required model/data protocol fields: {missing_protocol}")

    init_paths = tuple(run / "weights_epoch_000.pt" for run in runs)
    for path in init_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing initialization checkpoint: {path}")
    init_states = tuple(load_checkpoint(path) for path in init_paths)
    validate_state_dicts(*init_states, check_dtype=True)
    for key in init_states[0]:
        if not torch.equal(init_states[0][key], init_states[1][key]):
            raise ValueError(f"Initialization tensor differs for {key}")

    total_epochs = int(config_sab["train"]["epochs_total"])
    final_paths = tuple(run / f"weights_epoch_{total_epochs:03d}.pt" for run in runs)
    for path in final_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing expected final checkpoint: {path}")
    final_states = tuple(load_checkpoint(path) for path in final_paths)
    validate_state_dicts(*final_states)
    return AnalysisPair(config_sab, config_sba, init_paths[0], init_paths[1], final_paths[0], final_paths[1])


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Tensor]:
    value = torch.load(Path(path), map_location=map_location, weights_only=True)
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict) or not all(isinstance(v, Tensor) for v in value.values()):
        raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
    return value


def build_model(config: Mapping[str, Any]) -> SimpleCNN:
    model_cfg = config["model"]
    if model_cfg.get("arch") != "simple_cnn":
        raise ValueError(f"Unsupported architecture: {model_cfg.get('arch')}")
    return SimpleCNN(model_cfg.get("norm", "none"), int(model_cfg.get("group_norm_groups", 8)),
                     model_cfg.get("activation", "relu"), float(model_cfg.get("leaky_relu_negative_slope", 0.01)))


def load_model_checkpoint(config, checkpoint, device="cpu") -> nn.Module:
    model = build_model(config).to(device)
    model.load_state_dict(load_checkpoint(checkpoint, device), strict=True)
    model.eval()
    return model


def extract_epoch(path: Path | str) -> int:
    match = EPOCH_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Not an epoch checkpoint filename: {path}")
    return int(match.group(1))


def sorted_checkpoints(run_dir: Path | str) -> list[Path]:
    return sorted(Path(run_dir).glob("weights_epoch_*.pt"), key=extract_epoch)


def validate_state_dicts(a: Mapping[str, Tensor], b: Mapping[str, Tensor], check_dtype: bool = False) -> None:
    if set(a) != set(b):
        raise ValueError(f"State-dict keys differ: only A={sorted(set(a)-set(b))}, only B={sorted(set(b)-set(a))}")
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(a[key].shape)} vs {tuple(b[key].shape)}")
        if check_dtype and a[key].dtype != b[key].dtype:
            raise ValueError(f"Dtype mismatch for {key}: {a[key].dtype} vs {b[key].dtype}")


def flatten_state_dict(state: Mapping[str, Tensor], floating_only: bool = True) -> Tensor:
    values = [state[key].detach().cpu().reshape(-1) for key in sorted(state)
              if not floating_only or state[key].is_floating_point()]
    return torch.cat(values) if values else torch.empty(0)


def flatten_model_parameters(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().cpu().reshape(-1) for _, parameter in sorted(model.named_parameters())])


def parameter_delta(state: Mapping[str, Tensor], initial: Mapping[str, Tensor]) -> Tensor:
    validate_state_dicts(state, initial)
    return flatten_state_dict(state) - flatten_state_dict(initial)


def interpolate_state_dicts(sab: Mapping[str, Tensor], sba: Mapping[str, Tensor], alpha: float) -> dict[str, Tensor]:
    """Return W(alpha)=(1-alpha) W_SAB + alpha W_SBA."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    validate_state_dicts(sab, sba)
    result = {}
    for key in sab:
        if sab[key].is_floating_point():
            result[key] = torch.lerp(sab[key], sba[key], alpha)
        else:
            if not torch.equal(sab[key], sba[key]):
                raise ValueError(f"Non-floating tensor differs for {key}")
            result[key] = sab[key].clone()
    return result


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def create_output_dir(path: Path | str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: Path | str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames or (list(rows[0]) if rows else []))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def final_checkpoint(run_dir: Path | str) -> Path:
    checkpoints = sorted_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def fixed_probe_loader(config: dict[str, Any], samples_per_class: int, download: bool = False) -> DataLoader:
    validate_samples_per_class(samples_per_class)
    _, test, *_ = get_mnist_datasets(config, download=download)
    if config["data"].get("protocol", "class_split") == "rotated_mnist":
        selected = balanced_rotated_indices(test, samples_per_class)
        return make_loader(test, selected, config, int(config["experiment"]["seed"]), False)
    selected = []
    for digit in range(10):
        candidates = indices_by_digits(test, [digit])
        if len(candidates) < samples_per_class:
            raise ValueError(f"Digit {digit} has only {len(candidates)} test examples")
        selected.extend(candidates[:samples_per_class])
    return make_loader(test, selected, config, int(config["experiment"]["seed"]), False)


def validate_samples_per_class(samples_per_class: int) -> None:
    """Validate a balanced-probe sample count without loading a dataset."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than 0")


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device, criterion: nn.Module | None = None) -> tuple[float, float]:
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_sum = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        count += y.size(0)
    return loss_sum / max(count, 1), correct / max(count, 1)
