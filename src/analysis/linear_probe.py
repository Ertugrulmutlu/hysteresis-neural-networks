"""Train deterministic, paired fresh linear probes on frozen representations."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.common import (create_output_dir, load_model_checkpoint,
                                 validate_common_relaxation_pair,
                                 validate_samples_per_class, write_csv, write_json)
from src.data import balanced_indices, get_mnist_datasets


DOMAINS = ("full", "A", "B")


def parse_checkpoint_selector(selector: str, steps: tuple[int, ...], metrics_a=None,
                              metrics_b=None, max_accuracy_gap: float = .002,
                              minimum_accuracy: float = .97,
                              numerical_tolerance: float = 1e-12) -> int:
    if selector == "final":
        return steps[-1]
    if selector.startswith("relaxation-step:"):
        step = int(selector.split(":", 1)[1])
        if step not in steps:
            raise ValueError(f"Relaxation step {step} is not available")
        return step
    if selector == "performance-matched":
        if metrics_a is None or metrics_b is None:
            raise ValueError("Performance-matched selection requires relaxation metrics")
        for step in steps:
            a, b = metrics_a[step], metrics_b[step]
            if (abs(a - b) <= max_accuracy_gap + numerical_tolerance
                    and min(a, b) >= minimum_accuracy):
                return step
        raise ValueError("No performance-matched checkpoint satisfies the declared criterion")
    raise ValueError("checkpoint must be performance-matched, final, or relaxation-step:<integer>")


def _accuracy_map(run: Path) -> dict[int, float]:
    with (run / "relaxation_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return {int(row["relaxation_step"]): float(row["test_acc_full"])
                for row in csv.DictReader(handle)}


def indices_hash(indices: Iterable[int]) -> str:
    """Stable hash independent of Python/container representation details."""
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def paired_indices(config, samples: int, train: bool):
    train_ds, test_ds, a_train, b_train, a_test, b_test, _ = get_mnist_datasets(config, download=False)
    dataset = train_ds if train else test_ds
    a_source, b_source = (a_train, b_train) if train else (a_test, b_test)
    protocol = config["data"].get("protocol", "class_split")
    if protocol == "class_split":
        full = balanced_indices(dataset, range(10), samples)
        a_set, b_set = set(a_source), set(b_source)
        a = [index for index in full if index in a_set]
        b = [index for index in full if index in b_set]
    else:
        half = len(dataset) // 2
        a = balanced_indices(dataset.datasets[0], range(10), samples)
        b = [half + index for index in balanced_indices(dataset.datasets[1], range(10), samples)]
        full = a + b
    return dataset, {"full": list(full), "A": list(a), "B": list(b)}


def indexed_loader(dataset, indices: list[int], batch_size: int) -> DataLoader:
    examples = [dataset[index] for index in indices]
    x = torch.stack([example[0] for example in examples])
    y = torch.as_tensor([int(example[1]) for example in examples], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def freeze_backbone(model: nn.Module) -> dict[str, Tensor]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_dict_exactly_equal(before: dict[str, Tensor], model: nn.Module) -> bool:
    after = model.state_dict()
    return before.keys() == after.keys() and all(
        torch.equal(value, after[name].detach().cpu()) for name, value in before.items())


@torch.no_grad()
def extract_features(model: nn.Module, loader, device, layer: str) -> tuple[Tensor, Tensor]:
    if layer not in {"fc1", "conv2"}:
        raise ValueError("feature layer must be fc1 or conv2")
    model.eval()
    features, labels = [], []
    for x, y in loader:
        _, activations = model(x.to(device), return_activations=True)
        feature = activations["fc1"] if layer == "fc1" else activations["conv2"].mean(dim=(2, 3))
        features.append(feature.detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(features), torch.cat(labels)


def clone_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def make_identical_heads(feature_dimension: int, seed: int) -> tuple[nn.Linear, nn.Linear, bool]:
    torch.manual_seed(seed)
    template = nn.Linear(feature_dimension, 10)
    initial = clone_state_dict(template.state_dict())
    heads = (nn.Linear(feature_dimension, 10), nn.Linear(feature_dimension, 10))
    for head in heads:
        head.load_state_dict(clone_state_dict(initial))
    equal = all(torch.equal(heads[0].state_dict()[key], heads[1].state_dict()[key]) for key in initial)
    return heads[0], heads[1], equal


def deterministic_epoch_order(size: int, seed: int, epoch: int) -> Tensor:
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch))


@torch.no_grad()
def _score(head: nn.Module, features: Tensor, labels: Tensor) -> tuple[float, float]:
    logits = head(features)
    return (float(nn.functional.cross_entropy(logits, labels)),
            float((logits.argmax(1) == labels).float().mean()))


def train_linear_probe(head: nn.Linear, train_data: tuple[Tensor, Tensor],
                       test_data: dict[str, tuple[Tensor, Tensor]], epochs: int, lr: float,
                       weight_decay: float, momentum: float, batch_size: int, seed: int):
    features, labels = train_data
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    history = []
    for epoch in range(1, epochs + 1):
        order = deterministic_epoch_order(len(labels), seed, epoch)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(head(features[batch]), labels[batch]).backward()
            optimizer.step()
        train_loss, train_accuracy = _score(head, features, labels)
        row = {"probe_epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy}
        for domain, data in test_data.items():
            row[f"test_accuracy_{domain}"] = _score(head, *data)[1]
        history.append(row)
    return history


def validate_probe_arguments(args) -> None:
    validate_samples_per_class(args.train_samples_per_class)
    validate_samples_per_class(args.test_samples_per_class)
    if args.probe_epochs <= 0 or args.probe_lr <= 0 or args.probe_batch_size <= 0:
        raise ValueError("probe epochs, learning rate, and batch size must be positive")
    if args.probe_weight_decay < 0 or args.probe_momentum < 0:
        raise ValueError("probe weight decay and momentum must be non-negative")
    if args.max_accuracy_gap < 0 or args.minimum_accuracy < 0 or args.numerical_tolerance < 0:
        raise ValueError("matching thresholds must be non-negative")


def add_probe_arguments(parser: argparse.ArgumentParser, include_runs: bool = True) -> None:
    if include_runs:
        parser.add_argument("--run-sabc", required=True); parser.add_argument("--run-sbac", required=True)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--max-accuracy-gap", type=float, default=.002)
    parser.add_argument("--minimum-accuracy", type=float, default=.97)
    parser.add_argument("--numerical-tolerance", type=float, default=1e-12)
    parser.add_argument("--feature-layer", choices=("conv2", "fc1"), default="fc1")
    parser.add_argument("--train-samples-per-class", type=int, default=500)
    parser.add_argument("--test-samples-per-class", type=int, default=200)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-lr", type=float, default=.01)
    parser.add_argument("--probe-weight-decay", type=float, default=0.)
    parser.add_argument("--probe-momentum", type=float, default=0.)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-seed", type=int, default=777)
    parser.add_argument("--device", default="cpu")


def run_pair(run_sabc: str | Path, run_sbac: str | Path, args, outdir: str | Path | None = None):
    validate_probe_arguments(args)
    pair = validate_common_relaxation_pair(run_sabc, run_sbac)
    step = parse_checkpoint_selector(
        args.checkpoint, pair.relaxation_steps, _accuracy_map(Path(run_sabc)), _accuracy_map(Path(run_sbac)),
        args.max_accuracy_gap, args.minimum_accuracy, args.numerical_tolerance)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    train_dataset, train_indices = paired_indices(pair.config_sabc, args.train_samples_per_class, True)
    test_dataset, test_indices = paired_indices(pair.config_sabc, args.test_samples_per_class, False)
    train_loader = indexed_loader(train_dataset, train_indices["full"], args.probe_batch_size)
    test_loaders = {domain: indexed_loader(test_dataset, indices, args.probe_batch_size)
                    for domain, indices in test_indices.items()}
    models = [load_model_checkpoint(config, checkpoints[step], device) for config, checkpoints in
              ((pair.config_sabc, pair.checkpoints_sabc), (pair.config_sbac, pair.checkpoints_sbac))]
    before = [freeze_backbone(model) for model in models]
    train_data = [extract_features(model, train_loader, device, args.feature_layer) for model in models]
    test_data = [{domain: extract_features(model, loader, device, args.feature_layer)
                  for domain, loader in test_loaders.items()} for model in models]
    if train_data[0][0].shape[1] != train_data[1][0].shape[1]:
        raise RuntimeError("SABC and SBAC feature dimensions differ")
    if not torch.equal(train_data[0][1], train_data[1][1]) or any(
            not torch.equal(test_data[0][d][1], test_data[1][d][1]) for d in DOMAINS):
        raise RuntimeError("SABC and SBAC label ordering differs")
    heads = make_identical_heads(train_data[0][0].shape[1], args.probe_seed)
    if not heads[2]:
        raise RuntimeError("Fresh head initializations are not exactly equal")
    histories = [train_linear_probe(head, data, tests, args.probe_epochs, args.probe_lr,
                                    args.probe_weight_decay, args.probe_momentum,
                                    args.probe_batch_size, args.probe_seed)
                 for head, data, tests in zip(heads[:2], train_data, test_data)]
    unchanged = [state_dict_exactly_equal(snapshot, model) for snapshot, model in zip(before, models)]
    if not all(unchanged):
        raise RuntimeError("A frozen backbone changed during linear probing")
    seed = int(pair.config_sabc["experiment"]["seed"])
    rows = []
    for scenario, history in zip(("SABC", "SBAC"), histories):
        for metrics in history:
            rows.append({"seed": seed, "scenario": scenario, "feature_layer": args.feature_layer,
                         "checkpoint_selector": args.checkpoint, "checkpoint_step": step, **metrics})
    final = {scenario: history[-1] for scenario, history in zip(("SABC", "SBAC"), histories)}
    signed = {domain: final["SABC"][f"test_accuracy_{domain}"] - final["SBAC"][f"test_accuracy_{domain}"]
              for domain in DOMAINS}
    selected_epochs = {str(epoch): {scenario: history[epoch - 1]
                                    for scenario, history in zip(("SABC", "SBAC"), histories)}
                       for epoch in (1, 2, 5, 10, 20) if epoch <= args.probe_epochs}
    summary = {
        "seed": seed, "selected_checkpoint": args.checkpoint, "checkpoint_step": step,
        "feature_layer": args.feature_layer, "feature_dimension": train_data[0][0].shape[1],
        "probe_seed": args.probe_seed,
        "probe_hyperparameters": {"epochs": args.probe_epochs, "lr": args.probe_lr,
                                  "weight_decay": args.probe_weight_decay, "momentum": args.probe_momentum,
                                  "batch_size": args.probe_batch_size},
        "sample_counts": {"train_full": len(train_indices["full"]),
                          **{f"test_{d}": len(v) for d, v in test_indices.items()}},
        "index_hashes": {"train_indices": indices_hash(train_indices["full"]),
                         **{f"test_{d}_indices": indices_hash(v) for d, v in test_indices.items()}},
        "final_accuracies": {scenario: {domain: metrics[f"test_accuracy_{domain}"] for domain in DOMAINS}
                             for scenario, metrics in final.items()},
        "signed_accuracy_difference_sabc_minus_sbac": signed,
        "absolute_accuracy_difference": {domain: abs(value) for domain, value in signed.items()},
        "selected_epoch_metrics": selected_epochs,
        "backbone_unchanged_sabc": unchanged[0], "backbone_unchanged_sbac": unchanged[1],
        "head_initializations_exactly_equal": heads[2],
    }
    if outdir is not None:
        out = create_output_dir(outdir)
        write_csv(out / "linear_probe_epoch_metrics.csv", rows)
        write_json(out / "linear_probe_summary.json", summary)
        _plots(out, rows, summary)
    return rows, summary


def _plots(out: Path, rows: list[dict], summary: dict) -> None:
    plt.figure()
    for scenario in ("SABC", "SBAC"):
        selected = [row for row in rows if row["scenario"] == scenario]
        plt.plot([r["probe_epoch"] for r in selected], [r["train_loss"] for r in selected], label=scenario)
    plt.xlabel("Probe epoch"); plt.ylabel("Training loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "linear_probe_training_curve.png", dpi=200); plt.close()
    labels = [f"{scenario}-{domain}" for scenario in ("SABC", "SBAC") for domain in DOMAINS]
    values = [summary["final_accuracies"][scenario][domain]
              for scenario in ("SABC", "SBAC") for domain in DOMAINS]
    plt.figure(); plt.bar(labels, values); plt.ylabel("Final probe accuracy"); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(out / "linear_probe_final_comparison.png", dpi=200); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(); add_probe_arguments(parser); parser.add_argument("--outdir", required=True)
    args = parser.parse_args(); run_pair(args.run_sabc, args.run_sbac, args, args.outdir)


if __name__ == "__main__":
    main()
