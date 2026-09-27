"""Analyze whether AB/BA history survives a shared balanced-MNIST relaxation phase."""
import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from src.analysis.activation_health import (LAYERS as HEALTH_LAYERS, collect_units, unit_statistics,
                                             validate_activation_arguments)
from src.analysis.cka import LAYERS, collect_activations, linear_cka
from src.analysis.common import (create_output_dir, fixed_probe_loader, flatten_state_dict, load_checkpoint,
                                 load_model_checkpoint, resolve_device, validate_common_relaxation_pair,
                                 write_csv, write_json)


def prediction_disagreement(probabilities_a: Tensor, probabilities_b: Tensor) -> float:
    if probabilities_a.shape != probabilities_b.shape or probabilities_a.ndim != 2:
        raise ValueError("Probability matrices must have the same [samples, classes] shape")
    return float((probabilities_a.argmax(1) != probabilities_b.argmax(1)).float().mean())


def mean_js_divergence(probabilities_a: Tensor, probabilities_b: Tensor, epsilon: float = 1e-12) -> float:
    if probabilities_a.shape != probabilities_b.shape or probabilities_a.ndim != 2:
        raise ValueError("Probability matrices must have the same [samples, classes] shape")
    p = probabilities_a.double().clamp_min(epsilon); q = probabilities_b.double().clamp_min(epsilon)
    p = p / p.sum(1, keepdim=True); q = q / q.sum(1, keepdim=True); midpoint = 0.5 * (p + q)
    divergence = 0.5 * ((p * (p.log() - midpoint.log())).sum(1) + (q * (q.log() - midpoint.log())).sum(1))
    return float(divergence.mean())


def normalized_weight_distance(
    state_a: Mapping[str, Tensor], state_b: Mapping[str, Tensor], initial: Mapping[str, Tensor]
) -> float:
    a, b, origin = flatten_state_dict(state_a), flatten_state_dict(state_b), flatten_state_dict(initial)
    if a.shape != b.shape or a.shape != origin.shape:
        raise ValueError("Flattened state dictionaries must have matching shapes")
    denominator = 0.5 * (torch.linalg.vector_norm(a - origin) + torch.linalg.vector_norm(b - origin))
    return 0.0 if float(denominator) == 0.0 else float(torch.linalg.vector_norm(a - b) / denominator)


def representation_history_score(cka_values: Mapping[str, float]) -> float:
    return 1.0 - 0.5 * (float(cka_values["conv2"]) + float(cka_values["fc1"]))


def select_performance_matched_endpoint(
    rows: Sequence[Mapping[str, Any]], max_gap: float, minimum_accuracy: float
) -> dict[str, Any] | None:
    if max_gap < 0 or not 0 <= minimum_accuracy <= 1:
        raise ValueError("Performance-match thresholds must satisfy max_gap >= 0 and minimum_accuracy in [0, 1]")
    for row in sorted(rows, key=lambda item: int(item["relaxation_step"])):
        a, b = float(row["full_accuracy_sabc"]), float(row["full_accuracy_sbac"])
        if abs(a - b) <= max_gap and min(a, b) >= minimum_accuracy:
            return dict(row)
    return None


def _load_relaxation_metrics(run: Path, expected_steps: Sequence[int]) -> dict[int, dict[str, str]]:
    path = run / "relaxation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing relaxation metrics: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row["relaxation_step"]) for row in rows]
    if steps != list(expected_steps):
        raise ValueError(f"Relaxation metrics steps in {path} are {steps}, expected {list(expected_steps)}")
    return {int(row["relaxation_step"]): row for row in rows}


@torch.no_grad()
def _probabilities(model, loader, device) -> torch.Tensor:
    model.eval(); values = []
    for x, _ in loader:
        values.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(values)


def _health_summary(activations, epsilon: float, dead_threshold: float) -> dict[str, dict[str, float]]:
    result = {}
    for layer in HEALTH_LAYERS:
        units = unit_statistics(activations[layer], epsilon, dead_threshold); count = len(units)
        below_threshold = sum(bool(row["dead"]) for row in units) / count
        positive_activity = sum(float(row["nonzero_rate"]) for row in units) / count
        result[layer] = {"dead_ratio": below_threshold,
                         "below_threshold_positive_activity_ratio": below_threshold,
                         "average_positive_activity_rate": positive_activity,
                         "average_sparsity": sum(float(row["sparsity_rate"]) for row in units) / count,
                         "average_activation_variance": sum(float(row["variance"]) for row in units) / count}
    return result


def _plot_series(path: Path, rows, columns, labels, ylabel: str) -> None:
    plt.figure()
    steps = [row["relaxation_step"] for row in rows]
    for column, label in zip(columns, labels):
        plt.plot(steps, [row[column] for row in rows], marker="o", label=label)
    plt.xscale("symlog", linthresh=1); plt.xlabel("relaxation updates"); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def compute_common_relaxation_rows(pair, metrics_a, metrics_b, loader, device, epsilon: float, dead_threshold: float):
    """Compute all existing common-relaxation metrics for a validated paired seed."""
    initial = load_checkpoint(pair.init_sabc); rows = []
    for step in pair.relaxation_steps:
        checkpoints = (pair.checkpoints_sabc[step], pair.checkpoints_sbac[step])
        models = (load_model_checkpoint(pair.config_sabc, checkpoints[0], device),
                  load_model_checkpoint(pair.config_sbac, checkpoints[1], device))
        probabilities = tuple(_probabilities(model, loader, device) for model in models)
        representations = tuple(collect_activations(model, loader, device) for model in models)
        cka_values = {layer: linear_cka(representations[0][layer], representations[1][layer]) for layer in LAYERS}
        health = tuple(_health_summary(collect_units(model, loader, device), epsilon, dead_threshold) for model in models)
        health_differences = {f"{layer}_{metric}_difference": abs(health[0][layer][metric] - health[1][layer][metric])
                              for layer in HEALTH_LAYERS for metric in (
                                  "dead_ratio", "below_threshold_positive_activity_ratio",
                                  "average_positive_activity_rate", "average_sparsity",
                                  "average_activation_variance")}
        state_a, state_b = load_checkpoint(checkpoints[0]), load_checkpoint(checkpoints[1]); ma, mb = metrics_a[step], metrics_b[step]
        signed_full_accuracy = float(ma["test_acc_full"]) - float(mb["test_acc_full"])
        signed_full_loss = float(ma["test_loss_full"]) - float(mb["test_loss_full"])
        rows.append({"relaxation_step": step, "full_accuracy_sabc": float(ma["test_acc_full"]), "full_accuracy_sbac": float(mb["test_acc_full"]),
            "signed_full_accuracy_difference_sabc_minus_sbac": signed_full_accuracy,
            "absolute_accuracy_gap": abs(signed_full_accuracy), "full_accuracy_difference": abs(signed_full_accuracy), "A_accuracy_sabc": float(ma["test_acc_A"]),
            "A_accuracy_sbac": float(mb["test_acc_A"]), "B_accuracy_sabc": float(ma["test_acc_B"]), "B_accuracy_sbac": float(mb["test_acc_B"]),
            "A_accuracy_difference": abs(float(ma["test_acc_A"])-float(mb["test_acc_A"])), "B_accuracy_difference": abs(float(ma["test_acc_B"])-float(mb["test_acc_B"])),
            "full_loss_difference_sabc_minus_sbac": signed_full_loss,
            "full_loss_difference": abs(signed_full_loss), "prediction_disagreement": prediction_disagreement(*probabilities),
            "js_divergence": mean_js_divergence(*probabilities), **{f"cka_{layer}": value for layer,value in cka_values.items()},
            "representation_history_score": representation_history_score(cka_values), "normalized_weight_distance": normalized_weight_distance(state_a,state_b,initial), **health_differences})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sabc", required=True); parser.add_argument("--run-sbac", required=True)
    parser.add_argument("--samples-per-class", type=int, default=200); parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-accuracy-gap", type=float, default=0.002); parser.add_argument("--minimum-accuracy", type=float, default=0.97)
    parser.add_argument("--epsilon", type=float, default=1e-6); parser.add_argument("--dead-threshold", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--download", action="store_true")
    args = parser.parse_args(); validate_activation_arguments(args.samples_per_class, args.epsilon, args.dead_threshold)
    pair = validate_common_relaxation_pair(args.run_sabc, args.run_sbac)
    metrics_a = _load_relaxation_metrics(Path(args.run_sabc), pair.relaxation_steps)
    metrics_b = _load_relaxation_metrics(Path(args.run_sbac), pair.relaxation_steps)
    device = resolve_device(args.device); loader = fixed_probe_loader(pair.config_sabc, args.samples_per_class, args.download)
    rows = compute_common_relaxation_rows(pair, metrics_a, metrics_b, loader, device, args.epsilon, args.dead_threshold)
    endpoint = select_performance_matched_endpoint(rows, args.max_accuracy_gap, args.minimum_accuracy)
    out = create_output_dir(args.outdir); write_csv(out / "common_relaxation_metrics.csv", rows)
    write_json(out / "common_relaxation_summary.json", {"performance_matched_endpoint": endpoint,
               "max_accuracy_gap": args.max_accuracy_gap, "minimum_accuracy": args.minimum_accuracy,
               "data_protocol": pair.config_sabc["data"].get("protocol", "class_split"),
               "activation": pair.config_sabc["model"].get("activation", "relu"),
               "relaxation_optimizer_policy": pair.config_sabc["train"].get("relaxation_optimizer_policy", "preserve"),
               "interpretation_note": "Persistence after common relaxation is descriptive and does not by itself prove hysteresis."})
    _plot_series(out/"accuracy_recovery.png", rows,
                 ("full_accuracy_sabc","full_accuracy_sbac","A_accuracy_sabc","A_accuracy_sbac","B_accuracy_sabc","B_accuracy_sbac"),
                 ("SABC full","SBAC full","SABC A","SBAC A","SABC B","SBAC B"), "test accuracy")
    _plot_series(out/"representation_history.png", rows, ("cka_conv1","cka_conv2","cka_fc1","cka_logits","representation_history_score"), ("conv1 CKA","conv2 CKA","fc1 CKA","logits CKA","history score"), "representation metric")
    _plot_series(out/"functional_history.png", rows, ("prediction_disagreement","js_divergence"), ("prediction disagreement","JS divergence"), "functional difference")
    _plot_series(out/"weight_history.png", rows, ("normalized_weight_distance",), ("normalized weight distance",), "normalized distance")
    activation_columns = tuple(f"{layer}_{metric}_difference" for layer in HEALTH_LAYERS for metric in ("dead_ratio","average_sparsity","average_activation_variance"))
    _plot_series(out/"activation_history.png", rows, activation_columns, activation_columns, "absolute activation-profile difference")


if __name__ == "__main__":
    main()
