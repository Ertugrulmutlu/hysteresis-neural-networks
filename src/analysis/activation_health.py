"""Post-ReLU unit activation-health statistics."""
import argparse

import matplotlib.pyplot as plt
import torch

from src.analysis.common import (create_output_dir, fixed_probe_loader, load_model_checkpoint, resolve_device,
                                 validate_analysis_pair, validate_samples_per_class, write_csv, write_json)

LAYERS = ("conv1", "conv2", "fc1")


def validate_activation_arguments(samples_per_class: int, epsilon: float, dead_threshold: float) -> None:
    validate_samples_per_class(samples_per_class)
    if epsilon < 0:
        raise ValueError("epsilon must be greater than or equal to 0")
    if not 0 <= dead_threshold <= 1:
        raise ValueError("dead_threshold must be between 0 and 1 inclusive")


@torch.no_grad()
def collect_units(model, loader, device) -> dict[str, torch.Tensor]:
    values = {layer: [] for layer in LAYERS}; model.eval()
    for x, _ in loader:
        _, acts = model(x.to(device), return_activations=True)
        for layer in LAYERS: values[layer].append(acts[layer].cpu())
    return {layer: torch.cat(parts) for layer, parts in values.items()}


def unit_statistics(values: torch.Tensor, epsilon: float, dead_threshold: float) -> list[dict[str, float | bool | int]]:
    if values.ndim == 4: values = values.permute(1, 0, 2, 3).flatten(1)
    elif values.ndim == 2: values = values.T
    else: raise ValueError("Expected NCHW or ND activations")
    rows = []
    for index, unit in enumerate(values):
        nonzero = float((unit > epsilon).float().mean())
        rows.append({"unit": index, "mean_activation": float(unit.mean()), "variance": float(unit.var(unbiased=False)),
                     "nonzero_rate": nonzero, "sparsity_rate": 1.0 - nonzero, "dead": nonzero < dead_threshold})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-sab", required=True); parser.add_argument("--run-sba", required=True)
    parser.add_argument("--samples-per-class", type=int, default=200); parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--dead-threshold", type=float, default=1e-4); parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    validate_activation_arguments(args.samples_per_class, args.epsilon, args.dead_threshold)
    pair = validate_analysis_pair(args.run_sab, args.run_sba)
    out, device = create_output_dir(args.outdir), resolve_device(args.device); unit_rows, summaries = [], []
    for scenario, config, checkpoint in (("SAB", pair.config_sab, pair.final_sab), ("SBA", pair.config_sba, pair.final_sba)):
        loader = fixed_probe_loader(config, args.samples_per_class, args.download)
        model = load_model_checkpoint(config, checkpoint, device); activations = collect_units(model, loader, device)
        for layer in LAYERS:
            stats = unit_statistics(activations[layer], args.epsilon, args.dead_threshold)
            unit_rows.extend({"scenario": scenario, "layer": layer, **row} for row in stats)
            dead = sum(bool(row["dead"]) for row in stats); n = len(stats)
            summaries.append({"scenario": scenario, "layer": layer, "number_of_units": n, "dead_unit_count": dead,
                              "dead_ratio": dead/n, "average_activation_mean": sum(float(r["mean_activation"]) for r in stats)/n,
                              "average_activation_variance": sum(float(r["variance"]) for r in stats)/n,
                              "average_nonzero_rate": sum(float(r["nonzero_rate"]) for r in stats)/n,
                              "average_sparsity": sum(float(r["sparsity_rate"]) for r in stats)/n})
    write_csv(out / "activation_health_units.csv", unit_rows); write_csv(out / "activation_health_summary.csv", summaries)
    write_json(out / "activation_health_summary.json", {"epsilon": args.epsilon, "dead_threshold": args.dead_threshold, "summaries": summaries})
    plt.figure(); x = range(len(LAYERS)); width=.35
    for offset, scenario in ((-.5, "SAB"), (.5, "SBA")):
        vals = [next(r["dead_ratio"] for r in summaries if r["scenario"] == scenario and r["layer"] == layer) for layer in LAYERS]
        plt.bar([i + offset*width for i in x], vals, width, label=scenario)
    plt.xticks(list(x), LAYERS); plt.ylabel("dead unit ratio"); plt.legend(); plt.tight_layout(); plt.savefig(out / "activation_health.png", dpi=200); plt.close()


if __name__ == "__main__": main()
