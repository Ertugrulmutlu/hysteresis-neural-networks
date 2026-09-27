"""Linear CKA comparison of final SAB and SBA representations."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.common import (create_output_dir, fixed_probe_loader, load_model_checkpoint, resolve_device,
                                 validate_analysis_pair, validate_samples_per_class, write_csv, write_json)

LAYERS = ("conv1", "conv2", "fc1", "logits")


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute feature-space linear CKA after column centering."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("CKA inputs must be 2D [samples, features]")
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have equal sample counts")
    x, y = x.double() - x.double().mean(0), y.double() - y.double().mean(0)
    numerator = torch.linalg.norm(x.T @ y).square()
    denominator = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    if not torch.isfinite(denominator) or float(denominator) <= eps:
        return 0.0
    value = numerator / denominator
    if not torch.isfinite(value):
        raise ValueError("CKA produced a non-finite value")
    return float(value.clamp(0.0, 1.0).item())


def activation_features(value: torch.Tensor) -> torch.Tensor:
    """Map NCHW activations to N x C using global average pooling; retain N x D values."""
    if value.ndim == 4:
        return value.mean(dim=(2, 3))
    if value.ndim == 2:
        return value
    raise ValueError(f"Unsupported activation shape: {tuple(value.shape)}")


@torch.no_grad()
def collect_activations(model, loader, device) -> dict[str, torch.Tensor]:
    collected = {layer: [] for layer in LAYERS}
    model.eval()
    for x, _ in loader:
        _, activations = model(x.to(device), return_activations=True)
        for layer in LAYERS:
            collected[layer].append(activation_features(activations[layer]).cpu())
    return {layer: torch.cat(values) for layer, values in collected.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sab", required=True); parser.add_argument("--run-sba", required=True)
    parser.add_argument("--samples-per-class", type=int, default=200); parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    validate_samples_per_class(args.samples_per_class)
    pair = validate_analysis_pair(args.run_sab, args.run_sba)
    out = create_output_dir(args.outdir)
    configs = [pair.config_sab, pair.config_sba]
    checkpoints = [pair.final_sab, pair.final_sba]
    device = resolve_device(args.device)
    loader = fixed_probe_loader(configs[0], args.samples_per_class, args.download)
    models = [load_model_checkpoint(config, checkpoint, device) for config, checkpoint in zip(configs, checkpoints)]
    sab, sba = [collect_activations(model, loader, device) for model in models]
    matrix = np.array([[linear_cka(sab[a], sba[b]) for b in LAYERS] for a in LAYERS])
    diagonal = [{"layer": layer, "cka": float(matrix[i, i])} for i, layer in enumerate(LAYERS)]
    write_csv(out / "cka_diagonal.csv", diagonal)
    write_csv(out / "cka_cross_layer.csv", [{"sab_layer": a, **{b: float(matrix[i, j]) for j, b in enumerate(LAYERS)}} for i, a in enumerate(LAYERS)])
    score = 1.0 - float(np.mean(np.diag(matrix)[:3]))
    write_json(out / "cka_summary.json", {"representation_hysteresis": score, "layers": list(LAYERS),
                                           "probe_samples_per_class": args.samples_per_class,
                                           "conv_feature_method": "global_average_pooling"})
    plt.figure(); plt.bar(LAYERS, np.diag(matrix)); plt.ylim(0, 1); plt.ylabel("linear CKA"); plt.tight_layout(); plt.savefig(out / "cka_diagonal.png", dpi=200); plt.close()
    plt.figure(); plt.imshow(matrix, vmin=0, vmax=1, cmap="viridis"); plt.colorbar(label="linear CKA"); plt.xticks(range(4), LAYERS); plt.yticks(range(4), LAYERS); plt.xlabel("SBA"); plt.ylabel("SAB"); plt.tight_layout(); plt.savefig(out / "cka_cross_layer_heatmap.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
