"""Raw, unaligned linear interpolation between final model states."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.common import (build_model, create_output_dir, evaluate, interpolate_state_dicts,
                                 load_checkpoint, resolve_device, validate_analysis_pair, write_csv, write_json)
from src.data import get_mnist_datasets, make_loader


def barrier_summary(alphas, losses) -> dict[str, float]:
    index = int(np.argmax(losses))
    endpoint = max(float(losses[0]), float(losses[-1]))
    maximum = float(losses[index])
    return {"barrier_height": maximum - endpoint, "alpha_at_maximum": float(alphas[index]),
            "endpoint_loss_sab": float(losses[0]), "endpoint_loss_sba": float(losses[-1]),
            "maximum_interpolated_loss": maximum}


def validate_num_points(num_points: int) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sab", required=True); parser.add_argument("--run-sba", required=True)
    parser.add_argument("--num-points", type=int, default=51); parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    validate_num_points(args.num_points)
    pair = validate_analysis_pair(args.run_sab, args.run_sba)
    out, device = create_output_dir(args.outdir), resolve_device(args.device)
    config = pair.config_sab
    sab, sba = load_checkpoint(pair.final_sab), load_checkpoint(pair.final_sba)
    _, test, _, _, a_idx, b_idx, full_idx = get_mnist_datasets(config, download=args.download)
    loaders = {name: make_loader(test, indices, config, int(config["experiment"]["seed"]), False)
               for name, indices in (("A", a_idx), ("B", b_idx), ("full", full_idx))}
    alphas, rows = np.linspace(0, 1, args.num_points), []
    model = build_model(config).to(device)
    for alpha in alphas:
        model.load_state_dict(interpolate_state_dicts(sab, sba, float(alpha))); model.eval()
        metrics = {name: evaluate(model, loader, device) for name, loader in loaders.items()}
        rows.append({"alpha": float(alpha), **{f"loss_{n}": v[0] for n, v in metrics.items()},
                     **{f"accuracy_{n}": v[1] for n, v in metrics.items()}})
    write_csv(out / "interpolation_metrics.csv", rows)
    summary = {name: barrier_summary(alphas, [row[f"loss_{name}"] for row in rows]) for name in loaders}
    summary["convention"] = "W(alpha)=(1-alpha)W_SAB+alpha W_SBA; raw, not permutation-aligned"
    write_json(out / "interpolation_summary.json", summary)
    for kind in ("loss", "accuracy"):
        plt.figure()
        for name in loaders: plt.plot(alphas, [r[f"{kind}_{name}"] for r in rows], label=name)
        plt.xlabel("alpha (0=SAB, 1=SBA)"); plt.ylabel(kind); plt.legend(); plt.tight_layout(); plt.savefig(out / f"interpolation_{kind}.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
