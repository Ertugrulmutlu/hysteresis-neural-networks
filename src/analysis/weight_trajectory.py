"""Shared-PCA projection of SAB and SBA weight trajectories."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.analysis.common import (create_output_dir, extract_epoch, flatten_state_dict, load_checkpoint,
                                 sorted_checkpoints, validate_analysis_pair, validate_state_dicts,
                                 write_csv, write_json)


def trajectory_matrix(run_dir: Path | str, initial: dict[str, torch.Tensor]):
    paths = sorted_checkpoints(run_dir)
    vectors = []
    for path in paths:
        state = load_checkpoint(path); validate_state_dicts(state, initial)
        vectors.append((flatten_state_dict(state) - flatten_state_dict(initial)).numpy())
    return paths, np.stack(vectors)


def shared_pca(sab: np.ndarray, sba: np.ndarray):
    combined = np.concatenate([sab, sba], axis=0)
    if combined.shape[0] < 2: raise ValueError("At least two trajectory points are required")
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)
    return projected[:len(sab)], projected[len(sab):], pca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sab", required=True); parser.add_argument("--run-sba", required=True); parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    pair = validate_analysis_pair(args.run_sab, args.run_sba)
    out = create_output_dir(args.outdir)
    init_sab = load_checkpoint(pair.init_sab)
    sab_paths, sab = trajectory_matrix(args.run_sab, init_sab); sba_paths, sba = trajectory_matrix(args.run_sba, init_sab)
    if [extract_epoch(p) for p in sab_paths] != [extract_epoch(p) for p in sba_paths]: raise ValueError("Checkpoint epoch sets differ")
    sab_xy, sba_xy, pca = shared_pca(sab, sba)
    configs = {"SAB": pair.config_sab, "SBA": pair.config_sba}
    rows = []
    for scenario, paths, vectors, coords, other in (("SAB", sab_paths, sab, sab_xy, sba), ("SBA", sba_paths, sba, sba_xy, sab)):
        phase_epochs = int(configs[scenario]["train"]["phase_epochs"])
        for i, (path, vector, xy) in enumerate(zip(paths, vectors, coords)):
            epoch = extract_epoch(path); phase = 0 if epoch == 0 else (1 if epoch <= phase_epochs else 2)
            phase_data = "init" if not phase else (("A", "B") if scenario == "SAB" else ("B", "A"))[phase - 1]
            rows.append({"scenario": scenario, "epoch": epoch, "phase": phase, "phase_data": phase_data,
                         "pc1": float(xy[0]), "pc2": float(xy[1]), "distance_from_init": float(np.linalg.norm(vector)),
                         "distance_to_other_scenario_same_epoch": float(np.linalg.norm(vector - other[i]))})
    write_csv(out / "trajectory_coordinates.csv", rows)
    epochs = [extract_epoch(path) for path in sab_paths]
    phase_boundary_epoch = int(pair.config_sab["train"]["phase_epochs"])
    write_json(out / "pca_metadata.json", {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                                            "n_points": len(rows), "parameter_count": int(flatten_state_dict(init_sab).numel()),
                                            "checkpoint_epochs": epochs, "phase_boundary_epoch": phase_boundary_epoch,
                                            "centered_on_common_initialization": True,
                                            "geometry_note": "The 2D PCA projection does not preserve all weight-space geometry."})
    plt.figure()
    for scenario, coords, paths in (("SAB", sab_xy, sab_paths), ("SBA", sba_xy, sba_paths)):
        path_epochs = [extract_epoch(path) for path in paths]
        plt.plot(coords[:, 0], coords[:, 1], marker="o", label=scenario)
        plt.scatter(*coords[0], s=110, marker="*", label=f"{scenario} start")
        plt.scatter(*coords[-1], s=90, marker="X", label=f"{scenario} final")
        if phase_boundary_epoch in path_epochs:
            boundary_index = path_epochs.index(phase_boundary_epoch)
            plt.scatter(*coords[boundary_index], s=110, marker="D", facecolors="none",
                        linewidths=1.8, label=f"{scenario} phase boundary")
        for index in range(0, len(coords), max(1, len(coords)//5)):
            plt.annotate(str(path_epochs[index]), coords[index])
    ratios = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({ratios[0]:.1f}% variance)"); plt.ylabel(f"PC2 ({ratios[1]:.1f}% variance)"); plt.legend(); plt.tight_layout(); plt.savefig(out / "weight_trajectory_pca.png", dpi=200); plt.close()


if __name__ == "__main__": main()
