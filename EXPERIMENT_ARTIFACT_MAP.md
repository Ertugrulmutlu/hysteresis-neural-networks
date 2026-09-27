# Experiment artifact map

## Raw training runs

The `results/` root intentionally remains flat and immutable because manifests and
reproduction commands reference those exact run paths.

- `mnist_v0_SAB_*`, `mnist_v0_SBA_*`: Part 1 legacy order experiment.
- `mnist_common_relaxation_*`: original seed-1337 common-relaxation experiment.
- `mnist_class_split_*_preserve_relu_seed1337_*`: optimizer-preserve pilot.
- `mnist_class_split_*_reset_relu_seed1337_*`: optimizer-reset pilot.
- `mnist_class_split_*_reset_relu_seed{101..2020}_*`: main 20-paired-seed experiment.
- `mnist_class_split_*_reset_relu_long50k_seed{101,202,303,404,505}_*`: long-horizon experiment.
- `mnist_class_split_*_reset_leaky_relu_long50k_*`: completed LeakyReLU mechanism-control runs. The final matched-LR experiment uses learning rate 0.04 and five paired seeds.
- `results/manifests/`: experiment-to-run mappings.
- `results/_index/`: generated inventory and cleanup reports.
- `results/_archive/stale_failed/`: failed startup artifacts only.

## Curated paper-facing analysis

- `paper_artifacts/01_pilots_and_legacy/`
- `paper_artifacts/02_main_20seed/`
- `paper_artifacts/03_linear_probe/`
- `paper_artifacts/04_long50k/`

These are copies of analysis outputs. Original `plots/` paths remain unchanged for
reproducibility.

## Logs

Completed logs are archived under:

- `logs/_archive/main_20seed/`
- `logs/_archive/long50k/`

## Safety rule

Do not manually move completed directories under `results/` unless every manifest,
README command, and analysis path is updated together. Moving stale/failed directories
and completed log directories is safe.

## Final completed controls

- `plots/auto_remaining/activation_condition_comparison_long50k_lr0400_5seeds/`: matched-LR ReLU versus LeakyReLU mechanism-control analysis.
- `plots/auto_remaining/aggregate_rotated_reset_relu_5seeds/`: five-paired-seed same-label rotated-MNIST aggregate analysis.
- `results/auto_remaining_status.json`: final orchestration status; expected `completed: true`.
- `results/manifests/auto_remaining/`: manifests for the final control runs.
- `configs/generated/auto_remaining/`: resolved/generated configs used by the final control package.
- `paper_artifacts/final_release/`: paper-facing release copy.
