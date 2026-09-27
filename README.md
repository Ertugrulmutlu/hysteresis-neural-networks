# Hysteresis in Neural Networks

This repository studies order dependence in sequential MNIST training. Part 1 trains the same `SimpleCNN` from the same initialization in two orders: **SAB** learns digits 0–4 (A), then 5–9 (B); **SBA** learns B, then A. Part 2 adds controlled common-relaxation, representation, activation, and fresh linear-probe analyses. The README reports the current empirical results while keeping every claim scoped to the tested protocol.

## Scientific scope and limitations

The experiment measures final order-dependent differences in weights, representations, losses, and accuracies. AB-versus-BA effects can overlap with catastrophic forgetting and ordinary last-task effects, so they are not by themselves universal proof of hysteresis or causality. Weight interpolation is raw and **not permutation-aligned**; a linear barrier does not prove complete basin disconnection. Conclusions should be replicated over multiple seeds.

## Repository structure

- `configs/`: the original legacy combined config plus explicit, protocol-matched SAB/SBA configs
- `src/`: training, data, model, tracking, and pair validation
- `src/analysis/`: training curves, shared-PCA trajectories, CKA, interpolation, and activation health
- `tests/`: fast offline tests using synthetic data and temporary files
- `results/`: run artifacts (ignored except `.gitkeep`)
- `plots/`: analysis artifacts (ignored except `.gitkeep`)
- `plot_metrics.py`: compatibility wrapper for `python -m src.analysis.training_curves`

## Environment setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The original `configs/mnist_ablation_v0.yaml` is retained as the legacy combined Part 1 config. The explicit configs preserve its seed (1337), normalization, architecture, SGD settings, learning rate, momentum, and 10+10 epoch phases. The pair differs only in scenario and run identity.

`SimpleCNN` retains the original `conv1`, `conv2`, `fc1`, and `fc2` state-dict names, so existing Part 1 checkpoints remain directly loadable without key rewriting.

## Reproduction commands

Run all commands from the repository root. Training downloads MNIST when it is absent. Run directories are protected: training refuses a non-empty target unless `logging.overwrite: true` is explicitly set. Overwrite clears only that exact run directory.

```bash
python -m src.train configs/mnist_v0_sab.yaml
python -m src.train configs/mnist_v0_sba.yaml

python -m src.validate_pair --run-a results/mnist_v0_SAB_seed1337_normnone --run-b results/mnist_v0_SBA_seed1337_normnone --json-out results/pair_validation.json

python -m src.analysis.training_curves --sab results/mnist_v0_SAB_seed1337_normnone/metrics.csv --sba results/mnist_v0_SBA_seed1337_normnone/metrics.csv --outdir plots/part1_seed1337
# Compatibility form: python plot_metrics.py with the same arguments

python -m src.analysis.weight_trajectory --run-sab results/mnist_v0_SAB_seed1337_normnone --run-sba results/mnist_v0_SBA_seed1337_normnone --outdir plots/part2_seed1337/weight_trajectory

python -m src.analysis.cka --run-sab results/mnist_v0_SAB_seed1337_normnone --run-sba results/mnist_v0_SBA_seed1337_normnone --samples-per-class 200 --outdir plots/part2_seed1337/cka

python -m src.analysis.interpolation --run-sab results/mnist_v0_SAB_seed1337_normnone --run-sba results/mnist_v0_SBA_seed1337_normnone --num-points 51 --outdir plots/part2_seed1337/interpolation

python -m src.analysis.activation_health --run-sab results/mnist_v0_SAB_seed1337_normnone --run-sba results/mnist_v0_SBA_seed1337_normnone --samples-per-class 200 --epsilon 1e-6 --dead-threshold 1e-4 --outdir plots/part2_seed1337/activation_health

pytest -q
```

CKA and activation-health use a deterministic balanced test probe and never train on it. If MNIST is not already present under `data/`, add `--download` to those commands and to interpolation.

## Outputs and metric definitions

Each training run contains `config_resolved.yaml`, `run_metadata.json`, `metrics.csv`, initialization checkpoint `weights_epoch_000.pt`, and epoch checkpoints.

- Weight trajectory: `trajectory_coordinates.csv`, `pca_metadata.json`, and `weight_trajectory_pca.png`. A single PCA is fit jointly after subtracting the common initialization.
- CKA: diagonal/cross-layer CSVs and plots plus `cka_summary.json`. Convolutional features use global average pooling. The paper-facing common-relaxation representation-history score is `H_repr = 1 - mean(CKA_conv2, CKA_fc1)`; logits and Conv1 are excluded from this declared primary score. Legacy exploratory analyses may report additional layer combinations separately.
- Interpolation: metrics CSV, summary JSON, loss plot, and accuracy plot. `W(alpha)=(1-alpha)W_SAB+alpha W_SBA`; barrier height is the maximum path loss minus the larger endpoint loss.
- Activation health: unit CSV, summary CSV/JSON, and plot. A unit is dead when its activation-above-epsilon rate is below the configured threshold.
- Training curves: accuracy/loss PNGs and `final_order_dependent_accuracy_gap.csv`. The gap is descriptive rather than definitive proof of hysteresis.

## Determinism notes

Both scenarios use seed 1337 and request deterministic PyTorch algorithms. Data-loader generators are seeded, evaluation is unshuffled, and pair validation requires bit-identical initialization tensors. Hardware, PyTorch, and CUDA differences can still affect reproducibility; run metadata records the available environment. CUDA requests fall back to CPU when CUDA is unavailable.

## Common Relaxation Experiment

Plain AB-versus-BA training overlaps with catastrophic forgetting and last-task effects. The common-relaxation protocol therefore compares `AB -> C` (`SABC`) with `BA -> C` (`SBAC`), where C is the same deterministic, balanced digits 0–9 training distribution for both histories. C is controlled by optimizer-update count rather than epochs, and both runs receive the same batch sequence and checkpoint schedule.

A representation gap that persists while behavioral performance becomes matched would be evidence that training history remains encoded after exposure to a common distribution. If the gap disappears, the chosen common-relaxation protocol erased the measured history dependence. Neither outcome alone establishes a universal hysteresis claim.

Run from the repository root in PowerShell:

```powershell
uv run python -m src.train configs\mnist_common_relaxation_sabc.yaml
uv run python -m src.train configs\mnist_common_relaxation_sbac.yaml

uv run python -m src.validate_pair `
  --run-a results\mnist_common_relaxation_SABC_seed1337_normnone `
  --run-b results\mnist_common_relaxation_SBAC_seed1337_normnone `
  --json-out results\common_relaxation_pair_validation.json

uv run python -m src.analysis.common_relaxation `
  --run-sabc results\mnist_common_relaxation_SABC_seed1337_normnone `
  --run-sbac results\mnist_common_relaxation_SBAC_seed1337_normnone `
  --samples-per-class 200 `
  --max-accuracy-gap 0.002 `
  --minimum-accuracy 0.97 `
  --outdir plots\common_relaxation_seed1337
```

Training writes the existing epoch artifacts for the A/B phases, plus `weights_relax_step_XXXXXX.pt` and `relaxation_metrics.csv` for C. Analysis writes `common_relaxation_metrics.csv`, `common_relaxation_summary.json`, and five history figures. The performance-matched endpoint is selected mechanically from the configured thresholds and is `null` when no checkpoint qualifies.

## Current status

Completed:

- 20-paired-seed `class_split_reset_relu` common-relaxation experiment
- optimizer-reset versus optimizer-preserve pilot
- final FC1 fresh linear probe with 500 examples per class
- low-data FC1 probe sweep with 25, 50, and 100 examples per class
- 50-examples-per-class robustness validation across five fresh-head probe seeds
- exploratory TOST practical-equivalence analysis for the 500-examples-per-class endpoint
- fresh five-paired-seed 50,000-update common-relaxation stress test

Completed final controls:

- five-paired-seed LeakyReLU mechanism control at matched LR=0.04
- five-paired-seed symmetric same-label rotated-MNIST control

Optional future extension:

- one wider architecture or second dataset

A cyclic mixture schedule remains optional and is only required if the final paper makes a strict cyclic-hysteresis claim rather than a narrower persistent-history-dependence claim.

## Paper Experiment Package

The additive paper package covers optimizer-state memory, activation-mediated plasticity, same-label domain shifts, fresh linear probes, and paired multi-seed estimation. These experiments remain descriptive controls and do not automatically establish physical hysteresis.

- Reset versus preserve isolates whether momentum/optimizer state contributes to persistence during C.
- LeakyReLU tests whether reduced positive-activity/dead-unit behavior changes the measured history effect. Activation health consistently defines positive activity as `activation > epsilon`; for LeakyReLU this is a positive-activity diagnostic, not a claim that negatively active units are biologically or computationally dead.
- Rotated MNIST uses the same labels in both histories, testing whether the effect depends on disjoint output classes.
- Linear probes test whether distinct frozen representations retain equally accessible label information.
- Paired-seed aggregation estimates repeatability without treating SABC and SBAC as independent samples.

Recommended execution order and current status:

1. **Completed — optimizer pilot:** seed 1337 for `class_split_preserve_relu` and `class_split_reset_relu`.
2. **Completed — initial seed sweep:** five predeclared paired seeds for the selected protocol.
3. **Completed — main seed sweep:** 20 paired `class_split_reset_relu` seeds.
4. **Completed — primary and low-data FC1 probes:** 25, 50, 100, and 500 examples per class.
5. **Completed — low-data readout robustness:** repeat the 50-examples-per-class probe across five fresh-head probe seeds while retaining 20 model seeds as the independent statistical units.
6. **Completed — exploratory practical equivalence:** test the 500-examples-per-class endpoint with TOST under a ±0.5 percentage-point margin.
7. **Completed — longer relaxation:** rerun five paired seeds from initialization with a 50,000-update C phase and analyze 10k, 25k, and 50k.
8. **Completed — mechanism control:** five paired seeds, ReLU versus LeakyReLU at matched learning rate 0.04, evaluated through 50,000 common-relaxation updates.
9. **Completed — same-label validation:** five paired seeds on symmetric rotated-MNIST.
10. **Conditional extension:** choose one wider architecture or second dataset only after the mechanism and same-label controls are known.

Generate a five-seed matrix without training (dry-run is also the default when `--execute` is absent):

```powershell
uv run python -m src.experiments.run_paper_matrix `
  --base-sabc configs\paper\mnist_class_split_reset_relu_sabc.yaml `
  --base-sbac configs\paper\mnist_class_split_reset_relu_sbac.yaml `
  --seeds 101 202 303 404 505 `
  --generated-config-dir configs\generated\class_split_reset_relu `
  --manifest-out results\manifests\class_split_reset_relu.json `
  --dry-run
```

After inspecting generated configs and commands, execute explicitly:

```powershell
uv run python -m src.experiments.run_paper_matrix `
  --base-sabc configs\paper\mnist_class_split_reset_relu_sabc.yaml `
  --base-sbac configs\paper\mnist_class_split_reset_relu_sbac.yaml `
  --seeds 101 202 303 404 505 `
  --generated-config-dir configs\generated\class_split_reset_relu `
  --manifest-out results\manifests\class_split_reset_relu.json `
  --execute
```

Use `--skip-existing` to leave completed runs untouched. `--overwrite` is required before replacing generated artifacts or a run directory.

Validate one generated pair:

```powershell
uv run python -m src.validate_pair `
  --run-a results\mnist_class_split_SABC_reset_relu_seed101_normnone `
  --run-b results\mnist_class_split_SBAC_reset_relu_seed101_normnone
```

Aggregate the paired matrix:

```powershell
uv run python -m src.analysis.aggregate_common_relaxation `
  --manifest results\manifests\class_split_reset_relu.json `
  --samples-per-class 200 `
  --bootstrap-samples 10000 `
  --bootstrap-seed 12345 `
  --max-accuracy-gap 0.002 `
  --minimum-accuracy 0.97 `
  --outdir plots\aggregate_class_split_reset_relu
```

Run a fresh frozen-representation probe:

```powershell
uv run python -m src.analysis.linear_probe `
  --run-sabc results\mnist_class_split_SABC_reset_relu_seed101_normnone `
  --run-sbac results\mnist_class_split_SBAC_reset_relu_seed101_normnone `
  --checkpoint relaxation-step:10000 `
  --feature-layer fc1 `
  --train-samples-per-class 500 `
  --test-samples-per-class 200 `
  --probe-epochs 20 `
  --probe-lr 0.01 `
  --probe-seed 777 `
  --outdir plots\linear_probe_class_split_seed1337
```

Compare aggregate conditions:

```powershell
uv run python -m src.analysis.compare_paper_conditions `
  --condition reset_relu=plots\aggregate_class_split_reset_relu\aggregate_summary.json `
  --condition preserve_relu=plots\aggregate_class_split_preserve_relu\aggregate_summary.json `
  --condition reset_leaky=plots\aggregate_class_split_reset_leaky_relu\aggregate_summary.json `
  --condition rotated_reset=plots\aggregate_rotated_reset_relu\aggregate_summary.json `
  --outdir plots\paper_condition_comparison
```

Paper configs live under `configs/paper/`. They inherit the validated common-relaxation protocol and resolve to complete configs before a run is created.
# Fresh Linear-Probe Analysis

The fresh linear-probe analysis asks whether history-dependent representations merely have different geometry or also make class information differently accessible to a linear readout. It reuses the trained common-relaxation CNN checkpoints: CNN backbones are never retrained. Each backbone is put in evaluation mode, frozen, and checked bit-for-bit after probing. Only a new `Linear(feature_dimension, 10)` head is optimized.

The SABC and SBAC heads start from exactly identical tensors and receive identical examples, labels, example order, and deterministic minibatch permutations. This removes initialization and sampling noise from the paired comparison. FC1 post-activation features are the predeclared primary representation. Conv2 global-average-pooled features are a secondary analysis.

Equal final probe accuracy means that the two histories make the retained class information similarly accessible to this linear probe under the declared training budget; it does not establish that their representation geometries are equal. Signed differences are always SABC minus SBAC: positive values favor linear accessibility after SABC, and negative values favor SBAC. Early-epoch differences measure accessibility/optimization speed and are exploratory; the predeclared final epoch, never a favorably selected best epoch, is the primary endpoint.

### Current results: 20-seed FC1 probe across data budgets

The FC1 probe reused the final 10,000-update checkpoints from all 20 paired `class_split_reset_relu` seeds. For every sample budget, SABC and SBAC used the same balanced MNIST examples, identical fresh-head initialization, the same deterministic minibatch sequence, the same optimizer hyperparameters, and the same 200 test examples per class. All 20 pairs completed successfully, the CNN backbones remained bit-identical, and only the fresh linear heads were trained.

The main full-test result changes systematically with the amount of labeled probe data:

| Train examples per class | Mean SABC accuracy | Mean SBAC accuracy | Mean paired difference (SABC - SBAC) | 95% t CI | 95% bootstrap CI | Paired t-test | Wilcoxon | Paired Cohen's dz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 94.080% | 93.075% | **+1.005 pp** | [-0.031, +2.041] pp | [+0.083, +1.950] pp | p = 0.056 | p = 0.076 | 0.454 |
| 50 | 95.915% | 95.335% | **+0.580 pp** | [+0.072, +1.088] pp | [+0.130, +1.058] pp | p = 0.027 | p = 0.028 | 0.535 |
| 100 | 96.623% | 96.335% | +0.288 pp | [-0.094, +0.669] pp | [-0.065, +0.633] pp | p = 0.131 | p = 0.077 | 0.353 |
| 500 | 97.550% | 97.460% | +0.090 pp | [-0.064, +0.244] pp | [-0.055, +0.220] pp | p = 0.237 | p = 0.083 | 0.273 |

The mean SABC-minus-SBAC gap decreases as the probe receives more labeled data:

```text
25/class  -> +1.005 pp
50/class  -> +0.580 pp
100/class -> +0.288 pp
500/class -> +0.090 pp
```

The cleanest nominal low-data signal occurs at 50 examples per class: 14 seeds favor SABC and 6 favor SBAC, both the paired t-test and Wilcoxon test are below 0.05, the two reported confidence intervals exclude zero, and the paired effect size is moderate (`dz = 0.535`). At 25 examples per class the mean effect is larger but substantially noisier: the bootstrap interval excludes zero while the t interval, paired t-test, and Wilcoxon test remain borderline. At 100 and 500 examples per class, the intervals include zero and the final accessibility difference is small.

The low-data advantage is strongest on the A-domain classes:

| Train examples per class | Mean A-domain difference | 95% t CI | Paired t-test | Wilcoxon | Paired Cohen's dz |
|---:|---:|---:|---:|---:|---:|
| 25 | **+0.935 pp** | [+0.330, +1.540] pp | p = 0.004 | p = 0.011 | 0.723 |
| 50 | **+0.530 pp** | [+0.169, +0.891] pp | p = 0.006 | p = 0.009 | 0.687 |
| 100 | +0.245 pp | [-0.053, +0.543] pp | p = 0.102 | p = 0.085 | 0.385 |
| 500 | +0.015 pp | [-0.171, +0.201] pp | p = 0.867 | p = 0.965 | 0.038 |

The B-domain differences are more variable and are not robust under the paired t-test and Wilcoxon test. The A/B asymmetry is therefore reported descriptively rather than treated as an established mechanism.

**Interpretation.** The common-relaxation analysis shows that SABC and SBAC retain different FC1 geometries. The probe sweep further suggests that this difference is not merely cosmetic: under low labeled-data budgets, the SABC representation is easier for an identically initialized linear head to learn from. As the amount of probe data increases, the difference shrinks toward a similar high-data endpoint. The current evidence therefore supports **different representation geometry with different low-data sample efficiency, but similar high-data linear accessibility**.

### Robustness across fresh-head probe seeds

The 50-examples-per-class condition was repeated with five independent probe seeds. Each probe seed changes the fresh linear-head initialization and deterministic minibatch ordering while reusing the same frozen SABC/SBAC backbones and paired model seeds.

| Probe seed | Mean full-test difference (SABC - SBAC) | Positive / negative model seeds | Paired t-test | Wilcoxon | Paired Cohen's dz |
|---:|---:|---:|---:|---:|---:|
| 777 | +0.580 pp | 14 / 6 | p = 0.027 | p = 0.028 | 0.535 |
| 1777 | +0.495 pp | 13 / 5 | p = 0.019 | p = 0.026 | 0.571 |
| 2777 | +0.405 pp | 15 / 5 | p = 0.017 | p = 0.021 | 0.582 |
| 3777 | +0.482 pp | 12 / 8 | p = 0.056 | p = 0.076 | 0.456 |
| 4777 | +0.520 pp | 14 / 5 | p = 0.013 | p = 0.021 | 0.610 |

The repeated probes are not treated as 100 independent experiments. The five readout repetitions are first averaged within each of the 20 model seeds, and the final paired inference is then performed over those 20 model-seed means.

| Repeated-probe statistic | Result |
|---|---:|
| Positive / negative model seeds | 15 / 5 |
| Mean paired difference | **+0.496 pp** |
| 95% t confidence interval | **[+0.127, +0.866] pp** |
| 95% bootstrap confidence interval | **[+0.160, +0.833] pp** |
| Paired t-test | **p = 0.0111** |
| Wilcoxon signed-rank test | **p = 0.0083** |
| Sign test | **p = 0.0414** |
| Paired Cohen's dz | **0.629** |

The direction is positive for all five probe seeds, four of the five per-probe-seed paired t-tests are below 0.05, and the model-seed-averaged confidence intervals exclude zero. The result is therefore robust to the tested fresh-head initialization and minibatch-order choices.

**Updated interpretation.** The evidence supports a distinction between asymptotic accessibility and low-data readout efficiency. With abundant probe data, SABC and SBAC reach nearly identical linear-probe accuracy. With 50 labeled examples per class, however, the SABC FC1 representation is consistently easier for a fresh linear classifier to decode. Training order therefore appears to affect the sample efficiency of a downstream linear readout, not merely the geometric arrangement of features.

This result remains scoped to the tested class-split MNIST protocol, FC1 features, linear readouts, and the fixed 50-examples-per-class subset. The 50-example condition was identified after inspecting a 25/50/100/500 sweep, so it is described as an exploratory discovery followed by robustness validation rather than as a preregistered confirmatory endpoint.

### Exploratory practical-equivalence analysis at 500 examples per class

The high-data endpoint was evaluated with two one-sided tests (TOST) using an equivalence margin of ±0.5 percentage points. Accuracy differences are defined as SABC minus SBAC.

| TOST quantity | Result |
|---|---:|
| Paired model seeds | 20 |
| Mean paired difference | +0.090 pp |
| Equivalence bounds | [-0.500, +0.500] pp |
| 90% confidence interval | **[-0.037, +0.217] pp** |
| Lower-bound one-sided test | **p = 8.26 × 10^-8** |
| Upper-bound one-sided test | **p = 1.14 × 10^-5** |
| Equivalent within the declared margin | **Yes** |

The complete 90% confidence interval lies inside the declared equivalence interval, and both one-sided tests reject effects outside the ±0.5 percentage-point bounds. Under this margin, the 500-examples-per-class FC1 probe endpoints are therefore practically equivalent.

This result does **not** imply exact equality of the two representations or exact equality of probe performance. It establishes equivalence only relative to the chosen ±0.5 percentage-point bound. Because the equivalence margin was selected after inspecting the existing results, this analysis is labeled exploratory rather than preregistered confirmatory evidence.

**Linear-probe conclusion.** The combined probe results support a two-regime interpretation: SABC representations provide a repeatable low-data linear-readout advantage at 50 examples per class, while SABC and SBAC become practically equivalent under the declared high-data margin at 500 examples per class. Training order therefore changes low-data readout sample efficiency without producing a practically meaningful asymptotic probe gap under the tested budget.


Single complete pair (replace the two run directories with a paired seed from the manifest):

```powershell
uv run python -m src.analysis.linear_probe `
  --run-sabc results\<SABC-run-directory> `
  --run-sbac results\<SBAC-run-directory> `
  --checkpoint final --feature-layer fc1 `
  --train-samples-per-class 500 --test-samples-per-class 200 `
  --probe-epochs 20 --probe-lr 0.01 --probe-weight-decay 0.0 `
  --probe-momentum 0.0 --probe-batch-size 128 --probe-seed 777 `
  --device cuda --outdir plots\linear_probe_single_pair_final_fc1
```

Primary 20-seed final-FC1 analysis:

```powershell
uv run python -m src.analysis.aggregate_linear_probe `
  --manifest results\manifests\class_split_reset_relu_20seeds_paired.json `
  --checkpoint final --feature-layer fc1 `
  --train-samples-per-class 500 --test-samples-per-class 200 `
  --probe-epochs 20 --probe-lr 0.01 --probe-weight-decay 0.0 `
  --probe-momentum 0.0 --probe-batch-size 128 --probe-seed 777 `
  --bootstrap-samples 10000 --bootstrap-seed 12345 --device cuda `
  --outdir plots\linear_probe_class_split_reset_relu_20seeds_final_fc1
```

Low-data FC1 sweep using the same frozen checkpoints:

```powershell
$sampleCounts = @(25, 50, 100)

foreach ($samplesPerClass in $sampleCounts) {
    uv run python -m src.analysis.aggregate_linear_probe `
      --manifest results\manifests\class_split_reset_relu_20seeds_paired.json `
      --checkpoint final --feature-layer fc1 `
      --train-samples-per-class $samplesPerClass --test-samples-per-class 200 `
      --probe-epochs 20 --probe-lr 0.01 --probe-weight-decay 0.0 `
      --probe-momentum 0.0 --probe-batch-size 128 --probe-seed 777 `
      --bootstrap-samples 10000 --bootstrap-seed 12345 --device cuda `
      --outdir "plots\linear_probe_class_split_reset_relu_20seeds_final_fc1_train_${samplesPerClass}_per_class"

    if ($LASTEXITCODE -ne 0) {
        throw "Linear probe failed for $samplesPerClass samples per class."
    }
}
```

Robustness replication for the 50-examples-per-class condition:

```powershell
$probeSeeds = @(777, 1777, 2777, 3777, 4777)

foreach ($probeSeed in $probeSeeds) {
    uv run python -m src.analysis.aggregate_linear_probe `
      --manifest results\manifests\class_split_reset_relu_20seeds_paired.json `
      --checkpoint final --feature-layer fc1 `
      --train-samples-per-class 50 --test-samples-per-class 200 `
      --probe-epochs 20 --probe-lr 0.01 --probe-weight-decay 0.0 `
      --probe-momentum 0.0 --probe-batch-size 128 --probe-seed $probeSeed `
      --bootstrap-samples 10000 --bootstrap-seed 12345 --device cuda `
      --outdir "plots\linear_probe_fc1_50_probe_seed_${probeSeed}"

    if ($LASTEXITCODE -ne 0) {
        throw "Linear probe failed for probe seed $probeSeed."
    }
}
```

The combined robustness summary averages the five repeated probe outcomes within each model seed before performing paired inference across the 20 model seeds. This avoids treating repeated readouts from the same frozen model pair as independent observations.

The 500-examples-per-class TOST analysis uses the final-epoch full-domain signed differences from `per_seed_linear_probe_differences.csv`, a ±0.5 percentage-point equivalence margin, and the standard 90% confidence interval associated with an alpha-0.05 TOST procedure. Its output is stored as:

```text
plots\linear_probe_class_split_reset_relu_20seeds_final_fc1\tost_equivalence_summary.json
```

Predeclared performance-matched FC1 secondary analysis (seeds without a qualifying endpoint are reported, not replaced by final checkpoints):

```powershell
uv run python -m src.analysis.aggregate_linear_probe `
  --manifest results\manifests\class_split_reset_relu_20seeds_paired.json `
  --checkpoint relaxation-step:10000 --max-accuracy-gap 0.002 `
  --minimum-accuracy 0.97 --numerical-tolerance 1e-12 --feature-layer fc1 `
  --train-samples-per-class 500 --test-samples-per-class 200 `
  --probe-epochs 20 --probe-lr 0.01 --probe-weight-decay 0.0 `
  --probe-momentum 0.0 --probe-batch-size 128 --probe-seed 777 `
  --bootstrap-samples 10000 --bootstrap-seed 12345 --device cuda `
  --outdir plots\linear_probe_class_split_reset_relu_20seeds_matched_fc1
```

Optional final-Conv2 analysis:

```powershell
uv run python -m src.analysis.aggregate_linear_probe `
  --manifest results\manifests\class_split_reset_relu_20seeds_paired.json `
  --checkpoint final --feature-layer conv2 `
  --train-samples-per-class 500 --test-samples-per-class 200 `
  --probe-epochs 20 --probe-lr 0.01 --probe-weight-decay 0.0 `
  --probe-momentum 0.0 --probe-batch-size 128 --probe-seed 777 `
  --bootstrap-samples 10000 --bootstrap-seed 12345 --device cuda `
  --outdir plots\linear_probe_class_split_reset_relu_20seeds_final_conv2
```
# Long Common-Relaxation Experiment

The 10,000-update experiment establishes residue only at that measured horizon; it cannot establish permanent hysteresis. The predeclared follow-up measures whether the same representation-history score decays, plateaus, or increases at 25,000 and 50,000 shared-C optimizer updates.

This repository uses a clean fresh 50,000-update rerun, not resume from the completed 10,000-update artifacts. Existing relaxation checkpoints contain model weights only. Exact resume would additionally require optimizer and scheduler state, the current update, Python/NumPy/PyTorch CPU and all-device CUDA RNG states, loader-generator and sampler/batch-position state, and tracker continuation state. Restarting from a weight-only checkpoint could repeat or skip C batches and is therefore not labeled exact.

The long experiment uses paired seeds `101 202 303 404 505`, class-split MNIST, the existing ReLU CNN without normalization, optimizer reset on entry to C, and the schedule `0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000`. Its names include `long50k`, so it cannot collide with the completed 10k runs.

Generate and inspect the five-seed matrix without starting training:

```powershell
uv run python -m src.experiments.run_paper_matrix `
  --base-sabc configs\paper\mnist_class_split_reset_relu_long50k_sabc.yaml `
  --base-sbac configs\paper\mnist_class_split_reset_relu_long50k_sbac.yaml `
  --seeds 101 202 303 404 505 `
  --generated-config-dir configs\generated\class_split_reset_relu_long50k_5seeds `
  --manifest-out results\manifests\class_split_reset_relu_long50k_5seeds.json `
  --dry-run
```

Start the ten fresh runs explicitly:

```powershell
uv run python -m src.experiments.run_paper_matrix `
  --base-sabc configs\paper\mnist_class_split_reset_relu_long50k_sabc.yaml `
  --base-sbac configs\paper\mnist_class_split_reset_relu_long50k_sbac.yaml `
  --seeds 101 202 303 404 505 `
  --generated-config-dir configs\generated\class_split_reset_relu_long50k_5seeds `
  --manifest-out results\manifests\class_split_reset_relu_long50k_5seeds.json `
  --execute
```

Aggregate every dynamically discovered relaxation checkpoint:

```powershell
uv run python -m src.analysis.aggregate_common_relaxation `
  --manifest results\manifests\class_split_reset_relu_long50k_5seeds.json `
  --samples-per-class 200 --bootstrap-samples 10000 --bootstrap-seed 12345 `
  --device cuda `
  --outdir plots\aggregate_class_split_reset_relu_long50k_5seeds
```

Calculate the focused paired long-horizon statistics:

```powershell
uv run python -m src.analysis.analyze_long_relaxation `
  --aggregate-csv plots\aggregate_class_split_reset_relu_long50k_5seeds\per_seed_relaxation_metrics.csv `
  --steps 10000 25000 50000 `
  --bootstrap-samples 10000 --bootstrap-seed 12345 `
  --outdir plots\long_relaxation_analysis_reset_relu_5seeds
```

Conclusions from this experiment must be phrased as persistence over the measured 50,000-update horizon, never as proof of permanent hysteresis.

### Current five-seed long-horizon result

All five predeclared paired seeds (`101, 202, 303, 404, 505`) completed the fresh 50,000-update protocol. No incomplete trajectories were present.

| Common-relaxation step | Mean `H_repr` | Median `H_repr` | Standard deviation | 95% bootstrap CI |
|---:|---:|---:|---:|---:|
| 10,000 | 0.1532 | 0.1577 | 0.0224 | [0.1357, 0.1708] |
| 25,000 | 0.1625 | 0.1537 | 0.0226 | [0.1456, 0.1811] |
| 50,000 | **0.1902** | 0.1898 | 0.0365 | **[0.1611, 0.2193]** |

The representation-history residue remains clearly above zero at 50,000 shared updates. It therefore does not disappear within the measured horizon.

Paired long-horizon changes were:

| Comparison | Mean signed change | Positive / negative seeds | 95% bootstrap CI | Paired t-test | Wilcoxon | Paired Cohen's `dz` |
|---|---:|---:|---:|---:|---:|---:|
| 10k → 25k | +0.0092 | 3 / 2 | [-0.0006, +0.0190] | p = 0.179 | p = 0.313 | 0.728 |
| 10k → 50k | **+0.0369** | **4 / 1** | **[+0.0087, +0.0592]** | p = 0.060 | p = 0.125 | **1.165** |
| 25k → 50k | +0.0277 | 4 / 1 | [-0.0076, +0.0586] | p = 0.227 | p = 0.188 | 0.638 |

The 10k-to-50k comparison shows a positive mean change and a bootstrap interval above zero, but the classical paired tests remain inconclusive with only five seeds. The result is therefore described as **persistent residue with suggestive evidence of long-horizon increase**, not as a proven monotonic increase.

Behavior remained closely matched while the internal residue persisted:

| Step | Mean SABC accuracy | Mean SBAC accuracy | Signed gap | Mean absolute gap | Prediction disagreement | JS divergence |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 98.624% | 98.582% | +0.042 pp | 0.130 pp | 2.25% | 0.0134 |
| 25,000 | 98.620% | 98.544% | +0.076 pp | 0.164 pp | 2.56% | 0.0156 |
| 50,000 | 98.480% | 98.316% | +0.164 pp | 0.180 pp | 2.93% | 0.0191 |

The layerwise CKA pattern suggests that the late increase in `H_repr` is driven mainly by Conv2:

| Step | Mean Conv1 CKA | Mean Conv2 CKA | Mean FC1 CKA | Mean logits CKA |
|---:|---:|---:|---:|---:|
| 10,000 | 0.9785 | 0.9008 | 0.7927 | 0.7561 |
| 25,000 | 0.9709 | 0.8990 | 0.7760 | 0.7456 |
| 50,000 | 0.9840 | **0.8304** | 0.7893 | 0.7143 |

**Long-horizon conclusion.** The two histories remain behaviorally close but do not converge to the same measured internal representation over 50,000 shared updates. The evidence supports **behavioral convergence without representational convergence over the measured horizon**. It does not establish permanent memory, strict physical hysteresis, or a universally increasing residue.
# LeakyReLU Mechanism Control

This mechanism-control experiment tests whether ReLU sparse/low-positive-activity behavior may reduce plasticity and help preserve history-dependent representations. The final comparison reruns both ReLU and LeakyReLU at the same stable learning rate of 0.04 after the original learning-rate-0.05 LeakyReLU condition showed numerical instability. The completed results are reported below.

The direct condition effect is `delta_activation = H_repr_LeakyReLU - H_repr_ReLU`, so negative values indicate reduced residue under LeakyReLU. Positive `H_repr` within either condition does not answer the mechanism question by itself. Conditions are strictly paired using seeds `101, 202, 303, 404, 505` and relaxation updates `0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000`. The primary endpoint is the paired condition difference at 50,000 updates; 10,000, 25,000, and change-over-time comparisons are secondary.

For LeakyReLU, units with negative activations are not described as dead. Shared numerical diagnostics are labeled positive-activity rate, below-threshold positive activity, sparsity, or activation variance. Historical ReLU dead-ratio fields remain readable for backward compatibility.

Prepare the configs and manifest without training:

```powershell
uv run python -m src.experiments.run_paper_matrix `
  --base-sabc configs\paper\mnist_class_split_reset_leaky_relu_long50k_sabc.yaml `
  --base-sbac configs\paper\mnist_class_split_reset_leaky_relu_long50k_sbac.yaml `
  --seeds 101 202 303 404 505 `
  --generated-config-dir configs\generated\class_split_reset_leaky_relu_long50k_5seeds `
  --manifest-out results\manifests\class_split_reset_leaky_relu_long50k_5seeds.json `
  --dry-run
```

Start the ten fresh LeakyReLU runs explicitly:

```powershell
uv run python -m src.experiments.run_paper_matrix `
  --base-sabc configs\paper\mnist_class_split_reset_leaky_relu_long50k_sabc.yaml `
  --base-sbac configs\paper\mnist_class_split_reset_leaky_relu_long50k_sbac.yaml `
  --seeds 101 202 303 404 505 `
  --generated-config-dir configs\generated\class_split_reset_leaky_relu_long50k_5seeds `
  --manifest-out results\manifests\class_split_reset_leaky_relu_long50k_5seeds.json `
  --execute
```

Aggregate LeakyReLU common-relaxation metrics:

```powershell
uv run python -m src.analysis.aggregate_common_relaxation `
  --manifest results\manifests\class_split_reset_leaky_relu_long50k_5seeds.json `
  --samples-per-class 200 --bootstrap-samples 10000 --bootstrap-seed 12345 `
  --device cuda `
  --outdir plots\aggregate_class_split_reset_leaky_relu_long50k_5seeds
```

Compare matched activation conditions:

```powershell
uv run python -m src.analysis.compare_activation_conditions `
  --relu-csv plots\aggregate_class_split_reset_relu_long50k_5seeds\per_seed_relaxation_metrics.csv `
  --leaky-csv plots\aggregate_class_split_reset_leaky_relu_long50k_5seeds\per_seed_relaxation_metrics.csv `
  --steps 10000 25000 50000 --primary-step 50000 `
  --bootstrap-samples 10000 --bootstrap-seed 12345 `
  --outdir plots\activation_condition_comparison_long50k_5seeds
```
# Numerical Stability Diagnostics

An optional fail-fast tracer was added because four completed LeakyReLU long50k runs—SABC seed303 and SBAC seeds101, 202, and 303—first have a known non-finite checkpoint at epoch 11, immediately after the history-domain switch. The tracer diagnoses a new run from initialization; it does not repair, overwrite, or include the affected completed artifacts. No stabilization choice, including clipping, learning-rate changes, or optimizer reset, is made before diagnosis.

Trace the primary failing configuration into an isolated directory:

```powershell
uv run python -m src.diagnostics.trace_training_failure `
  --config configs\generated\class_split_reset_leaky_relu_long50k_5seeds\seed101_sbac.yaml `
  --trace-from-epoch 10 --stop-after-epoch 11 `
  --stop-on-first-nonfinite `
  --outdir diagnostics\leaky_seed101_sbac_failure_trace
```

Trace the finite seed404-SBAC comparison control:

```powershell
uv run python -m src.diagnostics.trace_training_failure `
  --config configs\generated\class_split_reset_leaky_relu_long50k_5seeds\seed404_sbac.yaml `
  --trace-from-epoch 10 --stop-after-epoch 11 `
  --stop-on-first-nonfinite `
  --outdir diagnostics\leaky_seed404_sbac_finite_control
```

Each diagnostic directory contains `numerical_trace.csv`, `numerical_failure.json`, `run_summary.json`, and isolated `training_artifacts/`. When a failure occurs it also contains `failure_context.pt` with the batch, pre-step state, RNG states, hashes, and failure stage. No diagnostic result is claimed until these commands are explicitly executed.

## Final evidence package

The predeclared experiment package is complete.

### Main empirical conclusion

Under the tested SimpleCNN/MNIST protocols, behavioral convergence does not imply representational convergence. After different training histories are followed by the same deterministic common-relaxation distribution, accuracy can become closely matched while a measurable representation-history score remains.

This is an empirical, protocol-scoped claim. It is not a universal mathematical proof that all neural networks exhibit hysteresis.

### Final controls

- **Main common-relaxation experiment:** 20 paired seeds.
- **Long-horizon stress test:** five paired seeds through 50,000 common-relaxation updates.
- **Fresh linear probes:** 25, 50, 100, and 500 labeled examples per class, plus repeated low-data probe seeds.
- **Activation-function mechanism control:** matched learning rate 0.04, five paired seeds each for ReLU and LeakyReLU, through 50,000 common-relaxation updates.
- **Same-label validation:** five paired seeds using rotated MNIST.

At 50,000 updates, the LeakyReLU-minus-ReLU difference in the declared representation-history score is approximately **-0.0398** across five paired seeds, with all five paired differences in the same direction. The bootstrap 95% interval reported by the analysis is approximately **[-0.0772, -0.0176]**. Because n=5 and the paired t-test is not below 0.05, this is reported as directional mechanism evidence rather than definitive causal proof.

For the rotated-MNIST control, all five paired seeds reached the predeclared behavioral-matching criterion while retaining non-zero representation-history scores at their matched endpoints.

### Release artifacts

Final paper-facing artifacts are stored under `paper_artifacts/01_pilots_and_legacy/` through `paper_artifacts/06_rotated_mnist/`. Raw run directories under `results/` remain the source of truth and should not be moved or rewritten after release packaging.
