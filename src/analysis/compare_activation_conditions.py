"""Strict paired comparison of ReLU and LeakyReLU common-relaxation residues."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.analysis.common import create_output_dir, write_csv, write_json

PRACTICAL_MARGIN = .01
CORE_METRICS = {
    "h_repr": "representation_history_score",
    "cka_conv2": "cka_conv2",
    "cka_fc1": "cka_fc1",
    "prediction_disagreement": "prediction_disagreement",
    "js_divergence": "js_divergence",
    "absolute_accuracy_gap": "absolute_accuracy_gap",
    "signed_accuracy_difference": "signed_full_accuracy_difference_sabc_minus_sbac",
}


def read_condition_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _keyed(rows, condition: str):
    keyed = {}
    for row in rows:
        key = (int(row["seed"]), int(row["relaxation_step"]))
        if key in keyed:
            raise ValueError(f"Duplicate {condition} row for seed/step {key}")
        keyed[key] = row
    return keyed


def align_condition_rows(relu_rows, leaky_rows, steps: list[int]):
    relu, leaky = _keyed(relu_rows, "ReLU"), _keyed(leaky_rows, "LeakyReLU")
    requested = set(steps)
    relu_seeds = {seed for seed, step in relu if step in requested}
    leaky_seeds = {seed for seed, step in leaky if step in requested}
    if relu_seeds != leaky_seeds:
        raise ValueError(f"Condition seed sets differ: ReLU={sorted(relu_seeds)}, LeakyReLU={sorted(leaky_seeds)}")
    if set(relu) != set(leaky):
        raise ValueError(f"Incomplete condition pairs; condition seed/step grids differ: ReLU-only={sorted(set(relu)-set(leaky))}, "
                         f"LeakyReLU-only={sorted(set(leaky)-set(relu))}")
    expected = {(seed, step) for seed in relu_seeds for step in steps}
    missing_relu, missing_leaky = sorted(expected - set(relu)), sorted(expected - set(leaky))
    if missing_relu or missing_leaky:
        raise ValueError(f"Incomplete condition pairs: missing ReLU={missing_relu}, missing LeakyReLU={missing_leaky}")
    if not relu_seeds:
        raise ValueError("No matched condition seeds")
    return [(key, relu[key], leaky[key]) for key in sorted(expected)]


def available_metrics(aligned) -> dict[str, str]:
    relu_columns = set(aligned[0][1]); leaky_columns = set(aligned[0][2])
    metrics = {name: column for name, column in CORE_METRICS.items()
               if column in relu_columns and column in leaky_columns}
    if "absolute_accuracy_gap" not in metrics and "full_accuracy_difference" in relu_columns & leaky_columns:
        metrics["absolute_accuracy_gap"] = "full_accuracy_difference"
    health_suffixes = ("_average_positive_activity_rate_difference",
                       "_below_threshold_positive_activity_ratio_difference",
                       "_average_activation_variance_difference", "_average_sparsity_difference")
    for column in sorted(relu_columns & leaky_columns):
        if column.endswith(health_suffixes):
            metrics[column] = column
    if "h_repr" not in metrics:
        raise ValueError("Both inputs must contain representation_history_score")
    return metrics


def bootstrap_ci(values, samples: int, seed: int):
    values = np.asarray(values, dtype=float)
    if samples <= 0 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite values and positive samples")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def tost_paired(values, margin: float = PRACTICAL_MARGIN):
    values = np.asarray(values, dtype=float); n = len(values)
    if margin <= 0:
        raise ValueError("TOST margin must be positive")
    sd = float(values.std(ddof=1)) if n > 1 else 0.; se = sd / math.sqrt(n) if n else math.nan
    if se == 0:
        equivalent = bool(abs(float(values.mean())) < margin)
        return {"tost_lower_pvalue": 0. if equivalent else 1., "tost_upper_pvalue": 0. if equivalent else 1.,
                "tost_equivalent": equivalent}
    lower_t = (float(values.mean()) + margin) / se
    upper_t = (float(values.mean()) - margin) / se
    lower_p = float(stats.t.sf(lower_t, n - 1)); upper_p = float(stats.t.cdf(upper_t, n - 1))
    return {"tost_lower_pvalue": lower_p, "tost_upper_pvalue": upper_p,
            "tost_equivalent": lower_p < .05 and upper_p < .05}


def paired_statistics(values, bootstrap_samples: int, bootstrap_seed: int):
    values = np.asarray(values, dtype=float); n = len(values); mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.
    ttest = stats.ttest_1samp(values, 0.) if n > 1 else None
    try:
        wilcoxon = stats.wilcoxon(values)
        w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        w_stat, w_p = 0., 1.
    return {"n": n, "mean": mean, "median": float(np.median(values)), "standard_deviation": sd,
            "bootstrap_ci_95": bootstrap_ci(values, bootstrap_samples, bootstrap_seed),
            "paired_t_statistic": float(ttest.statistic) if ttest else math.nan,
            "paired_t_pvalue": float(ttest.pvalue) if ttest else math.nan,
            "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_p,
            "paired_cohens_dz": mean / sd if sd > 0 else math.nan,
            "positive_count": int((values > 0).sum()), "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum())}


def build_comparison(relu_rows, leaky_rows, steps, primary_step, bootstrap_samples, bootstrap_seed):
    if primary_step not in steps:
        raise ValueError("primary step must be included in requested steps")
    aligned = align_condition_rows(relu_rows, leaky_rows, steps); metrics = available_metrics(aligned)
    per_seed = []
    for (seed, step), relu, leaky in aligned:
        row = {"seed": seed, "relaxation_step": step}
        for name, column in metrics.items():
            rv, lv = float(relu[column]), float(leaky[column])
            row[f"{name}_relu"] = rv; row[f"{name}_leaky_relu"] = lv
            row[f"delta_{name}_leaky_minus_relu"] = lv - rv
        per_seed.append(row)
    statistics = []
    for step in steps:
        for index, name in enumerate(metrics):
            values = [row[f"delta_{name}_leaky_minus_relu"] for row in per_seed if row["relaxation_step"] == step]
            summary = paired_statistics(values, bootstrap_samples, bootstrap_seed + step + index)
            statistics.append({"relaxation_step": step, "metric": name,
                               "analysis_role": "primary" if step == primary_step and name == "h_repr" else "secondary",
                               **summary})
    first, last = steps[0], primary_step
    by_seed = {(row["seed"], row["relaxation_step"]): row for row in per_seed}
    seeds = sorted({row["seed"] for row in per_seed})
    temporal = [by_seed[seed, last]["delta_h_repr_leaky_minus_relu"]
                - by_seed[seed, first]["delta_h_repr_leaky_minus_relu"] for seed in seeds]
    temporal_stats = paired_statistics(temporal, bootstrap_samples, bootstrap_seed + 999999)
    primary = next(row for row in statistics if row["analysis_role"] == "primary")
    tost = tost_paired([row["delta_h_repr_leaky_minus_relu"] for row in per_seed
                        if row["relaxation_step"] == primary_step])
    low, high = primary["bootstrap_ci_95"]
    if high < -PRACTICAL_MARGIN:
        classification = "LeakyReLU meaningfully reduces residue over the measured horizon"
    elif low > PRACTICAL_MARGIN:
        classification = "LeakyReLU increases residue over the measured horizon"
    elif tost["tost_equivalent"]:
        classification = "LeakyReLU and ReLU are practically similar under the declared margin"
    else:
        classification = "Results are too variable for a clear condition conclusion"
    summary = {"primary_endpoint": f"delta_H_repr at {primary_step} updates",
               "delta_definition": "LeakyReLU minus ReLU", "practical_margin": PRACTICAL_MARGIN,
               "primary_statistics": primary, "primary_tost": tost,
               "secondary_delta_h_repr_change": {"start_step": first, "end_step": last, **temporal_stats},
               "classification": classification,
               "interpretation_limit": "Evidence for or against activation-mediated contribution; not a causal or sole-mechanism claim.",
               "seeds": seeds, "steps": steps, "metrics_compared": metrics}
    return per_seed, statistics, summary


def make_plots(out, rows, steps, primary_step):
    seeds = sorted({row["seed"] for row in rows})
    plt.figure()
    for seed in seeds:
        selected = [r for r in rows if r["seed"] == seed]
        plt.plot(steps, [r["delta_h_repr_leaky_minus_relu"] for r in selected], marker="o", alpha=.6)
    plt.axhline(0, color="black", linewidth=.8); plt.xlabel("Relaxation updates"); plt.ylabel("LeakyReLU - ReLU H_repr")
    plt.tight_layout(); plt.savefig(out / "activation_effect_history_curve.png", dpi=200); plt.close()
    primary = [r for r in rows if r["relaxation_step"] == primary_step]
    plt.figure(); plt.bar([str(r["seed"]) for r in primary], [r["delta_h_repr_leaky_minus_relu"] for r in primary])
    plt.axhline(0, color="black", linewidth=.8); plt.ylabel("LeakyReLU - ReLU H_repr")
    plt.tight_layout(); plt.savefig(out / "activation_effect_by_seed_50k.png", dpi=200); plt.close()
    plt.figure(); plt.scatter([r["h_repr_relu"] for r in primary], [r["h_repr_leaky_relu"] for r in primary])
    limits = plt.xlim(); plt.plot(limits, limits, "--", color="black"); plt.xlabel("ReLU H_repr"); plt.ylabel("LeakyReLU H_repr")
    plt.tight_layout(); plt.savefig(out / "relu_vs_leaky_h_repr_50k.png", dpi=200); plt.close()
    functional = [name for name in ("prediction_disagreement", "js_divergence", "absolute_accuracy_gap")
                  if f"delta_{name}_leaky_minus_relu" in primary[0]]
    plt.figure(); plt.bar(functional, [np.mean([r[f"delta_{name}_leaky_minus_relu"] for r in primary]) for name in functional])
    plt.axhline(0, color="black", linewidth=.8); plt.ylabel("Mean LeakyReLU - ReLU effect"); plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(out / "activation_functional_metrics.png", dpi=200); plt.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--relu-csv", required=True); parser.add_argument("--leaky-csv", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[10000, 25000, 50000]); parser.add_argument("--primary-step", type=int, default=50000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000); parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--outdir", required=True); args = parser.parse_args()
    if args.steps != sorted(set(args.steps)):
        raise ValueError("steps must be unique and increasing")
    rows, statistics, summary = build_comparison(read_condition_csv(args.relu_csv), read_condition_csv(args.leaky_csv),
                                                  args.steps, args.primary_step, args.bootstrap_samples, args.bootstrap_seed)
    out = create_output_dir(args.outdir); write_csv(out / "per_seed_activation_condition_metrics.csv", rows)
    write_csv(out / "activation_condition_statistics.csv", statistics)
    write_json(out / "activation_condition_summary.json", summary); make_plots(out, rows, args.steps, args.primary_step)


if __name__ == "__main__":
    main()
