"""Aggregate paired fresh-linear-probe results across a paper manifest."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.analysis.common import create_output_dir, write_csv, write_json
from src.analysis.linear_probe import DOMAINS, add_probe_arguments, run_pair, validate_probe_arguments


DECLARED_EPOCHS = (1, 2, 5, 10, 20)


def _run_records(value):
    """Yield manifest leaves supporting the repository's existing run-list formats."""
    if isinstance(value, list):
        for item in value:
            yield from _run_records(item)
    elif isinstance(value, dict):
        paired_paths = {
            "SABC": value.get("run_sabc", value.get("sabc")),
            "SBAC": value.get("run_sbac", value.get("sbac")),
        }
        if any(path is not None for path in paired_paths.values()):
            for paired_scenario, paired_path in paired_paths.items():
                if paired_path is not None:
                    yield paired_scenario, value.get("seed"), str(paired_path)
            return
        scenario = str(value.get("scenario", value.get("sequence", value.get("condition", "")))).upper()
        path = value.get("run_dir", value.get("run", value.get("path", value.get("output_dir"))))
        if scenario in {"SABC", "SBAC"} and path is not None:
            yield scenario, value.get("seed"), str(path)
        else:
            for child in value.values():
                yield from _run_records(child)


def manifest_pairs(manifest: str | Path):
    manifest_path = Path(manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped = {}
    for scenario, seed, path in _run_records(data):
        resolved = Path(path)
        if not resolved.is_absolute():
            candidates = (manifest_path.parent / resolved, Path.cwd() / resolved)
            resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        if seed is None:
            config_path = resolved / "config.json"
            if config_path.exists():
                seed = json.loads(config_path.read_text(encoding="utf-8"))["experiment"]["seed"]
        key = int(seed) if seed is not None else f"unknown:{resolved}"
        grouped.setdefault(key, {})[scenario] = resolved
    return grouped


def signed_differences(rows: list[dict]) -> list[dict]:
    indexed = {(int(row["seed"]), row["scenario"], int(row["probe_epoch"])): row for row in rows}
    result = []
    for seed, _, epoch in sorted(indexed):
        if (seed, "SABC", epoch) not in indexed or (seed, "SBAC", epoch) not in indexed:
            continue
        if any(item["seed"] == seed and item["probe_epoch"] == epoch for item in result):
            continue
        a, b = indexed[(seed, "SABC", epoch)], indexed[(seed, "SBAC", epoch)]
        for domain in DOMAINS:
            difference = a[f"test_accuracy_{domain}"] - b[f"test_accuracy_{domain}"]
            result.append({"seed": seed, "probe_epoch": epoch, "domain": domain,
                           "accuracy_sabc": a[f"test_accuracy_{domain}"],
                           "accuracy_sbac": b[f"test_accuracy_{domain}"],
                           "signed_accuracy_difference_sabc_minus_sbac": difference,
                           "absolute_accuracy_difference": abs(difference)})
    return result


def bootstrap_ci(values, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def paired_statistics(values, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    values = np.asarray(values, dtype=float); n = len(values)
    mean = float(np.mean(values)); standard_deviation = float(np.std(values, ddof=1)) if n > 1 else math.nan
    sem = standard_deviation / math.sqrt(n) if n > 1 else math.nan
    t_critical = float(stats.t.ppf(.975, n - 1)) if n > 1 else math.nan
    nonzero = values[values != 0]; positive = int(np.sum(values > 0)); negative = int(np.sum(values < 0))
    if len(nonzero):
        sign = stats.binomtest(positive, len(nonzero), .5, alternative="two-sided")
        sign_stat, sign_p = positive, float(sign.pvalue)
    else:
        sign_stat, sign_p = 0, 1.
    try:
        wilcoxon = stats.wilcoxon(values, alternative="two-sided")
        wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0., 1.
    ttest = stats.ttest_1samp(values, 0.) if n > 1 else None
    boot_low, boot_high = bootstrap_ci(values, bootstrap_samples, bootstrap_seed)
    return {"n": n, "positive_count": positive, "negative_count": negative,
            "zero_count": int(np.sum(values == 0)), "mean": mean, "median": float(np.median(values)),
            "standard_deviation": standard_deviation,
            "t_ci_95_low": mean - t_critical * sem, "t_ci_95_high": mean + t_critical * sem,
            "bootstrap_ci_95_low": boot_low, "bootstrap_ci_95_high": boot_high,
            "paired_t_statistic": float(ttest.statistic) if ttest else math.nan,
            "paired_t_pvalue": float(ttest.pvalue) if ttest else math.nan,
            "wilcoxon_statistic": wilcoxon_stat, "wilcoxon_pvalue": wilcoxon_p,
            "sign_test_statistic": sign_stat, "sign_test_pvalue": sign_p,
            "paired_cohens_dz": mean / standard_deviation if standard_deviation > 0 else math.nan,
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "trimmed_mean_10_percent": float(stats.trim_mean(values, .1))}


def aggregate_statistics(differences, final_epoch: int, bootstrap_samples: int, bootstrap_seed: int):
    rows = []
    available = sorted({row["probe_epoch"] for row in differences})
    for epoch in [epoch for epoch in DECLARED_EPOCHS if epoch in available]:
        for domain in DOMAINS:
            values = [row["signed_accuracy_difference_sabc_minus_sbac"] for row in differences
                      if row["probe_epoch"] == epoch and row["domain"] == domain]
            if values:
                rows.append({"probe_epoch": epoch, "domain": domain,
                             "analysis_role": "primary_final" if epoch == final_epoch else "exploratory_epoch_wise",
                             **paired_statistics(values, bootstrap_samples, bootstrap_seed + epoch)})
    return rows


def _plots(out: Path, metrics, differences, final_epoch: int):
    final_metrics = [r for r in metrics if r["probe_epoch"] == final_epoch]
    plt.figure()
    for scenario in ("SABC", "SBAC"):
        rows = [r for r in final_metrics if r["scenario"] == scenario]
        plt.plot([r["seed"] for r in rows], [r["test_accuracy_full"] for r in rows], "o-", label=scenario)
    plt.xlabel("Seed"); plt.ylabel("Final full accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "probe_accuracy_by_seed.png", dpi=200); plt.close()
    final_diffs = [r for r in differences if r["probe_epoch"] == final_epoch and r["domain"] == "full"]
    plt.figure(); plt.bar([str(r["seed"]) for r in final_diffs],
                          [r["signed_accuracy_difference_sabc_minus_sbac"] for r in final_diffs])
    plt.axhline(0, color="black", linewidth=.8); plt.xlabel("Seed"); plt.ylabel("SABC - SBAC")
    plt.tight_layout(); plt.savefig(out / "probe_signed_difference_by_seed.png", dpi=200); plt.close()
    plt.figure()
    for scenario in ("SABC", "SBAC"):
        epochs = sorted({r["probe_epoch"] for r in metrics})
        means = [np.mean([r["test_accuracy_full"] for r in metrics
                         if r["scenario"] == scenario and r["probe_epoch"] == epoch]) for epoch in epochs]
        plt.plot(epochs, means, label=scenario)
    plt.xlabel("Probe epoch"); plt.ylabel("Mean full accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "probe_learning_curves.png", dpi=200); plt.close()
    plt.figure()
    labels, values = [], []
    for scenario in ("SABC", "SBAC"):
        for domain in ("A", "B"):
            labels.append(f"{scenario}-{domain}")
            values.append(np.mean([r[f"test_accuracy_{domain}"] for r in final_metrics if r["scenario"] == scenario]))
    plt.bar(labels, values); plt.ylabel("Mean final accuracy"); plt.tight_layout()
    plt.savefig(out / "probe_A_B_comparison.png", dpi=200); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True)
    add_probe_arguments(parser, include_runs=False)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--outdir", required=True); args = parser.parse_args()
    validate_probe_arguments(args)
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    pairs = manifest_pairs(args.manifest); metrics, pair_status = [], []
    for seed, runs in sorted(pairs.items(), key=lambda item: str(item[0])):
        missing = [scenario for scenario in ("SABC", "SBAC") if scenario not in runs]
        if missing:
            pair_status.append({"seed": seed, "status": "incomplete", "reason": f"missing {','.join(missing)}"})
            continue
        try:
            rows, _ = run_pair(runs["SABC"], runs["SBAC"], args)
            metrics.extend(rows); pair_status.append({"seed": seed, "status": "matched", "reason": ""})
        except Exception as error:  # record every failed/matched-selection pair; never silently discard
            pair_status.append({"seed": seed, "status": "failed", "reason": str(error)})
    differences = signed_differences(metrics)
    statistics = aggregate_statistics(differences, args.probe_epochs, args.bootstrap_samples, args.bootstrap_seed)
    out = create_output_dir(args.outdir)
    write_csv(out / "per_seed_linear_probe_metrics.csv", metrics)
    write_csv(out / "per_seed_linear_probe_differences.csv", differences)
    write_csv(out / "aggregate_linear_probe_statistics.csv", statistics)
    summary = {"manifest": str(args.manifest), "checkpoint": args.checkpoint, "feature_layer": args.feature_layer,
               "declared_pairs": len(pairs), "matched_pairs": sum(r["status"] == "matched" for r in pair_status),
               "unmatched_or_failed_pairs": sum(r["status"] != "matched" for r in pair_status),
               "pair_status": pair_status, "final_epoch": args.probe_epochs,
               "epoch_wise_tests_are_exploratory": True, "statistics": statistics}
    write_json(out / "aggregate_linear_probe_summary.json", summary)
    if metrics:
        _plots(out, metrics, differences, args.probe_epochs)


if __name__ == "__main__":
    main()
