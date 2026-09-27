"""Focused paired analysis of representation residue at long relaxation horizons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.analysis.common import create_output_dir, write_csv, write_json

HISTORY_COLUMN = "representation_history_score"
PLATEAU_EQUIVALENCE_MARGIN = 0.01


def read_long_metrics(path: str | Path, steps: list[int]) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    wanted = set(steps)
    rows = []
    for row in source:
        step = int(row["relaxation_step"])
        if step in wanted:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "relaxation_step": step,
                    HISTORY_COLUMN: float(row[HISTORY_COLUMN]),
                }
            )
    duplicates = [(r["seed"], r["relaxation_step"]) for r in rows]
    if len(duplicates) != len(set(duplicates)):
        raise ValueError("Duplicate seed/step rows in aggregate CSV")
    return rows


def deterministic_bootstrap_ci(values, samples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values) or samples <= 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite values and a positive sample count")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(
        axis=1
    )
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize_level(values, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "bootstrap_ci_95": deterministic_bootstrap_ci(
            values, bootstrap_samples, bootstrap_seed
        ),
    }


def summarize_paired_change(
    changes, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    changes = np.asarray(changes, dtype=float)
    n = len(changes)
    try:
        wilcoxon = stats.wilcoxon(changes)
        wilcoxon_statistic, wilcoxon_pvalue = (
            float(wilcoxon.statistic),
            float(wilcoxon.pvalue),
        )
    except ValueError:
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    ttest = stats.ttest_1samp(changes, 0.0) if n > 1 else None
    standard_deviation = float(changes.std(ddof=1)) if n > 1 else 0.0
    mean = float(changes.mean())
    return {
        "n": n,
        "mean_change": mean,
        "median_change": float(np.median(changes)),
        "standard_deviation": standard_deviation,
        "bootstrap_ci_95": deterministic_bootstrap_ci(
            changes, bootstrap_samples, bootstrap_seed
        ),
        "paired_t_statistic": float(ttest.statistic) if ttest else math.nan,
        "paired_t_pvalue": float(ttest.pvalue) if ttest else math.nan,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "paired_cohens_dz": mean / standard_deviation
        if standard_deviation > 0
        else math.nan,
        "positive_count": int((changes > 0).sum()),
        "negative_count": int((changes < 0).sum()),
        "zero_count": int((changes == 0).sum()),
    }


def build_long_analysis(
    rows: list[dict], steps: list[int], bootstrap_samples: int, bootstrap_seed: int
):
    table = {(row["seed"], row["relaxation_step"]): row[HISTORY_COLUMN] for row in rows}
    all_seeds = sorted({row["seed"] for row in rows})
    complete = [
        seed for seed in all_seeds if all((seed, step) in table for step in steps)
    ]
    incomplete = {
        str(seed): [step for step in steps if (seed, step) not in table]
        for seed in all_seeds
        if seed not in complete
    }
    if not complete:
        raise ValueError(
            f"No complete paired seed trajectories; incomplete={incomplete}"
        )
    per_seed = [
        {"seed": seed, **{f"h_repr_step_{step}": table[seed, step] for step in steps}}
        for seed in complete
    ]
    statistics = []
    level_stats = {}
    for step in steps:
        summary = summarize_level(
            [table[seed, step] for seed in complete],
            bootstrap_samples,
            bootstrap_seed + step,
        )
        level_stats[str(step)] = summary
        statistics.append(
            {"statistic_family": "level", "start_step": "", "end_step": step, **summary}
        )
    changes, change_stats = [], {}
    for index, (start, end) in enumerate(
        ((steps[0], steps[1]), (steps[0], steps[2]), (steps[1], steps[2]))
    ):
        values = [table[seed, end] - table[seed, start] for seed in complete]
        for seed, value in zip(complete, values):
            changes.append(
                {
                    "seed": seed,
                    "start_step": start,
                    "end_step": end,
                    "signed_h_repr_change_end_minus_start": value,
                }
            )
        summary = summarize_paired_change(
            values, bootstrap_samples, bootstrap_seed + 100000 + index
        )
        change_stats[f"{start}_to_{end}"] = summary
        statistics.append(
            {
                "statistic_family": "paired_change",
                "start_step": start,
                "end_step": end,
                **summary,
            }
        )
    conclusion = classify_horizon(
        level_stats[str(steps[-1])], change_stats[f"{steps[1]}_to_{steps[2]}"]
    )
    return (
        per_seed,
        changes,
        statistics,
        {
            "complete_seeds": complete,
            "incomplete_seeds": incomplete,
            "level_statistics": level_stats,
            "change_statistics": change_stats,
            "plateau_equivalence_margin": PLATEAU_EQUIVALENCE_MARGIN,
            "measured_horizon_conclusion": conclusion,
        },
    )


def classify_horizon(final_level: dict, late_change: dict) -> str:
    low, _ = final_level["bootstrap_ci_95"]
    change_low, change_high = late_change["bootstrap_ci_95"]
    residue = (
        "residue remains clearly above zero over the measured horizon"
        if low > 0
        else "residue at 50,000 is too variable for a clear above-zero conclusion"
    )
    if change_high < 0:
        trend = "residue continues decaying from 25,000 to 50,000"
    elif change_low > 0:
        trend = "residue increases from 25,000 to 50,000"
    elif (
        change_low >= -PLATEAU_EQUIVALENCE_MARGIN
        and change_high <= PLATEAU_EQUIVALENCE_MARGIN
    ):
        trend = "residue reaches an empirical plateau from 25,000 to 50,000"
    else:
        trend = "late-horizon changes are too variable for a clear conclusion"
    return f"{residue}; {trend}. This describes persistence only through 50,000 measured updates."


def make_plots(out: Path, per_seed, changes, steps):
    plt.figure()
    for row in per_seed:
        plt.plot(
            steps, [row[f"h_repr_step_{step}"] for step in steps], marker="o", alpha=0.6
        )
    means = [
        np.mean([row[f"h_repr_step_{step}"] for row in per_seed]) for step in steps
    ]
    plt.plot(steps, means, color="black", linewidth=2.5, marker="o", label="mean")
    plt.xlabel("Common-relaxation updates")
    plt.ylabel("H_repr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "long_relaxation_history_curve.png", dpi=200)
    plt.close()
    plt.figure()
    width = 0.25
    x = np.arange(len(per_seed))
    for index, step in enumerate(steps):
        plt.bar(
            x + (index - 1) * width,
            [r[f"h_repr_step_{step}"] for r in per_seed],
            width,
            label=str(step),
        )
    plt.xticks(x, [str(r["seed"]) for r in per_seed])
    plt.xlabel("Seed")
    plt.ylabel("H_repr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "long_relaxation_per_seed.png", dpi=200)
    plt.close()
    selected = [
        r for r in changes if r["start_step"] == steps[0] and r["end_step"] == steps[-1]
    ]
    plt.figure()
    plt.bar(
        [str(r["seed"]) for r in selected],
        [r["signed_h_repr_change_end_minus_start"] for r in selected],
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Seed")
    plt.ylabel("H_repr change")
    plt.tight_layout()
    plt.savefig(out / "long_relaxation_change_10k_to_50k.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--steps", nargs=3, type=int, default=[10000, 25000, 50000])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    if args.steps != sorted(set(args.steps)):
        raise ValueError("steps must contain three unique increasing update counts")
    rows = read_long_metrics(args.aggregate_csv, args.steps)
    per_seed, changes, statistics, summary = build_long_analysis(
        rows, args.steps, args.bootstrap_samples, args.bootstrap_seed
    )
    out = create_output_dir(args.outdir)
    write_csv(out / "per_seed_long_relaxation_metrics.csv", per_seed)
    write_csv(out / "paired_long_relaxation_changes.csv", changes)

    # ``statistics`` intentionally mixes level summaries and paired-change
    # summaries. These row families have different keys, so relying on the
    # first row to define the CSV schema causes csv.DictWriter to reject the
    # later paired-change fields. Use one explicit, stable union schema.
    statistics_fieldnames = [
        "statistic_family",
        "start_step",
        "end_step",
        "n",
        "mean",
        "mean_change",
        "median",
        "median_change",
        "standard_deviation",
        "bootstrap_ci_95",
        "paired_t_statistic",
        "paired_t_pvalue",
        "wilcoxon_statistic",
        "wilcoxon_pvalue",
        "paired_cohens_dz",
        "positive_count",
        "negative_count",
        "zero_count",
    ]
    write_csv(
        out / "long_relaxation_statistics.csv",
        statistics,
        fieldnames=statistics_fieldnames,
    )
    write_json(out / "long_relaxation_summary.json", {"steps": args.steps, **summary})
    make_plots(out, per_seed, changes, args.steps)


if __name__ == "__main__":
    main()
