"""Descriptive comparison of aggregate paper-condition summaries."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.analysis.common import create_output_dir, write_csv, write_json


def parse_condition(specification: str) -> tuple[str, Path]:
    if "=" not in specification: raise ValueError("Condition must use label=aggregate_summary.json")
    label,path=specification.split("=",1)
    if not label or not path: raise ValueError("Condition label and path must be non-empty")
    return label,Path(path)


def compare_conditions(specifications: list[str]):
    parsed=[parse_condition(value) for value in specifications];labels=[label for label,_ in parsed]
    if len(labels)!=len(set(labels)): raise ValueError("Duplicate condition names are not allowed")
    rows=[];details={}
    for label,path in parsed:
        summary=json.loads(path.read_text(encoding="utf-8"));seeds=set(summary.get("seeds",[]));primary=summary.get("primary_metric") or {}
        rows.append({"condition":label,"seed_count":len(seeds),"primary_metric_mean":primary.get("mean"),"primary_metric_ci":primary.get("mean_ci_95"),
                     "prediction_disagreement_mean":(summary.get("matched_prediction_disagreement") or {}).get("mean"),
                     "activation_difference_mean":(summary.get("matched_fc1_dead_ratio_difference") or {}).get("mean"),
                     "matched_fraction":summary.get("matched_endpoint",{}).get("matched_fraction")})
        endpoint_values={int(row["seed"]):row.get("representation_history_score") for row in summary.get("per_seed_matched_endpoints",[]) if row.get("matched")}
        details[label]={"seeds":sorted(seeds),"source":str(path),"matched_primary_by_seed":endpoint_values}
    overlaps={f"{a}__{b}":sorted(set(details[a]["seeds"])&set(details[b]["seeds"])) for i,a in enumerate(labels) for b in labels[i+1:]}
    paired_differences={}
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            shared=sorted(set(details[a]["matched_primary_by_seed"])&set(details[b]["matched_primary_by_seed"]))
            paired_differences[f"{a}__minus__{b}"]=[{"seed":seed,"difference":details[a]["matched_primary_by_seed"][seed]-details[b]["matched_primary_by_seed"][seed]} for seed in shared]
    return rows,{"conditions":details,"overlapping_seeds":overlaps,"paired_primary_differences":paired_differences,"descriptive_only":True}


def _bar(path,rows,column,ylabel):
    plt.figure();plt.bar([r["condition"] for r in rows],[r[column] or 0 for r in rows]);plt.ylabel(ylabel);plt.xticks(rotation=30);plt.tight_layout();plt.savefig(path,dpi=200);plt.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--condition",action="append",required=True);parser.add_argument("--outdir",required=True);args=parser.parse_args()
    rows,details=compare_conditions(args.condition);out=create_output_dir(args.outdir);write_csv(out/"condition_comparison.csv",rows);write_json(out/"condition_comparison.json",details)
    _bar(out/"condition_primary_metric.png",rows,"primary_metric_mean","representation history score")
    _bar(out/"condition_disagreement.png",rows,"prediction_disagreement_mean","prediction disagreement")
    _bar(out/"condition_activation_difference.png",rows,"activation_difference_mean","FC1 dead-ratio difference")


if __name__=="__main__":main()
