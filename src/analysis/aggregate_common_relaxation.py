"""Paired-seed aggregation for common-relaxation experiments."""
import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.common import (create_output_dir, fixed_probe_loader, resolve_device,
                                 validate_common_relaxation_pair, write_csv, write_json)
from src.analysis.common_relaxation import (_load_relaxation_metrics, compute_common_relaxation_rows,
                                             select_performance_matched_endpoint)


def bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int):
    array=np.asarray(values,dtype=float)
    if not len(array): raise ValueError("Cannot bootstrap zero seeds")
    if not np.isfinite(array).all(): raise ValueError("Bootstrap values contain NaN or infinity")
    if len(array)==1: return None
    if samples<=0: raise ValueError("bootstrap_samples must be positive")
    rng=np.random.default_rng(seed); means=array[rng.integers(0,len(array),size=(samples,len(array)))].mean(1)
    return [float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]


def summarize(values: Sequence[float], bootstrap_samples: int, seed: int):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all(): raise ValueError("Summary requires finite paired-seed values")
    return {"count":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std(ddof=1)) if len(a)>1 else 0.0,
            "minimum":float(a.min()),"maximum":float(a.max()),"mean_ci_95":bootstrap_mean_ci(a,bootstrap_samples,seed)}


def pairs_from_manifest(path: Path):
    manifest=json.loads(path.read_text(encoding="utf-8")); grouped={}
    for entry in manifest["entries"]: grouped.setdefault(int(entry["seed"]),{})[entry["scenario"]]=entry["expected_run_directory"]
    missing={seed:sorted({"SABC","SBAC"}-set(items)) for seed,items in grouped.items() if set(items)!={"SABC","SBAC"}}
    if missing: raise ValueError(f"Manifest has missing run pairs: {missing}")
    return [(seed,Path(items["SABC"]),Path(items["SBAC"])) for seed,items in sorted(grouped.items())]


def _plot(path, rows, metric, bootstrap_samples, bootstrap_seed):
    plt.figure(); seeds=sorted({row["seed"] for row in rows})
    for seed in seeds:
        subset=[row for row in rows if row["seed"]==seed];plt.plot([r["relaxation_step"] for r in subset],[r[metric] for r in subset],alpha=.25)
    steps=sorted({row["relaxation_step"] for row in rows});groups=[[r[metric] for r in rows if r["relaxation_step"]==step] for step in steps];means=[np.mean(group) for group in groups]
    intervals=[bootstrap_mean_ci(group,bootstrap_samples,bootstrap_seed+i) for i,group in enumerate(groups)]
    plt.plot(steps,means,linewidth=2,label=f"mean (n={len(seeds)})")
    if all(interval is not None for interval in intervals): plt.fill_between(steps,[x[0] for x in intervals],[x[1] for x in intervals],alpha=.2,label="95% bootstrap CI")
    plt.xscale("symlog",linthresh=1);plt.xlabel("relaxation updates");plt.ylabel(metric);plt.legend();plt.tight_layout();plt.savefig(path,dpi=200);plt.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest");parser.add_argument("--pair",nargs=3,action="append",metavar=("SEED","RUN_SABC","RUN_SBAC"));parser.add_argument("--samples-per-class",type=int,default=200)
    parser.add_argument("--bootstrap-samples",type=int,default=10000);parser.add_argument("--bootstrap-seed",type=int,default=12345)
    parser.add_argument("--max-accuracy-gap",type=float,default=.002);parser.add_argument("--minimum-accuracy",type=float,default=.97);parser.add_argument("--outdir",required=True)
    parser.add_argument("--device",default="cpu");args=parser.parse_args();device=resolve_device(args.device)
    if bool(args.manifest)==bool(args.pair): raise ValueError("Provide exactly one of --manifest or --pair")
    input_pairs=pairs_from_manifest(Path(args.manifest)) if args.manifest else [(int(seed),Path(a),Path(b)) for seed,a,b in args.pair]
    all_rows=[];endpoints=[];compatibility=None
    for seed,run_a,run_b in input_pairs:
        pair=validate_common_relaxation_pair(run_a,run_b); signature=(pair.config_sabc["data"].get("protocol","class_split"),pair.config_sabc["model"]["activation"],pair.config_sabc["train"].get("relaxation_optimizer_policy","preserve"),pair.relaxation_steps)
        if compatibility is None: compatibility=signature
        elif signature!=compatibility: raise ValueError(f"Incompatible run pair for seed {seed}: {signature} != {compatibility}")
        loader=fixed_probe_loader(pair.config_sabc,args.samples_per_class,False);rows=compute_common_relaxation_rows(pair,_load_relaxation_metrics(run_a,pair.relaxation_steps),_load_relaxation_metrics(run_b,pair.relaxation_steps),loader,device,1e-6,1e-4)
        all_rows.extend({"seed":seed,**row} for row in rows);endpoint=select_performance_matched_endpoint(rows,args.max_accuracy_gap,args.minimum_accuracy)
        endpoints.append({"seed":seed,"matched":endpoint is not None,**({} if endpoint is None else endpoint)})
    metrics=[key for key in all_rows[0] if key not in {"seed","relaxation_step"}];aggregate=[]
    for step in sorted({row["relaxation_step"] for row in all_rows}):
        subset=[row for row in all_rows if row["relaxation_step"]==step]
        for metric in metrics: aggregate.append({"relaxation_step":step,"metric":metric,**summarize([r[metric] for r in subset],args.bootstrap_samples,args.bootstrap_seed+step)})
    matched=[row for row in endpoints if row["matched"]];out=create_output_dir(args.outdir);write_csv(out/"per_seed_relaxation_metrics.csv",all_rows);write_csv(out/"per_seed_matched_endpoints.csv",endpoints);write_csv(out/"aggregate_relaxation_metrics.csv",aggregate)
    matched_summary=[{"seed_count":len(endpoints),"matched_count":len(matched),"matched_fraction":len(matched)/len(endpoints)}];write_csv(out/"aggregate_matched_endpoint_summary.csv",matched_summary)
    write_json(out/"aggregate_summary.json",{"seeds":[row["seed"] for row in endpoints],"compatibility_signature":compatibility,"matched_endpoint":matched_summary[0],
        "per_seed_matched_endpoints":endpoints,
        "primary_metric":summarize([row["representation_history_score"] for row in matched],args.bootstrap_samples,args.bootstrap_seed) if matched else None,
        "matched_prediction_disagreement":summarize([row["prediction_disagreement"] for row in matched],args.bootstrap_samples,args.bootstrap_seed) if matched else None,
        "matched_fc1_dead_ratio_difference":summarize([row["fc1_dead_ratio_difference"] for row in matched],args.bootstrap_samples,args.bootstrap_seed) if matched else None})
    for filename,metric in (("aggregate_accuracy_recovery.png","full_accuracy_difference"),("aggregate_representation_history.png","representation_history_score"),("aggregate_functional_history.png","prediction_disagreement"),("aggregate_weight_history.png","normalized_weight_distance"),("aggregate_activation_history.png","fc1_dead_ratio_difference")):_plot(out/filename,all_rows,metric,args.bootstrap_samples,args.bootstrap_seed)
    plt.figure();plt.hist([row["relaxation_step"] for row in matched]);plt.xlabel("matched endpoint step");plt.tight_layout();plt.savefig(out/"matched_endpoint_distribution.png",dpi=200);plt.close()


if __name__=="__main__":main()
