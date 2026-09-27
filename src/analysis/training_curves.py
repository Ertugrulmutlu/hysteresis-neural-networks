"""Validated Part 1 training curves and final order-dependent accuracy gaps."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.analysis.common import create_output_dir, write_csv


def load_metrics(path):
    frame = pd.read_csv(path); frame["epoch"] = pd.to_numeric(frame["epoch"], errors="raise")
    if frame["epoch"].duplicated().any() or not frame["epoch"].is_monotonic_increasing: raise ValueError(f"Duplicate or non-monotonic epochs in {path}")
    return frame


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--sab", required=True); parser.add_argument("--sba", required=True); parser.add_argument("--outdir", default="plots"); args=parser.parse_args()
    out=create_output_dir(args.outdir); frames={"SAB":load_metrics(args.sab),"SBA":load_metrics(args.sba)}
    for scenario, df in frames.items():
        boundary=df.loc[df["phase"].eq(2),"epoch"].min()
        for kind, cols in (("acc",["test_acc_full","test_acc_A","test_acc_B"]),("loss",["train_loss","test_loss_full","test_loss_A","test_loss_B"])):
            plt.figure()
            for col in cols: plt.plot(df["epoch"],df[col],label=col)
            if pd.notna(boundary): plt.axvline(boundary-.5,ls="--",label="phase boundary")
            plt.xlabel("epoch");plt.ylabel(kind);plt.legend();plt.tight_layout();plt.savefig(out/f"{kind}_{scenario}.png",dpi=200);plt.close()
    rows=[]
    for subset,col in (("A","test_acc_A"),("B","test_acc_B"),("full","test_acc_full")):
        a=float(frames["SAB"].iloc[-1][col]);b=float(frames["SBA"].iloc[-1][col]);rows.append({"subset":subset,"sab_final_accuracy":a,"sba_final_accuracy":b,"absolute_order_dependent_accuracy_gap":abs(a-b)})
    write_csv(out/"final_order_dependent_accuracy_gap.csv",rows)


if __name__ == "__main__": main()
