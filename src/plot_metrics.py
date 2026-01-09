import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure numeric
    for c in ["epoch", "phase", "train_loss", "test_acc_full", "test_acc_A", "test_acc_B",
              "test_loss_full", "test_loss_A", "test_loss_B"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def mark_phase_boundary(df: pd.DataFrame) -> int:
    # returns first epoch index of phase 2 (boundary line x)
    phase2 = df[df["phase"] == 2]
    if len(phase2) == 0:
        return -1
    return int(phase2["epoch"].iloc[0])

def plot_accuracy_curves(df: pd.DataFrame, title: str, outpath: Path):
    boundary = mark_phase_boundary(df)

    plt.figure()
    plt.plot(df["epoch"], df["test_acc_full"], label="acc_full")
    plt.plot(df["epoch"], df["test_acc_A"], label="acc_A")
    plt.plot(df["epoch"], df["test_acc_B"], label="acc_B")
    if boundary != -1:
        plt.axvline(boundary, linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_losses(df: pd.DataFrame, title: str, outpath: Path):
    boundary = mark_phase_boundary(df)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["test_loss_full"], label="test_loss_full")
    plt.plot(df["epoch"], df["test_loss_A"], label="test_loss_A")
    plt.plot(df["epoch"], df["test_loss_B"], label="test_loss_B")
    if boundary != -1:
        plt.axvline(boundary, linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_hysteresis_bars(df_sab: pd.DataFrame, df_sba: pd.DataFrame, outpath: Path):
    # Use final epoch
    sab_last = df_sab.iloc[-1]
    sba_last = df_sba.iloc[-1]

    hyst_A = abs(float(sab_last["test_acc_A"]) - float(sba_last["test_acc_A"]))
    hyst_B = abs(float(sab_last["test_acc_B"]) - float(sba_last["test_acc_B"]))
    hyst_full = abs(float(sab_last["test_acc_full"]) - float(sba_last["test_acc_full"]))

    labels = ["Hyst(A)", "Hyst(B)", "Hyst(full)"]
    vals = [hyst_A, hyst_B, hyst_full]

    plt.figure()
    plt.bar(labels, vals)
    plt.ylim(0, 1.0)
    plt.ylabel("absolute accuracy gap")
    plt.title("Hysteresis score (final epoch)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sab", required=True, help="Path to SAB metrics.csv")
    ap.add_argument("--sba", required=True, help="Path to SBA metrics.csv")
    ap.add_argument("--outdir", default="plots", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_sab = load_metrics(args.sab)
    df_sba = load_metrics(args.sba)

    # Accuracy plots (separate figures)
    plot_accuracy_curves(df_sab, "SAB: accuracy vs epoch", outdir / "acc_SAB.png")
    plot_accuracy_curves(df_sba, "SBA: accuracy vs epoch", outdir / "acc_SBA.png")

    # Loss plots (optional but good for blog)
    plot_losses(df_sab, "SAB: loss vs epoch", outdir / "loss_SAB.png")
    plot_losses(df_sba, "SBA: loss vs epoch", outdir / "loss_SBA.png")

    # Hysteresis summary bar
    plot_hysteresis_bars(df_sab, df_sba, outdir / "hysteresis_final.png")

    print("Saved plots to:", outdir.resolve())
    print("Files:")
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
