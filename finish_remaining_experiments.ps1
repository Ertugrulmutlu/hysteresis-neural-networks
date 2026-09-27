$ErrorActionPreference = "Stop"

$repo = "D:\DEV\Projects\00_ACTIVE\hysteresis-neural-networks-main"
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { throw "Python not found. Create/activate .venv first." }
    $python = $cmd.Source
}

$orchestrator = Join-Path $repo "run_remaining_experiments.py"
@'
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml

from src.config import load_experiment_config

ROOT = Path.cwd()
PYTHON = sys.executable
SEEDS = [101, 202, 303, 404, 505]
LR_CANDIDATES = [0.04, 0.03, 0.02, 0.01, 0.005]
GENERATED_ROOT = ROOT / "configs" / "generated" / "auto_remaining"
MANIFEST_ROOT = ROOT / "results" / "manifests" / "auto_remaining"
DIAG_ROOT = ROOT / "diagnostics" / "auto_stability"
PLOTS_ROOT = ROOT / "plots" / "auto_remaining"
STATUS_PATH = ROOT / "results" / "auto_remaining_status.json"


def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100, flush=True)


def run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def lr_token(lr: float) -> str:
    return f"lr{int(round(lr * 10000)):04d}"


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Refusing to replace a different generated config: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def make_config(base_path: Path, *, seed: int, scenario: str, lr: float, run_name: str) -> dict:
    cfg = load_experiment_config(base_path)
    cfg["experiment"]["seed"] = int(seed)
    cfg["train"]["scenario"] = scenario
    cfg["train"]["lr"] = float(lr)
    cfg["logging"]["run_name"] = run_name
    cfg["logging"]["overwrite"] = False
    return cfg


def make_pilot_config(lr: float, seed: int, scenario: str) -> Path:
    token = lr_token(lr)
    base = ROOT / "configs" / "paper" / f"mnist_class_split_reset_leaky_relu_long50k_{scenario.lower()}.yaml"
    run_name = f"diag_class_split_{scenario}_reset_leaky_relu_{token}_seed{seed}_normnone"
    cfg = make_config(base, seed=seed, scenario=scenario, lr=lr, run_name=run_name)
    path = GENERATED_ROOT / "stability" / token / f"seed{seed}_{scenario.lower()}.yaml"
    write_yaml(path, cfg)
    return path


def pilot_is_finite(lr: float) -> bool:
    """Outcome-blind numerical screen: all 10 paired history runs must remain finite through epoch 11."""
    token = lr_token(lr)
    banner(f"STABILITY PILOT {token}: LeakyReLU, 5 paired seeds, through epoch 11")
    all_ok = True
    for seed in SEEDS:
        for scenario in ("SABC", "SBAC"):
            cfg = make_pilot_config(lr, seed, scenario)
            outdir = DIAG_ROOT / token / f"seed{seed}_{scenario.lower()}"
            summary_path = outdir / "run_summary.json"
            if not summary_path.exists():
                if outdir.exists():
                    shutil.rmtree(outdir)
                run([
                    PYTHON, "-m", "src.diagnostics.trace_training_failure",
                    "--config", str(cfg),
                    "--trace-from-epoch", "10",
                    "--stop-after-epoch", "11",
                    "--stop-on-first-nonfinite",
                    "--outdir", str(outdir),
                ])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ok = bool(summary.get("requested_epoch_range_completed_without_nonfinite_values"))
            print(f"  {token} seed={seed} {scenario}: {'FINITE' if ok else 'NON-FINITE'}", flush=True)
            all_ok &= ok
    return all_ok


def create_base_config(activation: str, lr: float, scenario: str) -> Path:
    token = lr_token(lr)
    if activation == "leaky_relu":
        base_name = f"mnist_class_split_reset_leaky_relu_long50k_{scenario.lower()}.yaml"
        activation_name = "leaky_relu"
    elif activation == "relu":
        base_name = f"mnist_class_split_reset_relu_long50k_{scenario.lower()}.yaml"
        activation_name = "relu"
    else:
        raise ValueError(activation)
    base = ROOT / "configs" / "paper" / base_name
    run_name = f"mnist_class_split_{scenario}_reset_{activation_name}_long50k_{token}_seed1337_normnone"
    cfg = make_config(base, seed=1337, scenario=scenario, lr=lr, run_name=run_name)
    path = GENERATED_ROOT / "bases" / f"{activation_name}_long50k_{token}_{scenario.lower()}.yaml"
    write_yaml(path, cfg)
    return path


def execute_matrix(activation: str, lr: float) -> tuple[Path, Path]:
    token = lr_token(lr)
    base_sabc = create_base_config(activation, lr, "SABC")
    base_sbac = create_base_config(activation, lr, "SBAC")
    label = f"class_split_reset_{activation}_long50k_{token}_5seeds"
    generated = GENERATED_ROOT / label
    manifest = MANIFEST_ROOT / f"{label}.json"
    banner(f"FULL MATRIX: {label}")
    run([
        PYTHON, "-m", "src.experiments.run_paper_matrix",
        "--base-sabc", str(base_sabc),
        "--base-sbac", str(base_sbac),
        "--seeds", *[str(x) for x in SEEDS],
        "--generated-config-dir", str(generated),
        "--manifest-out", str(manifest),
        "--execute", "--skip-existing",
    ])
    return manifest, generated


def load_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint is not a state dict: {path}")
    return value


def checkpoint_is_finite(path: Path) -> bool:
    for value in load_state(path).values():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all().item():
                return False
    return True


def csv_is_finite(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for value in row.values():
                if value in (None, ""):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    return False
    return True


def audit_manifest(manifest_path: Path) -> tuple[bool, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest["entries"]:
        run_dir = Path(entry["expected_run_directory"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        checkpoints = sorted(run_dir.glob("weights_epoch_*.pt")) + sorted(run_dir.glob("weights_relax_step_*.pt"))
        if not checkpoints:
            failures.append(f"{run_dir.name}: no checkpoints")
            continue
        for checkpoint in checkpoints:
            if not checkpoint_is_finite(checkpoint):
                failures.append(f"{run_dir.name}: non-finite {checkpoint.name}")
                break
        for csv_name in ("metrics.csv", "relaxation_metrics.csv"):
            if not csv_is_finite(run_dir / csv_name):
                failures.append(f"{run_dir.name}: missing/non-finite {csv_name}")
    return not failures, failures


def aggregate(manifest: Path, outdir: Path) -> None:
    run([
        PYTHON, "-m", "src.analysis.aggregate_common_relaxation",
        "--manifest", str(manifest),
        "--samples-per-class", "200",
        "--bootstrap-samples", "10000",
        "--bootstrap-seed", "12345",
        "--max-accuracy-gap", "0.002",
        "--minimum-accuracy", "0.97",
        "--outdir", str(outdir),
        "--device", "cuda",
    ])


def analyze_long(aggregate_dir: Path, outdir: Path) -> None:
    run([
        PYTHON, "-m", "src.analysis.analyze_long_relaxation",
        "--aggregate-csv", str(aggregate_dir / "per_seed_relaxation_metrics.csv"),
        "--steps", "10000", "25000", "50000",
        "--bootstrap-samples", "10000",
        "--bootstrap-seed", "12345",
        "--outdir", str(outdir),
    ])


def run_rotated() -> tuple[Path, Path]:
    label = "rotated_reset_relu_5seeds"
    manifest = MANIFEST_ROOT / f"{label}.json"
    generated = GENERATED_ROOT / label
    banner("ROTATED-MNIST SAME-LABEL CONTROL")
    run([
        PYTHON, "-m", "src.experiments.run_paper_matrix",
        "--base-sabc", str(ROOT / "configs" / "paper" / "mnist_rotated_reset_relu_sabc.yaml"),
        "--base-sbac", str(ROOT / "configs" / "paper" / "mnist_rotated_reset_relu_sbac.yaml"),
        "--seeds", *[str(x) for x in SEEDS],
        "--generated-config-dir", str(generated),
        "--manifest-out", str(manifest),
        "--execute", "--skip-existing",
    ])
    ok, failures = audit_manifest(manifest)
    if not ok:
        raise RuntimeError("Rotated-MNIST finiteness audit failed:\n" + "\n".join(failures))
    outdir = PLOTS_ROOT / "aggregate_rotated_reset_relu_5seeds"
    aggregate(manifest, outdir)
    return manifest, outdir


def main() -> None:
    required = [
        ROOT / "src" / "train.py",
        ROOT / "src" / "experiments" / "run_paper_matrix.py",
        ROOT / "src" / "diagnostics" / "trace_training_failure.py",
        ROOT / "configs" / "paper" / "mnist_class_split_reset_leaky_relu_long50k_sabc.yaml",
        ROOT / "configs" / "paper" / "mnist_rotated_reset_relu_sabc.yaml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Run this from the repository root. Missing:\n" + "\n".join(missing))

    banner("PRE-FLIGHT TESTS")
    print(f"Python: {PYTHON}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run([
        PYTHON, "-m", "pytest", "-q",
        "tests/test_model.py",
        "tests/test_rotated_protocol.py",
        "tests/test_activation_condition_control.py",
        "tests/test_paper_matrix.py",
        "tests/test_numerical_stability_diagnostics.py",
        "tests/test_optimizer_policy.py",
    ])

    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    chosen_lr: float | None = None
    leaky_manifest: Path | None = None

    for lr in LR_CANDIDATES:
        if not pilot_is_finite(lr):
            print(f"{lr_token(lr)} rejected by history-phase numerical screen.", flush=True)
            continue

        candidate_manifest, _ = execute_matrix("leaky_relu", lr)
        ok, failures = audit_manifest(candidate_manifest)
        if not ok:
            print(f"{lr_token(lr)} passed the short pilot but failed the long50k audit:")
            for item in failures:
                print("  -", item)
            print("Trying the next lower predeclared LR candidate.", flush=True)
            continue

        chosen_lr = lr
        leaky_manifest = candidate_manifest
        break

    if chosen_lr is None or leaky_manifest is None:
        raise RuntimeError("No LR candidate kept all LeakyReLU runs finite. No mechanism conclusion was produced.")

    token = lr_token(chosen_lr)
    banner(f"SELECTED STABLE LR: {chosen_lr} ({token})")

    # Symmetric matched control: ReLU is rerun at exactly the selected LeakyReLU LR.
    relu_manifest, _ = execute_matrix("relu", chosen_lr)
    ok, failures = audit_manifest(relu_manifest)
    if not ok:
        raise RuntimeError("Matched ReLU finiteness audit failed:\n" + "\n".join(failures))

    leaky_agg = PLOTS_ROOT / f"aggregate_class_split_reset_leaky_relu_long50k_{token}_5seeds"
    relu_agg = PLOTS_ROOT / f"aggregate_class_split_reset_relu_long50k_{token}_5seeds"
    aggregate(leaky_manifest, leaky_agg)
    aggregate(relu_manifest, relu_agg)

    analyze_long(leaky_agg, PLOTS_ROOT / f"long_analysis_leaky_relu_{token}")
    analyze_long(relu_agg, PLOTS_ROOT / f"long_analysis_relu_{token}")

    activation_compare = PLOTS_ROOT / f"activation_condition_comparison_long50k_{token}_5seeds"
    run([
        PYTHON, "-m", "src.analysis.compare_activation_conditions",
        "--relu-csv", str(relu_agg / "per_seed_relaxation_metrics.csv"),
        "--leaky-csv", str(leaky_agg / "per_seed_relaxation_metrics.csv"),
        "--steps", "10000", "25000", "50000",
        "--primary-step", "50000",
        "--bootstrap-samples", "10000",
        "--bootstrap-seed", "12345",
        "--outdir", str(activation_compare),
    ])

    rotated_manifest, rotated_agg = run_rotated()

    status = {
        "selected_stable_learning_rate": chosen_lr,
        "selected_lr_token": token,
        "seeds": SEEDS,
        "leaky_manifest": str(leaky_manifest),
        "relu_manifest": str(relu_manifest),
        "activation_comparison": str(activation_compare / "activation_condition_summary.json"),
        "rotated_manifest": str(rotated_manifest),
        "rotated_aggregate": str(rotated_agg / "aggregate_summary.json"),
        "completed": True,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    banner("ALL REMAINING PREDECLARED CONTROLS COMPLETED")
    print(json.dumps(status, indent=2))
    print("\nPrimary files to inspect:")
    print(" -", activation_compare / "activation_condition_summary.json")
    print(" -", rotated_agg / "aggregate_summary.json")
    print(" -", STATUS_PATH)


if __name__ == "__main__":
    main()

'@ | Set-Content -Path $orchestrator -Encoding UTF8

Write-Host "Created: $orchestrator"
& $python $orchestrator
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
