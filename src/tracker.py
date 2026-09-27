"""Safe experiment artifact tracking."""
import csv
import json
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


class Tracker:
    """Create an isolated run directory and write checkpoints and metrics."""

    def __init__(self, run_dir: Path | str, config: Mapping[str, Any], overwrite: bool = False) -> None:
        self.run_dir = Path(run_dir)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Run directory is not empty: {self.run_dir}; use explicit overwrite")
            for child in self.run_dir.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
        self.metrics_path = self.run_dir / "metrics.csv"
        self._fieldnames: list[str] | None = None
        self.relaxation_metrics_path = self.run_dir / "relaxation_metrics.csv"
        self._relaxation_fieldnames: list[str] | None = None
        self._write_metadata(config)

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=2)
            return result.stdout.strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _write_metadata(self, config: Mapping[str, Any]) -> None:
        experiment, train = config.get("experiment", {}), config.get("train", {})
        data, model = config.get("data", {}), config.get("model", {})
        metadata = {"seed": experiment.get("seed"), "scenario": train.get("scenario"),
                    "device_request": experiment.get("device"), "pytorch_version": torch.__version__,
                    "python_version": platform.python_version(), "cuda_available": torch.cuda.is_available(),
                    "hostname": socket.gethostname(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "data_protocol": data.get("protocol", "class_split"), "activation": model.get("activation", "relu"),
                    "leaky_relu_negative_slope": model.get("leaky_relu_negative_slope"), "optimizer": train.get("optimizer"),
                    "momentum": train.get("momentum"), "relaxation_optimizer_policy": train.get("relaxation_optimizer_policy", "preserve"),
                    "relaxation_checkpoints": train.get("relaxation_checkpoints"), "input_normalization": data.get("normalize"),
                    "network_normalization": model.get("norm"), "rotation_degrees_A": data.get("rotation_degrees_A"),
                    "rotation_degrees_B": data.get("rotation_degrees_B"), "c_ordering_seed": experiment.get("seed"),
                    "git_commit": self._git_commit()}
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def save_weights(self, model: torch.nn.Module, epoch: int) -> Path:
        path = self.run_dir / f"weights_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), path)
        return path

    def save_relaxation_weights(self, model: torch.nn.Module, step: int) -> Path:
        if step < 0:
            raise ValueError("relaxation step must be non-negative")
        path = self.run_dir / f"weights_relax_step_{step:06d}.pt"
        torch.save(model.state_dict(), path)
        return path

    def log_metrics(self, row: Mapping[str, Any]) -> None:
        keys = list(row)
        if self._fieldnames is None:
            self._fieldnames = keys
        elif keys != self._fieldnames:
            raise ValueError(f"Metric columns changed: expected {self._fieldnames}, got {keys}")
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def log_relaxation_metrics(self, row: Mapping[str, Any]) -> None:
        keys = list(row)
        if self._relaxation_fieldnames is None:
            self._relaxation_fieldnames = keys
        elif keys != self._relaxation_fieldnames:
            raise ValueError(f"Relaxation metric columns changed: expected {self._relaxation_fieldnames}, got {keys}")
        with self.relaxation_metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._relaxation_fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
