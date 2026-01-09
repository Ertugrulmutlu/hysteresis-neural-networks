# src/tracker.py
from pathlib import Path
import torch
import yaml

class Tracker:
    def __init__(self, run_dir, config):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)

        self.metrics_path = self.run_dir / "metrics.csv"
        self._metrics_header_written = self.metrics_path.exists()

    def save_weights(self, model, epoch):
        path = self.run_dir / f"weights_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), path)

    def log_metrics(self, row: dict):
        # row must always have the same keys
        if not self._metrics_header_written:
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                f.write(",".join(row.keys()) + "\n")
            self._metrics_header_written = True

        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")
