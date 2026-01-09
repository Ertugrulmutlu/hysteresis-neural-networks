# src/tracker.py
from pathlib import Path
import torch
import yaml

class Tracker:
    def __init__(self, run_dir, config):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config_resolved.yaml", "w") as f:
            yaml.dump(config, f)

    def save_weights(self, model, epoch):
        path = self.run_dir / f"weights_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), path)

    def log_metrics(self, epoch, metrics: dict):
        path = self.run_dir / "metrics.csv"
        header = not path.exists()
        with open(path, "a") as f:
            if header:
                f.write(",".join(metrics.keys()) + "\n")
            f.write(",".join(str(v) for v in metrics.values()) + "\n")
