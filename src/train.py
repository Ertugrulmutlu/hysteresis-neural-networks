# src/train.py
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from utils_seed import set_seed
from data import get_mnist_datasets, make_loader
from model import SimpleCNN
from tracker import Tracker

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total += int(y.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return avg_loss, acc

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        total += int(y.size(0))

    return total_loss / max(total, 1)

def resolve_device(requested: str):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)

def main(config_path):
    print(">>> FAZ 1 TRAIN LOOP AKTİF <<<")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"], config["experiment"]["strict_determinism"])
    device = resolve_device(config["experiment"]["device"])

    # Data
    train_ds, test_ds, A_train_idx, B_train_idx, A_test_idx, B_test_idx, full_test_idx = get_mnist_datasets(config)

    scenario = config["train"]["scenario"]  # "SAB" or "SBA"
    phase_epochs = int(config["train"]["phase_epochs"])
    epochs_total = int(config["train"]["epochs_total"])
    assert phase_epochs * 2 == epochs_total, "For FAZ1 we assume epochs_total == 2 * phase_epochs"

    if scenario == "SAB":
        first_train_idx, second_train_idx = A_train_idx, B_train_idx
        first_name, second_name = "A", "B"
    else:
        first_train_idx, second_train_idx = B_train_idx, A_train_idx
        first_name, second_name = "B", "A"

    train_loader_1 = make_loader(train_ds, first_train_idx, config, config["experiment"]["seed"], shuffle=True)
    train_loader_2 = make_loader(train_ds, second_train_idx, config, config["experiment"]["seed"], shuffle=True)

    # Evaluation loaders (no shuffle)
    test_loader_full = make_loader(test_ds, full_test_idx, config, config["experiment"]["seed"], shuffle=False)
    test_loader_A    = make_loader(test_ds, A_test_idx,   config, config["experiment"]["seed"], shuffle=False)
    test_loader_B    = make_loader(test_ds, B_test_idx,   config, config["experiment"]["seed"], shuffle=False)

    # Model
    model = SimpleCNN(
        norm=config["model"]["norm"],
        gn_groups=config["model"]["group_norm_groups"]
    ).to(device)

    # Optimizer
    opt_name = config["train"]["optimizer"].lower()
    lr = float(config["train"]["lr"])
    wd = float(config["train"]["weight_decay"])

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config["train"]["momentum"]),
            weight_decay=wd
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    criterion = nn.CrossEntropyLoss()

    # Run dir + tracker
    run_name = f"{config['experiment']['name']}_{scenario}_seed{config['experiment']['seed']}_norm{config['model']['norm']}"
    run_dir = Path(config["logging"]["save_dir"]) / run_name
    tracker = Tracker(run_dir, config)

    # Epoch 0 snapshot (init)
    tracker.save_weights(model, epoch=0)

    # Train loop
    for epoch in range(1, epochs_total + 1):
        in_phase1 = epoch <= phase_epochs
        phase = 1 if in_phase1 else 2
        phase_data = first_name if in_phase1 else second_name
        loader = train_loader_1 if in_phase1 else train_loader_2

        train_loss = train_one_epoch(model, loader, device, optimizer, criterion)

        # Save weights every epoch (as per config)
        if config["logging"]["save_weights_every_epoch"]:
            tracker.save_weights(model, epoch=epoch)

        # Eval on full/A/B
        test_loss_full, test_acc_full = evaluate(model, test_loader_full, device, criterion)
        test_loss_A,    test_acc_A    = evaluate(model, test_loader_A, device, criterion)
        test_loss_B,    test_acc_B    = evaluate(model, test_loader_B, device, criterion)

        tracker.log_metrics({
            "epoch": epoch,
            "phase": phase,
            "phase_data": phase_data,
            "train_loss": round(train_loss, 6),
            "test_loss_full": round(test_loss_full, 6),
            "test_acc_full": round(test_acc_full, 6),
            "test_loss_A": round(test_loss_A, 6),
            "test_acc_A": round(test_acc_A, 6),
            "test_loss_B": round(test_loss_B, 6),
            "test_acc_B": round(test_acc_B, 6),
        })

        print(f"[{scenario}] epoch {epoch:02d}/{epochs_total} | phase {phase}({phase_data}) "
              f"| train_loss {train_loss:.4f} | acc_full {test_acc_full:.4f} | acc_A {test_acc_A:.4f} | acc_B {test_acc_B:.4f}")

    print("FAZ 1 complete: training + logging done.")
    print(f"Run dir: {run_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/train.py configs/mnist_ablation_v0.yaml")
    main(sys.argv[1])
