"""Part 1 sequential MNIST training entry point."""
import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from src.analysis.common import evaluate, resolve_device
from src.config import load_experiment_config
from src.data import get_mnist_datasets, make_loader, make_relaxation_loader, validate_data_protocol
from src.model import SimpleCNN
from src.optimizer import apply_relaxation_optimizer_policy, build_optimizer
from src.tracker import Tracker
from src.utils_seed import set_seed


def train_one_epoch(model, loader, device, optimizer, criterion, log_every_steps: int = 0,
                    numerical_tracer=None, epoch: int | None = None, phase: int | None = None,
                    phase_data: str | None = None) -> float:
    model.train()
    loss_sum, count = 0.0, 0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        context = {"epoch": epoch, "phase": phase, "phase_data": phase_data, "batch_index": step,
                   "learning_rate": float(optimizer.param_groups[0]["lr"])}
        if numerical_tracer is not None:
            numerical_tracer.begin_batch(context, x, y)
            numerical_tracer.check_named(context, "before_forward", {"input": x, **dict(model.named_parameters())},
                                           model=model, optimizer=optimizer, x=x, y=y)
            numerical_tracer.check_optimizer(context, "before_forward_optimizer_state", optimizer,
                                               model=model, x=x, y=y)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if numerical_tracer is not None:
            numerical_tracer.check_forward(context, logits, model=model, optimizer=optimizer, x=x, y=y)
        loss = criterion(logits, y)
        if numerical_tracer is not None:
            numerical_tracer.check_named(context, "after_loss", {"loss": loss}, model=model,
                                           optimizer=optimizer, x=x, y=y)
        loss.backward()
        if numerical_tracer is not None:
            numerical_tracer.check_gradients(context, model, optimizer=optimizer, x=x, y=y)
            pre_model_state = numerical_tracer.clone_model_state(model)
            pre_optimizer_state = numerical_tracer.clone_optimizer_state(optimizer)
            numerical_tracer.check_named(context, "before_optimizer_step", dict(model.named_parameters()),
                                           model=model, optimizer=optimizer, x=x, y=y,
                                           pre_model_state=pre_model_state, pre_optimizer_state=pre_optimizer_state)
            numerical_tracer.check_optimizer(context, "before_optimizer_step", optimizer, model=model, x=x, y=y,
                                               pre_model_state=pre_model_state, pre_optimizer_state=pre_optimizer_state)
        optimizer.step()
        if numerical_tracer is not None:
            numerical_tracer.check_named(context, "after_optimizer_step", dict(model.named_parameters()),
                                           model=model, optimizer=optimizer, x=x, y=y,
                                           pre_model_state=pre_model_state, pre_optimizer_state=pre_optimizer_state)
            numerical_tracer.check_optimizer(context, "after_optimizer_step", optimizer, model=model, x=x, y=y,
                                               pre_model_state=pre_model_state, pre_optimizer_state=pre_optimizer_state)
        loss_sum += float(loss.item()) * y.size(0)
        count += y.size(0)
        if log_every_steps and step % log_every_steps == 0:
            print(f"  step {step}: train_loss={loss_sum / count:.4f}")
    return loss_sum / max(count, 1)


@torch.no_grad()
def evaluate_with_numerical_trace(model, loader, device, criterion, tracer, epoch, phase, phase_data, domain):
    model.eval(); loss_sum = correct = count = 0
    for batch_index, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        context = {"epoch": epoch, "phase": phase, "phase_data": phase_data,
                   "batch_index": batch_index, "evaluation_domain": domain, "learning_rate": None}
        tracer.begin_batch(context, x, y); logits = model(x); tracer.check_forward(context, logits, model=model, x=x, y=y)
        loss = criterion(logits, y); tracer.check_named(context, "after_evaluation_loss", {"loss": loss}, model=model, x=x, y=y)
        loss_sum += float(loss.item()) * y.size(0); correct += int((logits.argmax(1) == y).sum()); count += y.size(0)
    return loss_sum / max(count, 1), correct / max(count, 1)


def _validate_config(config: dict[str, Any]) -> None:
    if config["data"].get("dataset") != "MNIST":
        raise ValueError("Only dataset=MNIST is supported")
    if config["model"].get("arch") != "simple_cnn":
        raise ValueError("Only model.arch=simple_cnn is supported")
    validate_data_protocol(config["data"])
    if config["train"]["scenario"] not in {"SAB", "SBA", "SABC", "SBAC"}:
        raise ValueError("train.scenario must be SAB, SBA, SABC, or SBAC")
    if int(config["train"]["phase_epochs"]) * 2 != int(config["train"]["epochs_total"]):
        raise ValueError("epochs_total must equal 2 * phase_epochs")
    if config["train"]["scenario"] in {"SABC", "SBAC"}:
        validate_relaxation_schedule(config["train"])
        if config["train"].get("relaxation_optimizer_policy", "preserve") not in {"preserve", "reset"}:
            raise ValueError("relaxation_optimizer_policy must be 'preserve' or 'reset'")


def scenario_phase_order(scenario: str) -> tuple[str, str, str | None]:
    schedules = {"SAB": ("A", "B", None), "SBA": ("B", "A", None),
                 "SABC": ("A", "B", "C"), "SBAC": ("B", "A", "C")}
    if scenario not in schedules:
        raise ValueError(f"Unsupported scenario: {scenario}")
    return schedules[scenario]


def validate_relaxation_schedule(train_config: dict[str, Any]) -> list[int]:
    steps = int(train_config.get("relaxation_steps", -1))
    checkpoints = [int(value) for value in train_config.get("relaxation_checkpoints", [])]
    if steps < 0:
        raise ValueError("relaxation_steps must be non-negative")
    if checkpoints != sorted(checkpoints):
        raise ValueError("relaxation_checkpoints must be sorted")
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("relaxation_checkpoints must be unique")
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] > steps:
        raise ValueError("relaxation_checkpoints must begin with 0 and stay within relaxation_steps")
    if steps not in checkpoints:
        raise ValueError("final relaxation_steps value must be included in relaxation_checkpoints")
    if int(train_config.get("relaxation_samples_per_class", 0)) <= 0:
        raise ValueError("relaxation_samples_per_class must be greater than 0")
    return checkpoints


def initialize_relaxation(tracker: Tracker, model: nn.Module, optimizer, train_config: dict[str, Any]):
    """Persist the exact post-history model before applying the optimizer-state policy."""
    tracker.save_relaxation_weights(model, 0)
    return apply_relaxation_optimizer_policy(optimizer, model, train_config)


def main(config_path: str | None = None, numerical_tracer=None, stop_after_epoch: int | None = None,
         diagnostic_save_dir: str | None = None) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        config_path = parser.parse_args().config
    config = load_experiment_config(config_path)
    _validate_config(config)
    set_seed(int(config["experiment"]["seed"]), bool(config["experiment"]["strict_determinism"]))
    device = resolve_device(config["experiment"]["device"])
    train_ds, test_ds, a_train, b_train, a_test, b_test, full_test = get_mnist_datasets(config)
    scenario, seed = config["train"]["scenario"], int(config["experiment"]["seed"])
    first_name, second_name, _ = scenario_phase_order(scenario)
    first, second = ((a_train, b_train) if first_name == "A" else (b_train, a_train))
    shuffle = bool(config["data"]["shuffle"])
    loaders = [make_loader(train_ds, indices, config, seed, shuffle) for indices in (first, second)]
    eval_loaders = [make_loader(test_ds, indices, config, seed, False) for indices in (full_test, a_test, b_test)]
    model = SimpleCNN(config["model"]["norm"], int(config["model"]["group_norm_groups"]), config["model"]["activation"],
                      float(config["model"].get("leaky_relu_negative_slope", 0.01))).to(device)
    train_cfg = config["train"]
    optimizer = build_optimizer(model.parameters(), train_cfg)
    logging = config["logging"]
    run_name = logging.get("run_name") or f"{config['experiment']['name']}_{scenario}_seed{seed}_norm{config['model']['norm']}"
    run_dir = Path(diagnostic_save_dir) if diagnostic_save_dir is not None else Path(logging["save_dir"]) / run_name
    tracker = Tracker(run_dir, config, bool(logging.get("overwrite", False)))
    if numerical_tracer is not None:
        numerical_tracer.install_activation_hooks(model)
        numerical_tracer.clone_model_state = lambda current: {
            name: value.detach().cpu().clone() for name, value in current.state_dict().items()}
        numerical_tracer.clone_optimizer_state = lambda current: copy.deepcopy(current.state_dict())
    tracker.save_weights(model, 0)
    criterion = nn.CrossEntropyLoss()
    phase_epochs = int(train_cfg["phase_epochs"])
    for epoch in range(1, int(train_cfg["epochs_total"]) + 1):
        phase = 1 if epoch <= phase_epochs else 2
        phase_data = first_name if phase == 1 else second_name
        train_loss = train_one_epoch(model, loaders[phase - 1], device, optimizer, criterion,
                                     int(train_cfg.get("log_every_steps", 0)), numerical_tracer,
                                     epoch, phase, phase_data)
        if logging.get("save_weights_every_epoch", True):
            tracker.save_weights(model, epoch)
        if numerical_tracer is None:
            full, a, b = [evaluate(model, loader, device, criterion) for loader in eval_loaders]
        else:
            full, a, b = [evaluate_with_numerical_trace(model, loader, device, criterion, numerical_tracer,
                                                        epoch, phase, phase_data, domain)
                          for loader, domain in zip(eval_loaders, ("full", "A", "B"))]
        tracker.log_metrics({"epoch": epoch, "phase": phase, "phase_data": phase_data, "train_loss": train_loss,
                             "test_loss_full": full[0], "test_acc_full": full[1], "test_loss_A": a[0],
                             "test_acc_A": a[1], "test_loss_B": b[0], "test_acc_B": b[1]})
        print(f"[{scenario}] epoch {epoch:02d}: full={full[1]:.4f} A={a[1]:.4f} B={b[1]:.4f}")
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            if numerical_tracer is not None:
                numerical_tracer.close_hooks()
            return
    if scenario in {"SABC", "SBAC"}:
        checkpoints = validate_relaxation_schedule(train_cfg)
        samples_per_class = int(train_cfg["relaxation_samples_per_class"])
        c_batches = make_relaxation_loader(train_ds, config, seed, samples_per_class)
        completed_steps = 0
        recent_losses: list[float] = []
        for target_step in checkpoints:
            if target_step == 0:
                optimizer = initialize_relaxation(tracker, model, optimizer, train_cfg)
            while completed_steps < target_step:
                x, y = next(c_batches); x, y = x.to(device), y.to(device)
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y); loss.backward(); optimizer.step()
                recent_losses.append(float(loss.item())); completed_steps += 1
            if target_step != 0:
                tracker.save_relaxation_weights(model, target_step)
            full, a, b = [evaluate(model, loader, device, criterion) for loader in eval_loaders]
            tracker.log_relaxation_metrics({"scenario": scenario, "relaxation_step": target_step,
                "relaxation_optimizer_policy": train_cfg.get("relaxation_optimizer_policy", "preserve"),
                "train_loss_C_recent": (sum(recent_losses) / len(recent_losses)) if recent_losses else None,
                "test_loss_full": full[0], "test_acc_full": full[1], "test_loss_A": a[0],
                "test_acc_A": a[1], "test_loss_B": b[0], "test_acc_B": b[1],
                "prediction_disagreement_optional": None})
            recent_losses.clear()


if __name__ == "__main__":
    main()
