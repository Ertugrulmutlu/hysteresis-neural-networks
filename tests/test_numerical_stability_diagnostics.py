"""Fast CPU-only tests for optional numerical diagnostics."""
import json

import pytest
import torch
from torch import nn

from src.diagnostics.numerical_stability import (NonFiniteDetected, NumericalTracer, batch_identity,
                                                 clone_model_state, clone_optimizer_state,
                                                 gradient_statistics, optimizer_tensor_statistics,
                                                 tensor_hash, tensor_summary)
from src.train import train_one_epoch


def test_tensor_summary_finite_and_nonfinite_counts():
    finite = tensor_summary(torch.tensor([-2., 1., 3.]))
    assert finite["finite"] and finite["minimum"] == -2 and finite["maximum"] == 3
    summary = tensor_summary(torch.tensor([float("nan"), float("inf"), -float("inf"), 4.]))
    assert not summary["finite"]
    assert (summary["nan_count"], summary["positive_inf_count"], summary["negative_inf_count"]) == (1, 1, 1)


def test_gradient_and_momentum_buffer_inspection():
    model = nn.Linear(2, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    model(torch.ones(3, 2)).sum().backward(); optimizer.step()
    global_stats, per_parameter = gradient_statistics(model)
    assert global_stats["global_gradient_l2_norm"] > 0 and set(per_parameter) == {"weight", "bias"}
    momentum = optimizer_tensor_statistics(optimizer, model)
    assert len(momentum) == 2 and all(row["state_name"] == "momentum_buffer" for row in momentum)
    assert {row["parameter_name"] for row in momentum} == {"weight", "bias"}


def test_batch_hashes_are_deterministic_and_label_sensitive():
    x = torch.arange(8.).reshape(2, 4); y = torch.tensor([1, 2])
    assert tensor_hash(x) == tensor_hash(x.clone())
    assert batch_identity(x, y)["label_sha256"] != batch_identity(x, y.flip(0))["label_sha256"]


class InfiniteForward(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(1.))
    def forward(self, x):
        return x * self.weight * float("inf")


def test_nonfinite_forward_classification_and_context_serialization(tmp_path):
    model = InfiniteForward(); optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    tracer = NumericalTracer(tmp_path / "trace", 1, True); x = torch.ones(2); y = torch.zeros(2, dtype=torch.long)
    context = {"epoch": 1, "phase": 1, "phase_data": "B", "batch_index": 1, "learning_rate": .1}
    tracer.begin_batch(context, x, y)
    with pytest.raises(NonFiniteDetected):
        tracer.check_forward(context, model(x), model=model, optimizer=optimizer, x=x, y=y)
    failure = json.loads((tmp_path / "trace/numerical_failure.json").read_text())
    assert failure["exact_stage"] == "after_forward" and failure["positive_inf_count"] == 2
    context_payload = torch.load(tmp_path / "trace/failure_context.pt", weights_only=False)
    assert context_payload["phase_data"] == "B" and context_payload["failure_stage"] == "after_forward"


class ExplodingSGD(torch.optim.SGD):
    def step(self, closure=None):
        result = super().step(closure)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("inf"))
        return result


def test_finite_backward_then_nonfinite_optimizer_step(tmp_path):
    model = nn.Linear(1, 1); optimizer = ExplodingSGD(model.parameters(), lr=.1, momentum=.9)
    x = torch.ones(2, 1); y = torch.ones(2, 1); loss = (model(x) - y).square().mean(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    tracer = NumericalTracer(tmp_path / "trace", 1, True); context = {"epoch": 1, "phase": 1,
        "phase_data": "A", "batch_index": 1, "learning_rate": .1}; tracer.begin_batch(context, x, y.flatten().long())
    before_model, before_optimizer = clone_model_state(model), clone_optimizer_state(optimizer); optimizer.step()
    with pytest.raises(NonFiniteDetected):
        tracer.check_named(context, "after_optimizer_step", dict(model.named_parameters()), model=model,
                           optimizer=optimizer, x=x, y=y, pre_model_state=before_model,
                           pre_optimizer_state=before_optimizer)
    assert tracer.failure["parameters_finite_before_optimizer_step"]
    assert tracer.failure["parameters_became_nonfinite_after_optimizer_step"]


def test_hooks_do_not_change_outputs_or_gradients(tmp_path):
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    # Give the generic model aliases expected by the diagnostic hook installer.
    aliased = nn.Module(); aliased.conv1=model[0]; aliased.relu1=model[1]; aliased.conv2=model[2]
    aliased.relu2=nn.Identity(); aliased.fc1=nn.Identity(); aliased.relu3=nn.Identity(); aliased.fc2=nn.Identity()
    def forward(x): return aliased.fc2(aliased.relu3(aliased.fc1(aliased.relu2(aliased.conv2(aliased.relu1(aliased.conv1(x)))))))
    aliased.forward = forward
    x = torch.randn(4, 2); baseline = aliased(x); baseline.sum().backward()
    gradients = {name: p.grad.clone() for name, p in aliased.named_parameters()}; aliased.zero_grad()
    tracer = NumericalTracer(tmp_path / "trace", 1, False); tracer.install_activation_hooks(aliased)
    traced = aliased(x); traced.sum().backward()
    assert torch.equal(baseline, traced)
    assert all(torch.equal(gradients[name], p.grad) for name, p in aliased.named_parameters())
    tracer.close_hooks()


def test_diagnostic_disabled_training_matches_existing_behavior():
    torch.manual_seed(4); a = nn.Linear(2, 2); b = nn.Linear(2, 2); b.load_state_dict(a.state_dict())
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 0, 1])), batch_size=3)
    oa, ob = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()
    train_one_epoch(a, data, "cpu", oa, criterion)
    # The explicit None exercises the optional path and must be identical.
    train_one_epoch(b, data, "cpu", ob, criterion, numerical_tracer=None)
    assert all(torch.equal(a.state_dict()[key], b.state_dict()[key]) for key in a.state_dict())
