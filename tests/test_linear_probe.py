"""Fast offline tests for the fresh paired linear-probe analysis."""
import json
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.aggregate_linear_probe import (bootstrap_ci, manifest_pairs,
                                                  paired_statistics, signed_differences)
from src.analysis.linear_probe import (deterministic_epoch_order, extract_features,
                                       freeze_backbone, indices_hash, make_identical_heads,
                                       parse_checkpoint_selector, state_dict_exactly_equal,
                                       train_linear_probe, validate_probe_arguments)


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(1, 3, 1); self.fc = nn.Linear(3, 4)

    def forward(self, x, return_activations=False):
        conv2 = torch.relu(self.conv(x)); fc1 = torch.relu(self.fc(conv2.mean((2, 3))))
        return (fc1, {"conv2": conv2, "fc1": fc1}) if return_activations else fc1


def loader():
    return DataLoader(TensorDataset(torch.arange(32.).reshape(8, 1, 2, 2), torch.arange(8) % 4), batch_size=3)


def test_freeze_eval_and_backbone_unchanged_while_only_head_updates():
    model = SyntheticBackbone(); model.train(); before = freeze_backbone(model)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    features, labels = extract_features(model, loader(), "cpu", "fc1")
    head, _, _ = make_identical_heads(4, 7); head_before = {k: v.clone() for k, v in head.state_dict().items()}
    tests = {domain: (features, labels) for domain in ("full", "A", "B")}
    train_linear_probe(head, (features, labels), tests, 1, .01, 0., 0., 4, 9)
    assert state_dict_exactly_equal(before, model)
    assert any(not torch.equal(head_before[k], head.state_dict()[k]) for k in head_before)


def test_identical_head_initialization_and_deterministic_batch_order():
    a, b, equal = make_identical_heads(5, 777)
    assert equal and all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert torch.equal(deterministic_epoch_order(25, 4, 2), deterministic_epoch_order(25, 4, 2))
    assert not torch.equal(deterministic_epoch_order(25, 4, 2), deterministic_epoch_order(25, 4, 3))


def test_indices_hash_is_deterministic_and_order_sensitive():
    assert indices_hash([1, 2, 3]) == indices_hash([1, 2, 3])
    assert indices_hash([1, 2, 3]) != indices_hash([3, 2, 1])


def test_fc1_and_conv2_global_average_pool_shapes_and_values():
    model = SyntheticBackbone(); freeze_backbone(model)
    fc1, labels1 = extract_features(model, loader(), "cpu", "fc1")
    conv2, labels2 = extract_features(model, loader(), "cpu", "conv2")
    x, _ = next(iter(loader())); _, activations = model(x, return_activations=True)
    assert fc1.shape == (8, 4) and conv2.shape == (8, 3)
    assert torch.allclose(conv2[:len(x)], activations["conv2"].mean(dim=(2, 3)))
    assert torch.equal(labels1, labels2)


def test_checkpoint_selection_modes_and_tolerance():
    steps = (0, 5, 10)
    assert parse_checkpoint_selector("final", steps) == 10
    assert parse_checkpoint_selector("relaxation-step:5", steps) == 5
    assert parse_checkpoint_selector("performance-matched", steps, {0: .98, 5: .9, 10: .9},
                                     {0: .9779999999995, 5: .9, 10: .9}, .002, .97, 1e-12) == 0
    with pytest.raises(ValueError, match="No performance-matched"):
        parse_checkpoint_selector("performance-matched", steps, {s: .8 for s in steps},
                                  {s: .8 for s in steps})
    with pytest.raises(ValueError):
        parse_checkpoint_selector("relaxation-step:3", steps)


def test_invalid_layer_and_probe_arguments_are_rejected():
    model = SyntheticBackbone()
    with pytest.raises(ValueError):
        extract_features(model, loader(), "cpu", "bad")
    args = Namespace(train_samples_per_class=1, test_samples_per_class=1, probe_epochs=0,
                     probe_lr=.01, probe_batch_size=2, probe_weight_decay=0., probe_momentum=0.,
                     max_accuracy_gap=.002, minimum_accuracy=.97, numerical_tolerance=1e-12)
    with pytest.raises(ValueError):
        validate_probe_arguments(args)
    args.probe_epochs = 1; args.probe_weight_decay = -1
    with pytest.raises(ValueError):
        validate_probe_arguments(args)


def test_signed_differences_preserve_direction():
    base = {"seed": 3, "probe_epoch": 1, "test_accuracy_full": .8,
            "test_accuracy_A": .7, "test_accuracy_B": .6}
    rows = [{**base, "scenario": "SABC"}, {**base, "scenario": "SBAC",
             "test_accuracy_full": .75, "test_accuracy_A": .8}]
    result = {r["domain"]: r for r in signed_differences(rows)}
    assert result["full"]["signed_accuracy_difference_sabc_minus_sbac"] == pytest.approx(.05)
    assert result["A"]["signed_accuracy_difference_sabc_minus_sbac"] == pytest.approx(-.1)
    assert result["A"]["absolute_accuracy_difference"] == pytest.approx(.1)


def test_paired_statistics_known_values_and_deterministic_bootstrap():
    values = np.array([-1., 1., 2., 2.])
    result = paired_statistics(values, 500, 12)
    assert result["n"] == 4 and result["positive_count"] == 3 and result["negative_count"] == 1
    assert result["mean"] == 1 and result["median"] == 1.5
    assert bootstrap_ci(values, 500, 12) == bootstrap_ci(values, 500, 12)


def test_manifest_reports_incomplete_pairs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": [
        {"scenario": "SABC", "seed": 1, "path": "one-a"},
        {"scenario": "SBAC", "seed": 1, "path": "one-b"},
        {"scenario": "SABC", "seed": 2, "path": "two-a"}]}), encoding="utf-8")
    pairs = manifest_pairs(manifest)
    assert set(pairs[1]) == {"SABC", "SBAC"}
    assert set(pairs[2]) == {"SABC"}
