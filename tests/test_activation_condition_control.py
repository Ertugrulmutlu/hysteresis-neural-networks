"""Fast offline tests for the LeakyReLU mechanism control."""
import json
from pathlib import Path

import pytest
import torch

from src.analysis.activation_condition_validation import (validate_bit_identical_initializations,
                                                          validate_condition_configs)
from src.analysis.compare_activation_conditions import (align_condition_rows, bootstrap_ci,
                                                        build_comparison, paired_statistics,
                                                        tost_paired)
from src.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
RELU = ROOT / "configs/paper/mnist_class_split_reset_relu_long50k_sabc.yaml"
LEAKY = ROOT / "configs/paper/mnist_class_split_reset_leaky_relu_long50k_sabc.yaml"
MANIFEST = ROOT / "results/manifests/class_split_reset_leaky_relu_long50k_5seeds.json"


def test_leaky_long_config_matches_relu_protocol_but_not_output_identity():
    relu, leaky = load_experiment_config(RELU), load_experiment_config(LEAKY)
    assert leaky["train"]["relaxation_steps"] == 50000
    assert leaky["train"]["relaxation_checkpoints"] == relu["train"]["relaxation_checkpoints"]
    assert leaky["logging"]["run_name"] != relu["logging"]["run_name"]
    assert "leaky_relu_long50k" in leaky["logging"]["run_name"]
    validate_condition_configs(relu, leaky)


def test_manifest_has_complete_five_seed_pairs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 10
    assert manifest["seeds"] == [101, 202, 303, 404, 505]
    for seed in manifest["seeds"]:
        assert {e["scenario"] for e in manifest["entries"] if e["seed"] == seed} == {"SABC", "SBAC"}
        assert all("leaky_relu_long50k" in e["expected_run_directory"]
                   for e in manifest["entries"] if e["seed"] == seed)


def test_config_validation_rejects_optimizer_and_data_changes():
    relu, leaky = load_experiment_config(RELU), load_experiment_config(LEAKY)
    leaky["train"]["lr"] = relu["train"]["lr"] * 2
    with pytest.raises(ValueError, match="Unexpected"):
        validate_condition_configs(relu, leaky)
    leaky = load_experiment_config(LEAKY); leaky["data"]["batch_size"] += 1
    with pytest.raises(ValueError, match="Unexpected"):
        validate_condition_configs(relu, leaky)


def test_bit_identical_initialization_validation():
    a = {"weight": torch.tensor([1., 2.]), "bias": torch.tensor([0.])}
    validate_bit_identical_initializations(a, {key: value.clone() for key, value in a.items()})
    with pytest.raises(ValueError, match="initialization tensors differ"):
        validate_bit_identical_initializations(a, {"weight": torch.tensor([1., 3.]), "bias": torch.tensor([0.])})


def metric_row(seed, step, h, offset=0.):
    return {"seed": str(seed), "relaxation_step": str(step), "representation_history_score": str(h),
            "cka_conv2": str(.8 + offset), "cka_fc1": str(.7 + offset),
            "prediction_disagreement": str(.1 + offset), "js_divergence": str(.02 + offset),
            "absolute_accuracy_gap": str(.01 + offset),
            "signed_full_accuracy_difference_sabc_minus_sbac": str(-.01 + offset)}


def test_alignment_delta_sign_and_known_paired_statistics():
    relu = [metric_row(seed, step, .2 + seed / 100) for seed in (1, 2) for step in (10000, 50000)]
    leaky = [metric_row(seed, step, .1 + seed / 100) for seed in (1, 2) for step in (10000, 50000)]
    rows, statistics, summary = build_comparison(relu, leaky, [10000, 50000], 50000, 200, 7)
    assert all(row["delta_h_repr_leaky_minus_relu"] == pytest.approx(-.1) for row in rows)
    assert summary["delta_definition"] == "LeakyReLU minus ReLU"
    known = paired_statistics([-1., 0., 1., 2.], 200, 3)
    assert known["mean"] == pytest.approx(.5)
    assert (known["positive_count"], known["negative_count"], known["zero_count"]) == (2, 1, 1)
    assert any(row["analysis_role"] == "primary" for row in statistics)


def test_duplicate_and_incomplete_condition_rows_are_rejected():
    row = metric_row(1, 50000, .2)
    with pytest.raises(ValueError, match="Duplicate"):
        align_condition_rows([row, dict(row)], [metric_row(1, 50000, .1)], [50000])
    with pytest.raises(ValueError, match="Incomplete"):
        align_condition_rows([metric_row(1, 10000, .2), metric_row(1, 50000, .2)],
                             [metric_row(1, 10000, .1)], [10000, 50000])
    with pytest.raises(ValueError, match="seed sets differ"):
        align_condition_rows([metric_row(1, 50000, .2)], [metric_row(2, 50000, .1)], [50000])


def test_bootstrap_and_tost_margin_logic():
    assert bootstrap_ci([-.02, -.01, 0.], 300, 8) == bootstrap_ci([-.02, -.01, 0.], 300, 8)
    assert tost_paired([0., 0., 0., 0.])["tost_equivalent"]
    assert not tost_paired([.02, .02, .02, .02])["tost_equivalent"]
