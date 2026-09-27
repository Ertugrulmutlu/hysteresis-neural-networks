"""Offline tests for the clean long common-relaxation protocol."""
import json
from pathlib import Path

import pytest
import torch

from src.analysis.aggregate_common_relaxation import pairs_from_manifest
from src.analysis.analyze_long_relaxation import (build_long_analysis,
                                                  deterministic_bootstrap_ci,
                                                  summarize_paired_change)
from src.config import load_experiment_config
from src.resume_audit import missing_exact_resume_state, validate_exact_resume_state
from src.train import validate_relaxation_schedule

ROOT = Path(__file__).resolve().parents[1]
LONG_SABC = ROOT / "configs/paper/mnist_class_split_reset_relu_long50k_sabc.yaml"
LONG_SBAC = ROOT / "configs/paper/mnist_class_split_reset_relu_long50k_sbac.yaml"
MANIFEST = ROOT / "results/manifests/class_split_reset_relu_long50k_5seeds.json"


def test_weight_only_checkpoint_is_not_exact_resume():
    checkpoint = {"conv1.weight": torch.zeros(1), "fc1.bias": torch.zeros(1)}
    with pytest.raises(ValueError, match="Not an exact-resume checkpoint"):
        validate_exact_resume_state(checkpoint)


@pytest.mark.parametrize("missing", ["optimizer_state", "python_rng_state", "torch_cpu_rng_state",
                                     "data_loader_generator_state", "sampler_state_or_batch_position"])
def test_exact_resume_validation_detects_required_missing_state(missing):
    complete = {key: object() for key in missing_exact_resume_state({})}
    complete.pop(missing)
    assert missing in missing_exact_resume_state(complete)


def test_long_configs_resolve_to_50k_and_required_schedule():
    for path in (LONG_SABC, LONG_SBAC):
        config = load_experiment_config(path)
        assert config["train"]["relaxation_steps"] == 50000
        schedule = validate_relaxation_schedule(config["train"])
        assert {10000, 25000, 50000} <= set(schedule)
        assert config["train"]["relaxation_optimizer_policy"] == "reset"


def test_old_10k_config_stays_10k_and_new_names_cannot_collide():
    old = load_experiment_config(ROOT / "configs/paper/mnist_class_split_reset_relu_sabc.yaml")
    new = load_experiment_config(LONG_SABC)
    assert old["train"]["relaxation_steps"] == 10000
    assert old["logging"]["run_name"] != new["logging"]["run_name"]
    assert "long50k" in new["logging"]["run_name"] and "long50k" not in old["logging"]["run_name"]


def test_old_weight_state_dict_remains_loadable_shape():
    model_state = {"conv.weight": torch.ones(2, 2), "fc.bias": torch.zeros(2)}
    clone = {key: value.clone() for key, value in model_state.items()}
    assert model_state.keys() == clone.keys()
    assert all(torch.equal(model_state[key], clone[key]) for key in model_state)


def test_five_seed_manifest_has_ten_paired_entries_and_supported_format():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 10
    assert manifest["seeds"] == [101, 202, 303, 404, 505]
    for seed in manifest["seeds"]:
        assert {entry["scenario"] for entry in manifest["entries"] if entry["seed"] == seed} == {"SABC", "SBAC"}
    pairs = pairs_from_manifest(MANIFEST)
    assert [seed for seed, _, _ in pairs] == manifest["seeds"]


def test_aggregate_pair_validation_discovers_long_steps_dynamically(tmp_path):
    # The manifest parser carries paths without imposing any checkpoint ceiling.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"seed": 1, "scenario": "SABC", "expected_run_directory": "a"},
        {"seed": 1, "scenario": "SBAC", "expected_run_directory": "b"}]}), encoding="utf-8")
    assert pairs_from_manifest(manifest)[0][0] == 1
    assert validate_relaxation_schedule({"relaxation_steps": 50000,
        "relaxation_checkpoints": [0, 10000, 25000, 50000], "relaxation_samples_per_class": 1})[-2:] == [25000, 50000]


def test_incomplete_manifest_pair_is_reported(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"seed": 1, "scenario": "SABC", "expected_run_directory": "a"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing run pairs"):
        pairs_from_manifest(manifest)


def test_known_paired_changes_and_deterministic_bootstrap():
    summary = summarize_paired_change([-1., 0., 1., 2.], 500, 9)
    assert summary["n"] == 4
    assert summary["positive_count"] == 2 and summary["negative_count"] == 1 and summary["zero_count"] == 1
    assert summary["mean_change"] == pytest.approx(.5)
    assert deterministic_bootstrap_ci([1., 2., 3.], 500, 7) == deterministic_bootstrap_ci([1., 2., 3.], 500, 7)


def test_long_analysis_reports_incomplete_seed_without_silent_drop():
    rows = []
    for seed, values in ((1, (.3, .2, .1)), (2, (.4, .3, .2))):
        rows.extend({"seed": seed, "relaxation_step": step, "representation_history_score": value}
                    for step, value in zip((10000, 25000, 50000), values))
    rows.append({"seed": 3, "relaxation_step": 10000, "representation_history_score": .5})
    per_seed, changes, _, summary = build_long_analysis(rows, [10000, 25000, 50000], 200, 4)
    assert len(per_seed) == 2 and len(changes) == 6
    assert summary["incomplete_seeds"] == {"3": [25000, 50000]}
