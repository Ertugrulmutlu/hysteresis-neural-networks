import pytest
import torch
from torch.utils.data import TensorDataset

from src.analysis.common_relaxation import (mean_js_divergence, normalized_weight_distance,
    prediction_disagreement, representation_history_score, select_performance_matched_endpoint)
from src.data import make_balanced_restartable_loader
from src.train import scenario_phase_order, validate_relaxation_schedule


def test_common_relaxation_phase_orders_and_invalid_scenario():
    assert scenario_phase_order("SABC") == ("A", "B", "C")
    assert scenario_phase_order("SBAC") == ("B", "A", "C")
    with pytest.raises(ValueError, match="Unsupported scenario"):
        scenario_phase_order("SCAB")


@pytest.mark.parametrize("config", [
    {"relaxation_steps": -1, "relaxation_checkpoints": [0]},
    {"relaxation_steps": 10, "relaxation_checkpoints": [0, 10, 5]},
    {"relaxation_steps": 10, "relaxation_checkpoints": [0, 5, 5, 10]},
    {"relaxation_steps": 10, "relaxation_checkpoints": [0, 5]},
])
def test_invalid_relaxation_schedules(config):
    with pytest.raises(ValueError):
        validate_relaxation_schedule(config)

def test_valid_relaxation_schedule():
    assert validate_relaxation_schedule({"relaxation_steps":10,"relaxation_checkpoints":[0,5,10],
                                         "relaxation_samples_per_class":2}) == [0,5,10]


def test_deterministic_balanced_restartable_loader():
    labels = torch.arange(10).repeat_interleave(2); dataset = TensorDataset(labels.float().unsqueeze(1), labels)
    config = {"data": {"batch_size": 4, "num_workers": 0}}
    iterator_a = make_balanced_restartable_loader(dataset, config, seed=7, samples_per_class=2)
    iterator_b = make_balanced_restartable_loader(dataset, config, seed=7, samples_per_class=2)
    sequence_a = [next(iterator_a)[1].tolist() for _ in range(10)]
    sequence_b = [next(iterator_b)[1].tolist() for _ in range(10)]
    assert sequence_a == sequence_b
    assert sequence_a[:5] == sequence_a[5:]


def test_functional_and_weight_metrics():
    p = torch.tensor([[.9, .1], [.2, .8]])
    q = torch.tensor([[.8, .2], [.7, .3]])
    assert prediction_disagreement(p, q) == pytest.approx(.5)
    assert mean_js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert mean_js_divergence(p, q) > 0
    initial = {"x": torch.tensor([0.])}; a = {"x": torch.tensor([1.])}; b = {"x": torch.tensor([-1.])}
    assert normalized_weight_distance(a, b, initial) == pytest.approx(2.0)
    assert normalized_weight_distance(initial, initial, initial) == 0.0
    assert representation_history_score({"conv2": .8, "fc1": .6}) == pytest.approx(.3)


def test_performance_matched_endpoint_selection_and_null():
    rows = [{"relaxation_step": 0, "full_accuracy_sabc": .96, "full_accuracy_sbac": .96},
            {"relaxation_step": 100, "full_accuracy_sabc": .971, "full_accuracy_sbac": .970},
            {"relaxation_step": 500, "full_accuracy_sabc": .98, "full_accuracy_sbac": .98}]
    assert select_performance_matched_endpoint(rows, .002, .97)["relaxation_step"] == 100
    assert select_performance_matched_endpoint(rows, .0001, .99) is None
