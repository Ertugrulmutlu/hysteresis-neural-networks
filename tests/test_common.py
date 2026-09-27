import pytest, torch
import yaml
from src.analysis.common import (extract_epoch, flatten_state_dict, interpolate_state_dicts,
                                 parameter_delta, sorted_checkpoints, validate_analysis_pair,
                                 validate_samples_per_class)
def test_flatten_delta():
    s={"z":torch.tensor([2.]),"a":torch.tensor([1.])}; assert torch.equal(flatten_state_dict(s),torch.tensor([1.,2.])); assert torch.equal(parameter_delta(s,{"z":torch.tensor([1.]),"a":torch.tensor([1.])}),torch.tensor([0.,1.]))
def test_sort(tmp_path):
    for n in [10,2,0]: (tmp_path/f"weights_epoch_{n:03d}.pt").touch()
    assert [extract_epoch(p) for p in sorted_checkpoints(tmp_path)]==[0,2,10]
def test_interpolation():
    a={"x":torch.tensor([0.,2.]),"n":torch.tensor([1])}; b={"x":torch.tensor([2.,4.]),"n":torch.tensor([1])}
    assert torch.equal(interpolate_state_dicts(a,b,0)["x"],a["x"]) and torch.equal(interpolate_state_dicts(a,b,1)["x"],b["x"]) and torch.equal(interpolate_state_dicts(a,b,.5)["x"],torch.tensor([1.,3.]))
    with pytest.raises(ValueError,match="Non-floating"): interpolate_state_dicts(a,{"x":b["x"],"n":torch.tensor([2])},.5)
def test_mismatch():
    with pytest.raises(ValueError,match="keys differ"): interpolate_state_dicts({"a":torch.ones(1)},{"b":torch.ones(1)},.5)
    with pytest.raises(ValueError,match="Shape mismatch"): interpolate_state_dicts({"a":torch.ones(1)},{"a":torch.ones(2)},.5)

def pair_config(scenario):
    return {"experiment":{"seed":1}, "data":{"dataset":"MNIST","batch_size":2,"normalize":True,
            "normalize_mean":[0.1],"normalize_std":[0.2],"augmentation":"none"},
            "split":{"A_digits":[0],"B_digits":[1]}, "model":{"arch":"simple_cnn","norm":"none",
            "group_norm_groups":8,"activation":"relu"}, "train":{"scenario":scenario,"optimizer":"sgd",
            "lr":.1,"momentum":.9,"weight_decay":0.,"epochs_total":1,"phase_epochs":1},
            "logging":{"run_name":scenario,"save_dir":"results"}}

def make_analysis_run(path, scenario, init=1., lr=.1):
    path.mkdir(); config=pair_config(scenario); config["train"]["lr"]=lr
    (path/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    torch.save({"x":torch.tensor([init])},path/"weights_epoch_000.pt")
    torch.save({"x":torch.tensor([2.])},path/"weights_epoch_001.pt")

def test_analysis_pair_rejects_initialization_and_config_difference(tmp_path):
    make_analysis_run(tmp_path/"sab","SAB"); make_analysis_run(tmp_path/"sba","SBA",init=2.)
    with pytest.raises(ValueError,match="Initialization tensor differs"): validate_analysis_pair(tmp_path/"sab",tmp_path/"sba")
    torch.save({"x":torch.tensor([1.])},tmp_path/"sba"/"weights_epoch_000.pt")
    config=pair_config("SBA"); config["train"]["lr"]=.2; (tmp_path/"sba"/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match="Disallowed config differences"): validate_analysis_pair(tmp_path/"sab",tmp_path/"sba")

def test_analysis_pair_rejects_two_sab_runs(tmp_path):
    make_analysis_run(tmp_path/"first", "SAB"); make_analysis_run(tmp_path/"second", "SAB")
    with pytest.raises(ValueError, match="ordered SAB then SBA"):
        validate_analysis_pair(tmp_path/"first", tmp_path/"second")

def test_probe_count_validation():
    with pytest.raises(ValueError,match="greater than 0"): validate_samples_per_class(0)
