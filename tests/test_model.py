import pytest
import torch
from src.model import LayerNorm2d, SimpleCNN

@pytest.mark.parametrize("norm", ["none", "group", "layer"])
def test_forward_and_activations(norm):
    model=SimpleCNN(norm=norm); logits, acts=model(torch.randn(2,1,28,28),True)
    assert logits.shape==(2,10) and set(acts)=={"conv1","conv2","fc1","logits"}
    assert acts["conv1"].shape==(2,32,28,28) and acts["conv2"].shape==(2,64,14,14) and acts["fc1"].shape==(2,128)
def test_normal_forward(): assert SimpleCNN()(torch.randn(3,1,28,28)).shape==(3,10)
@pytest.mark.parametrize("size", [7,19])
def test_layer_norm_sizes(size): assert LayerNorm2d(4)(torch.randn(2,4,size,size)).shape==(2,4,size,size)
def test_bad_norm():
    with pytest.raises(ValueError,match="Unsupported normalization"): SimpleCNN(norm="batch")

def test_legacy_state_dict_keys():
    keys = set(SimpleCNN().state_dict())
    legacy = {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
              "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert legacy.issubset(keys)
    assert not any(key.startswith(("conv1_layer", "conv2_layer", "fc1_layer")) for key in keys)

def test_leaky_relu_and_slope_validation():
    model=SimpleCNN(activation="leaky_relu",leaky_relu_negative_slope=.02); logits,acts=model(torch.randn(2,1,28,28),True)
    assert logits.shape==(2,10) and set(acts)=={"conv1","conv2","fc1","logits"}
    assert set(model.state_dict())==set(SimpleCNN().state_dict())
    with pytest.raises(ValueError,match="non-negative"): SimpleCNN(activation="leaky_relu",leaky_relu_negative_slope=-.1)
