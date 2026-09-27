import pytest
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from src.data import FixedRotationDataset, balanced_indices, balanced_rotated_indices, validate_data_protocol

class Tiny(Dataset):
    targets=torch.tensor([0,0,1,1])
    def __len__(self): return 4
    def __getitem__(self,index): return Image.new("L",(3,3),color=index*30),int(self.targets[index])

class TinyTen(Dataset):
    targets=torch.arange(10).repeat_interleave(2)
    def __len__(self): return 20
    def __getitem__(self,index): return Image.new("L",(3,3)),int(self.targets[index])

def test_fixed_rotation_is_deterministic_and_labels_unchanged():
    dataset=FixedRotationDataset(Tiny(),90,lambda image:torch.tensor(list(image.getdata())))
    assert torch.equal(dataset[1][0],dataset[1][0]) and dataset[1][1]==Tiny()[1][1]
def test_protocol_and_balancing_validation():
    assert validate_data_protocol({})=="class_split"; assert balanced_indices(Tiny(),[0,1],2)==[0,1,2,3]
    with pytest.raises(ValueError): validate_data_protocol({"protocol":"unknown"})
    with pytest.raises(ValueError): validate_data_protocol({"protocol":"rotated_mnist","rotation_degrees_A":0})
def test_rotated_c_balances_domains_and_digits():
    combined=ConcatDataset([TinyTen(),TinyTen()]);indices=balanced_rotated_indices(combined,2)
    assert len(indices)==40 and sum(index<20 for index in indices)==20
