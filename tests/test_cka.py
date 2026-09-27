import pytest, torch
from src.analysis.cka import linear_cka
def test_identical_scaled():
    x=torch.randn(30,5); assert linear_cka(x,x)==pytest.approx(1,abs=1e-10) and linear_cka(x,3*x)==pytest.approx(1,abs=1e-10)
def test_unrelated():
    g=torch.Generator().manual_seed(1); assert linear_cka(torch.randn(200,5,generator=g),torch.randn(200,5,generator=g))<.5
def test_invalid_degenerate():
    with pytest.raises(ValueError): linear_cka(torch.ones(2,3),torch.ones(3,3))
    assert linear_cka(torch.ones(5,2),torch.ones(5,3))==0
