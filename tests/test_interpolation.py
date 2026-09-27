import pytest
from src.analysis.interpolation import barrier_summary, validate_num_points
def test_barrier():
    r=barrier_summary([0,.5,1],[1,3,2]); assert r["barrier_height"]==1 and r["alpha_at_maximum"]==.5
def test_invalid_num_points():
    with pytest.raises(ValueError,match="at least 2"): validate_num_points(1)
