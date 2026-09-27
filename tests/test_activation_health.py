import pytest
from src.analysis.activation_health import validate_activation_arguments

@pytest.mark.parametrize("values, message", [((0, 0.0, 0.1), "greater than 0"),
                                               ((1, -1.0, 0.1), "epsilon"),
                                               ((1, 0.0, 1.1), "dead_threshold")])
def test_invalid_activation_arguments(values, message):
    with pytest.raises(ValueError, match=message):
        validate_activation_arguments(*values)
