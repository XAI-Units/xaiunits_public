import pytest
import torch
from xaiunits.model.uncertainty_model import UncertaintyNN


@pytest.fixture
def test_weight_n_features():
    with pytest.raises(AssertionError):
        UncertaintyNN(2, torch.rand((1, 2)))


def test_expected_output_and_type():
    w = torch.Tensor([[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().T
    result = UncertaintyNN(4, weights=w)(torch.Tensor([[1, 0, 0, 0]]))
    expected = torch.ones(1, 4) * 0.25
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(expected, result)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
