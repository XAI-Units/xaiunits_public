import pytest
import torch
from xaiunits.model.pertinent_negative import PertinentNN


@pytest.fixture
def random_model_pertinent_negative():
    weights = torch.rand(10)
    data = torch.rand(10)
    c = torch.ones(10)
    model = PertinentNN(10, weights, c, 1)
    return model, data


def test_foward_scalar_shape(random_model_pertinent_negative):
    """Test the shape after a forward pass with a scalar"""
    model, data = random_model_pertinent_negative
    assert model.forward(data).shape[0] == 1


def test_forward_type(random_model_pertinent_negative):
    """Test the feature type after one forward pass"""
    model, data = random_model_pertinent_negative
    assert isinstance(model.forward(data), torch.Tensor)


def test_all_non_zero():
    """test when there no pertinent negative the model is equivalent to the continuous one"""
    w = torch.rand(10)
    c = torch.zeros(10)
    x = torch.ones(10)
    result = PertinentNN(10, w, c, 5)(x)
    expected = w @ x
    assert torch.allclose(expected, result)


def test_one_pertinent_negative():
    """test when there pertinent negatives are equal to one the model is equivalent to the continuous one"""
    w = torch.rand(10)
    c = torch.ones(10)
    x = torch.ones(10)
    result = PertinentNN(10, w, c, 5)(x)
    expected = w @ x
    assert torch.allclose(expected, result)


def test_pertinent_negative_weight():
    """test that when pertinent negatives are 0 their weights increases"""
    w = torch.Tensor([3, 4, 7, 8])
    c = torch.Tensor([1, 1, 0, 0])
    x = torch.Tensor([0, 1, 1, 1])
    result = PertinentNN(4, w, c, 5)(x)
    expected = torch.Tensor([34])
    assert torch.allclose(expected, result)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
