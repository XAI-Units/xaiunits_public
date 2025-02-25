import pytest
import torch
from xaiunits.model.interaction_features import InteractingFeaturesNN


def test_interaction_one():
    """test when the interaction feature is one the weight should be the first one"""
    w = [(3, 5), 5, 6, 4]
    x = torch.Tensor([[9, 1, 0, 0]])
    result = InteractingFeaturesNN(4, w, [(1, 0)])(x)
    expected = torch.Tensor([[9 * 5 + 1 * 5]])
    assert torch.allclose(expected, result)


def test_interaction_zero():
    """test when the interaction feature is zero the weight should be the second one"""
    w = [(3, 5), 5, 6, 4]
    x = torch.Tensor([[9, 0, 0, 0]])
    result = InteractingFeaturesNN(4, w, [(1, 0)])(x)
    expected = torch.Tensor([[9 * 3]])
    assert torch.allclose(expected, result)


def test_foward_scalar_shape():
    """Test the shape after a forward pass with a scalar"""
    n_sample = 10
    w = [(3, 5), 5, 6, 4, 4]
    x = torch.rand((n_sample, len(w)))
    model = InteractingFeaturesNN(len(w), w, [(1, 0)])
    output = model(x)
    assert output.shape[0] == n_sample
    assert output.shape[1] == 1


def test_forward_type():
    """Test the feature type after one forward pass"""
    w = [(3, 7), 5, 6, 4]
    x = torch.Tensor([9, 0, 4, 2])
    model = InteractingFeaturesNN(4, w, [(1, 0)])
    assert isinstance(model(x), torch.Tensor)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__]))
