import pytest
import torch
from xaiunits.model.conflicting import ConflictingFeaturesNN


@pytest.fixture
def random_model_conflicting():
    weights = torch.rand(10)
    model = ConflictingFeaturesNN(10, weights)
    return model, weights


def test_l0_weights(random_model_conflicting):
    """Test the weight values for the first layer"""
    model, weights = random_model_conflicting
    w0 = torch.zeros((20, 20))
    d = 10
    w0[:d, :d] = torch.diag_embed(weights)
    w0[:d, d:] = -100 * torch.eye(d)
    w0[d:, :d] = -torch.diag_embed(weights)
    w0[d:, d:] = -100 * torch.eye(d)
    assert torch.equal(model[0].weight.data, w0)


def test_l1_weights(random_model_conflicting):
    """Test the weight values for the second layer"""
    model, _ = random_model_conflicting
    d = 10
    w1 = torch.zeros((1, d * 2))
    w1[0, :d] = 1.0
    w1[0, d : 2 * d] = -1.0
    assert torch.equal(model[2].weight.data, w1)


def test_forward_result_values():
    """Test the final output of the model"""
    weights = torch.tensor([0.5, 2.0, 1.5])
    input = torch.tensor([[1.0, 2.0, -3.0, 0.0, 1.0, 0.0]])
    model = ConflictingFeaturesNN(3, weights)
    expected_result = torch.tensor([-4.0])

    # Assert if the actual output is close to the expected output
    assert torch.allclose(model(input), expected_result, atol=1e-6)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
