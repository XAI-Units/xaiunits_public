import pytest
import torch
from xaiunits.model.continuous import ContinuousFeaturesNN


@pytest.fixture
def random_model():
    data, weights = torch.rand((5, 10)), torch.rand(10)
    model = ContinuousFeaturesNN(data.shape[1], weights)
    return data, weights, model


@pytest.fixture
def random_scalar_model():
    data, weights = torch.rand(10), torch.rand(10)
    model = ContinuousFeaturesNN(len(data), weights)
    return data, weights, model


class TestContinuousFeaturesNN:
    def test_subclass(self):
        """Test that the layer is a subclass of torch.nn.Module"""
        assert issubclass(ContinuousFeaturesNN, torch.nn.Module)

    def test_forward_shape(self, random_model):
        """Test the shape after a forward pass"""
        data, _, model = random_model
        assert model.forward(data).shape[0] == 5
        assert model.forward(data).shape[1] == 1

    def test_foward_scalar_shape(self, random_scalar_model):
        """Test the shape after a forward pass with a scalar"""
        data, _, model = random_scalar_model
        assert model.forward(data).shape[0] == 1

    def test_forward_value(self, random_model):
        """Test the value after one random pass"""
        data, weights, model = random_model
        expected_value = torch.unsqueeze(data @ weights, 1)
        actual_value = model.forward(data)
        assert torch.allclose(actual_value, expected_value)

    def test_forward_type(self, random_model):
        """Test the feature type after one forward pass"""
        data, _, model = random_model
        assert isinstance(model.forward(data), torch.Tensor)

    def test_l0_value(self, random_model):
        """Test the value after the first layer"""
        data, weights, model = random_model
        M = weights.shape[0]
        expected_value = data @ torch.diag(weights)
        actual_value = model[0](data)
        assert torch.allclose(actual_value[:, :M], expected_value)
        assert torch.allclose(actual_value[:, M:], -expected_value)

    def test_act1_pos(self, random_model):
        """Test the value after the relu"""
        data, _, model = random_model
        x = model[0](data)
        assert torch.all(torch.ge(model[1](x), 0))

    def test_l0_weights_shape(self, random_model):
        """Test the shape of the weights for the first layer"""
        data, _, model = random_model
        M = data.shape[1]
        assert model[0].weight.shape[0] == 2 * M
        assert model[0].weight.shape[1] == M

    def test_l0_weights_value(self, random_model):
        """Test the value of the weights for the first layer"""
        _, weights, model = random_model
        M = weights.shape[0]
        expected_matrix = torch.diag(weights)
        actual_matrix = model[0].weight
        assert torch.allclose(actual_matrix[:M], expected_matrix)
        assert torch.allclose(actual_matrix[M:], -expected_matrix)

    def test_l2_weights_shape(self, random_model):
        """Test the shape of the weights for the second layer"""
        data, _, model = random_model
        M = data.shape[1]
        assert model[2].weight.shape[0] == 1
        assert model[2].weight.shape[1] == 2 * M

    def test_l2_weights_value(self, random_model):
        """Test the value of the weights for the second layer"""
        _, weights, model = random_model
        M = weights.shape[0]
        expected_matrix = torch.ones(M)
        actual_matrix = model[2].weight
        assert torch.allclose(actual_matrix[:, :M], expected_matrix)
        assert torch.allclose(actual_matrix[:, M:], -expected_matrix)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
