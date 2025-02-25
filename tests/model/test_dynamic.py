import pytest
import torch
from xaiunits.model.dynamic import DynamicNN


@pytest.fixture
def random_model():
    data, weights = torch.rand((5, 10)), torch.rand(10)
    N = data.shape[1]
    config = [
        {"type": "Linear", "in_features": N, "out_features": 2 * N, "bias": False},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 2 * N, "out_features": 1, "bias": False},
    ]
    model = DynamicNN(config)
    return data, weights, model


class TestDynamicNN:
    def test_subclass(self):
        """Test that the layer is a subclass of torch.nn.Module"""
        assert issubclass(DynamicNN, torch.nn.Module)

    def test_forward_shape(self, random_model):
        data, _, model = random_model
        assert model.forward(data).shape[0] == 5
        assert model.forward(data).shape[1] == 1

    def test_forward_type(self, random_model):
        data, _, model = random_model
        assert isinstance(model.forward(data), torch.Tensor)

    def test_act1_pos(self, random_model):
        data, _, model = random_model
        x = model[0](data)
        assert torch.all(torch.ge(model[1](x), 0))

    def test_l1_weights_shape(self, random_model):
        data, _, model = random_model
        M = data.shape[1]
        assert model[0].weight.shape[0] == 2 * M
        assert model[0].weight.shape[1] == M

    def test_l2_weights_shape(self, random_model):
        data, _, model = random_model
        M = data.shape[1]
        assert model[2].weight.shape[0] == 1
        assert model[2].weight.shape[1] == 2 * M

    def test_ReLU_only_forward_values(self):
        config = [{"type": "relu"}]
        model = DynamicNN(config)
        x_pos = torch.rand(10)
        x_neg = -x_pos
        assert model[0].__class__.__name__ == "ReLU"
        assert torch.allclose(model.forward(x_pos), x_pos)
        assert torch.allclose(model.forward(x_neg), torch.zeros(10))


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
