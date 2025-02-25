import pytest
import torch
from xaiunits.model.shattered_gradients import ShatteredGradientsNN


@pytest.fixture
def weights():
    # Example weights for testing
    return torch.randn(10, 5)


# class TestShatteredGradientsNN:
def test_invalid_activation_function():
    # Test that an error is raised for an invalid activation function
    with pytest.raises(KeyError):
        ShatteredGradientsNN(weights=torch.randn(5, 5), act_fun="Invalid")


def test_default_activation_function():
    # Test default activation function mapping
    model = ShatteredGradientsNN(weights=torch.randn(5, 5))
    assert isinstance(model[1], torch.nn.ReLU)  # Second layer should be ReLU by default


def test_model_forward(weights):
    # Test forward pass of the model
    model = ShatteredGradientsNN(weights=weights)
    input_data = torch.randn(1, 5)  # Input data of shape (batch_size, n_features)
    output = model(input_data)
    assert output.shape == (1, 10)  # Check output shape


def test_different_activation_functions():
    activation_functions = "Relu"  # Use a single activation function name
    model = ShatteredGradientsNN(
        weights=torch.randn(5, 5), act_fun=activation_functions
    )
    assert isinstance(model[1], torch.nn.ReLU)  # Check first activation function

    activation_functions = "Gelu"  # Use a single activation function name
    model = ShatteredGradientsNN(
        weights=torch.randn(5, 5), act_fun=activation_functions
    )
    assert isinstance(model[1], torch.nn.GELU)  # Check first activation function

    activation_functions = "Sigmoid"  # Use a single activation function name
    model = ShatteredGradientsNN(
        weights=torch.randn(5, 5), act_fun=activation_functions
    )
    assert isinstance(model[1], torch.nn.Sigmoid)  # Check first activation function


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
