import pytest
from xaiunits.datagenerator import (
    WeightedFeaturesDataset,
    ConflictingDataset,
    InteractingFeatureDataset,
    PertinentNegativesDataset,
    UncertaintyAwareDataset,
    ShatteredGradientsDataset,
    BooleanDataset,
)
import torch
from sympy import symbols


@pytest.fixture
def create_datasets():
    outputs = []
    samples = 100
    outputs.append(WeightedFeaturesDataset(n_samples=samples))
    outputs.append(ConflictingDataset(n_samples=samples))
    outputs.append(InteractingFeatureDataset(n_samples=samples))
    outputs.append(PertinentNegativesDataset(n_samples=samples))
    outputs.append(ShatteredGradientsDataset(n_samples=samples))

    x, y, z, w = symbols("x y z w")
    k = x | (y & z & x) | ~w
    outputs.append(BooleanDataset(k, n_samples=samples, atoms=(x, y, z, w)))

    return outputs


def test_model(create_datasets):
    """Test that the model output and dataset results are consistent"""
    for dataset in create_datasets:
        model = dataset.generate_model()
        input_output = dataset[:]
        model_output = model(input_output[0]).squeeze()

        try:
            assert torch.allclose(input_output[1].squeeze(), model_output, rtol=1e-5)
        except AssertionError:
            print(f"Data and Model has conflict for {dataset.__class__.__name__}")
            raise AssertionError


def test_uncertainty_model():
    """Test that the model output and dataset results are consistent for the
    uncertainty model"""
    data = UncertaintyAwareDataset(n_features=10, n_samples=100)
    label = data[:][1]
    model = data.generate_model()
    input_output = data[:]
    model_output = model(input_output[0]).squeeze()
    arg = torch.max(model_output, dim=1)[1]
    try:
        assert torch.allclose(arg, label, rtol=1e-5)
    except AssertionError:
        print(f"Data and Model has conflict for {data.__class__.__name__}")
        raise AssertionError


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
