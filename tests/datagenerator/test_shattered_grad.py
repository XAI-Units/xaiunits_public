import pytest
import torch
from xaiunits.datagenerator import ShatteredGradientsDataset, WeightedFeaturesDataset


def test_shattered_gradients_dataset_inheritance():
    """Test inheritance of ShatteredGradientsDataset."""
    assert issubclass(ShatteredGradientsDataset, WeightedFeaturesDataset)


def test_shattered_gradients_dataset_default_metric():
    """Test default_metric property in ShatteredGradientsDataset."""
    dataset = ShatteredGradientsDataset(n_samples=100)
    default_metric = dataset.default_metric
    assert default_metric is not None
    assert callable(default_metric)


def test_shattered_gradients_dataset_weighted_samples():
    """Test weighted_samples attribute in ShatteredGradientsDataset."""
    dataset = ShatteredGradientsDataset(n_samples=100)
    weighted_samples = dataset.weighted_samples
    assert isinstance(weighted_samples, torch.Tensor)
    assert weighted_samples.shape[0] == len(dataset)


def test_shattered_gradients_dataset_creation():
    """Test creation of ShatteredGradientsDataset."""
    dataset = ShatteredGradientsDataset(n_samples=100)
    assert len(dataset) == 100
    assert isinstance(dataset[0], tuple)
    assert isinstance(dataset.weights, torch.Tensor)


def test_shattered_gradients_dataset_weights():
    """Test weights generation in ShatteredGradientsDataset."""
    dataset = ShatteredGradientsDataset(n_samples=100)
    assert dataset.weights.sum().item() != 0  # Ensure weights are non-zero


def test_shattered_gradients_dataset_getitem():
    """Test __getitem__ method in ShatteredGradientsDataset."""
    dataset = ShatteredGradientsDataset(n_samples=100)
    sample, label = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_shattered_gradients_dataset_generate_model():
    """Test generate_model method in ShatteredGradientsDataset."""
    dataset = ShatteredGradientsDataset(n_samples=100)
    model = dataset.generate_model()
    assert model is not None
    assert isinstance(model, torch.nn.Module)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
