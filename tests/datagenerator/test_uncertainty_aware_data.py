import pytest
from xaiunits.datagenerator import UncertaintyAwareDataset
import torch


@pytest.fixture
def datasample():
    return UncertaintyAwareDataset()


@pytest.fixture
def custom_weights_dataset():
    n_features = 5
    weights = torch.rand(n_features, n_features)
    return UncertaintyAwareDataset(n_features=n_features, weights=weights)


class TestUncertaintyAwareData:
    def test_default_initialization(self, datasample):
        """Test the initialization shape"""
        assert datasample.samples.shape == (len(datasample), datasample.n_features)
        assert datasample.labels.shape == (len(datasample),)
        assert datasample.weights.shape == (
            datasample.n_features - datasample.common_features,
            datasample.n_features,
        )

    def test_custom_weights_initialization(self, custom_weights_dataset):
        """Test the weights shape initialization"""
        assert custom_weights_dataset.weights.shape == (5, 5)

    def test_weights_effect_on_labels(self, custom_weights_dataset):
        """Test the weight effect on labels"""
        weighted_samples = (
            custom_weights_dataset.samples @ custom_weights_dataset.weights.T
        )
        labels = torch.max(weighted_samples, dim=1)[1]
        assert torch.equal(custom_weights_dataset.weighted_samples, weighted_samples)
        assert torch.equal(custom_weights_dataset.labels, labels)

    def test_assertions_for_invalid_weights(self):
        """Test that an assertion error is raised with an incorrect shape"""
        n_features = 5
        invalid_weights = torch.rand(3, 3)
        with pytest.raises(AssertionError):
            UncertaintyAwareDataset(n_features=n_features, weights=invalid_weights)

    def test_assertions_for_invalid_common_features(self):
        """Test that an assertion error is raised when a non int common_feature is inputted"""
        n_features = 5
        invalid_common_features = "not_an_int"
        with pytest.raises(AssertionError):
            UncertaintyAwareDataset(
                n_features=n_features, common_features=invalid_common_features
            )

    def test_default_metric_implemented(self, custom_weights_dataset):
        """Tests whether default_metric method is implemented for the class."""
        assert hasattr(custom_weights_dataset, "default_metric")

    def test_default_metric_correctness(self, custom_weights_dataset):
        """Tests whether the default_metric property is correctly specified."""
        from xaiunits.metrics import wrap_metric

        custom_weights_dataset.default_metric == wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.sqrt(
                torch.sum(x[:, list(range(-1, -self.common_features - 1, -1))], dim=1)
            ),
        )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
