import numpy as np
import pytest
import torch
from xaiunits.datagenerator import ConflictingDataset


@pytest.fixture
def datasample():
    data = ConflictingDataset()
    return data


class TestConflictingData:
    def test_feature(self, datasample):
        """Test the initial cancellation features"""
        assert datasample.cancellation_features == [0, 1]

    def test_cancellation_features_not_list(self, datasample):
        """Test that a assertion error is raised when the cancellation feature is not list"""
        datasample.cancellation_features = "not a list"
        with pytest.raises(AssertionError) as exc_info:
            datasample._initialize_cancellation_features()
        assert "input must be a list" in str(exc_info.value)

    def test_cancellation_features_not_all_integers(self, datasample):
        """Test that when an non integers is inputted an assertion error is raised"""
        datasample.cancellation_features = [1, "two", 3]
        with pytest.raises(AssertionError) as exc_info:
            datasample._initialize_cancellation_features()
        assert "cancellation features should be integers" in str(exc_info.value)

    def test_cancellation_features_out_of_range(self, datasample):
        """Test that an assertion error is raised when an index is out of range"""
        datasample.cancellation_features = [0, 10]
        with pytest.raises(AssertionError) as exc_info:
            datasample._initialize_cancellation_features()
        assert "cancellation features must be within the number of features" in str(
            exc_info.value
        )

    def test_cancellation_features_empty(self, datasample):
        """Test that an assetion error is raised when no cancellation features are inputted"""
        datasample.cancellation_features = []
        with pytest.raises(AssertionError) as exc_info:
            datasample._initialize_cancellation_features()
        assert "cancellation features should be at least one" in str(exc_info.value)

    def test_cancellation_likelihood_not_float_or_out_of_range(self, datasample):
        """Test that an assertion error is raised when a likelihood above one is inputted"""
        datasample.cancellation_likelihood = 1.1
        with pytest.raises(AssertionError) as exc_info:
            datasample._initialize_cancellation_features()
        assert "must be between zero and one" in str(exc_info.value)

    def test_get_cancellations_values(self, datasample):
        """Test that the cancellation values are below 1"""
        cancellations = datasample._get_cancellations()
        assert all(cancellations.unique() <= 1)

    def test_cancellations_follow_likelihood(self, datasample):
        """Test that cancellations values follow the likelihood"""
        datasample.cancellation_likelihood = 1.0
        cancellations = datasample._get_cancellations()
        assert all(cancellations.unique() == 1)

        datasample.cancellation_likelihood = 0.0
        cancellations = datasample._get_cancellations()
        assert all(cancellations.unique() == 0)

    def test_cancellation_shape(self, datasample):
        """Test the shape of the cancellation features"""
        cancellation = datasample._get_cancellation_samples()
        assert cancellation.shape == torch.Size([10, 4])

    def test_attribution_shape(self, datasample):
        """Test the shape of the attributions features"""
        cancellation = datasample._get_cancellation_attributions()
        assert cancellation.shape == datasample.weighted_samples.shape

    def test_cancellation_distribution_statistical(self, datasample):
        """Test that the distribution of the cancellation matches the expected likelihood"""
        datasample.n_samples = 1000  # Increase sample size for statistical testing
        datasample.cancellation_likelihood = 0.5
        cancellations = datasample._get_cancellations()
        mean_cancellation = cancellations.float().mean().item()
        # Check if the mean cancellation is within a reasonable tolerance of the expected likelihood
        np.testing.assert_almost_equal(
            mean_cancellation,
            0.5,
            decimal=1,
            err_msg="Cancellations do not statistically match the likelihood",
        )

    def test_interaction_cancellations_weights(self, datasample):
        """Test that the all elements are cancelled when the likelihood is 1"""
        # Ensure weighted samples interact correctly with cancellations
        datasample.cancellation_likelihood = 1.0  # All features are cancelled
        datasample.cancellation_outcomes = datasample._get_cancellations()
        cancellations = datasample._get_cancellations()
        expected_attribution = (
            -datasample.weighted_samples
        )  # All contributions should be negated
        calculated_attribution = datasample._get_cancellation_attributions()
        assert torch.allclose(
            calculated_attribution, expected_attribution
        ), "Cancellation attributions do not match expected negated weights"

    def test_non_cancellation_feature_handling(self, datasample):
        """Test the cancelled features"""
        datasample.cancellation_features = [0]  # Only the first feature can be canceled
        datasample.cancellation_likelihood = 0.5
        cancellations = datasample._get_cancellations()
        # Ensure only the first feature has cancellations
        assert (
            cancellations[:, 1:].sum() == 0
        ), "Features that should not be canceled are being canceled"

    def test_cancellation_samples_integration(self, datasample):
        """Test that the cancellation data integrates in combined data"""
        datasample.cancellation_likelihood = 0.5
        combined_samples = datasample._get_cancellation_samples()
        # Verify that the combined samples contain both original and cancellation data
        original_samples_part = combined_samples[:, : datasample.n_features]
        cancellation_part = combined_samples[:, datasample.n_features :]
        assert torch.equal(
            original_samples_part, datasample.samples
        ), "Original samples not correctly integrated in combined data"
        assert torch.equal(
            cancellation_part, datasample.cancellation_outcomes
        ), "Cancellation data not correctly integrated in combined data"

    def test_model_end_to_end_with_cancellations(self, datasample):
        """Test that the model output"""
        model = datasample.generate_model()
        predictions = model(datasample.cancellation_samples.float())
        # Simple check to ensure predictions are being made
        assert (
            predictions is not None and predictions.numel() == datasample.n_samples
        ), "Model does not correctly handle cancellation augmented samples"


if __name__ == "__main__":

    import sys

    import pytest

    d = ConflictingDataset()
    d.cancellation_likelihood = 1.0  # All features are cancelled

    d.cancellation_outcomes = d._get_cancellations()
    expected_attribution = -d.weighted_samples  # All contributions should be negated
    calculated_attribution = d._get_cancellation_attributions()
    print(calculated_attribution)
    print(expected_attribution)
    print(d.cancellation_outcomes)

    # sys.exit(pytest.main([__file__]))
