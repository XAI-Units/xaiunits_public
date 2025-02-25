import pytest
from xaiunits.datagenerator import InteractingFeatureDataset
import torch


@pytest.fixture
def datasample():
    data = InteractingFeatureDataset()
    return data


class TestFeatureInteractingData:
    def test_get_flat_weights_with_mixed_types(self, datasample):
        """Test the get flat weights method"""
        weights = [(1, 2), 3, (4, 5)]
        expected_result = torch.tensor([1, 2, 3, 4, 5])
        result = datasample._get_flat_weights(weights)
        assert torch.equal(result, expected_result)

    def test_interacting_features(self, datasample):
        """Test the interacting features initialization"""
        f = datasample.interacting_features
        assert f == [[1, 0], [3, 2]]

    def test_weight_range(self, datasample):
        """Test the higher bound of the weight range"""
        w = datasample.weights
        assert all(w) <= 1

    def test_weight_range_2(self, datasample):
        """Test the lower bound of the weight range"""
        w = datasample.weights
        assert -1 <= all(w)

    def test_dynamic_feature_interaction_effect(self, datasample):
        """Verify that the weights and samples are correctly modified based
        on the interactions specified. This involves checking if the values in
        weighted_samples reflect the expected interactions."""
        for impacts, impacted in datasample.interacting_features:
            impacts_value = datasample.samples[:, impacts]
            impacted_value = datasample.samples[:, impacted]
            for index in range(datasample.n_samples):
                if impacts_value[index] == 0:
                    assert (
                        datasample.weighted_samples[index, impacted]
                        == datasample.weights[impacted][0] * impacted_value[index]
                    )
                else:
                    assert (
                        datasample.weighted_samples[index, impacted]
                        == datasample.weights[impacted][1] * impacted_value[index]
                    )

    def test_zero_likelihood_handling(self, datasample):
        data = InteractingFeatureDataset(
            zero_likelihood=1.0
        )  # Setting likelihood of zero to 100%
        for impacts, impacted in data.interacting_features:
            assert torch.all(data.samples[:, impacts] == 0)

    def test_model_integration(self, datasample):
        model = datasample.generate_model()
        predictions = model(datasample.samples.float())
        assert predictions is not None, "Model should output predictions"
        assert (
            predictions.shape[0] == datasample.n_samples
        ), "Output predictions count should match the number of samples"


if __name__ == "__main__":

    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
