import pytest
from xaiunits.datagenerator import PertinentNegativesDataset
import torch
from xaiunits.model.pertinent_negative import PertinentNN


@pytest.fixture
def datasample():
    data = PertinentNegativesDataset()
    return data


class TestPertinentNegativeData:
    def test_pn_features(self, datasample):
        """Test initial pertinent negative index"""
        assert datasample.pn_features == [0]

        assert datasample._intialize_pn_features([0, 2]) == [0, 2]

    def test_weight_shape(self, datasample):
        """Test the shape of the initial weights"""
        assert datasample.weighted_samples.shape == torch.Size([10, 5])

    def test_result(self, datasample):
        """Test the result of the model and data generator"""
        expected = datasample[1][1]
        model = PertinentNN(
            datasample.n_features,
            datasample.weights,
            torch.Tensor([1, 0, 0, 0, 0]),
            datasample.pn_weight_factor,
        )
        result = model(datasample[1][0])
        assert torch.allclose(expected, result)

    def test_generate_model(self, datasample):
        """Test generate model method with pertinent negative of 1"""
        model = datasample.generate_model()
        res = model(torch.tensor([1, 1.0, 2.0, 3.0, 4.0])).detach()
        expected = datasample.weights @ torch.tensor([1, 1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(res, expected)

    def test_generate_model_2(self, datasample):
        """Test generate model method with pertinent negative of 0"""
        model = datasample.generate_model()
        res = model(torch.tensor([0, 1.0, 2.0, 3.0, 4.0])).detach()
        expected = datasample.weights @ torch.tensor(
            [datasample.pn_weight_factor, 1.0, 2.0, 3.0, 4.0]
        )
        assert torch.allclose(res, expected)

    def test_pn_features_not_list(self, datasample):
        """Test if a value error is raised when something else than a list is inputted"""
        with pytest.raises(ValueError) as exc_info:
            datasample._intialize_pn_features("not a list")
        assert "pn_features must be a list of integers." in str(exc_info.value)

    def test_pn_features_not_all_integers(self, datasample):
        """Test if a value error is raised when a non integer value is inputted"""
        with pytest.raises(ValueError) as exc_info:
            datasample._intialize_pn_features([1, "two", 3])
        assert "All elements in pn_features must be integers." in str(exc_info.value)

    def test_pn_features_empty_or_out_of_range(self, datasample):
        """Test that a value error is raised when an empty is list is inputted"""
        with pytest.raises(ValueError) as exc_info:
            datasample._intialize_pn_features([])
        assert (
            "pn_features cannot be empty and must be within the range of avaialable features."
            in str(exc_info.value)
        )

        with pytest.raises(ValueError) as exc_info:
            datasample._intialize_pn_features([0, 5])
        assert (
            "pn_features cannot be empty and must be within the range of avaialable features."
            in str(exc_info.value)
        )

    def test_statistical_validity_of_pn_zeros(self):
        data = PertinentNegativesDataset(n_samples=1000)
        zero_count = (data.samples[:, data.pn_features] == 0).sum()
        expected_zeros = (
            data.pn_zero_likelihood * len(data.pn_features) * data.n_samples
        )
        tolerance = 0.005 * expected_zeros  # 0.5% tolerance
        assert (
            abs(zero_count - expected_zeros) <= tolerance
        ), "Zero introduction rate does not match likelihood"

    def test_weight_adjustment_due_to_zeros(self, datasample):
        original_weights = datasample.weights.clone()
        datasample._initialize_zeros_for_PN()
        datasample._get_new_weighted_samples()
        for i, sample in enumerate(datasample.samples):
            for idx, pn_feature in enumerate(datasample.pn_features):
                if sample[pn_feature] == 0:
                    expected_weight = (
                        original_weights[pn_feature] * datasample.pn_weight_factor
                    )
                    assert torch.isclose(
                        datasample.weighted_samples[i, pn_feature], expected_weight
                    ), "Weights not adjusted correctly for zeros"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
