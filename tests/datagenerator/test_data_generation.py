import os
import pickle
import numpy as np
import pytest
import random
import csv
import torch
from tempfile import NamedTemporaryFile
from unittest.mock import mock_open, patch
from xaiunits.datagenerator.data_generation import BaseFeaturesDataset
from xaiunits.datagenerator.data_generation import WeightedFeaturesDataset
from xaiunits.datagenerator.data_generation import load_dataset
from xaiunits.datagenerator.data_generation import generate_csv


@pytest.fixture
def base_features():
    data = BaseFeaturesDataset()
    return data


@pytest.fixture
def weighted_features():
    data = WeightedFeaturesDataset()
    return data


###### Test inheritance ########

# First we test that inheritance is working properly. That way, any tests
# for the base features dataset also apply to the weighted features dataset
# and other inheritor classes.


class TestInheritance:
    def test_subclasses(self):
        """Test that the inheritance via subclasses is correct"""
        assert issubclass(BaseFeaturesDataset, torch.utils.data.Dataset)
        assert issubclass(WeightedFeaturesDataset, torch.utils.data.Dataset)
        assert issubclass(WeightedFeaturesDataset, BaseFeaturesDataset)

    def test_inherited_attributes(self, base_features, weighted_features):
        """Make sure default attribute values are identical or different as expected"""
        # Expected identical
        assert base_features.seed == weighted_features.seed
        assert base_features.n_features == weighted_features.n_features
        assert base_features.n_samples == weighted_features.n_samples
        assert (
            base_features.distribution.mean == weighted_features.distribution.mean
        ).all()
        assert (
            base_features.distribution.covariance_matrix
            == weighted_features.distribution.covariance_matrix
        ).all()
        assert base_features.sample_std_dev == weighted_features.sample_std_dev
        assert base_features.label_std_dev == weighted_features.label_std_dev
        assert base_features.features == weighted_features.features
        assert (base_features.samples == weighted_features.samples).all()
        assert (base_features.label_noise == weighted_features.label_noise).all()
        assert base_features.cat_features == weighted_features.cat_features
        # Expected different
        assert (
            base_features.ground_truth_attribute
            != weighted_features.ground_truth_attribute
        )
        assert base_features.subset_data != weighted_features.subset_data
        assert base_features.subset_attribute != weighted_features.subset_attribute

    def test_overriding_attributes(self):
        """Test that each explicit attribute is correctly passed to the base class"""
        weighted_dataset = WeightedFeaturesDataset(
            seed=4, n_features=20, n_samples=100, distribution="poisson"
        )
        assert weighted_dataset.seed == 4
        assert weighted_dataset.n_features == 20
        assert weighted_dataset.n_samples == 100
        assert weighted_dataset.distribution.mean == 3.0

    def test_overriding_methods(self):
        """Test each class is overriding properly the functions of its parent class"""
        base_method = BaseFeaturesDataset.__dict__.get("__init__", None)
        sub_method = WeightedFeaturesDataset.__dict__.get("__init__", None)
        assert (sub_method is not None and base_method is not sub_method) and True

    def test_same_samples_from_seed(self):
        """Test that using the same seed gives the same samples"""
        random_seed = random.randint(0, 100)
        CF_df = BaseFeaturesDataset(seed=random_seed)
        WF_df = WeightedFeaturesDataset(seed=random_seed)
        assert all([torch.equal(a, b) for a, b in zip(CF_df.samples, WF_df.samples)])

    # Testing overriden methods
    def test_overridden_initialization(self):
        weights = torch.tensor([0.5, 1.5])
        dataset = WeightedFeaturesDataset(weights=weights)
        assert torch.equal(
            dataset.weights, weights
        ), "WeightedFeaturesDataset does not correctly initialize weights"
        assert torch.equal(
            dataset.samples * dataset.weights, dataset.weighted_samples
        ), "Weighted samples are not calculated correctly upon initialization"

    def test_overridden_getitem(self):
        dataset = WeightedFeaturesDataset(n_samples=10)
        sample, label, additional_data = dataset.__getitem__(0, others=["weights"])
        assert (
            "weights" in additional_data
        ), "WeightedFeaturesDataset does not return weights in __getitem__ as expected"

    # Testing abstract methods
    def test_implementation_of_abstract_methods(self):
        try:
            model = (
                WeightedFeaturesDataset().generate_model()
            )  # Assuming generate_model is abstract in base
            assert (
                model is not None
            ), "generate_model should be implemented and return a model instance"
        except NotImplementedError:
            pytest.fail("generate_model is not implemented in WeightedFeaturesDataset")

    def test_functionality_of_abstract_methods(self):
        dataset = WeightedFeaturesDataset(n_samples=5, weights=torch.tensor([1.0, 2.0]))
        model = dataset.generate_model()
        # Assume we can simulate a simple forward pass or its equivalent
        inputs = torch.randn(5, 2)
        outputs = model(inputs)
        expected_outputs = inputs @ torch.tensor([1.0, 2.0]).unsqueeze(
            -1
        )  # Simplified expected behavior
        assert torch.allclose(
            outputs, expected_outputs
        ), "Model does not compute outputs as expected from weights"

    # Test properties
    def test_properties(self):
        dataset = WeightedFeaturesDataset(
            n_samples=10, weights=torch.tensor([0.5, 1.0])
        )
        assert (
            dataset.default_metric is not None
        ), "default_metric should be properly defined and not return None"

    def test_dynamic_properties(self):
        dataset = WeightedFeaturesDataset(n_samples=10)
        initial_metric = dataset.default_metric
        dataset.weights = torch.tensor([0.3, 0.7])  # Change that influences the metric
        assert (
            dataset.default_metric != initial_metric
        ), "default_metric should reflect changes in dataset properties or configuration"


###### Base Features Dataset Tests ########


class TestInputValidation:
    def test_seed_input(self):
        """Ensure that the seed has to be an integer"""
        with pytest.raises(ValueError, match="Seed must be an integer"):
            BaseFeaturesDataset(seed=1.4)
        with pytest.raises(ValueError, match="Seed must be an integer"):
            BaseFeaturesDataset(seed="hello")

    def test_n_features_input(self):
        """Ensure that n features is a positive integer."""
        with pytest.raises(
            ValueError, match="Number of features must be a positive integer"
        ):
            BaseFeaturesDataset(n_features=0.7)
        with pytest.raises(
            ValueError, match="Number of features must be a positive integer"
        ):
            BaseFeaturesDataset(n_features=-4)
        with pytest.raises(
            ValueError, match="Number of features must be a positive integer"
        ):
            BaseFeaturesDataset(n_features="text")

    def test_n_samples_input(self):
        """Ensure that n samples must also be positive integers."""
        with pytest.raises(
            ValueError, match="Number of samples must be a positive integer"
        ):
            BaseFeaturesDataset(n_samples=0.7)
        with pytest.raises(
            ValueError, match="Number of samples must be a positive integer"
        ):
            BaseFeaturesDataset(n_samples=-4)
        with pytest.raises(
            ValueError, match="Number of samples must be a positive integer"
        ):
            BaseFeaturesDataset(n_samples="text")

    def test_invalid_parameter_types(self):
        """Test we get Errors when expected"""
        with pytest.raises(ValueError):
            BaseFeaturesDataset(n_samples="a lot")
        with pytest.raises(ValueError):
            BaseFeaturesDataset(n_features={"number": 5})
        with pytest.raises(ValueError):
            BaseFeaturesDataset(sample_std_dev="high")
        with pytest.raises(AssertionError):
            WeightedFeaturesDataset(weights="heavy")


class TestStandardDeviations:
    def test_default_std(self, base_features):
        """Checks that the base class is defaulting to expected std values"""
        assert base_features.sample_std_dev == 1.0
        assert base_features.label_std_dev == 0.0

    def test_valid_std_values(self):
        """Checks that we do not permit invalid values for noise"""
        with pytest.raises(
            ValueError, match="Sample standard deviation must be a positive number"
        ):
            BaseFeaturesDataset(sample_std_dev=-1)
        with pytest.raises(
            ValueError, match="Sample standard deviation must be a positive number"
        ):
            BaseFeaturesDataset(sample_std_dev="hello")

        with pytest.raises(
            ValueError, match="Label standard deviation must be a non-negative number"
        ):
            BaseFeaturesDataset(label_std_dev=-1)
        with pytest.raises(
            ValueError, match="Label standard deviation must be a non-negative number"
        ):
            BaseFeaturesDataset(label_std_dev="hello")


class TestDistributions:
    def test_distribution_string_classes(self):
        """Checks how a string passed to the constructor is handled"""
        assert isinstance(
            BaseFeaturesDataset(distribution="normal").distribution,
            torch.distributions.multivariate_normal.MultivariateNormal,
        )
        assert isinstance(
            BaseFeaturesDataset(distribution="poisson").distribution,
            torch.distributions.poisson.Poisson,
        )
        assert isinstance(
            BaseFeaturesDataset(distribution="uniform").distribution,
            torch.distributions.uniform.Uniform,
        )

    def test_distribution_default_parameters(self):
        """Tests that distributions are initialized with correct parameters."""
        normal_dataset = BaseFeaturesDataset(distribution="normal")
        assert torch.allclose(
            normal_dataset.distribution.mean,
            torch.zeros(normal_dataset.n_features),
            atol=1e-5,
        )
        assert torch.allclose(
            normal_dataset.distribution.covariance_matrix,
            torch.diag(torch.ones(normal_dataset.n_features)),
            atol=1e-5,
        )

        poisson_dataset = BaseFeaturesDataset(
            distribution="poisson", distribution_params={"rate": 3.0}
        )
        assert poisson_dataset.distribution.rate == 3.0

        uniform_dataset = BaseFeaturesDataset(
            distribution="uniform", distribution_params={"low": -1.0, "high": 1.0}
        )
        assert uniform_dataset.distribution.low == -1.0
        assert uniform_dataset.distribution.high == 1.0

    def test_normal_distribution_params(self):
        """Test normal distribution with custom parameters"""
        dataset = BaseFeaturesDataset(
            n_samples=100,
            distribution="normal",
            distribution_params={"mean": 0, "stddev": 2},
        )
        expected_stddev = 2 * torch.ones(dataset.n_features)
        actual_stddev = torch.diag(dataset.distribution.covariance_matrix)
        assert torch.allclose(
            dataset.distribution.mean, torch.zeros(dataset.n_features)
        ), "Mean is incorrect"
        assert torch.allclose(
            actual_stddev, expected_stddev
        ), "Standard deviations are incorrect"

    def test_uniform_distribution_params(self):
        """Test uniform distribution with custom parameters"""
        dataset = BaseFeaturesDataset(
            n_samples=100,
            distribution="uniform",
            distribution_params={"low": -2, "high": 2},
        )
        assert dataset.distribution.low == -2
        assert dataset.distribution.high == 2

    def test_poisson_distribution_params(self):
        """Test Poisson distribution with custom rate"""
        dataset = BaseFeaturesDataset(
            n_samples=100, distribution="poisson", distribution_params={"rate": 1.0}
        )
        assert dataset.distribution.rate == 1.0

    def test_invalid_distribution(self):
        """Ensure that invalid distributions raise an error"""
        with pytest.raises(ValueError):
            BaseFeaturesDataset(
                distribution="binomial", distribution_params={"n": 10, "p": 0.5}
            )

    def test_accepts_torch_distribution(self):
        """Test the dataset can accept a torch.distributions.Distribution instance."""
        dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        dataset = BaseFeaturesDataset(n_samples=100, distribution=dist)
        assert isinstance(
            dataset.distribution, torch.distributions.Normal
        ), "Failed to accept a Normal distribution instance"

        dist = torch.distributions.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        dataset = BaseFeaturesDataset(n_samples=100, distribution=dist)
        assert isinstance(
            dataset.distribution, torch.distributions.Uniform
        ), "Failed to accept a Uniform distribution instance"


class TestGetItem:
    def test_get_item(self):
        """Test functionality of getitem"""
        tests = BaseFeaturesDataset().__getitem__(3)
        truths = (torch.tensor([-0.34136, 1.85301]), torch.tensor(1.51160))
        for test, truth in zip(tests, truths):
            assert torch.allclose(input=test, other=truth, atol=1e-4)
        # Checking index errors
        with pytest.raises(IndexError):
            test = BaseFeaturesDataset().samples.__getitem__(20)
        with pytest.raises(IndexError):
            test = BaseFeaturesDataset().samples.__getitem__(-11)

    def test_zero_samples(self):
        with pytest.raises(ValueError):
            BaseFeaturesDataset(n_samples=0)

    def test_single_sample_dataset(self):
        dataset = BaseFeaturesDataset(n_samples=1)
        assert len(dataset) == 1
        assert isinstance(
            dataset.__getitem__(0), tuple
        )  # Make sure getitem returns correctly


class TestLen:
    """Ensure that the length of the dataset matches the number of samples"""

    def test_default_len(self):
        assert len(BaseFeaturesDataset()) == 10

    def test_len_matches_samples(self):
        for _ in range(10):
            sample_num = random.randint(1, 1000)
            assert len(BaseFeaturesDataset(n_samples=sample_num)) == sample_num

    def test_very_large_dataset(self):
        # This test might need adjustments based on system capabilities
        dataset = BaseFeaturesDataset(n_samples=1000000)
        assert len(dataset) == 1000000


class TestReproducibility:
    """Several tests to make sure that the random seeding works as expected."""

    def test_seed_based_reproducibility(self):
        seed = 42
        dataset1 = BaseFeaturesDataset(seed=seed, n_samples=100)
        dataset2 = BaseFeaturesDataset(seed=seed, n_samples=100)
        assert torch.equal(
            dataset1.samples, dataset2.samples
        ), "Samples do not match with the same seed"
        assert torch.equal(
            dataset1.labels, dataset2.labels
        ), "Labels do not match with the same seed"

    def test_deterministic_splits(self):
        seed = 123
        dataset = BaseFeaturesDataset(seed=seed, n_samples=50)
        split_ratio = [0.7, 0.3]
        train1, test1 = dataset.split(split_ratio)
        train2, test2 = dataset.split(split_ratio)

        # Check if the samples in splits are identical
        assert torch.equal(
            train1.samples, train2.samples
        ), "Train samples differ between splits with the same seed"
        assert torch.equal(
            test1.samples, test2.samples
        ), "Test samples differ between splits with the same seed"

    def test_reproducible_weights(self):
        seed = 45
        dataset1 = WeightedFeaturesDataset(seed=seed)
        dataset2 = WeightedFeaturesDataset(seed=seed)
        assert torch.equal(
            dataset1.weights, dataset2.weights
        ), "Weights are not reproducible with the same seed"


class TestSaveDataset:
    def test_save_dataset(self, tmp_path):
        dataset = BaseFeaturesDataset(n_samples=10, n_features=2, distribution="normal")
        # Define file name and path using pytest's tmp_path fixture for temporary file
        file_name = "test_dataset.pkl"
        file_path = tmp_path / file_name
        # Save the dataset
        dataset.save_dataset(file_name=file_name, directory_path=str(tmp_path))

        # Verify the file exists
        assert os.path.exists(file_path), "Dataset file was not created."

        # Load the dataset
        with open(file_path, "rb") as file:
            loaded_dataset = pickle.load(file)

        # Verify the loaded dataset's attributes match the original dataset
        assert (
            loaded_dataset.n_samples == dataset.n_samples
        ), "Loaded dataset n_samples attribute does not match."
        assert (
            loaded_dataset.n_features == dataset.n_features
        ), "Loaded dataset n_features attribute does not match."
        # Add more checks as necessary for other attributes

        # Optionally, verify the data itself if applicable
        assert torch.equal(
            loaded_dataset.samples, dataset.samples
        ), "Loaded dataset samples do not match original."
        assert torch.equal(
            loaded_dataset.labels, dataset.labels
        ), "Loaded dataset labels do not match original."


class TestSplit:
    def test_train_test_split(self):
        seed = 42
        n_samples = 100
        n_features = 2
        test_split = 0.2
        dataset = BaseFeaturesDataset(
            seed=seed, n_features=n_features, n_samples=n_samples, distribution="normal"
        )
        train_dataset, test_dataset = dataset.split([1 - test_split, test_split])

        # Check split ratios
        expected_test_size = int(n_samples * test_split)
        expected_train_size = n_samples - expected_test_size
        assert len(test_dataset) == expected_test_size
        assert len(train_dataset) == expected_train_size

        # Check mutual exclusivity of train and test datasets
        combined_samples = torch.cat(
            (train_dataset.samples, test_dataset.samples), dim=0
        )
        unique_samples, _ = combined_samples.unique(dim=0, return_counts=True)
        assert combined_samples.size(0) == unique_samples.size(
            0
        ), "Train and test datasets are not mutually exclusive."

        # Check reproducibility
        train_dataset_repeated, test_dataset_repeated = dataset.split(
            [1 - test_split, test_split]
        )
        np.testing.assert_array_equal(
            train_dataset.samples,
            train_dataset_repeated.samples,
            err_msg="Train splits are not the same across runs.",
        )
        np.testing.assert_array_equal(
            test_dataset.samples,
            test_dataset_repeated.samples,
            err_msg="Test splits are not the same across runs.",
        )


class TestPerturbFunction:
    def test_perturb_function_method(self, base_features):
        """Tests whether the class has the generate_model() abstract method."""
        from xaiunits.metrics import perturb_func_constructor

        noise_scale = 0.5
        cat_resample_prob = 0.2
        run_infidelity_decorator = True
        multipy_by_inputs = True

        perturbed1 = base_features.perturb_function(
            noise_scale=noise_scale,
            cat_resample_prob=cat_resample_prob,
            run_infidelity_decorator=run_infidelity_decorator,
            multipy_by_inputs=multipy_by_inputs,
        )
        perturbed2 = perturb_func_constructor(
            noise_scale=noise_scale,
            cat_resample_prob=cat_resample_prob,
            cat_features=base_features.cat_features,
            run_infidelity_decorator=run_infidelity_decorator,
            multipy_by_inputs=multipy_by_inputs,
        )

        inputs = base_features[:][0]
        torch.manual_seed(42)
        pf_out1 = perturbed1(inputs)
        torch.manual_seed(42)
        pf_out2 = perturbed2(inputs)

        print(pf_out1)

        assert [
            torch.isclose(pf_out1[i][0], pf_out2[i][0]) for i in range(len(pf_out1))
        ]

    def test_perturb_function_method_no_decorator(self, base_features):
        """Tests whether the class has the generate_model() abstract method."""
        from xaiunits.metrics import perturb_func_constructor

        noise_scale = 0.5
        cat_resample_prob = 0.2
        run_infidelity_decorator = False
        multipy_by_inputs = True

        perturbed1 = base_features.perturb_function(
            noise_scale=noise_scale,
            cat_resample_prob=cat_resample_prob,
            run_infidelity_decorator=run_infidelity_decorator,
            multipy_by_inputs=multipy_by_inputs,
        )
        perturbed2 = perturb_func_constructor(
            noise_scale=noise_scale,
            cat_resample_prob=cat_resample_prob,
            cat_features=base_features.cat_features,
            run_infidelity_decorator=run_infidelity_decorator,
            multipy_by_inputs=multipy_by_inputs,
        )

        inputs = base_features[:][0]
        torch.manual_seed(42)
        pf_out1 = perturbed1(inputs)
        torch.manual_seed(42)
        pf_out2 = perturbed2(inputs)

        print(pf_out1)

        assert [
            torch.isclose(pf_out1[i][0], pf_out2[i][0]) for i in range(len(pf_out1))
        ]


class TestAbstractMethods:
    def test_generate_model_not_implemented(self, base_features):
        """Tests whether the class has the generate_model() abstract method."""
        with pytest.raises(NotImplementedError):
            base_features.generate_model()

    def test_default_metric_not_implemented(self, base_features):
        """Tests whether the class has the default_metric() abstract method."""
        with pytest.raises(NotImplementedError):
            base_features.default_metric()


###### Weighted Features Dataset Tests ########


class TestWeightsInputValidation:
    def test_weights_range(self):
        """Weight range must be a lower and upper bound"""
        with pytest.raises(AssertionError):
            WeightedFeaturesDataset(weight_range=(-1.0, 0.0, 1.0))
        with pytest.raises(AssertionError):
            WeightedFeaturesDataset(weight_range=(0))
        with pytest.raises(AssertionError):
            WeightedFeaturesDataset(weight_range=([-1, 1]))
        with pytest.raises(AssertionError):
            WeightedFeaturesDataset(weight_range=("-1.0", "1.0"))

    def test_weight_override(self):
        """Ensure that weight_range is ignored when manual weights are specified"""
        dataset = WeightedFeaturesDataset(
            weight_range=(-2.0, 1.0), weights=torch.tensor([0.1, 0.2])
        )
        assert torch.equal(
            dataset.weights, torch.tensor([0.1, 0.2])
        ), "Weights were not correctly initialized."


class TestWeightEdgeCases:
    def test_highly_skewed_weights(self):
        # Skewed towards the upper limit
        dataset = WeightedFeaturesDataset(weight_range=(0.99, 1.0))
        assert (dataset.weights > 0.99).all() and (dataset.weights <= 1.0).all()

    def test_weight_boundary_initialization(self):
        dataset_min = WeightedFeaturesDataset(
            weight_range=(0.0, 0.0)
        )  # All weights are zero
        dataset_max = WeightedFeaturesDataset(
            weight_range=(1.0, 1.0)
        )  # All weights are at their maximum
        assert torch.all(dataset_min.weights == 0)
        assert torch.all(dataset_max.weights == 1)


class TestWeightAttributeCalculations:
    def test_weighted_samples(self, weighted_features):
        """Ensure that the weighted sample procedure is correct"""
        assert (
            weighted_features.weighted_samples
            == (weighted_features.samples * weighted_features.weights)
        ).all()
        assert (weighted_features.label_noise == 0.0).all()
        assert (
            weighted_features.labels
            == weighted_features.weighted_samples.sum(axis=1)
            + weighted_features.label_noise
        ).all()


##### Test ancillary functions ########


class TestLoadDataset:
    def test_load_dataset_success(self):
        """
        Tests that `load_dataset` correctly loads and returns a dataset object from a valid file path.
        This test simulates the scenario where the dataset file exists and contains valid pickled data.
        It checks that the file is read correctly, pickle loads the data, and the function returns the expected dataset object.
        """
        fake_dataset = {"data": "this is a test"}
        with patch(
            "builtins.open", mock_open(read_data=pickle.dumps(fake_dataset))
        ) as mocked_file:
            with patch("pickle.load", return_value=fake_dataset):
                result = load_dataset("fakefile.pkl", "fakedir")
                # Verify the file was opened correctly
                mocked_file.assert_called_with(
                    os.path.join("fakedir", "fakefile.pkl"), "rb"
                )
                # Check the return value is as expected
                assert (
                    result == fake_dataset
                ), "The dataset should be loaded successfully"

    def test_load_dataset_file_not_found(self):
        """
        Tests that `load_dataset` correctly handles and logs an error when the specified file does not exist.
        This test simulates the scenario where the file path provided does not point to an existing file.
        It verifies that the function returns None and logs an appropriate error message indicating the file was not found.
        """
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with patch("logging.error") as mock_log_error:
                result = load_dataset("nonexistent.pkl", "fakedir")
                assert (
                    result is None
                ), "The function should return None for non-existent files"
                mock_log_error.assert_called_once()  # Check if the log error was called

    def test_load_dataset_general_exception(self):
        """
        Tests that `load_dataset` correctly handles and logs any unexpected errors during the file loading process.
        This test introduces a general exception to simulate an error like file corruption or a reading error.
        It checks that the function returns None and logs an error message detailing the unexpected error.
        """
        with patch("builtins.open", mock_open()) as mocked_file:
            mocked_file.side_effect = Exception("Unexpected error")
            with patch("logging.error") as mock_log_error:
                result = load_dataset("corruptfile.pkl", "fakedir")
                assert result is None, "The function should return None on error"
                mock_log_error.assert_called_once()  # Ensure the error is logged


class TestDefaultMetric:
    def test_default_metric_implemented(self, weighted_features):
        """Tests whether default_metric method is implemented for the class."""
        assert hasattr(weighted_features, "default_metric")

    def test_default_metric_correctness(self, weighted_features):
        """Tests whether the default_metric property is correctly specified."""
        from xaiunits.metrics import wrap_metric

        weighted_features.default_metric == wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.sqrt(torch.sum(x, dim=1)),
        )


class TestCSVGeneration:
    def test_csv_generation_and_deletion(self):
        with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
        generate_csv(
            file_label=file_path.replace(".csv", ""), num_rows=100, num_features=10
        )
        assert os.path.exists(file_path), "CSV file should exist after generation"
        # Open and check the CSV content
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            headers = next(reader)
            assert len(headers) == 11, "Should have 11 columns including label"
            data = list(reader)
            assert len(data) == 100, "Should have 100 rows as specified"
        os.remove(file_path)

    def test_csv_correct_number_of_features(self):
        with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
        generate_csv(
            file_label=file_path.replace(".csv", ""), num_rows=50, num_features=5
        )
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            headers = next(reader)
            assert len(headers) == 6, "There should be 6 columns (1 label + 5 features)"
        os.remove(file_path)

    def test_csv_empty_case(self):
        with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
        with pytest.raises(ValueError) as excinfo:
            generate_csv(
                file_label=file_path.replace(".csv", ""), num_rows=0, num_features=10
            )
        assert "num_rows must be a positive integer" in str(
            excinfo.value
        ), "Should raise an error for zero num_rows"

    def test_data_types(self):
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
        generate_csv(file_path.replace(".csv", ""), num_rows=10, num_features=5)
        with open(file_path, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                assert isinstance(
                    int(row["label"]), int
                ), "Label should be convertible to int"
                for i in range(5):
                    assert isinstance(
                        float(row[f"c{i}"]), float
                    ), f"c{i} should be convertible to float"
        os.remove(file_path)

    def test_edge_case_negative_numbers(self):
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
        with pytest.raises(ValueError):
            generate_csv(file_path.replace(".csv", ""), num_rows=-1, num_features=-5)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                assert (
                    file.read() == ""
                ), "File should be empty due to error in generation parameters"
            os.remove(file_path)


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
