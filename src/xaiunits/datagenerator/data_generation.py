import csv
import logging
import os
import pickle
import random
from typing import List, Tuple, Optional, Any, Callable, Dict, Union

import torch
from torch.distributions import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset, random_split
from xaiunits.metrics import perturb_func_constructor


class BaseFeaturesDataset(Dataset):
    """
    Generic synthetic dataset of continuous features for AI explainability.

    This class creates a dataset of continuous features based on a specified distribution,
    which can be used for training and evaluating AI models. It allows for reproducible
    sample creation, customizable features and sample sizes, and supports various distributions.

    Attributes:
        seed (int): Seed for random number generators to ensure reproducibility.
        n_features (int): Number of features in the dataset.
        n_samples (int): Number of samples in the dataset.
        distribution (str | torch.distributions.Distribution): Distribution used for generating the samples.
            Defaults to 'normal' which uses a multivariate normal distribution.
        sample_std_dev (float): Standard deviation of the noise added to the samples.
        label_std_dev (float): Standard deviation of the noise added to generate labels.
        samples (torch.Tensor): Generated samples.
        labels (torch.Tensor): Generated labels with optional noise.
        ground_truth_attribute (str): Name of the attribute considered as ground truth.
        subset_data (list[str]): List of attributes to be included in subsets.
        subset_attribute (list[str]): Additional attributes to be considered in subsets.
        cat_features (list[str]): List of categorical feature names, used in perturbations.
    """

    def __init__(
        self,
        seed: int = 0,
        n_features: int = 2,
        n_samples: int = 10,
        distribution: Union[str, Distribution] = "normal",
        distribution_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a dataset of continuous features based on a specified distribution.

        Args:
            seed (int): For sample creation reproducibility. Defaults to 0.
            n_features (int): Number of features for each sample. Defaults to 2.
            n_samples (int): Total number of samples. Defaults to 10.
            distribution (str | torch.distributions.Distribution): Distribution to use for generating samples.
                Defaults to "normal", which indicates multivariate normal distribution.
            distribution_params (dict, optional): Parameters for the distribution if a string identifier
                is used. Defaults to None.
            **kwargs: Arbitrary keyword arguments, including:

                - sample_std_dev (float): Standard deviation for sample creation noise. Defaults to 1.
                - label_std_dev (float): Noise standard deviation to generate labels. Defaults to 0.

        Raises:
            ValueError: If an unsupported string identifier is provided.
            TypeError: If 'distribution' is neither a string nor a torch.distributions.Distribution instance.
        """
        self.seed, self.n_features, self.n_samples = self._validate_inputs(
            seed, n_features, n_samples
        )
        random.seed(seed)
        torch.manual_seed(seed)
        self.sample_std_dev, self.label_std_dev = self._init_noise_parameters(kwargs)
        self.samples, self.distribution = self._init_samples(
            n_samples, distribution, distribution_params
        )
        self.label_noise = torch.randn(n_samples) * self.label_std_dev
        self.features = "samples"
        self.labels = self.samples.sum(dim=1) + self.label_noise
        self.ground_truth_attribute = "samples"
        self.subset_data = ["samples"]
        self.subset_attribute = ["perturb_function", "name"]
        self.cat_features = []
        self.name = self.__class__.__name__

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.samples)

    def __getitem__(
        self, idx: int, others: List[str] = ["ground_truth_attribute"]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Retrieves a sample and its label, along with optional attributes, by index.

        Args:
            idx (int): Index of the sample to retrieve.
            others (list[str]): Additional attributes to be retrieved with the sample and label.
                Defaults to ["ground_truth_attribute"].

        Returns:
            tuple: A tuple containing the sample and label at the specified index,
                and optionally, a dictionary of additional attributes if requested.

        Raises:
            IndexError: If the specified index is out of the bounds of the dataset.
        """
        if len(others) == 0:
            return (getattr(self, self.features)[idx], self.labels[idx])
        else:
            subset_ret = {}
            for class_attr in others:
                if class_attr == "ground_truth_attribute":
                    subset_ret[class_attr] = getattr(self, self.ground_truth_attribute)[
                        idx
                    ]
                else:
                    subset_ret[class_attr] = getattr(self, class_attr)[idx]
            return (getattr(self, self.features)[idx], self.labels[idx], subset_ret)

    def split(
        self, split_lengths: List[float] = [0.7, 0.3]
    ) -> Tuple["BaseFeaturesDataset", "BaseFeaturesDataset"]:
        """
        Splits the dataset into subsets based on specified proportions.

        Args:
            split_lengths (list[float]): Proportions to split the dataset into. The values
                must sum up to 1. Defaults to [0.7, 0.3] for a 70%/30% split.

        Returns:
            tuple[BaseFeaturesDataset]: A tuple containing the split subsets
                of the dataset.
        """
        subset_list = random_split(
            self,
            lengths=split_lengths,
            generator=torch.Generator().manual_seed(self.seed),
        )
        for sub in subset_list:
            data = self.__getitem__(sub.indices, others=self.subset_data)
            sub.labels = data[1]
            if len(data) == 3:
                for class_attr, values in data[2].items():
                    setattr(sub, class_attr, values)
            for class_attr in self.subset_attribute:
                setattr(sub, class_attr, getattr(self, class_attr))

        return tuple(subset_list)

    def save_dataset(self, file_name: str, directory_path: str = os.getcwd()) -> None:
        """
        Saves the dataset to a pickle file in the specified directory.

        Args:
            file_name (str): Name of the file to save the dataset.
            directory_path (str): Path to the directory where the file will be saved.
                Defaults to the current working directory.
        """
        full_path = os.path.join(directory_path, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as file:
            pickle.dump(self, file)
        print("model saved successfully")

    def _validate_inputs(
        self, seed: int, n_features: int, n_samples: int
    ) -> Tuple[int, int, int]:
        """
        Validates the input parameters for dataset initialization.

        Args:
            seed (int): Seed for random number generation.
            n_features (int): Number of features.
            n_samples (int): Number of samples.

        Returns:
            tuple[int, int]: Validated seed and number of features.

        Raises:
            ValueError: If any input is not an integer or is out of an expected range.
        """
        if not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("Number of features must be a positive integer")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("Number of samples must be a positive integer")
        return seed, n_features, n_samples

    def _init_noise_parameters(self, kwargs: Dict[str, Any]) -> Tuple[float, float]:
        """
        Initializes noise parameters from keyword arguments.

        Args:
            kwargs: Keyword arguments passed to the initializer.

        Returns:
            tuple: Initialized sample and label standard deviations.

        Raises:
            ValueError: If the standard deviations are not positive numbers.
        """
        sample_std_dev = kwargs.get("sample_std_dev", 1.0)
        if not isinstance(sample_std_dev, (int, float)) or sample_std_dev <= 0:
            raise ValueError("Sample standard deviation must be a positive number")

        label_std_dev = kwargs.get("label_std_dev", 0.0)
        if not isinstance(label_std_dev, (int, float)) or label_std_dev < 0:
            raise ValueError("Label standard deviation must be a non-negative number")

        return sample_std_dev, label_std_dev

    def _init_samples(
        self,
        n_samples: int,
        distribution: Union[str, Distribution],
        distribution_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Distribution]:
        """
        Initializes samples based on the specified distribution and sample size.

        This method supports initialization using either a predefined distribution name (string) or directly
        with a torch.distributions.Distribution instance.

        Args:
            n_samples (int): Number of samples to generate, must be positive.
            distribution (str | torch.distributions.Distribution): The distribution to use for
                generating samples. Can be a string for predefined distributions ('normal', 'uniform', 'poisson')
                or an instance of torch.distributions.Distribution.
            distribution_params (dict, optional): Parameters for the distribution if a string identifier
                is used. Examples:
                - For 'normal': {'mean': torch.zeros(n_features), 'stddev': torch.ones(n_features)}
                - For 'uniform': {'low': -1.0, 'high': 1.0}
                - For 'poisson': {'rate': 3.0}

        Returns:
            tuple: A tuple containing generated samples (torch.Tensor) with shape [n_samples, n_features]
                and the distribution instance used.

        Raises:
            ValueError: If 'distribution' is a string and is not one of the supported identifiers or
                necessary parameters are missing.
            TypeError: If 'distribution' is neither a string identifier nor a torch.distributions.Distribution instance,
                or if the provided Distribution instance cannot generate a torch.Tensor.
            RuntimeError: If the generated samples do not match the expected shape and cannot be adjusted.
        """
        if isinstance(distribution, str):
            if distribution_params is None:
                distribution_params = {}

            if distribution == "normal":
                mean = distribution_params.get("mean", 0)
                if not isinstance(mean, torch.Tensor):
                    mean = torch.full((self.n_features,), float(mean))
                stddev = distribution_params.get("stddev", 1)
                if not isinstance(stddev, torch.Tensor):
                    stddev = torch.full((self.n_features,), float(stddev))
                covariance_matrix = torch.diag(stddev)
                distribution = MultivariateNormal(mean, covariance_matrix)
            elif distribution == "uniform":
                low = distribution_params.get("low", -1.0)
                high = distribution_params.get("high", 1.0)
                distribution = Uniform(low, high)
            elif distribution == "poisson":
                rate = distribution_params.get("rate", 3.0)
                distribution = Poisson(rate)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
        elif isinstance(distribution, Distribution):
            if not isinstance(distribution.sample((1,)), torch.Tensor):
                raise TypeError(
                    "The provided distribution must be able to generate a torch.Tensor"
                )
        else:
            raise TypeError(
                "Distribution must be a string identifier or a torch.distributions.Distribution instance"
            )

        # Sample and ensure correct shaping
        samples = distribution.sample((n_samples,))
        if samples.dim() == 1:
            samples = samples.unsqueeze(-1)  # Ensure at least 2D

        # Ensure the tensor has the correct number of features
        if samples.shape[-1] != self.n_features:
            if samples.shape[-1] == 1:  # Single feature expanded to multiple
                samples = samples.expand(-1, self.n_features)
            else:
                raise RuntimeError(
                    f"Expected sample shape [n_samples, n_features] but got {samples.shape}"
                )

        return samples, distribution

    def perturb_function(
        self,
        noise_scale: float = 0.01,
        cat_resample_prob: float = 0.2,
        run_infidelity_decorator: bool = True,
        multipy_by_inputs: bool = False,
    ) -> Callable:
        """
        Generates perturb function to be used for feature attribution method evaluation. Applies Gaussian noise
        for continuous features, and resampling for categorical features.

        Args:
            noise_scale (float): A standard deviation of the Gaussian noise added to the continuous features.
                Defaults to 0.01.
            cat_resample_prob (float): Probability of resampling a categorical feature. Defaults to 0.2.
            run_infidelity_decorator (bool): Set to True if you want the returned fns to be compatible with infidelity.
                Set flag to False for sensitivity. Defaults to True.
            multiply_by_inputs (bool): Parameters for decorator. Defaults to False.

        Returns:
            perturb_func (function): A perturbation function compatible with Captum.
        """
        return perturb_func_constructor(
            noise_scale=noise_scale,
            cat_resample_prob=cat_resample_prob,
            cat_features=self.cat_features,
            run_infidelity_decorator=run_infidelity_decorator,
            multipy_by_inputs=multipy_by_inputs,
        )

    def generate_model(self) -> Any:
        """
        Generates a corresponding model for current dataset.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def default_metric(self) -> Callable:
        """
        The default metric for evaluating the performance of explanation methods applied
        to this dataset.

        Raises:
            NotImplementedError: If the property is not implemented by a subclass.
        """
        raise NotImplementedError


class WeightedFeaturesDataset(BaseFeaturesDataset):
    """
    A class extending BaseFeaturesDataset with support for weighted features.

    This class allows for creating a synthetic dataset with continuous features,
    where each feature can be weighted differently. This is particularly useful for
    scenarios where the impact of different features on the labels needs to be
    artificially manipulated or studied.

    Inherits from:
        BaseFeaturesDataset: The base class for creating continuous feature datasets.

    Attributes:
        weights (torch.Tensor): Weights applied to each feature.
        weight_range (tuple): The range (min, max) within which random weights are generated.
        weighted_samples (torch.Tensor): The samples after applying weights.
    """

    def __init__(
        self,
        seed: int = 0,
        n_features: int = 2,
        n_samples: int = 10,
        distribution: Union[str, Distribution] = "normal",
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a WeightedFeaturesDataset object.

        Args:
            seed (int): Seed for reproducibility. Defaults to 0.
            n_features (int): Number of features. Defaults to 2.
            n_samples (int): Number of samples. Defaults to 10.
            distribution (str): Type of distribution to use for generating samples. Defaults to "normal".
            weight_range (tuple): Range (min, max) for generating random weights. Defaults to (-1.0, 1.0).
            weights (torch.Tensor, optional): Specific weights for each feature.
                If None, weights are generated randomly within `weight_range`. Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the base class constructor, including:

                - sample_std_dev (float): Standard deviation for sample creation noise. Defaults to 1.
                - label_std_dev (float): Noise standard deviation to generate labels. Defaults to 0.
        """
        super().__init__(
            seed=seed,
            n_features=n_features,
            n_samples=n_samples,
            distribution=distribution,
            **kwargs,
        )
        self.weights, self.weight_range = self._initialize_weights(
            weights, weight_range
        )
        self.weighted_samples = self.samples * self.weights
        self.label_noise = torch.randn(self.n_samples) * self.label_std_dev
        self.labels = self.weighted_samples.sum(dim=1) + self.label_noise
        self.features = "samples"
        self.ground_truth_attribute = "weighted_samples"
        self.subset_data = ["samples", "weighted_samples"]
        self.subset_attribute = [
            "weights",
            "default_metric",
            "generate_model",
        ] + self.subset_attribute

    def _initialize_weights(
        self, weights: Optional[torch.Tensor], weight_range: Tuple[float, float]
    ) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """
        Initializes or validates the weights for each feature.

        If weights are not provided, they are randomly generated within the specified range.

        Args:
            weights (torch.Tensor | NoneType): If provided, these weights are used directly for the features.
                Must be a Tensor with a length equal to `n_features`.
            weight_range (tuple): Specifies the minimum and maximum values used to generate weights if `weights` is None.
                Expected format: (min_value, max_value), where both are floats.

        Returns:
            tuple[torch.Tensor, tuple]: The validated or generated weights and the effective weight range used.

        Raises:
            AssertionError: If the provided weights do not match the number of features or are not a torch.Tensor when provided.
            ValueError: If `weight_range` is improperly specified.
        """
        if weights is not None:
            assert (
                len(weights) == self.n_features
            ), f"Check you have the correct number of weights for features ({self.n_features})"
            assert torch.is_tensor(weights), "weights must be of torch.tensor() class"
            # Simply use provided weights and ignore weight_range without warning
            return weights, weight_range

        if weights is None:
            assert isinstance(weight_range, tuple), "Weight_range must be a tuple"
            assert (
                len(weight_range) == 2
            ), "weight_range must consist of a lower and upper bound"
            assert isinstance(weight_range[0], float) and isinstance(
                weight_range[1], float
            ), "Tuple elements must be floats"
            assert (
                weight_range[0] <= weight_range[1]
            ), "Lower bound should be lower than upper bound."
            # Randomly generate feature weights
            b1, b2 = weight_range
            weights = (b2 - b1) * torch.rand(self.n_features) + b1
            return weights, weight_range

    def generate_model(self) -> Any:
        """
        Generates and returns a neural network model configured to use the weighted features of this dataset.

        The model is designed to reflect the differential impact of each feature as specified by the weights.

        Returns:
            model.ContinuousFeaturesNN: A neural network model that includes mechanisms to account for feature weights,
                suitable for tasks requiring understanding of feature importance.
        """
        from xaiunits.model.continuous import ContinuousFeaturesNN

        return ContinuousFeaturesNN(self.n_features, self.weights)

    @property
    def default_metric(self) -> Callable:
        """
        The default metric for evaluating the performance of explanation methods applied
        to this dataset.

        For this dataset, the default metric is the Mean Squared Error (MSE) loss function.

        Returns:
            type: A class that wraps around the default metric to be instantiated
                within the pipeline.
        """
        from xaiunits.metrics import wrap_metric

        return wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        )


def load_dataset(
    file_path: str, directory_path: str = os.getcwd()
) -> Optional[Union[BaseFeaturesDataset, WeightedFeaturesDataset]]:
    """
    Loads a previously saved dataset from a binary pickle file.

    This function is designed to retrieve datasets that have been saved to disk, facilitating
    easy sharing and reloading of data for analysis or model training.

    Args:
        file_path (str): The name of the file to load.
        directory_path (str): The directory where the file is located. Defaults to the current working directory.

    Returns:
        Object | NoneType: The loaded dataset object, or None, if the file does not exist or an error occurs.
    """
    logging.basicConfig(level=logging.INFO)
    full_path = os.path.join(directory_path, file_path)
    try:
        with open(full_path, "rb") as file:
            obj = pickle.load(file)
            logging.info("Successfully loaded: %s", obj)
            return obj
    except FileNotFoundError:
        logging.error("The file '%s' does not exist.", file_path)
        return None
    except Exception as e:
        logging.error("Failed to load the file due to: %s", str(e))
        return None


def generate_csv(file_label: str, num_rows: int = 5000, num_features: int = 20) -> None:
    """
    Generates a CSV file with random data for a specified number of rows and features.

    This function helps create synthetic datasets for testing or development purposes.
    Each row will have a random label and a specified number of features filled with random values.

    Args:
        file_label (str): The base name for the CSV file.
        num_rows (int): Number of rows (samples) to generate. Defaults to 5000.
        num_features (int): Number of features to generate for each sample. Defaults to 20.

     Raises:
        ValueError: If num_rows or num_features are non-positive.
    """
    if num_rows <= 0:
        raise ValueError("num_rows must be a positive integer")
    if num_features <= 0:
        raise ValueError("num_features must be a positive integer")

    fieldnames = ["label"] + [f"c{i}" for i in range(num_features)]
    with open(f"{file_label}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_rows):
            row_data = {col: random.random() for col in fieldnames}
            row_data["label"] = random.randint(0, 9)
            writer.writerow(row_data)


if __name__ == "__main__":
    data = BaseFeaturesDataset()
    train, test = data.split()
    print(len(train))
    print(len(test))
    print(data[:])
    print(data.name)
