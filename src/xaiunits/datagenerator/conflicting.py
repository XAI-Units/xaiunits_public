from typing import Tuple, List, Optional, Callable, Any

import torch
from xaiunits.datagenerator import WeightedFeaturesDataset
from torch import nn


class ConflictingDataset(WeightedFeaturesDataset):
    """
    Generic synthetic dataset with feature cancellation capabilities.

    Feature cancellations are based on likelihood. If cancellation_features are not provided,
    all features in each sample are candidates for cancellation, with a specified likelihood of
    each feature being canceled. Canceled features are negated in their contributions to the dataset,
    allowing for the analysis of model behavior under feature absence scenarios.

    Inherits from:
        WeightedFeaturesDataset: Class extending BaseFeaturesDataset with support for weighted features

    Attributes:
        cancellation_features (list of int, optional): Indices of features subject to cancellation.
        cancellation_likelihood (float): Likelihood of feature cancellation, between 0 and 1.
        cancellation_outcomes (torch.Tensor): Binary tensor indicating whether each feature in each sample is canceled.
        cancellation_samples (torch.Tensor): Concatenation of samples with their cancellation outcomes.
        cancellation_attributions (torch.Tensor): The attribution of each feature considering the cancellation.
        cat_features (list): Categorical features derived from the cancellation samples.
        ground_truth_attributions (torch.Tensor): Combined tensor of weighted samples and cancellation attributions for ground truth analysis.
    """

    def __init__(
        self,
        seed: int = 0,
        n_features: int = 2,
        n_samples: int = 10,
        distribution: str = "normal",
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        weights: Optional[torch.Tensor] = None,
        cancellation_features: Optional[List[int]] = None,
        cancellation_likelihood: float = 0.5,
    ):
        """
        Initializes a ConflictingDataset object.

        Args:
            seed (int): Seed for random number generation, ensuring reproducibility. Defaults to 0.
            n_features (int): Number of features in each sample. Defaults to 2.
            n_samples (int): Number of samples to generate. Defaults to 10.
            distribution (str): Type of distribution to use for generating samples. Defaults to 'normal'.
            weight_range (tuple[float]): Range (min, max) for generating random feature weights. Defaults to (-1.0, 1.0).
            weights (torch.Tensor, optional): Predefined weights for each feature. Defaults to None.
            cancellation_features (list[int], optional): Specific features to apply cancellations to.
                Defaults to None, applying to all features.
            cancellation_likelihood (float): Probability of each feature being canceled. Defaults to 0.5.
        """
        super().__init__(
            seed=seed,
            n_features=n_features,
            n_samples=n_samples,
            distribution=distribution,
            weight_range=weight_range,
            weights=weights,
        )
        self.cancellation_features = cancellation_features
        self.cancellation_likelihood = cancellation_likelihood
        self._initialize_cancellation_features()
        self.cancellation_outcomes = self._get_cancellations()
        self.cancellation_samples = self._get_cancellation_samples()

        non_cancellation = self.cancellation_outcomes ^ 1
        labels = list(torch.sum(non_cancellation * self.samples * self.weights, dim=1))
        self.labels = torch.tensor([float(tensor) for tensor in labels])
        self.cancellation_attributions = self._get_cancellation_attributions()
        self.cat_features = list(range(-self.cancellation_samples.shape[1] // 2, 0))
        self.ground_truth_attributions = torch.cat(
            (self.weighted_samples, self.cancellation_attributions), dim=1
        )
        self.features = "cancellation_samples"
        self.ground_truth_attribute = "ground_truth_attributions"
        self.subset_data = [
            "weighted_samples",
            "cancellation_outcomes",
            "cancellation_samples",
            "cancellation_attributions",
            "ground_truth_attributions",
        ]
        # self.subset_attribute = ["weights"] # Default to BaseClass

    def _initialize_cancellation_features(self) -> None:
        """
        Validates and initializes the list of features subject to cancellation. If no specific features
        are provided, all features are considered candidates for cancellation.

        Raises:
            AssertionError: If cancellation_features is not a list, any element in cancellation_features is not an integer,
                the maximum element in cancellation_features is greater than the number of features, or cancellation_features
                is empty. Also, if cancellation_likelihood is not a float or is outside the range [0, 1].
        """
        if self.cancellation_features is not None:
            assert isinstance(self.cancellation_features, list), "input must be a list"
            assert all(
                [isinstance(x, int) for x in self.cancellation_features]
            ), "cancellation features should be integers"
            assert (
                max(self.cancellation_features, default=0) <= self.n_features
            ), "cancellation features must be within the number of features"
            assert (
                len(self.cancellation_features) > 0
            ), "cancellation features should be at least one"

        assert isinstance(self.cancellation_likelihood, float)
        assert (
            self.cancellation_likelihood >= 0.0
        ), "likelihood must be between zero and one"
        assert (
            self.cancellation_likelihood <= 1.0
        ), "likelihood must be between zero and one"

        if self.cancellation_features is None:
            self.cancellation_features = [n for n in range(self.n_features)]

    def _get_cancellations(self) -> torch.Tensor:
        """
        Generates a binary mask indicating whether each feature in each sample is canceled
        based on the specified likelihood.

        This method considers only the features specified in cancellation_features for possible cancellation.

        Returns:
            torch.Tensor: An integer tensor of shape (n_samples, n_features) where 1 represents a canceled feature,
                and 0 represents an active feature.
        """
        cancellation_feature_mask = torch.zeros(self.n_features, dtype=torch.bool)
        cancellation_feature_mask[self.cancellation_features] = True
        cancel_probs = torch.rand(len(self), self.n_features)
        cancellations = (
            cancel_probs < self.cancellation_likelihood
        ) & cancellation_feature_mask
        return cancellations.int()

    def _get_cancellation_samples(self) -> torch.Tensor:
        """
        Concatenates the original samples with their cancellation outcomes to form a comprehensive dataset.

        This allows for analyzing the impact of feature cancellations directly alongside the original features.

        Returns:
            torch.Tensor: A tensor containing the original samples augmented with their corresponding cancellation outcomes.
        """
        cancellation_samples = torch.cat(
            (self.samples, self.cancellation_outcomes), dim=1
        )
        cancellation_samples = torch.stack([row for row in cancellation_samples])
        return cancellation_samples

    def _get_cancellation_attributions(self) -> torch.Tensor:
        """
        Computes the attribution of each feature by negating the effect of canceled features.

        This method helps understand the impact of each feature on the model output when certain features are
        systematically canceled.

        Returns:
            torch.Tensor: A tensor of the same shape as the weighted samples, where the values of canceled features are
                negated to reflect their absence.
        """
        cancellation_attributions = -(
            self.weighted_samples * self.cancellation_outcomes
        )
        zero_mask = torch.eq(cancellation_attributions, 0)
        cancellation_attributions[zero_mask] = torch.abs(
            cancellation_attributions[zero_mask]
        )
        return cancellation_attributions

    def generate_model(self) -> torch.nn.Module:
        """
        Instantiates and returns a neural network model for analyzing datasets with conflicting features.

        The model is configured to use the specified features and weights, allowing for experimentation
        with feature cancellations.

        Returns:
            model.ConflictingFeaturesNN: A neural network model designed to work with the specified features and weights.
        """
        from xaiunits.model.conflicting import ConflictingFeaturesNN

        return ConflictingFeaturesNN(self.n_features, self.weights)


if __name__ == "__main__":
    data = ConflictingDataset()
    data.perturb_function()
    print(data[0][0])
