from typing import List, Tuple, Optional, Any

import torch
from xaiunits.datagenerator import WeightedFeaturesDataset


class PertinentNegativesDataset(WeightedFeaturesDataset):
    """
    A dataset designed to investigate the impact of pertinent negative (PN) features
    on model predictions by introducing zero values in selected features, which are
    expected to significantly impact the output.

    This dataset is useful for scenarios where the absence of certain features
    (indicated by zero values) provides important information for model predictions.

    Inherits from:
        WeightedFeaturesDataset: Class extending BaseFeaturesDataset with support for weighted features

    Attributes:
        pn_features (list[int]): Indices of features considered as pertinent negatives.
        pn_zero_likelihood (float): Likelihood of a pertinent negative feature being set to zero.
        pn_weight_factor (float): Weight factor applied to the pertinent negative features to emphasize their impact.
        cat_features (list): Categorical features derived from the pertinent negatives.
        labels (torch.Tensor): Generated labels with optional noise.
        features (str): Name of the attribute representing the input features.
        ground_truth_attribute (str): Name of the attribute considered as ground truth for analysis.
        subset_data (list[str]): List of attributes to be included in subsets.
        subset_attribute (list[str]): Additional attributes to be considered in subsets.
    """

    def __init__(
        self,
        seed: int = 0,
        n_features: int = 5,
        n_samples: int = 10,
        distribution: str = "normal",
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        weights: Optional[torch.Tensor] = None,
        pn_features: Optional[List[int]] = None,
        pn_zero_likelihood: float = 0.5,
        pn_weight_factor: float = 10,
        baseline: str = "zero",
    ):
        super().__init__(
            seed=seed,
            n_features=n_features,
            n_samples=n_samples,
            distribution=distribution,
            weight_range=weight_range,
            weights=weights,
        )
        """
        Initializes a PertinentNegativesDataset object.

        Args:
            seed (int): Seed for random number generation, ensuring reproducibility. Defaults to 0.
            n_features (int): Number of features in each sample. Defaults to 5.
            n_samples (int): Number of samples to generate. Defaults to 10.
            distribution (str): Type of distribution to use for generating samples. Defaults to 'normal'.
            weight_range (tuple): Range (min, max) for generating random feature weights. 
                Defaults to (-1.0, 1.0).
            weights (torch.Tensor, optional): Predefined weights for each feature. Defaults to None.
            pn_features (list[int], optional): Indices of features to be considered as pertinent negatives.
            pn_zero_likelihood (float): Probability of a pertinent negative feature being zero. Defaults to 0.5.
            pn_weight_factor (float): Factor to enhance the impact of pertinent negatives. Defaults to 10.
            baseline (string) : Determines which version of baseline and thus ground truth. "zero" or "one". Defaults to "zero" 
        """
        self.pn_zero_likelihood = pn_zero_likelihood
        self.pn_weight_factor = pn_weight_factor
        self.pn_features = self._intialize_pn_features(pn_features)
        self._initialize_zeros_for_PN()
        self._get_new_weighted_samples()
        self._create_ground_truth_baseline(baseline)
        self.cat_features = self.pn_features

        self.label_noise = torch.randn(n_samples) * self.label_std_dev
        self.labels = self.weighted_samples.sum(dim=1) + self.label_noise

        self.features = "samples"
        self.ground_truth_attribute = "ground_truth"

        self.subset_data = ["samples", "weighted_samples", "ground_truth"]
        self.subset_attribute = [
            "pn_zero_likelihood",
            "pn_weight_factor",
            "pn_features",
            "baseline",
        ] + self.subset_attribute

    def _intialize_pn_features(self, pn_features: Optional[List[int]]) -> List[int]:
        """
        Validates and initializes the indices of features to be considered as pertinent negatives (PN).

        Ensures that specified pertinent negative features are within the valid range of feature indices.
        Falls back to the first feature if pn_features is not specified or invalid.

        Args:
            pn_features (list of int, optional): Indices of features specified as pertinent negatives.

        Returns:
            list[int]: The validated list of indices for pertinent negative features.

        Raises:
            ValueError: If any specified pertinent negative feature index is out of the valid range or if the input is not a list.
        """
        if pn_features is None:
            return [0]  # Defaulting to the first feature if None.

        if not isinstance(pn_features, list):
            raise ValueError("pn_features must be a list of integers.")

        if not all(isinstance(x, int) for x in pn_features):
            raise ValueError("All elements in pn_features must be integers.")

        if len(pn_features) == 0 or max(pn_features) >= self.n_features:
            raise ValueError(
                "pn_features cannot be empty and must be within the range of avaialable features."
            )

        return pn_features

    def _initialize_zeros_for_PN(self) -> None:
        """
        Sets the values of pertinent negative (PN) features to zero with a specified likelihood,
        across all samples in a vectorized manner.

        This modification is performed directly on the `samples` attribute.
        """
        pn_random_matrix = (
            torch.rand(len(self), len(self.pn_features)) < self.pn_zero_likelihood
        )
        for feature_index, pn_feature in enumerate(self.pn_features):
            self.samples[:, pn_feature] = pn_random_matrix[:, feature_index]

    def _get_new_weighted_samples(self) -> None:
        """
        Recalculates the weighted samples considering the introduction of zeros
        for pertinent negative features in a vectorized manner.

        Adjusts the weight of features set to zero to emphasize their impact by using the pn_weight_factor.
        Updates the `weighted_samples` attribute with the new calculations.
        """
        weighted_samples = self.weights * self.samples

        # Vectorised adjustment for pertinent negatives
        pn_adjustments = (
            (1 - self.samples[:, self.pn_features])
            * self.pn_weight_factor
            * self.weights[self.pn_features]
        )
        weighted_samples[:, self.pn_features] += pn_adjustments

        self.weighted_samples = weighted_samples

    def _create_ground_truth_baseline(self, baseline: str) -> None:
        """
        Creates the ground truth baseline based on the specified baseline type ("zero" or "one").

        Args:
            baseline (str): Specifies the type of baseline to use. Must be either "zero" or "one".

        Raises:
            KeyError: If the specified baseline is not "zero" or "one".
        """
        if baseline == "zero":
            baseline_value = 0.0
        elif baseline == "one":
            baseline_value = 1.0
        else:
            raise KeyError("Baseline must be either 'zero' or 'one'.")

        self.baseline = torch.zeros((self.samples.shape[0], self.n_features))
        self.baseline[:, self.pn_features] = float(baseline_value)

        ground_truth = self.weighted_samples.detach().clone()
        ground_truth[:, self.pn_features] = ground_truth[
            :, self.pn_features
        ] - self.weights.unsqueeze(0)[:, self.pn_features] * (
            self.pn_weight_factor if baseline_value == 0 else 1
        )
        self.ground_truth = ground_truth

    def __getitem__(
        self, idx: int, others: List[str] = ["ground_truth_attribute", "baseline"]
    ) -> Tuple[Any, ...]:
        """
        Retrieve a sample and its associated label by index.

        Args:
            idx (int): Index of the sample to retrieve.
            others (list): Additional items to retrieve. Defaults to [].

        Returns:
            tuple: Tuple containing the sample and its label.
        """
        return super().__getitem__(idx, others=others)

    def generate_model(self) -> torch.nn.Module:
        """
        Generates and returns a neural network model tailored for analyzing the impact of pertinent negatives.

        The model is configured to incorporate the weights, pertinent negatives,
        and the pertinent negative weight factor.

        Returns:
            model.PertinentNN: A neural network model designed to work with the dataset's specific configuration,
                including the pertinent negatives and their associated weight factor.
        """
        from xaiunits.model.pertinent_negative import PertinentNN

        new_pn = torch.zeros((self.n_features,))
        for pn in self.pn_features:
            new_pn[pn] = 1.0

        return PertinentNN(self.n_features, self.weights, new_pn, self.pn_weight_factor)


if __name__ == "__main__":
    data = PertinentNegativesDataset()
    train, test = data.split()
    print(train.weights)
    print(data[:])
