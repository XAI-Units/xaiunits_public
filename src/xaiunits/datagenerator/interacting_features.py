import math
from typing import List, Tuple, Optional, Any

import torch
from xaiunits.datagenerator import WeightedFeaturesDataset


class InteractingFeatureDataset(WeightedFeaturesDataset):
    """
    A dataset subclass for modeling interactions between categorical and continuous features within weighted datasets.

    This class extends WeightedFeaturesDataset to support scenarios where the influence of one feature on the model
    is conditional on the value of another, typically categorical, feature. For instance, the model may include terms like
    `w_i(x_j) * x_i + w_j * x_j`, where the weight `w_i(x_j)` changes based on the value of `x_j`.

    Inherits from:
        WeightedFeaturesDataset: Class extending BaseFeaturesDataset with support for weighted features

    Attributes:
        interacting_features (list[list[int]]): Pairs of indices where the first index is the feature whose
            weight is influenced by the second, categorical feature.
        zero_likelihood (float): The likelihood of the categorical feature being zero.
        seed (int): Random seed for reproducibility.
        n_features (int): Number of features in the dataset.
        n_samples (int): Number of samples in the dataset.
        weight_range (tuple[float]): Min and max values for generating weights.
        weights (list | NoneType): Initial weight values for features.
        subset_attribute (list[str]): List of attributes that define the subset of the data with specific characteristics.
    """

    def __init__(
        self,
        seed: int = 0,
        n_features: int = 4,
        n_samples: int = 50,
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        weights: Optional[List[float]] = None,
        zero_likelihood: float = 0.5,
        interacting_features: List[List[int]] = [
            [1, 0],
            [3, 2],
        ],  # Value of second col determines weight of first
        **kwargs: Any,
    ):
        flat_weights = self._get_flat_weights(weights)
        super().__init__(
            seed=seed,
            n_features=n_features + len(interacting_features),
            n_samples=n_samples,
            weight_range=weight_range,
            weights=flat_weights,
            **kwargs,
        )
        self.interacting_features = interacting_features
        self.zero_likelihood = zero_likelihood
        self.make_cat()
        self.subset_attribute = [
            "interacting_features",
            "zero_likelihood",
        ] + self.subset_attribute
        self.cat_features = [x[0] for x in interacting_features]

    def make_cat(self) -> None:
        """
        Modifies the dataset to incorporate the specified categorical-to-continuous feature interactions.

        The method ensures that the dataset is correctly modified to reflect the specified feature
        interactions and their impact on weights and samples.
        """
        fill_weights = list(range(self.n_features - len(self.interacting_features)))
        for impacts, impacted in self.interacting_features:
            fill_weights.append(impacted)
        fill_weights.sort()

        for impacts, impacted in self.interacting_features:
            impacts_indices = [i for i, x in enumerate(fill_weights) if x == impacts]
            impacted_indices = [i for i, x in enumerate(fill_weights) if x == impacted]
            assert len(impacts_indices) == 1
            assert len(impacted_indices) == 2
            new_col = (torch.rand((len(self), 1)) > self.zero_likelihood).to(
                self.samples.dtype
            )
            self.samples[:, impacts_indices] = new_col
            self.samples[:, impacted_indices[1]] = self.samples[:, impacted_indices[0]]

        self.weighted_samples = self.samples * self.weights

        weights = [math.inf] * (self.n_features - len(self.interacting_features))

        for impacts, impacted in self.interacting_features:
            impacts_indices = [i for i, x in enumerate(fill_weights) if x == impacts]
            impacted_indices = [i for i, x in enumerate(fill_weights) if x == impacted]
            col_mask = self.samples[:, impacts_indices]
            ws1 = self.weighted_samples[:, [impacted_indices[0]]] * (1 - col_mask)
            ws2 = self.weighted_samples[:, [impacted_indices[1]]] * (col_mask)
            new_ws = ws1 + ws2
            self.weighted_samples[:, [impacted_indices[0]]] = new_ws
            self.weighted_samples[:, [impacted_indices[1]]] = new_ws
            weights[impacted] = (
                self.weights[impacted_indices[0]].item(),
                self.weights[impacted_indices[1]].item(),
            )

        for id, w in enumerate(weights):
            if w == math.inf:
                index = [i for i, x in enumerate(fill_weights) if x == id]
                assert len(index) == 1
                weights[id] = self.weights[index[0]].item()

        mask = [0]
        for idx in range(1, len(fill_weights)):
            if fill_weights[idx] != fill_weights[idx - 1]:
                mask.append(idx)

        self.samples = self.samples[:, mask]
        self.weighted_samples = self.weighted_samples[:, mask]
        self.weights = weights
        self.n_features = self.n_features - len(self.interacting_features)
        self.labels = self.weighted_samples.sum(dim=1) + self.label_noise

    def _get_flat_weights(
        self, weights: Optional[List[float]]
    ) -> Optional[torch.Tensor]:
        """
        Convert the weights into a flat tensor.

        This method takes a list of weights, which can be tuples representing ranges,
        and converts them into a flat tensor. If the input weights are None, the method returns None.

        Args:
            weights (list | NoneType): List of weights or None if weights are not specified.

        Returns:
            torch.Tensor | NoneType: Flat tensor of weights if weights are provided, else None.
        """
        flat_weights = []

        if weights is None:
            return None

        for w in weights:
            if type(w) == tuple:
                flat_weights.append(w[0])
                flat_weights.append(w[1])
            else:
                flat_weights.append(w)

        return torch.tensor(flat_weights)

    def generate_model(self) -> torch.nn.Module:
        """
        Generates a neural network model for interacting features analysis.

        This method instantiates and returns a neural network model specifically designed
        for analyzing datasets with interacting features. The model is configured using the
        specified number of features, feature weights, and interacting features information.

        Returns:
            model.InteractingFeaturesNN: An instance of the InteractingFeaturesNN class, representing
                the neural network model designed for interacting features analysis.
        """
        from xaiunits.model.interaction_features import InteractingFeaturesNN

        return InteractingFeaturesNN(
            self.n_features,
            self.weights,
            self.interacting_features,
        )


if __name__ == "__main__":
    data = InteractingFeatureDataset()
    model = data.generate_model()
    print(data[0])
    print(data.weights)
    print(model(data[0][0]))
