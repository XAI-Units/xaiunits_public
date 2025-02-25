import torch
from xaiunits.datagenerator import BaseFeaturesDataset
from torch import nn
from typing import Optional, Tuple, Any, Callable


class UncertaintyAwareDataset(BaseFeaturesDataset):
    """
    A dataset designed to investigate how feature attribution methods treat inputs
    features that equally impact model prediction.

    In particular, uncertainty/common features are input features that contribution equally
    to output class prediction. feature attribution method is expected not to assign any attribution
    score to these uncertainty inputs. The last columns of the dataset are uncertainty/common features.

    Users can also pass in their own weights if they wish to test for more complex uncertainty
    behavior, e.g. uncertainty/common feature only contribution equally to a subset of output classes.

    Inherits from:
        BaseFeaturesDataset: Base class for generating datasets with features and labels.

    Attributes:
        weighted_samples (torch.Tensor): Samples multiplied by weights.
        weights (torch.Tensor): Weights matrix for feature transformation.
        labels (torch.Tensor): Softmax output of weighted samples.
    """

    def __init__(
        self,
        n_features: int = 5,
        weights: Optional[torch.Tensor] = None,
        common_features: int = 1,
        seed: int = 0,
        n_samples: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an UncertaintyAwareDataset object.

        Args:
            n_features (int): Number of features in the dataset. Defaults to 5.
            weights (torch.Tensor, optional): Custom weights matrix for feature transformation. Defaults to None.
            common_features (int): Number of uncertainty/common features present. Defaults to 1.
            seed (int): Seed for random number generation. Defaults to 0.
            n_samples (int): Number of samples in the dataset. Defaults to 10.
            **kwargs: Additional keyword arguments for the base class constructor.
        """
        weights = self._create_weights(n_features, weights, common_features)
        super().__init__(seed, n_features=n_features, n_samples=n_samples, **kwargs)

        self.common_features = common_features
        self.weighted_samples = self.samples @ weights.T
        self.weights = weights
        self.labels = torch.max(self.weighted_samples, dim=1)[1]
        self.mask = torch.ones_like(self.samples)
        self.mask[:, list(range(-1, -common_features - 1, -1))] = 0.0
        self.features = "samples"
        self.ground_truth_attribute = "mask"
        self.subset_data = ["samples", "weighted_samples", "mask"]
        self.subset_attribute = [
            "weights",
            "common_features",
            "default_metric",
            "generate_model",
        ] + self.subset_attribute

    def _create_weights(
        self,
        n_features: int,
        weights: Optional[torch.Tensor],
        common_features: int,
    ) -> torch.Tensor:
        """
        Creates weights matrix based on common features.

        Args:
            n_features (int): Number of features in the dataset.
            weights (torch.Tensor): Custom weights matrix for feature transformation.
            common_features (list): List of indices representing common features.

        Returns:
            weights (torch.Tensor): Weights matrix for feature transformation.
        """
        if weights is not None:
            print("Ignoring Common Feature Arg as weights is provided")
            assert weights.shape[1] == n_features
            return weights

        # create gen weights
        assert type(common_features) is int
        assert common_features is not None
        assert common_features < n_features

        weights = torch.eye(n_features)
        for common_feature in range(common_features):
            col_weights = torch.ones((n_features,))
            weights[:, -(common_feature + 1)] = col_weights
        weights = weights[:-common_features, :]

        return weights

    def __getitem__(
        self, idx: int, others: list[str] = ["ground_truth_attribute"]
    ) -> Tuple[Any, ...]:
        """
        Retrieve a sample and its associated label by index.

        Args:
            idx (int): Index of the sample to retrieve.
            others (list): Additional items to retrieve. Defaults to ["ground_truth_attribute"].

        Returns:
            tuple: Tuple containing the sample and its label.
        """
        return super().__getitem__(idx, others=others)

    def generate_model(self, softmax_layer: bool = True) -> torch.nn.Module:
        """
        Generates an UncertaintyNN model based on the dataset.

        Returns:
            model.UncertaintyNN: Instance of UncertaintyNN model.
        """
        from xaiunits.model import UncertaintyNN

        return UncertaintyNN(self.samples.shape[1], self.weights, softmax_layer)

    @property
    def default_metric(self) -> Callable:
        """
        The default metric for evaluating the performance of explanation methods applied
        to this dataset.

        For this dataset, the default metric is modified Mean Squared Error (MSE) loss function.
        This metric measures the MSE for common/uncertainty features which should be 0.

        Returns:
            type: A class that wraps around the default metric to be instantiated
                within the pipeline.
        """
        from xaiunits.metrics import wrap_metric

        return wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(
                x[:, list(range(-1, -self.common_features - 1, -1))].flatten(1), dim=1
            ),
        )


if __name__ == "__main__":

    # w = torch.Tensor([[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().T
    w = None
    data = UncertaintyAwareDataset(4, w)
    print(data.weights)
    model = data.generate_model().float()
    print(data[:1])
    print(data[:1][1])
    print(model(data[:1][0]))
